//! Trace drawing on the GPU: each row is one line strip, drawn by a wgpu pipeline inside egui's
//! render pass (an egui_wgpu paint callback). Vertices arrive already in the row's normalized
//! device coordinates; the GPU only rasterizes. On high-DPI screens the strip is drawn three times,
//! offset by one physical pixel right and down, so lines are ~2 px wide like Qt's cosmetic pens.

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
}

const SHADER: &str = r#"
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> };
@vertex fn vs(@location(0) p: vec2<f32>, @location(1) c: vec4<f32>, @location(2) off: vec2<f32>) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(p + off, 0.0, 1.0);
    o.color = c;
    return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> { return i.color; }
"#;

struct Slot {
    verts: wgpu::Buffer,
    cap: u64,
    offsets: wgpu::Buffer,
    n: u32,
    instances: u32,
}

pub struct LineResources {
    pipeline: wgpu::RenderPipeline,
    slots: Vec<Option<Slot>>,
    srgb: bool,
}

/// Create the pipeline once (call from the eframe creation context).
pub fn init(rs: &egui_wgpu::RenderState) {
    let device = &rs.device;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("syncviewr lines"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("syncviewr lines"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("syncviewr lines"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x4],
                }),
                Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![2 => Float32x2],
                }),
            ],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: rs.target_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::LineStrip, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let srgb = rs.target_format.is_srgb();
    rs.renderer.write().callback_resources.insert(LineResources { pipeline, slots: vec![], srgb });
}

/// Convert an egui (sRGB, gamma-space) colour for the render target.
fn rgba(c: egui::Color32, srgb: bool) -> [f32; 4] {
    let f = |v: u8| {
        let v = v as f32 / 255.0;
        if srgb { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } } else { v }
    };
    [f(c.r()), f(c.g()), f(c.b()), c.a() as f32 / 255.0]
}

/// One row's line strip: points (x, y) already mapped to [-1, 1] in the row's rect.
pub struct LineStrip {
    pub slot: usize,
    pub points: Vec<[f32; 2]>,
    pub color: egui::Color32,
    /// Instance offsets: none, one physical pixel right, one down (from `offsets_for`).
    pub offsets: [[f32; 2]; 3],
}

impl egui_wgpu::CallbackTrait for LineStrip {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sd: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        res: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = res.get_mut::<LineResources>() else { return vec![] };
        let color = rgba(self.color, r.srgb);
        let verts: Vec<Vertex> = self.points.iter().map(|p| Vertex { pos: *p, color }).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&verts);
        if r.slots.len() <= self.slot {
            r.slots.resize_with(self.slot + 1, || None);
        }
        let need = (bytes.len() as u64).max(1024).next_power_of_two();
        let slot = &mut r.slots[self.slot];
        if slot.as_ref().is_none_or(|s| s.cap < need) {
            *slot = Some(Slot {
                verts: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("trace vertices"),
                    size: need,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                cap: need,
                offsets: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("trace offsets"),
                    contents: &[0u8; 24],
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }),
                n: 0,
                instances: 1,
            });
        }
        let s = slot.as_mut().unwrap();
        if !bytes.is_empty() {
            queue.write_buffer(&s.verts, 0, bytes);
        }
        queue.write_buffer(&s.offsets, 0, bytemuck::cast_slice(&self.offsets));
        s.n = verts.len() as u32;
        s.instances = if sd.pixels_per_point >= 1.5 { 3 } else { 1 };
        vec![]
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let Some(r) = res.get::<LineResources>() else { return };
        let Some(Some(s)) = r.slots.get(self.slot) else { return };
        if s.n < 2 {
            return;
        }
        pass.set_pipeline(&r.pipeline);
        pass.set_vertex_buffer(0, s.verts.slice(..));
        pass.set_vertex_buffer(1, s.offsets.slice(..));
        pass.draw(0..s.n, 0..s.instances);
    }
}

/// One physical pixel right and down, in the NDC of a callback covering `rect`.
pub fn offsets_for(rect: egui::Rect, ppp: f32) -> [[f32; 2]; 3] {
    let (w, h) = ((rect.width() * ppp).max(1.0), (rect.height() * ppp).max(1.0));
    [[0.0, 0.0], [2.0 / w, 0.0], [0.0, -2.0 / h]]
}
