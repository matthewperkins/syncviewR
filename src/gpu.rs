//! Trace drawing on the GPU: each row is one polyline, drawn by a wgpu pipeline inside egui's
//! render pass (an egui_wgpu paint callback). Points arrive already in the row's normalized
//! device coordinates. Each segment becomes a quad a little wider than the line, and the fragment
//! shader antialiases its sides, so lines have the same width at any slope and never break up the
//! way 1-pixel hardware lines do on short segments (the min/max columns of a quiet trace).

use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};

const SHADER: &str = r#"
struct U { color: vec4<f32>, px: vec2<f32>, half: f32, pad: f32 };
@group(0) @binding(0) var<uniform> u: U;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) d: f32 };
// One instance per segment a -> b; six vertices make its quad. Work in physical pixels.
@vertex fn vs(@builtin(vertex_index) vi: u32, @location(0) a: vec2<f32>, @location(1) b: vec2<f32>) -> VOut {
    var ends = array<f32, 6>(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    var sides = array<f32, 6>(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    let pa = a / u.px;
    let pb = b / u.px;
    var dir = vec2<f32>(1.0, 0.0);
    let len = length(pb - pa);
    if (len > 1e-4) { dir = (pb - pa) / len; }
    let nrm = vec2<f32>(-dir.y, dir.x);
    let ext = u.half + 1.0;
    let e = ends[vi];
    let s = sides[vi];
    let p = mix(pa, pb, e) + dir * (2.0 * e - 1.0) * u.half + nrm * s * ext;
    var o: VOut;
    o.pos = vec4<f32>(p * u.px, 0.0, 1.0);
    o.d = s * ext;
    return o;
}
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> {
    let cover = clamp(u.half + 0.5 - abs(i.d), 0.0, 1.0);
    return vec4<f32>(u.color.rgb, u.color.a * cover);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    color: [f32; 4],
    px: [f32; 2],
    half: f32,
    pad: f32,
}

struct Slot {
    points: wgpu::Buffer,
    cap: u64,
    uniforms: wgpu::Buffer,
    bind: wgpu::BindGroup,
    n: u32,
}

pub struct LineResources {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
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
    let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("syncviewr lines"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("syncviewr lines"),
        bind_group_layouts: &[Some(&bind_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("syncviewr lines"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            // The point buffer is bound twice, the second one point later: instance i reads
            // point i as `a` and point i + 1 as `b`.
            buffers: &[
                Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2],
                }),
                Some(wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![1 => Float32x2],
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
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let srgb = rs.target_format.is_srgb();
    rs.renderer.write().callback_resources.insert(LineResources { pipeline, layout: bind_layout, slots: vec![], srgb });
}

/// Convert an egui (sRGB, gamma-space) colour for the render target.
fn rgba(c: egui::Color32, srgb: bool) -> [f32; 4] {
    let f = |v: u8| {
        let v = v as f32 / 255.0;
        if srgb { if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) } } else { v }
    };
    [f(c.r()), f(c.g()), f(c.b()), c.a() as f32 / 255.0]
}

/// Line width in physical pixels: one logical pixel (1 on ordinary screens, 2 on Retina).
/// SYNCVIEWR_LINE_PX overrides it (any width stays continuous; thinner lines just get fainter).
fn width_px(ppp: f32) -> f32 {
    static OVERRIDE: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    let o = *OVERRIDE.get_or_init(|| std::env::var("SYNCVIEWR_LINE_PX").ok().and_then(|v| v.parse().ok()).filter(|w: &f32| *w > 0.0));
    o.unwrap_or(ppp.max(1.0))
}

/// One row's polyline: points (x, y) already mapped to [-1, 1] in the row's rect.
pub struct LineStrip {
    pub slot: usize,
    pub points: Vec<[f32; 2]>,
    pub color: egui::Color32,
    /// One physical pixel in the NDC of the callback's rect (from `px_for`).
    pub px: [f32; 2],
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
        let u = Uniforms { color: rgba(self.color, r.srgb), px: self.px, half: width_px(sd.pixels_per_point) / 2.0, pad: 0.0 };
        let bytes: &[u8] = bytemuck::cast_slice(&self.points);
        if r.slots.len() <= self.slot {
            r.slots.resize_with(self.slot + 1, || None);
        }
        let need = (bytes.len() as u64).max(1024).next_power_of_two();
        let slot = &mut r.slots[self.slot];
        if slot.as_ref().is_none_or(|s| s.cap < need) {
            let uniforms = match slot.take() {
                Some(s) => s.uniforms,
                None => device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("trace uniforms"),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            };
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("trace uniforms"),
                layout: &r.layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() }],
            });
            *slot = Some(Slot {
                points: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("trace points"),
                    size: need,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                cap: need,
                uniforms,
                bind,
                n: 0,
            });
        }
        let s = slot.as_mut().unwrap();
        if !bytes.is_empty() {
            queue.write_buffer(&s.points, 0, bytes);
        }
        queue.write_buffer(&s.uniforms, 0, bytemuck::bytes_of(&u));
        s.n = self.points.len() as u32;
        vec![]
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let Some(r) = res.get::<LineResources>() else { return };
        let Some(Some(s)) = r.slots.get(self.slot) else { return };
        if s.n < 2 {
            return;
        }
        pass.set_pipeline(&r.pipeline);
        pass.set_bind_group(0, &s.bind, &[]);
        pass.set_vertex_buffer(0, s.points.slice(..));
        pass.set_vertex_buffer(1, s.points.slice(8..));
        pass.draw(0..6, 0..s.n - 1);
    }
}

/// One physical pixel, in the NDC of a callback covering `rect`.
pub fn px_for(rect: egui::Rect, ppp: f32) -> [f32; 2] {
    let (w, h) = ((rect.width() * ppp).max(1.0), (rect.height() * ppp).max(1.0));
    [2.0 / w, 2.0 / h]
}
