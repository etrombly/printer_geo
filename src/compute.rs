use crate::geo::*;
use std::{default::Default, iter::Map, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    descriptor::{
        descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract,
    },
    device::{Device, DeviceExtensions, Features, Queue},
    instance::{Instance, InstanceExtensions, PhysicalDevice},
    pipeline::ComputePipeline,
    sync::GpuFuture,
};

pub struct Vk {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}
#[derive(Default, Copy, Clone)]
pub struct Point_vk {
    pub position: [f32; 3],
}
#[derive(Default, Copy, Clone)]
pub struct Triangle_vk {
    pub p1: Point_vk,
    pub p2: Point_vk,
    pub p3: Point_vk,
}
#[derive(Default, Copy, Clone)]
pub struct Line_vk {
    pub p1: Point_vk,
    pub p2: Point_vk,
}

pub fn init_vk() -> Vk {
    let instance = Instance::new(None, &InstanceExtensions::none(), None)
        .expect("failed to create instance");
    let physical = PhysicalDevice::enumerate(&instance)
        .next()
        .expect("no device available");
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics())
        .expect("couldn't find a graphical queue family");
    let (device, mut queues) = {
        Device::new(
            physical,
            &Features::none(),
            &DeviceExtensions::none(),
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };
    let queue = queues.next().unwrap();

    Vk { device, queue }
}

pub fn compute_bbox(tris: &Vec<Triangle_vk>, vk: &Vk) -> Vec<Line3d> {
    let mut results = Vec::new();
    let shader = cs::Shader::load(vk.device.clone())
        .expect("failed to create shader module");
    let compute_pipeline = Arc::new(
        ComputePipeline::new(
            vk.device.clone(),
            &shader.main_entry_point(),
            &(),
        )
        .expect("failed to create compute pipeline"),
    );
    let chunks = tris.chunks_exact(1024);
    for chunk in chunks {
    let dest_content = (0..tris.len()).map(|_| Line_vk {
        p1: Default::default(),
        p2: Default::default(),
    });
    let mut src_content: [Triangle_vk;1024] = [Default::default(); 1024];

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
    let out_layout =
        compute_pipeline.layout().descriptor_set_layout(1).unwrap();
    let source = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        BufferUsage::all(),
        false,
        src_content.iter(),
    )
    .expect("failed to create buffer");
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(source.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let dest = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        BufferUsage::all(),
        false,
        dest_content,
    )
    .expect("failed to create buffer");
    let out_set = Arc::new(
        PersistentDescriptorSet::start(out_layout.clone())
            .add_buffer(dest.clone())
            .unwrap()
            .build()
            .unwrap(),
    );
    let mut builder =
        AutoCommandBufferBuilder::new(vk.device.clone(), vk.queue.family())
            .unwrap();
    builder
        .dispatch(
            [1024, 1, 1],
            compute_pipeline.clone(),
            (set.clone(), out_set.clone()),
            (),
        )
        .unwrap();
    let command_buffer = builder.build().unwrap();
    let finished = command_buffer.execute(vk.queue.clone()).unwrap();
    finished
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
    let dest_content = dest.read().unwrap();
    }
    results
}

pub fn to_tri_vk(tris: &Vec<Triangle3d>) -> Vec<Triangle_vk> {
    tris.iter()
        .map(|tri| Triangle_vk {
            p1: Point_vk {
                position: [tri.p1.x, tri.p1.y, tri.p1.z],
            },
            p2: Point_vk {
                position: [tri.p2.x, tri.p2.y, tri.p2.z],
            },
            p3: Point_vk {
                position: [tri.p3.x, tri.p3.y, tri.p3.z],
            },
        })
        .collect()
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer Data {
    uint data[];
} buf;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    buf.data[idx] *= 12;
}"
    }
}
