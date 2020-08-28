use crate::geo::*;
use std::{default::Default, fmt, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract},
    device::{Device, DeviceExtensions, Features, Queue},
    instance::{Instance, InstanceExtensions, PhysicalDevice},
    pipeline::ComputePipeline,
    sync::GpuFuture,
};

pub struct Vk {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

#[repr(C, align(16))]
#[derive(Default, Copy, Clone)]
pub struct PointVk {
    pub position: [f32; 3],
}

impl PointVk {
    pub fn new(x: f32, y: f32, z: f32) -> PointVk {
        PointVk {
            position: [x, y, z],
        }
    }
}

impl fmt::Debug for PointVk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PointVk")
            .field("x", &self.position[0])
            .field("y", &self.position[1])
            .field("z", &self.position[2])
            .finish()
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct TriangleVk {
    pub p1: PointVk,
    pub p2: PointVk,
    pub p3: PointVk,
}

#[derive(Default, Debug, Copy, Clone)]
pub struct LineVk {
    pub p1: PointVk,
    pub p2: PointVk,
}

pub fn init_vk() -> Vk {
    let instance =
        Instance::new(None, &InstanceExtensions::none(), None).expect("failed to create instance");
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
            &DeviceExtensions {
                khr_storage_buffer_storage_class: true,
                ..DeviceExtensions::none()
            },
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };
    let queue = queues.next().unwrap();

    Vk { device, queue }
}

pub fn compute_bbox(tris: &[TriangleVk], vk: &Vk) -> Vec<LineVk> {
    let shader = cs::Shader::load(vk.device.clone()).expect("failed to create shader module");
    let compute_pipeline = Arc::new(
        ComputePipeline::new(vk.device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );
    let dest_content = (0..tris.len()).map(|_| LineVk {
        ..Default::default()
    });

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();

    let mut source_usage = BufferUsage::transfer_source();
    source_usage.storage_buffer = true;
    let (source, source_future) =
        ImmutableBuffer::from_iter(tris.iter().copied(), source_usage, vk.queue.clone())
            .expect("failed to create buffer");

    source_future
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
    let mut dest_usage = BufferUsage::transfer_destination();
    dest_usage.storage_buffer = true;
    let dest =
        CpuAccessibleBuffer::from_iter(vk.device.clone(), dest_usage, false, dest_content)
            .expect("failed to create buffer");
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(source)
            .unwrap()
            .add_buffer(dest.clone())
            .unwrap()
            .build()
            .unwrap(),
    );

    let mut builder = AutoCommandBufferBuilder::new(vk.device.clone(), vk.queue.family()).unwrap();
    builder
        .dispatch(
            [tris.len() as u32 / 128, 1, 1],
            compute_pipeline.clone(),
            set,
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
    dest_content.to_vec()
}

pub fn to_tri_vk(tris: &[Triangle3d]) -> Vec<TriangleVk> {
    tris.iter()
        .map(|tri| TriangleVk {
            p1: PointVk::new(tri.p1.x, tri.p1.y, tri.p1.z),
            p2: PointVk::new(tri.p2.x, tri.p2.y, tri.p2.z),
            p3: PointVk::new(tri.p3.x, tri.p3.y, tri.p3.z),
        })
        .collect()
}

pub fn to_line3d(line: &LineVk) -> Line3d {
    Line3d {
        p1: Point3d::new(
            line.p1.position[0],
            line.p1.position[1],
            line.p1.position[2],
        ),
        p2: Point3d::new(
            line.p2.position[0],
            line.p2.position[1],
            line.p2.position[2],
        ),
    }
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/bbox.comp" 
    }
}
