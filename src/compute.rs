pub use crate::geo_vulkan::*;
use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract},
    device::{Device, DeviceExtensions, Features, Queue},
    instance::{Instance, InstanceExtensions, PhysicalDevice},
    pipeline::ComputePipeline,
    sync::GpuFuture,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VkError {
    #[error("Unable to create vulkan instance")]
    Instance(#[from] vulkano::instance::InstanceCreationError),
    #[error("No vulkan device available")]
    PhysicalDevice,
    #[error("Device does not support graphics operations")]
    Graphics,
    #[error("Could not create vulkan device")]
    Device(#[from] vulkano::device::DeviceCreationError),
    #[error("No queue available for vulkan device")]
    Queue,
}

pub struct Vk {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl Vk {
    pub fn new() -> Result<Vk, VkError> {
        let instance =
            Instance::new(None, &InstanceExtensions::none(), None)?;
        let physical = PhysicalDevice::enumerate(&instance)
            .next().ok_or_else(|| VkError::PhysicalDevice)?;
        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics()).ok_or_else(|| VkError::Graphics)?;
        let (device, mut queues) = {
            Device::new(
                physical,
                &Features::none(),
                &DeviceExtensions {
                    khr_storage_buffer_storage_class: true,
                    ..DeviceExtensions::none()
                },
                [(queue_family, 0.5)].iter().cloned(),
            )?
        };
        let queue = queues.next().ok_or_else(|| VkError::Queue)?;

        Ok(Vk { device, queue })
    }
}

pub fn compute_drop(
    tris: &[TriangleVk],
    dest_content: &[PointVk],
    tool: &Tool,
    vk: &Vk,
) -> Vec<PointVk> {
    let shader = drop::Shader::load(vk.device.clone()).expect("failed to create shader module");
    let compute_pipeline = Arc::new(
        ComputePipeline::new(vk.device.clone(), &shader.main_entry_point(), &())
            .expect("failed to create compute pipeline"),
    );

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();

    let mut usage = BufferUsage::transfer_source();
    usage.storage_buffer = true;
    let (source, source_future) =
        ImmutableBuffer::from_iter(tris.iter().copied(), usage, vk.queue.clone())
            .expect("failed to create buffer");

    source_future
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();

    let dest = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        usage,
        false,
        dest_content.iter().copied(),
    )
    .expect("failed to create buffer");

    let (tool_buffer, tool_future) =
        ImmutableBuffer::from_iter(tool.points.iter().copied(), usage, vk.queue.clone())
            .expect("failed to create buffer");
    tool_future
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(source)
            .unwrap()
            .add_buffer(dest.clone())
            .unwrap()
            .add_buffer(tool_buffer)
            .unwrap()
            .build()
            .unwrap(),
    );

    let mut builder = AutoCommandBufferBuilder::new(vk.device.clone(), vk.queue.family()).unwrap();
    builder
        .dispatch(
            [
                (dest_content.len() as u32 / 64) + 1,
                tool.points.len() as u32,
                1,
            ],
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

mod drop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/drop.comp"
    }
}
