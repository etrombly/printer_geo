pub use crate::geo_vulkan::*;
use rayon::prelude::*;
use std::sync::Arc;
use thiserror::Error;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    descriptor::{descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract},
    device::{Device, DeviceExtensions, Features, Queue},
    instance::{Instance, InstanceExtensions, PhysicalDevice},
    pipeline::ComputePipeline,
    sync::GpuFuture,
};

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

#[derive(Error, Debug)]
pub enum ComputeError {
    #[error("Failed to create compute shader")]
    Shader(#[from] vulkano::OomError),
    #[error("Failed to create compute pipeline")]
    Pipeline(#[from] vulkano::pipeline::ComputePipelineCreationError),
    #[error("Could not allocate graphics memory")]
    Malloc(#[from] vulkano::memory::DeviceMemoryAllocError),
    #[error("Vulkan flush error")]
    Flush(#[from] vulkano::sync::FlushError),
    #[error("Error creating persistent descriptor set")]
    Set(#[from] vulkano::descriptor::descriptor_set::PersistentDescriptorSetError),
    #[error("Error building persistent descriptor set")]
    BuildSet(#[from] vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuildError),
    #[error("Error dispatching compute shader")]
    Dispatch(#[from] vulkano::command_buffer::DispatchError),
    #[error("Error building command buffer")]
    CommandBuild(#[from] vulkano::command_buffer::BuildError),
    #[error("Error executing command buffer")]
    CommandBufferExec(#[from] vulkano::command_buffer::CommandBufferExecError),
    #[error("Error locking buffer for reading")]
    ReadLock(#[from] vulkano::buffer::cpu_access::ReadLockError),
    #[error("Could not retreive descriptor set layout")]
    Layout,
}

pub struct Vk {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

impl Vk {
    pub fn new() -> Result<Vk, VkError> {
        let instance = Instance::new(None, &InstanceExtensions::none(), None)?;
        let physical = PhysicalDevice::enumerate(&instance)
            .next()
            .ok_or_else(|| VkError::PhysicalDevice)?;
        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics())
            .ok_or_else(|| VkError::Graphics)?;
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
    vk: &Vk,
) -> Result<Vec<PointVk>, ComputeError> {
    let shader = drop::Shader::load(vk.device.clone())?;
    let compute_pipeline = Arc::new(ComputePipeline::new(
        vk.device.clone(),
        &shader.main_entry_point(),
        &(),
    )?);

    let layout = compute_pipeline
        .layout()
        .descriptor_set_layout(0)
        .ok_or_else(|| ComputeError::Layout)?;

    let mut usage = BufferUsage::transfer_source();
    usage.storage_buffer = true;

    let (source, source_future) =
        ImmutableBuffer::from_iter(tris.iter().copied(), usage, vk.queue.clone())?;

    source_future.then_signal_fence_and_flush()?.wait(None)?;
    let dest = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        usage,
        false,
        dest_content.iter().copied(),
    )?;

    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(source.clone())?
            .add_buffer(dest.clone())?
            .build()?,
    );
    let mut builder = AutoCommandBufferBuilder::new(vk.device.clone(), vk.queue.family())?;
    builder.dispatch(
        [
            (tris.len() as u32 / 32) + 1,
            (dest_content.len() as u32 / 32) + 1,
            1,
        ],
        compute_pipeline.clone(),
        set,
        (),
    )?;
    let command_buffer = builder.build()?;
    let finished = command_buffer.execute(vk.queue.clone())?;
    finished.then_signal_fence_and_flush()?.wait(None)?;
    let dest_content = dest.read()?;

    Ok(dest_content.to_vec())
}

mod drop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/drop.comp"
    }
}
