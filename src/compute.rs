//! # Compute
//!
//! Module for running compute shaders on vulkan

pub use crate::geo::*;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::{sync::Arc, time::Instant};
use thiserror::Error;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    descriptor::{
        descriptor_set::PersistentDescriptorSet, pipeline_layout::PipelineLayout,
        PipelineLayoutAbstract,
    },
    device::{Device, DeviceExtensions, Features, Queue},
    instance::{
        debug::{DebugCallback, MessageSeverity, MessageType},
        Instance, InstanceExtensions, PhysicalDevice,
    },
    pipeline::ComputePipeline,
    sync::GpuFuture,
};

#[derive(Error, Debug)]
/// Error types for vulkan devices
pub enum VkError {
    #[error("Unable to create vulkan instance")]
    Instance(#[from] vulkano::instance::InstanceCreationError),
    #[error("No vulkan device available")]
    PhysicalDevice,
    #[error("Device does not support graphics operations")]
    Graphics,
    #[error("Could not create vulkan device")]
    Device(#[from] vulkano::device::DeviceCreationError),
    #[error("Failed to create compute shader")]
    Shader(#[from] vulkano::OomError),
    #[error("Failed to create compute pipeline")]
    Pipeline(#[from] vulkano::pipeline::ComputePipelineCreationError),
    #[error("No queue available for vulkan device")]
    Queue,
}

#[derive(Error, Debug)]
/// Error types for vulkan compute operations
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

/// Holds vulkan device and queue
#[cfg_attr(feature = "python", pyclass)]
pub struct Vk {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub debug_callback: Option<DebugCallback>,
    cp: Arc<ComputePipeline<PipelineLayout<drop::Layout>>>,
}

impl Vk {
    /// Create a new vulkan instance
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
        let shader = drop::Shader::load(device.clone())?;
        let cp = Arc::new(ComputePipeline::new(
            device.clone(),
            &shader.main_entry_point(),
            &(),
        )?);

        Ok(Vk {
            device,
            queue,
            debug_callback: None,
            cp,
        })
    }

    /// Create a new vulkan instance with validation layers
    pub fn new_debug() -> Result<Vk, VkError> {
        let extensions = InstanceExtensions {
            ext_debug_utils: true,
            ..InstanceExtensions::none()
        };

        let layer = "VK_LAYER_KHRONOS_validation";
        let layers = vec![layer];
        let instance = Instance::new(None, &extensions, layers)?;

        let severity = MessageSeverity {
            error: true,
            warning: true,
            information: true,
            verbose: true,
        };

        let ty = MessageType::all();

        let debug_callback = DebugCallback::new(&instance, severity, ty, |msg| {
            let severity = if msg.severity.error {
                "error"
            } else if msg.severity.warning {
                "warning"
            } else if msg.severity.information {
                "information"
            } else if msg.severity.verbose {
                "verbose"
            } else {
                panic!("no-impl");
            };

            let ty = if msg.ty.general {
                "general"
            } else if msg.ty.validation {
                "validation"
            } else if msg.ty.performance {
                "performance"
            } else {
                panic!("no-impl");
            };

            println!(
                "{} {} {}: {}",
                msg.layer_prefix, ty, severity, msg.description
            );
        })
        .ok();

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
        let shader = drop::Shader::load(device.clone())?;
        let cp = Arc::new(ComputePipeline::new(
            device.clone(),
            &shader.main_entry_point(),
            &(),
        )?);

        Ok(Vk {
            device,
            queue,
            debug_callback,
            cp,
        })
    }
}

/// Calculate intersection of points and triangles
///
/// # Arguments
///
/// * `tris` - Model to calculate intersections
/// * `points` - List of points to intersect
/// * `vk` - Vulkan instance
pub fn intersect_tris(
    tris: &[Triangle3d],
    points: &[Point3d],
    vk: &Vk,
) -> Result<Vec<Point3d>, ComputeError> {
    // load compute shader
    let compute_pipeline = vk.cp.clone();
    let layout = compute_pipeline
        .layout()
        .descriptor_set_layout(0)
        .ok_or_else(|| ComputeError::Layout)?;

    // set up ssbo buffer
    let mut usage = BufferUsage::transfer_source();
    usage.storage_buffer = true;

    // copy tris into source buffer
    let (source, source_future) =
        ImmutableBuffer::from_iter(tris.iter().copied(), usage, vk.queue.clone())?;
    source_future.then_signal_fence_and_flush()?.wait(None)?;

    // copy points into dest buffer, used for input and output because
    // the length and type of inputs and outputs are the same
    // scale z to an int so we can use atomicMax
    let dest = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        usage,
        false,
        points
            .iter()
            .map(|point| (point.pos[0], point.pos[1], (point.pos[2] * 1000.) as i32)),
    )?;

    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(source.clone())?
            .add_buffer(dest.clone())?
            .build()?,
    );
    let mut builder = AutoCommandBufferBuilder::new(vk.device.clone(), vk.queue.family())?;

    // using 32 as local workgroup size. Most documentation says to set it to 64,
    // but it ran slower that way
    builder.dispatch(
        [
            (tris.len() as u32 / 32) + 1,
            (points.len() as u32 / 32) + 1,
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

    Ok(dest_content
        .to_vec()
        .iter()
        .map(|x| Point3d::new(x.0, x.1, x.2 as f32 / 1000.))
        .collect())
}

pub fn heightmap(tris: &[Triangle3d], vk: &Vk) -> Result<Vec<f32>, ComputeError> {
    // load compute shader
    let shader = drop_single::Shader::load(vk.device.clone())?;
    let now = Instant::now();
    let compute_pipeline = Arc::new(ComputePipeline::new(
        vk.device.clone(),
        &shader.main_entry_point(),
        &(),
    )?);
    println!("pipeline {:?}", now.elapsed());
    let now = Instant::now();
    let layout = compute_pipeline
        .layout()
        .descriptor_set_layout(0)
        .ok_or(ComputeError::Layout)?;
    println!("layout {:?}", now.elapsed());

    // set up ssbo buffer
    let mut usage = BufferUsage::transfer_source();
    usage.storage_buffer = true;

    let now = Instant::now();
    // copy tris into source buffer
    let (source, source_future) =
        ImmutableBuffer::from_iter(tris.iter().copied(), usage, vk.queue.clone())?;
    source_future.then_signal_fence_and_flush()?.wait(None)?;
    println!("copy tris {:?}", now.elapsed());
    let now = Instant::now();
    let mut dest_dummy: Vec<f32> = Vec::with_capacity(tris.len());
    unsafe {
        dest_dummy.set_len(tris.len());
    }
    // copy points into dest buffer, used for input and output because
    // the length and type of inputs and outputs are the same
    let dest = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        usage,
        false,
        dest_dummy.iter().copied(),
    )?;
    println!("copy dest {:?}", now.elapsed());
    let now = Instant::now();
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(source.clone())?
            .add_buffer(dest.clone())?
            .build()?,
    );
    println!("create set {:?}", now.elapsed());
    let now = Instant::now();
    let mut results = Vec::new();
    for i in 0..10000 {
        let mut builder = AutoCommandBufferBuilder::new(vk.device.clone(), vk.queue.family())?;
        let i = i as f32 / 100.;
        let push_constants = drop_single::ty::PushConstantData { x: i, y: i };

        builder.dispatch(
            [(tris.len() as u32 / 64) + 1, 1, 1],
            compute_pipeline.clone(),
            set.clone(),
            push_constants,
        )?;
        let command_buffer = builder.build()?;
        let finished = command_buffer.execute(vk.queue.clone())?;
        finished.then_signal_fence_and_flush()?.wait(None)?;

        let dest_content = dest.read()?;
    }
    println!("10000 {:?}", now.elapsed());
    Ok(results)
}

/// Split triangles into columns they are contained in
///
/// # Arguments
///
/// * `tris` - List of triangles to partition
/// * `columns` - List of bounding boxes to partition with
/// * `vk` - Vulkan instance
pub fn partition_tris(
    tris: &[Triangle3d],
    columns: &[Line3d],
    vk: &Vk,
) -> Result<Vec<Vec<Triangle3d>>, ComputeError> {
    // load compute shader
    let shader = partition::Shader::load(vk.device.clone())?;
    let compute_pipeline = Arc::new(ComputePipeline::new(
        vk.device.clone(),
        &shader.main_entry_point(),
        &(),
    )?);

    let layout = compute_pipeline
        .layout()
        .descriptor_set_layout(0)
        .ok_or(ComputeError::Layout)?;

    // set up ssbo buffer
    let mut usage = BufferUsage::transfer_source();
    usage.storage_buffer = true;

    let (source, source_future) =
        ImmutableBuffer::from_iter(tris.iter().copied(), usage, vk.queue.clone())?;

    source_future.then_signal_fence_and_flush()?.wait(None)?;

    let (columns_buffer, columns_future) =
        ImmutableBuffer::from_iter(columns.iter().copied(), usage, vk.queue.clone())?;

    columns_future.then_signal_fence_and_flush()?.wait(None)?;

    // booleans in glsl are 32 bits, so using a bitmask here to hold what columns
    // each triangle is partitioned into instead. We can pack 32 bools into a
    // u32, and need one set of columns per tri
    let count = ((tris.len() as f32 - 1.) + ((columns.len() as f32 - 1.) * tris.len() as f32) / 32.)
        .ceil() as usize;
    // save some time by not initializing the vec, wasn't sure if this would be a
    // problem setting up the buffer but it seems to be working well
    let mut dest_content: Vec<u32> = Vec::with_capacity(count);
    unsafe {
        dest_content.set_len(count);
    }

    let dest = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        usage,
        false,
        dest_content.iter().copied(),
    )?;

    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(source)?
            .add_buffer(columns_buffer)?
            .add_buffer(dest.clone())?
            .build()?,
    );
    let mut builder = AutoCommandBufferBuilder::new(vk.device.clone(), vk.queue.family())?;
    // using 32 as local workgroup size. Most documentation says to set it to 64,
    // but it ran slower that way
    builder.dispatch(
        [
            (tris.len() as u32 / 32) + 1,
            (columns.len() as u32 / 32) + 1,
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
    let dest_content = dest_content.to_vec();

    // have to unpack the bitmask to determine what columns each tri belongs in
    let result = (0..columns.len())
        .map(|column| {
            (0..tris.len())
                .filter_map(|tri| {
                    let index = (tri + (column * tris.len())) / 32;
                    let pos = (tri + (column * tris.len())) % 32;
                    if dest_content[index] & (1 << pos) == (1 << pos) {
                        Some(tris[tri])
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    Ok(result)
}

mod drop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/drop.comp"
    }
}

mod drop_single {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/drop_single.comp"
    }
}

mod partition {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/partition.comp"
    }
}
