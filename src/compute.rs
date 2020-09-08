pub use crate::geo_vulkan::*;
use rayon::prelude::*;
use std::{ffi::CStr, fs::File, io::Read, sync::Arc};
use thiserror::Error;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, ImmutableBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    descriptor::{
        descriptor::{DescriptorBufferDesc, DescriptorDesc, DescriptorDescTy, ShaderStages},
        descriptor_set::PersistentDescriptorSet,
        pipeline_layout::{PipelineLayoutDesc, PipelineLayoutDescPcRange},
        PipelineLayoutAbstract,
    },
    device::{Device, DeviceExtensions, Features, Queue},
    instance::{Instance, InstanceExtensions, PhysicalDevice},
    pipeline::{shader::ShaderModule, ComputePipeline},
    sync::GpuFuture,
};
use vulkano::instance;
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use std::include_bytes;

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
    pub debug_callback: Option<DebugCallback>,
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

        Ok(Vk { device, queue, debug_callback: None })
    }

    pub fn new_debug() -> Result<Vk, VkError> {
        let extensions = InstanceExtensions {
            ext_debug_utils: true,
            ..InstanceExtensions::none()
        };

        println!("List of Vulkan debugging layers available to use:");
    let mut layers = instance::layers_list().unwrap();
    while let Some(l) = layers.next() {
        println!("\t{}", l.name());
    }

        let layer = "VK_LAYER_KHRONOS_validation";
        let layers = vec![layer];
        let instance =
        Instance::new(None, &extensions, layers)?;

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

        Ok(Vk { device, queue, debug_callback })
    }
}

pub fn compute_drop(
    tris: &[TriangleVk],
    dest_content: &[PointVk],
    vk: &Vk,
) -> Result<Vec<PointVk>, ComputeError> {
    let shader = drop::Shader::load(vk.device.clone())?;

    /*
    let shader = {
        //let mut f = File::open("shaders/drop.spv").expect("Can't find file shaders/drop.spv");
        let v = include_bytes!("../shaders/drop.spv");
        //let mut v = vec![];
        //f.read_to_end(&mut v).unwrap();
        // Create a ShaderModule on a device the same Shader::load does it.
        // NOTE: You will have to verify correctness of the data by yourself!
        unsafe { ShaderModule::new(vk.device.clone(), v) }.unwrap()
    };

    let shader_main = unsafe {
        shader.compute_entry_point(
            CStr::from_bytes_with_nul_unchecked(b"main\0"),
            DropLayout(ShaderStages {
                compute: true,
                ..ShaderStages::none()
            }),
        )
    };
    */

    let compute_pipeline = Arc::new(ComputePipeline::new(
        vk.device.clone(),
        &shader.main_entry_point(),
        //&shader_main,
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
            (tris.len() as f32 / 32.).ceil() as u32,
            (dest_content.len() as f32 / 32.).ceil() as u32,
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

pub fn partition_tris(
    tris: &[TriangleVk],
    columns: &[LineVk],
    vk: &Vk,
) -> Result<Vec<Vec<TriangleVk>>, ComputeError> {
    let shader = partition::Shader::load(vk.device.clone())?;
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

    let (columns_buffer, columns_future) =
        ImmutableBuffer::from_iter(columns.iter().copied(), usage, vk.queue.clone())?;

    columns_future.then_signal_fence_and_flush()?.wait(None)?;

    // Have to multiply by 4 since vulkan booleans are u32
    let mut dest_content = Vec::with_capacity(tris.len() * columns.len() * 4);
    unsafe {
        dest_content.set_len(tris.len() * columns.len() * 4);
    }
    let dest = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        usage,
        false,
        dest_content.iter().copied(),
    )?;

    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(source.clone())?
            .add_buffer(columns_buffer.clone())?
            .add_buffer(dest.clone())?
            .build()?,
    );
    let mut builder = AutoCommandBufferBuilder::new(vk.device.clone(), vk.queue.family())?;
    builder.dispatch(
        [
            (tris.len() as u32 / 32) + 1,
            (columns.len() as u32 / 32) + 1,
            1,
        ],
        compute_pipeline.clone(),
        set.clone(),
        (),
    )?;
    let command_buffer = builder.build()?;
    let finished = command_buffer.execute(vk.queue.clone())?;
    finished.then_signal_fence_and_flush()?.wait(None)?;
    let dest_content = dest.read()?;
    let dest_content = dest_content.to_vec();
    // println!("tris: {:?} columns: {:?} length: {:?} actual length:{:?}",
    // tris.len(), columns.len(), tris.len() * columns.len(), dest_content.len());
    // println!("{:?}", dest_content);
    let result = (0..columns.len())
        .into_par_iter()
        .map(|column| {
            (0..tris.len())
                .filter_map(|tri| {
                    if dest_content[tri * 4 + (column * tris.len())] {
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

mod partition {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/partition.comp"
    }
}

/*
pub struct DropLayout(pub ShaderStages);
#[automatically_derived]
#[allow(unused_qualifications)]
impl ::core::fmt::Debug for DropLayout {
    fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
        match *self {
            DropLayout(ref __self_0_0) => {
                let mut debug_trait_builder = f.debug_tuple("Layout");
                let _ = debug_trait_builder.field(&&(*__self_0_0));
                debug_trait_builder.finish()
            },
        }
    }
}
#[automatically_derived]
#[allow(unused_qualifications)]
impl ::core::clone::Clone for DropLayout {
    #[inline]
    fn clone(&self) -> DropLayout {
        match *self {
            DropLayout(ref __self_0_0) => DropLayout(::core::clone::Clone::clone(&(*__self_0_0))),
        }
    }
}
#[allow(unsafe_code)]
unsafe impl PipelineLayoutDesc for DropLayout {
    fn num_sets(&self) -> usize { 1usize }

    fn num_bindings_in_set(&self, set: usize) -> Option<usize> {
        match set {
            0usize => Some(2usize),
            _ => None,
        }
    }

    fn descriptor(&self, set: usize, binding: usize) -> Option<DescriptorDesc> {
        match (set, binding) {
            (0usize, 0usize) => Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: Some(false),
                    storage: true,
                }),
                array_count: 1u32,
                stages: self.0.clone(),
                readonly: true,
            }),
            (0usize, 1usize) => Some(DescriptorDesc {
                ty: DescriptorDescTy::Buffer(DescriptorBufferDesc {
                    dynamic: Some(false),
                    storage: true,
                }),
                array_count: 1u32,
                stages: self.0.clone(),
                readonly: true,
            }),
            _ => None,
        }
    }

    fn num_push_constants_ranges(&self) -> usize { 0usize }

    fn push_constants_range(&self, num: usize) -> Option<PipelineLayoutDescPcRange> {
        if num != 0 || 0usize == 0 {
            None
        } else {
            Some(PipelineLayoutDescPcRange {
                offset: 0,
                size: 0usize,
                stages: ShaderStages::all(),
            })
        }
    }
}
*/