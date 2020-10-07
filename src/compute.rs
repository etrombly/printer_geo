//! # Compute
//!
//! Module for running compute shaders on vulkan

pub use crate::geo::*;
pub use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{
    util::*,
    vk,
    vk::{
        DescriptorPoolCreateInfo, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
        PhysicalDeviceMemoryProperties, Queue, ShaderStageFlags, WriteDescriptorSet,
    },
    Device, Entry, Instance,
};
#[cfg(feature = "python")]
use pyo3::prelude::*;
use std::{
    ffi::CString,
    io::Cursor,
    mem::{self, align_of},
    os::raw::c_void,
    sync::Arc,
};
use thiserror::Error;

#[derive(Error, Debug)]
/// Error types for vulkan devices
pub enum VkError {
    #[error("Unable to load ash")]
    Loading(#[from] ash::LoadingError),
    #[error("Unable to get vulkan instance")]
    Instance(#[from] ash::InstanceError),
    #[error("Vulkan error")]
    VkResult(#[from] ash::vk::Result),
    #[error("Vulkan error")]
    Graphics,
    #[error("Could not create vulkan device")]
    Device,
}

#[derive(Error, Debug)]
/// Error types for vulkan compute operations
pub enum ComputeError {
    #[error("Vulkan error")]
    VkResult(#[from] ash::vk::Result),
    #[error("Unable to find suitable memory type")]
    MemoryType,
    #[error("Null string")]
    Nul(#[from] std::ffi::NulError),
    #[error("IO error")]
    IO(#[from] std::io::Error),
    //#[error("Failed to create compute shader")]
    //Shader(#[from] vulkano::OomError),
    //#[error("Failed to create compute pipeline")]
    //Pipeline(#[from] vulkano::pipeline::ComputePipelineCreationError),
    //#[error("Could not allocate graphics memory")]
    //Malloc(#[from] vulkano::memory::DeviceMemoryAllocError),
    //#[error("Vulkan flush error")]
    //Flush(#[from] vulkano::sync::FlushError),
    //#[error("Error creating persistent descriptor set")]
    //Set(#[from] vulkano::descriptor::descriptor_set::PersistentDescriptorSetError),
    //#[error("Error building persistent descriptor set")]
    //BuildSet(#[from] vulkano::descriptor::descriptor_set::PersistentDescriptorSetBuildError),
    //#[error("Error dispatching compute shader")]
    //Dispatch(#[from] vulkano::command_buffer::DispatchError),
    //#[error("Error building command buffer")]
    //CommandBuild(#[from] vulkano::command_buffer::BuildError),
    //#[error("Error executing command buffer")]
    //CommandBufferExec(#[from] vulkano::command_buffer::CommandBufferExecError),
    //#[error("Error locking buffer for reading")]
    //ReadLock(#[from] vulkano::buffer::cpu_access::ReadLockError),
    //#[error("Could not retreive descriptor set layout")]
    //Layout,
}

/// Holds vulkan device and queue
#[cfg_attr(feature = "python", pyclass)]
pub struct Vk {
    pub instance: Instance,
    pub device: Device,
    pub device_memory_properties: PhysicalDeviceMemoryProperties,
    pub queue: Queue,
    /*pub debug_callback: Option<DebugCallback>,
     *cp: Arc<ComputePipeline<PipelineLayout<drop::Layout>>>, */
}

impl Vk {
    /// Create a new vulkan instance
    pub unsafe fn new() -> Result<Vk, VkError> {
        let entry = Entry::new()?;
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_version(1, 0, 0),
            ..Default::default()
        };
        let create_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            ..Default::default()
        };
        let instance = entry.create_instance(&create_info, None)?;
        let pdevices = instance.enumerate_physical_devices()?;
        let (pdevice, queue_family_index) = pdevices
            .iter()
            .map(|pdevice| {
                instance
                    .get_physical_device_queue_family_properties(*pdevice)
                    .iter()
                    .enumerate()
                    .filter_map(|(index, ref info)| {
                        if info.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                            Some((*pdevice, index))
                        } else {
                            None
                        }
                    })
                    .next()
            })
            .filter_map(|v| v)
            .next()
            .ok_or(VkError::Device)?;
        let queue_family_index = queue_family_index as u32;
        let device_extension_names_raw = [];
        let features = vk::PhysicalDeviceFeatures { ..Default::default() };
        let priorities = [1.0];

        let queue_info = [vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)
            .build()];

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_extension_names_raw)
            .enabled_features(&features);

        let device: Device = instance.create_device(pdevice, &device_create_info, None)?;
        let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);

        let queue = device.get_device_queue(queue_family_index as u32, 0);
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);

        let pool = device.create_command_pool(&pool_create_info, None)?;

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(2)
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY);

        let command_buffers = device.allocate_command_buffers(&command_buffer_allocate_info)?;
        let intersect_command_buffer = command_buffers[0];
        let partition_command_buffer = command_buffers[1];

        let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

        let intersect_commands_reuse_fence = device.create_fence(&fence_create_info, None)?;
        let partition_commands_reuse_fence = device.create_fence(&fence_create_info, None)?;
        /*
        let instance = Instance::new(None, &InstanceExtensions::none(), None)?;
        let physical = PhysicalDevice::enumerate(&instance)
            .next()
            .ok_or(VkError::PhysicalDevice)?;
        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics())
            .ok_or(VkError::Graphics)?;
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
        let queue = queues.next().ok_or(VkError::Queue)?;
        let shader = drop::Shader::load(device.clone())?;
        let cp = Arc::new(ComputePipeline::new(
            device.clone(),
            &shader.main_entry_point(),
            &(),
        )?);
        */

        Ok(Vk {
            instance,
            device,
            device_memory_properties,
            queue,
        })
    }

    /*
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
            .ok_or(VkError::PhysicalDevice)?;
        let queue_family = physical
            .queue_families()
            .find(|&q| q.supports_graphics())
            .ok_or(VkError::Graphics)?;
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
        let queue = queues.next().ok_or(VkError::Queue)?;
        let shader = drop::Shader::load(device.clone())?;
        let cp = Arc::new(ComputePipeline::new(
            device.clone(),
            &shader.main_entry_point(),
            &(),
        )?);

        Ok(Vk {
        })
    }
    */
}

/*
impl Drop for Vk {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().expect("error waiting for idle");
            //self.device
            //    .destroy_semaphore(self.present_complete_semaphore, None);
            //self.device
            //    .destroy_semaphore(self.rendering_complete_semaphore, None);
            //self.device
            //    .destroy_fence(self.draw_commands_reuse_fence, None);
            //self.device
            //    .destroy_fence(self.setup_commands_reuse_fence, None);
            //self.device.free_memory(self.depth_image_memory, None);
            //self.device.destroy_image_view(self.depth_image_view, None);
            //self.device.destroy_image(self.depth_image, None);
            //for &image_view in self.present_image_views.iter() {
            //    self.device.destroy_image_view(image_view, None);
            //}
            //self.device.destroy_command_pool(self.pool, None);
            //self.swapchain_loader
            //    .destroy_swapchain(self.swapchain, None);
        //self.device.destroy_device(None);
            //self.surface_loader.destroy_surface(self.surface, None);
            //self.debug_utils_loader
            //    .destroy_debug_utils_messenger(self.debug_call_back, None);
        //self.instance.destroy_instance(None);
        }
    }
}
*/

/// Calculate intersection of points and triangles
///
/// # Arguments
///
/// * `tris` - Model to calculate intersections
/// * `points` - List of points to intersect
/// * `vk` - Vulkan instance
pub fn intersect_tris(tris: &[Triangle3d], points: &[Point3d], vk: &Vk) -> Result<Vec<Point3d>, ComputeError> {
    let descriptor_sizes = [
        vk::DescriptorPoolSize {
            ty: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        },
        vk::DescriptorPoolSize {
            ty: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
        },
    ];
    let descriptor_pool_info = DescriptorPoolCreateInfo::builder()
        .pool_sizes(&descriptor_sizes)
        .max_sets(1);

    let descriptor_pool = unsafe { vk.device.create_descriptor_pool(&descriptor_pool_info, None)? };
    let desc_layout_bindings = [
        DescriptorSetLayoutBinding {
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
        DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1,
            stage_flags: ShaderStageFlags::COMPUTE,
            ..Default::default()
        },
    ];
    let descriptor_info = DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings);

    let desc_set_layouts = unsafe { [vk.device.create_descriptor_set_layout(&descriptor_info, None)?] };

    let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&desc_set_layouts);
    let descriptor_sets = unsafe { vk.device.allocate_descriptor_sets(&desc_alloc_info)? };

    let tri_buffer_info = vk::BufferCreateInfo {
        size: std::mem::size_of_val(&tris) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let tri_buffer = unsafe { vk.device.create_buffer(&tri_buffer_info, None)? };
    let tri_buffer_memory_req = unsafe { vk.device.get_buffer_memory_requirements(tri_buffer) };
    let tri_buffer_memory_index = find_memorytype_index(
        &tri_buffer_memory_req,
        &vk.device_memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or(ComputeError::MemoryType)?;
    let tri_allocate_info = vk::MemoryAllocateInfo {
        allocation_size: tri_buffer_memory_req.size,
        memory_type_index: tri_buffer_memory_index,
        ..Default::default()
    };
    let tri_buffer_memory = unsafe { vk.device.allocate_memory(&tri_allocate_info, None)? };
    let tri_ptr: *mut c_void = unsafe {
        vk.device.map_memory(
            tri_buffer_memory,
            0,
            tri_buffer_memory_req.size,
            vk::MemoryMapFlags::empty(),
        )?
    };
    // TODO: this alignment isn't right
    let mut tri_slice = unsafe { Align::new(tri_ptr, align_of::<u32>() as u64, tri_buffer_memory_req.size) };
    tri_slice.copy_from_slice(&tris);
    unsafe { vk.device.unmap_memory(tri_buffer_memory) };
    unsafe { vk.device.bind_buffer_memory(tri_buffer, tri_buffer_memory, 0)? };

    let point_buffer_info = vk::BufferCreateInfo {
        size: std::mem::size_of_val(&points) as u64,
        usage: vk::BufferUsageFlags::STORAGE_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let point_buffer = unsafe { vk.device.create_buffer(&point_buffer_info, None)? };
    let point_buffer_memory_req = unsafe { vk.device.get_buffer_memory_requirements(point_buffer) };
    let point_buffer_memory_index = find_memorytype_index(
        &point_buffer_memory_req,
        &vk.device_memory_properties,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .ok_or(ComputeError::MemoryType)?;
    let point_allocate_info = vk::MemoryAllocateInfo {
        allocation_size: point_buffer_memory_req.size,
        memory_type_index: point_buffer_memory_index,
        ..Default::default()
    };
    let point_buffer_memory = unsafe { vk.device.allocate_memory(&point_allocate_info, None)? };
    let point_ptr: *mut c_void = unsafe {
        vk.device.map_memory(
            point_buffer_memory,
            0,
            point_buffer_memory_req.size,
            vk::MemoryMapFlags::empty(),
        )?
    };
    // TODO: this alignment isn't right
    let mut point_slice = unsafe { Align::new(point_ptr, align_of::<u32>() as u64, point_buffer_memory_req.size) };
    point_slice.copy_from_slice(&points);
    unsafe { vk.device.unmap_memory(point_buffer_memory) };
    unsafe { vk.device.bind_buffer_memory(point_buffer, point_buffer_memory, 0)? };

    let tri_buffer_descriptor = vk::DescriptorBufferInfo {
        buffer: tri_buffer,
        offset: 0,
        range: mem::size_of_val(&tris) as u64,
    };

    let point_buffer_descriptor = vk::DescriptorBufferInfo {
        buffer: point_buffer,
        offset: 0,
        range: mem::size_of_val(&points) as u64,
    };

    let write_desc_sets = [
        WriteDescriptorSet {
            dst_set: descriptor_sets[0],
            descriptor_count: 1,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &tri_buffer_descriptor,
            ..Default::default()
        },
        WriteDescriptorSet {
            dst_set: descriptor_sets[0],
            dst_binding: 1,
            descriptor_count: 1,
            descriptor_type: DescriptorType::STORAGE_BUFFER,
            p_buffer_info: &point_buffer_descriptor,
            ..Default::default()
        },
    ];
    unsafe { vk.device.update_descriptor_sets(&write_desc_sets, &[]) };

    let mut drop_spv_file = Cursor::new(&include_bytes!("../shaders/drop.spv")[..]);
    let drop_code = read_spv(&mut drop_spv_file)?;
    let drop_shader_info = vk::ShaderModuleCreateInfo::builder().code(&drop_code);

    let vertex_shader_module = unsafe { vk.device.create_shader_module(&drop_shader_info, None)? };

    let layout_create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&desc_set_layouts);

    let pipeline_layout = unsafe { vk.device.create_pipeline_layout(&layout_create_info, None)? };

    let shader_entry_name = CString::new("main")?;

    /*
    // load compute shader
    let compute_pipeline = vk.cp.clone();
    let layout = compute_pipeline
        .layout()
        .descriptor_set_layout(0)
        .ok_or(ComputeError::Layout)?;

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
            .add_buffer(source)?
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
        */
    Ok(Vec::new())
}

/*
pub fn heightmap(tris: &[Triangle3d], vk: &Vk) -> Result<Vec<f32>, ComputeError> {
    // load compute shader
    let shader = drop_single::Shader::load(vk.device.clone())?;

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

    // copy tris into source buffer
    let (source, source_future) =
        ImmutableBuffer::from_iter(tris.iter().copied(), usage, vk.queue.clone())?;
    source_future.then_signal_fence_and_flush()?.wait(None)?;

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

    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(source)?
            .add_buffer(dest.clone())?
            .build()?,
    );

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
    let dest_content: Vec<u32> = vec![0u32; count];

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
*/

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    // Try to find an exactly matching memory flag
    let best_suitable_index = find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
        property_flags == flags
    });
    if best_suitable_index.is_some() {
        return best_suitable_index;
    }
    // Otherwise find a memory flag that works
    find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
        property_flags & flags == flags
    })
}

pub fn find_memorytype_index_f<F: Fn(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags) -> bool>(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
    f: F,
) -> Option<u32> {
    let mut memory_type_bits = memory_req.memory_type_bits;
    for (index, ref memory_type) in memory_prop.memory_types.iter().enumerate() {
        if memory_type_bits & 1 == 1 && f(memory_type.property_flags, flags) {
            return Some(index as u32);
        }
        memory_type_bits >>= 1;
    }
    None
}
