//! # Compute
//!
//! Module for running compute shaders on vulkan

pub use crate::geo::*;
use crate::vulkan::{utils::to_vec32, vkstate::VulkanState, *};
pub use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::vk;
#[cfg(feature = "python")]
use pyo3::prelude::*;
use rayon::prelude::*;
use std::{cell::RefCell, ffi::CString, rc::Rc};
use thiserror::Error;

#[derive(Error, Debug)]
/// Error types for vulkan compute operations
pub enum ComputeError {
    #[error("Vulkan error")]
    VkResult(#[from] ash::vk::Result),
    #[error("Unable to find suitable memory type")]
    NoMem,
    #[error("Unable to get buffer")]
    NoBuffer,
    #[error("No vulkan pipeline")]
    NoPipe,
    #[error("No vulkan descriptor")]
    NoDescriptor,
    #[error("Null string")]
    Nul(#[from] std::ffi::NulError),
    #[error("IO error")]
    Io(#[from] std::io::Error),
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
    vk: Rc<VulkanState>,
) -> Result<Vec<Point3d>, ComputeError> {
    let shader = to_vec32(include_bytes!("../../shaders/spirv/drop.spv").to_vec());

    // Map Buffers
    let buffer_sizes: Vec<u64> = vec![
        (tris.len() * std::mem::size_of::<Triangle3d>()) as u64,
        (points.len() * std::mem::size_of::<Point3d>()) as u64,
    ];

    let mut buffers = buffer_sizes
        .iter()
        .map(|size| vkmem::VkBuffer::new(vk.clone(), *size))
        .collect::<Vec<_>>();
    let (mem_size, offsets) = vkmem::compute_non_overlapping_buffer_alignment(&buffers);
    let memory = vkmem::VkMem::find_mem(vk.clone(), mem_size).ok_or(ComputeError::NoMem)?;

    let mbuf = buffers.get_mut(0).ok_or(ComputeError::NoBuffer)?;
    mbuf.bind(memory.mem, offsets[0]);
    memory.map_buffer(tris, mbuf);
    let mbuf = buffers.get_mut(1).ok_or(ComputeError::NoBuffer)?;
    mbuf.bind(memory.mem, offsets[1]);
    let points: Vec<_> = points
        .iter()
        .map(|point| (point.pos[0], point.pos[1], (point.pos[2] * 1000000.) as i32))
        .collect();
    memory.map_buffer(&points, mbuf);

    // Shaders
    let shader = Rc::new(RefCell::new(vkshader::VkShader::new(
        vk.clone(),
        shader,
        CString::new("main")?,
    )));

    for i in 0..buffers.len() {
        shader.borrow_mut().add_layout_binding(
            i as u32,
            1,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::COMPUTE,
        );
    }
    shader.borrow_mut().create_pipeline_layout();
    let shad_pipeline_layout = shader.borrow().pipeline.ok_or(ComputeError::NoPipe)?;
    let shad_pip_vec = vkpipeline::VkComputePipeline::new(vk.clone(), &shader.borrow());
    let mut descriptor = vkdescriptor::VkDescriptor::new(vk.clone(), shader);

    descriptor.add_pool_size(buffers.len() as u32, vk::DescriptorType::STORAGE_BUFFER);
    descriptor.create_pool(1);
    descriptor.create_set();

    let mut descriptor_set = vkdescriptor::VkWriteDescriptor::new(vk.clone());

    let mut buffers_nfos: Vec<Vec<vk::DescriptorBufferInfo>> = Vec::new();
    for i in 0..buffers.len() {
        descriptor_set.add_buffer(buffers[i].buffer, 0, buffers[i].size);
        buffers_nfos.push(vec![descriptor_set.buffer_descriptors[i]]);
        descriptor_set.add_write_descriptors(
            *descriptor.get_first_set().ok_or(ComputeError::NoDescriptor)?,
            vk::DescriptorType::STORAGE_BUFFER,
            &buffers_nfos[i],
            i as u32,
            0,
        );
    }
    descriptor_set.update_descriptors_sets();

    // Command buffers
    let mut cmd_pool = vkcmd::VkCmdPool::new(vk.clone());
    let cmd_buffer = cmd_pool.create_cmd_buffer(vk::CommandBufferLevel::PRIMARY);

    cmd_pool.begin_cmd(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, cmd_buffer);
    cmd_pool.bind_pipeline(shad_pip_vec.pipeline, vk::PipelineBindPoint::COMPUTE, cmd_buffer);
    cmd_pool.bind_descriptor(shad_pipeline_layout, vk::PipelineBindPoint::COMPUTE, &descriptor.set, 0);

    cmd_pool.dispatch((tris.len() / 32) as u32 + 1, (points.len() / 32) as u32 + 1, 1, 0);

    // Memory barrier
    let mut buffer_barrier: Vec<vk::BufferMemoryBarrier> = Vec::new();
    for buffer in &buffers {
        buffer_barrier.push(
            vk::BufferMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(buffer.buffer)
                .size(vk::WHOLE_SIZE)
                .build(),
        );
    }
    unsafe {
        vk.device.cmd_pipeline_barrier(
            cmd_pool.cmd_buffers[0],
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &buffer_barrier,
            &[],
        );
    }

    cmd_pool.end_cmd(0);

    // Execution
    let fence = vkfence::VkFence::new(vk.clone(), false);
    let queue = unsafe { vk.device.get_device_queue(vk.queue_family_index, 0) };
    cmd_pool.submit(queue, Some(fence.fence));

    while fence.status() == vkfence::FenceStates::Unsignaled {}

    let output: Vec<Point3d> = memory
        .get_buffer::<(f32, f32, i32)>(&buffers[1])
        .iter()
        .map(|x| Point3d::new(x.0, x.1, x.2 as f32 / 1000000.))
        .collect();

    Ok(output)
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
    vk: Rc<VulkanState>,
) -> Result<Vec<Vec<Triangle3d>>, ComputeError> {
    let shader = to_vec32(include_bytes!("../../shaders/spirv/partition.spv").to_vec());

    // Map Buffers
    let count = ((tris.len() as f32 - 1.) + ((columns.len() as f32 - 1.) * tris.len() as f32) / 32.).ceil() as usize;
    let dest_content: Vec<u32> = vec![0u32; count];

    let buffer_sizes: Vec<u64> = vec![
        (tris.len() * std::mem::size_of::<Triangle3d>()) as u64,
        (columns.len() * std::mem::size_of::<Line3d>()) as u64,
        (dest_content.len() * std::mem::size_of::<u32>()) as u64,
    ];

    let mut buffers = buffer_sizes
        .iter()
        .map(|size| vkmem::VkBuffer::new(vk.clone(), *size))
        .collect::<Vec<_>>();
    let (mem_size, offsets) = vkmem::compute_non_overlapping_buffer_alignment(&buffers);
    let memory = vkmem::VkMem::find_mem(vk.clone(), mem_size).ok_or(ComputeError::NoMem)?;

    let mbuf = buffers.get_mut(0).ok_or(ComputeError::NoBuffer)?;
    mbuf.bind(memory.mem, offsets[0]);
    memory.map_buffer(tris, mbuf);
    let mbuf = buffers.get_mut(1).ok_or(ComputeError::NoBuffer)?;
    mbuf.bind(memory.mem, offsets[1]);
    memory.map_buffer(&columns, mbuf);
    let mbuf = buffers.get_mut(2).ok_or(ComputeError::NoBuffer)?;
    mbuf.bind(memory.mem, offsets[2]);
    memory.map_buffer(&dest_content, mbuf);

    // Shaders
    let shader = Rc::new(RefCell::new(vkshader::VkShader::new(
        vk.clone(),
        shader,
        CString::new("main")?,
    )));

    for i in 0..buffers.len() {
        shader.borrow_mut().add_layout_binding(
            i as u32,
            1,
            vk::DescriptorType::STORAGE_BUFFER,
            vk::ShaderStageFlags::COMPUTE,
        );
    }
    shader.borrow_mut().create_pipeline_layout();
    let shad_pipeline_layout = shader.borrow().pipeline.ok_or(ComputeError::NoPipe)?;
    let shad_pip_vec = vkpipeline::VkComputePipeline::new(vk.clone(), &shader.borrow());
    let mut descriptor = vkdescriptor::VkDescriptor::new(vk.clone(), shader);

    descriptor.add_pool_size(buffers.len() as u32, vk::DescriptorType::STORAGE_BUFFER);
    descriptor.create_pool(1);
    descriptor.create_set();

    let mut descriptor_set = vkdescriptor::VkWriteDescriptor::new(vk.clone());

    let mut buffers_nfos: Vec<Vec<vk::DescriptorBufferInfo>> = Vec::new();
    for i in 0..buffers.len() {
        descriptor_set.add_buffer(buffers[i].buffer, 0, buffers[i].size);
        buffers_nfos.push(vec![descriptor_set.buffer_descriptors[i]]);
        descriptor_set.add_write_descriptors(
            *descriptor.get_first_set().ok_or(ComputeError::NoDescriptor)?,
            vk::DescriptorType::STORAGE_BUFFER,
            &buffers_nfos[i],
            i as u32,
            0,
        );
    }
    descriptor_set.update_descriptors_sets();

    // Command buffers
    let mut cmd_pool = vkcmd::VkCmdPool::new(vk.clone());
    let cmd_buffer = cmd_pool.create_cmd_buffer(vk::CommandBufferLevel::PRIMARY);

    cmd_pool.begin_cmd(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT, cmd_buffer);
    cmd_pool.bind_pipeline(shad_pip_vec.pipeline, vk::PipelineBindPoint::COMPUTE, cmd_buffer);
    cmd_pool.bind_descriptor(shad_pipeline_layout, vk::PipelineBindPoint::COMPUTE, &descriptor.set, 0);

    cmd_pool.dispatch((tris.len() / 32) as u32 + 1, (columns.len() / 32) as u32 + 1, 1, 0);

    // Memory barrier
    let mut buffer_barrier: Vec<vk::BufferMemoryBarrier> = Vec::new();
    for buffer in &buffers {
        buffer_barrier.push(
            vk::BufferMemoryBarrier::builder()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .buffer(buffer.buffer)
                .size(vk::WHOLE_SIZE)
                .build(),
        );
    }
    unsafe {
        vk.device.cmd_pipeline_barrier(
            cmd_pool.cmd_buffers[0],
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &buffer_barrier,
            &[],
        );
    }

    cmd_pool.end_cmd(0);

    // Execution
    let fence = vkfence::VkFence::new(vk.clone(), false);
    let queue = unsafe { vk.device.get_device_queue(vk.queue_family_index, 0) };
    cmd_pool.submit(queue, Some(fence.fence));

    while fence.status() == vkfence::FenceStates::Unsignaled {}

    let dest_content = memory.get_buffer::<u32>(&buffers[2]);

    // have to unpack the bitmask to determine what columns each tri belongs in
    let result = (0..columns.len())
        .into_par_iter()
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
*/
