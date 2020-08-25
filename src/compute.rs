use crate::geo::*;
use std::{default::Default, sync::Arc};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    descriptor::{
        descriptor_set::PersistentDescriptorSet, PipelineLayoutAbstract,
    },
    device::{Device, DeviceExtensions, Features, Queue},
    instance::{Instance, InstanceExtensions, PhysicalDevice},
    pipeline::{
        vertex::{VertexMember, VertexMemberTy},
        ComputePipeline,
    },
    sync::GpuFuture,
};

pub struct Vk {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

#[derive(Default, Debug, Copy, Clone)]
pub struct PointVk {
    pub position: [f32; 3],
}

unsafe impl VertexMember for PointVk {
    fn format() -> (VertexMemberTy, usize) { (VertexMemberTy::F32, 3) }
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

pub fn compute_bbox(tris: &Vec<TriangleVk>, vk: &Vk) -> Vec<Line3d> {
    vulkano::impl_vertex!(PointVk, position);
    vulkano::impl_vertex!(LineVk, p1, p2);
    vulkano::impl_vertex!(TriangleVk, p1, p2, p3);
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
    println!("calculating {:?} chunks", chunks.len());
    let dest_content = (0..tris.len()).map(|_| LineVk {
        p1: Default::default(),
        p2: Default::default(),
    });
    let src_content = (0..1024).map(|_| TriangleVk {
        ..Default::default()
    });

    let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();

    let source = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        BufferUsage::all(),
        false,
        src_content,
    )
    .expect("failed to create buffer");

    let dest = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        BufferUsage::all(),
        false,
        dest_content,
    )
    .expect("failed to create buffer");
    let set = Arc::new(
        PersistentDescriptorSet::start(layout.clone())
            .add_buffer(source.clone())
            .unwrap()
            .add_buffer(dest.clone())
            .unwrap()
            .build()
            .unwrap(),
    );
    for chunk in chunks {
        println!("new chunk");

        {
            let mut content = source.write().unwrap();
            for (index, item) in chunk.iter().enumerate() {
                content[index] = *item;
            }
        }
        let mut builder =
            AutoCommandBufferBuilder::new(vk.device.clone(), vk.queue.family())
                .unwrap();
        builder
            .dispatch([32, 1, 1], compute_pipeline.clone(), set.clone(), ())
            .unwrap();
        let command_buffer = builder.build().unwrap();
        let finished = command_buffer.execute(vk.queue.clone()).unwrap();
        finished
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
        let dest_content = dest.read().unwrap();
        for item in dest_content.iter() {
            // println!("{:?}", item);
            results.push(to_line3d(item));
        }
    }
    results
}

pub fn to_tri_vk(tris: &Vec<Triangle3d>) -> Vec<TriangleVk> {
    tris.iter()
        .map(|tri| TriangleVk {
            p1: PointVk {
                position: [tri.p1.x, tri.p1.y, tri.p1.z],
            },
            p2: PointVk {
                position: [tri.p2.x, tri.p2.y, tri.p2.z],
            },
            p3: PointVk {
                position: [tri.p3.x, tri.p3.y, tri.p3.z],
            },
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
        src: "
#version 450
//#extension GL_EXT_nonuniform_qualifier : enable

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer TriangleVk {
    vec3 p1;
    vec3 p2;
    vec3 p3;
} tris[1];

layout(binding = 1) buffer LineVk {
    vec3 p1;
    vec3 p2;
} lines[1];

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float x_max = max(max(tris[idx].p1.x,tris[idx].p2.x), tris[idx].p3.x);
    float y_max = max(max(tris[idx].p1.y,tris[idx].p2.y), tris[idx].p3.y);
    float z_max = max(max(tris[idx].p1.z,tris[idx].p2.z), tris[idx].p3.z);
    float x_min = min(min(tris[idx].p1.x,tris[idx].p2.x), tris[idx].p3.x);
    float y_min = min(min(tris[idx].p1.y,tris[idx].p2.y), tris[idx].p3.y);
    float z_min = min(min(tris[idx].p1.z,tris[idx].p2.z), tris[idx].p3.z);
    lines[idx].p1 = vec3(x_min, y_min, z_min);
    lines[idx].p2 = vec3(x_max, y_max, z_max);
}"
    }
}
