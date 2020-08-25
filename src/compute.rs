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
use vulkano::pipeline::vertex::{VertexMember, VertexMemberTy};

pub struct Vk {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
}

#[derive(Default, Copy, Clone)]
pub struct Point_vk {
    pub position: [f32; 3],
}

unsafe impl VertexMember for Point_vk {
    fn format() -> (VertexMemberTy, usize) {
        (VertexMemberTy::F32, 3)
    }
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
            &DeviceExtensions{khr_storage_buffer_storage_class:true, ..DeviceExtensions::none()},
            [(queue_family, 0.5)].iter().cloned(),
        )
        .expect("failed to create device")
    };
    let queue = queues.next().unwrap();

    Vk { device, queue }
}

pub fn compute_bbox(tris: &Vec<Triangle_vk>, vk: &Vk) -> Vec<Line3d> {
    vulkano::impl_vertex!(Point_vk, position);
    vulkano::impl_vertex!(Line_vk, p1, p2);
    vulkano::impl_vertex!(Triangle_vk, p1, p2, p3);
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
    for chunk in chunks {
        println!("new chunk");
        let dest_content = (0..tris.len()).map(|_| Line_vk {
            p1: Default::default(),
            p2: Default::default(),
        });
        let src_content = (0..1024).map(|_| Triangle_vk {
            ..Default::default()
        });

        let layout =
            compute_pipeline.layout().descriptor_set_layout(0).unwrap();

        let source = CpuAccessibleBuffer::from_iter(
            vk.device.clone(),
            BufferUsage::all(),
            false,
            src_content,
        )
        .expect("failed to create buffer");
        {
            let mut content = source.write().unwrap();
            for (index, item) in chunk.iter().enumerate() {
                content[index] = *item;
            }
        }

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
        let mut builder =
            AutoCommandBufferBuilder::new(vk.device.clone(), vk.queue.family())
                .unwrap();
        builder
            .dispatch([1024, 1, 1], compute_pipeline.clone(), set.clone(), ())
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
            results.push(to_line3d(item));
        }
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

pub fn to_line3d(line: &Line_vk) -> Line3d {
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

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) buffer Triangle_vk {
    vec3 p1;
    vec3 p2;
    vec3 p3;
} tris;

layout(binding = 1) buffer Line_vk {
    vec3 p1;
    vec3 p2;
} lines;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    float x_max = max(max(tris.p1.x,tris.p2.x), tris.p3.x);
    float y_max = max(max(tris.p1.y,tris.p2.y), tris.p3.y);
    float z_max = max(max(tris.p1.z,tris.p2.z), tris.p3.z);
    float x_min = min(min(tris.p1.x,tris.p2.x), tris.p3.x);
    float y_min = min(min(tris.p1.y,tris.p2.y), tris.p3.y);
    float z_min = min(min(tris.p1.z,tris.p2.z), tris.p3.z);
    lines.p1 = vec3(x_min, y_min, z_min);
    lines.p2 = vec3(x_max, y_max, z_max);
}"
    }
}