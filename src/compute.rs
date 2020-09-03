use crate::geo::*;
use std::{default::Default, fmt, ops::Add, sync::Arc};
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

impl Add<PointVk> for PointVk {
    type Output = PointVk;

    fn add(self, other: PointVk) -> PointVk {
        PointVk {
            position: [
                self.position[0] + other.position[0],
                self.position[1] + other.position[1],
                self.position[2] + other.position[2],
            ],
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct TriangleVk {
    pub p1: PointVk,
    pub p2: PointVk,
    pub p3: PointVk,
}

impl TriangleVk {
    pub fn new(p1: (f32, f32, f32), p2: (f32, f32, f32), p3: (f32, f32, f32)) -> TriangleVk {
        TriangleVk {
            p1: PointVk::new(p1.0, p1.1, p1.2),
            p2: PointVk::new(p2.0, p2.1, p2.2),
            p3: PointVk::new(p3.0, p3.1, p3.2),
        }
    }

    pub fn in_2d_bounds(&self, bbox: &LineVk) -> bool {
        bbox.in_2d_bounds(&self.p1) || bbox.in_2d_bounds(&self.p2) || bbox.in_2d_bounds(&self.p3)
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct LineVk {
    pub p1: PointVk,
    pub p2: PointVk,
}

impl LineVk {
    pub fn new(p1: (f32, f32, f32), p2: (f32, f32, f32)) -> LineVk {
        LineVk {
            p1: PointVk::new(p1.0, p1.1, p1.2),
            p2: PointVk::new(p2.0, p2.1, p2.2),
        }
    }

    pub fn in_2d_bounds(&self, point: &PointVk) -> bool {
        point.position[0] >= self.p1.position[0]
            && point.position[0] <= self.p2.position[0]
            && point.position[1] >= self.p1.position[1]
            && point.position[1] <= self.p2.position[1]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CircleVk {
    pub center: PointVk,
    pub radius: f32,
}

impl CircleVk {
    pub fn new(center: PointVk, radius: f32) -> CircleVk { CircleVk { center, radius } }

    pub fn in_2d_bounds(&self, point: &PointVk) -> bool {
        let dx = f32::abs(point.position[0] - self.center.position[0]);
        let dy = f32::abs(point.position[1] - self.center.position[1]);
        if dx > self.radius {
            false
        } else if dy > self.radius {
            false
        } else if dx + dy <= self.radius {
            true
        } else {
            dx.powi(2) + dy.powi(2) <= self.radius.powi(2)
        }
    }

    pub fn bbox(self) -> LineVk {
        LineVk {
            p1: PointVk::new(
                self.min_x() - self.center.position[0] - self.radius * 2.,
                self.min_y() - self.center.position[1] - self.radius * 2.,
                self.min_z() - self.center.position[2],
            ),
            p2: PointVk::new(
                self.max_x() - self.center.position[0] + self.radius * 2.,
                self.max_y() - self.center.position[1] + self.radius * 2.,
                self.max_z() - self.center.position[2],
            ),
        }
    }

    fn min_x(self) -> f32 { self.center.position[0] - self.radius }

    fn min_y(self) -> f32 { self.center.position[1] - self.radius }

    fn min_z(self) -> f32 { self.center.position[2] }

    fn max_x(self) -> f32 { self.center.position[0] + self.radius }

    fn max_y(self) -> f32 { self.center.position[1] + self.radius }

    fn max_z(self) -> f32 { self.center.position[2] }
}

#[derive(Default, Clone)]
pub struct Tool {
    pub bbox: LineVk,
    pub points: Vec<PointVk>,
}

impl Tool {
    pub fn new_endmill(radius: f32) -> Tool {
        let circle = CircleVk::new(PointVk::new(radius, radius, 0.0), radius);
        let points: Vec<PointVk> = (0..=(radius * 20.0) as i32)
            .flat_map(|x| {
                (0..=(radius * 20.0) as i32)
                    .map(move |y| PointVk::new(x as f32 / 10.0, y as f32 / 10.0, 0.0))
            })
            .filter(|x| circle.in_2d_bounds(&x))
            .map(|x| {
                PointVk::new(
                    x.position[0] - radius,
                    x.position[1] - radius,
                    x.position[2],
                )
            })
            .collect();
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }

    pub fn new_v_bit(radius: f32, angle: f32) -> Tool {
        let circle = CircleVk::new(PointVk::new(radius, radius, 0.0), radius);
        let percent = (90. - (angle / 2.)).to_radians().tan();
        let points: Vec<PointVk> = (0..=(radius * 20.0) as i32)
            .flat_map(|x| {
                (0..=(radius * 20.0) as i32).filter_map(move |y| {
                    let x = x as f32 / 10.0;
                    let y = y as f32 / 10.0;
                    if circle.in_2d_bounds(&PointVk::new(x, y, 0.)) {
                        let x = x - radius;
                        let y = y - radius;
                        let distance = (x.powi(2) + y.powi(2)).sqrt();
                        let z = distance * percent;
                        Some(PointVk::new(x, y, z))
                    } else {
                        None
                    }
                })
            })
            .collect();
        Tool {
            bbox: circle.bbox(),
            points,
        }
    }
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
    let shader = bbox::Shader::load(vk.device.clone()).expect("failed to create shader module");
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
    let dest = CpuAccessibleBuffer::from_iter(vk.device.clone(), dest_usage, false, dest_content)
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
    let dest = CpuAccessibleBuffer::from_iter(
        vk.device.clone(),
        dest_usage,
        false,
        dest_content.iter().copied(),
    )
    .expect("failed to create buffer");

    let mut bbox_usage = BufferUsage::transfer_source();
    bbox_usage.storage_buffer = true;
    let (bbox_buffer, bbox_future) =
        ImmutableBuffer::from_data(tool.bbox.clone(), source_usage, vk.queue.clone())
            .expect("failed to create buffer");
    bbox_future
        .then_signal_fence_and_flush()
        .unwrap()
        .wait(None)
        .unwrap();
    let mut tool_usage = BufferUsage::transfer_source();
    tool_usage.storage_buffer = true;
    let (tool_buffer, tool_future) =
        ImmutableBuffer::from_iter(tool.points.iter().copied(), source_usage, vk.queue.clone())
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
            .add_buffer(bbox_buffer.clone())
            .unwrap()
            .add_buffer(tool_buffer.clone())
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

mod bbox {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/bbox.comp"
    }
}

mod drop {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "shaders/drop.comp"
    }
}
