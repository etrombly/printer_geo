use ash::{version::DeviceV1_0, vk};
use std::rc::Rc;

use crate::vulkan::vkstate::VulkanState;

#[derive(Eq, PartialEq, Debug)]
pub enum FenceStates {
    Signaled,
    Unsignaled,
    Lost,
    Unkown,
}

pub struct VkFence {
    pub fence: vk::Fence,
    state: Rc<VulkanState>,
}

impl VkFence {
    pub fn new(state: Rc<VulkanState>, signaled: bool) -> Self {
        let mut fence_info = vk::FenceCreateInfo::builder();
        if signaled {
            fence_info = fence_info.flags(vk::FenceCreateFlags::SIGNALED);
        }

        let fence = unsafe {
            state
                .device
                .create_fence(&fence_info, None)
                .expect("[ERR] Could not create fence.")
        };

        VkFence { fence, state }
    }

    pub fn status(&self) -> FenceStates {
        let status = unsafe {
            self.state
                .device
                .fp_v1_0()
                .get_fence_status(self.state.device.handle(), self.fence)
        };
        match status {
            vk::Result::SUCCESS => FenceStates::Signaled,
            vk::Result::NOT_READY => FenceStates::Unsignaled,
            vk::Result::ERROR_DEVICE_LOST => FenceStates::Lost,
            _ => FenceStates::Unkown,
        }
    }

    pub fn reset(&self) {
        unsafe {
            self.state
                .device
                .reset_fences(&[self.fence])
                .expect("[ERR] Could not reset fence");
        }
    }

    pub fn wait(&self, timeout: u64) -> FenceStates {
        let res = unsafe { self.state.device.wait_for_fences(&[self.fence], true, timeout) };

        if res.is_ok() {
            return FenceStates::Signaled;
        }

        FenceStates::Unsignaled
    }
}

impl Drop for VkFence {
    fn drop(&mut self) {
        unsafe {
            self.state.device.destroy_fence(self.fence, None);
        }
    }
}
