use std::sync::Arc;

use anyhow::{Context, Result};
use vulkano::device::DeviceFeatures;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        DeviceExtensions, QueueFlags,
    },
    instance::Instance,
    swapchain::Surface,
};

// Returns the names of Vulkan physical devices
// Copyright for this function:
// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
pub(crate) fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
    device_features: &DeviceFeatures,
    preferred_gpu: &str,
) -> Result<(Arc<PhysicalDevice>, u32, u32)> {
    log::info!(
        "Detected GPUs: {:?}",
        instance
            .enumerate_physical_devices()
            .unwrap()
            .map(|x| x.properties().device_name.clone())
            .collect::<Vec<String>>()
    );
    let (selected_gpu, graphics_queue_family_index) = instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices")
        .filter(|p| {
            if !p.supported_extensions().contains(device_extensions) {
                let missing = device_extensions.difference(&p.supported_extensions());
                log::warn!(
                    "GPU '{}' rejected: missing required extensions: {:?}",
                    p.properties().device_name,
                    missing
                );
                false
            } else {
                true
            }
        })
        .filter(|p| {
            if !p.supported_features().contains(device_features) {
                let missing = device_features.difference(&p.supported_features());
                log::warn!(
                    "GPU '{}' rejected: missing required features: {:?}",
                    p.properties().device_name,
                    missing
                );
                false
            } else {
                true
            }
        })
        .filter_map(|device| {
            let result = device
                .queue_family_properties()
                .iter()
                .enumerate()
                .position(|(queue_family_index, queue)| {
                    queue.queue_flags.contains(QueueFlags::GRAPHICS)
                        && device
                            .surface_support(queue_family_index as u32, surface)
                            .unwrap_or(false)
                });

            match result {
                Some(queue_family_index) => Some((device, queue_family_index as u32)),
                None => {
                    log::warn!(
                        "GPU '{}' rejected: no suitable graphics queue family with surface support",
                        device.properties().device_name
                    );
                    None
                }
            }
        })
        .min_by_key(|(p, _)| {
            if !preferred_gpu.is_empty()
                && p.properties()
                    .device_name
                    .to_lowercase()
                    .starts_with(&preferred_gpu.to_lowercase())
            {
                return 0;
            };
            let type_score = match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 2,
                PhysicalDeviceType::IntegratedGpu => 4,
                PhysicalDeviceType::VirtualGpu => 6,
                PhysicalDeviceType::Cpu => 8,
                PhysicalDeviceType::Other => {
                    log::warn!(
                        "Unknown GPU type detected for device '{}'",
                        p.properties().device_name
                    );
                    10
                }
                _ => {
                    log::warn!(
                        "Novel and unknown GPU type detected for device '{}', please file a bug!",
                        p.properties().device_name
                    );
                    12
                }
            };
            let portability_score = if p.supported_extensions().khr_portability_subset {
                1
            } else {
                0
            };
            type_score + portability_score
        })
        .with_context(|| "no device available")?;
    let transfer_queue = selected_gpu
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(i, q)| {
            q.queue_flags.contains(QueueFlags::TRANSFER)
                && i != (graphics_queue_family_index as usize)
        });
    let transfer_queue_family_index = match transfer_queue {
        Some(q) => q as u32,
        None => selected_gpu
            .queue_family_properties()
            .iter()
            .position(|q| q.queue_flags.contains(QueueFlags::TRANSFER))
            .context("no transfer queue available")? as u32,
    };
    log::info!(
        "GPU queue families: {:?}",
        selected_gpu.queue_family_properties()
    );
    log::info!(
        "Selected GPU: {:?}, graphics queue family index: {}, transfer queue family index: {}",
        selected_gpu.properties().device_name,
        graphics_queue_family_index,
        transfer_queue_family_index
    );
    if selected_gpu.supported_extensions().khr_portability_subset {
        log::info!("Selected a portability subset GPU.")
    }
    Ok((
        selected_gpu,
        graphics_queue_family_index,
        transfer_queue_family_index,
    ))
}
