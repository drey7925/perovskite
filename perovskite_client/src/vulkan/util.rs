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
        .filter(|p| p.supported_extensions().contains(device_extensions))
        .filter(|p| p.supported_features().contains(device_features))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
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
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 1,
                PhysicalDeviceType::IntegratedGpu => 2,
                PhysicalDeviceType::VirtualGpu => 3,
                PhysicalDeviceType::Cpu => 4,
                _ => 5,
            }
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
    Ok((
        selected_gpu,
        graphics_queue_family_index,
        transfer_queue_family_index,
    ))
}
