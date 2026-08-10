# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic source-view rendering for static InstantNuRec Gaussians.

This module deliberately has no dependency on an external 3DGS rasterizer.
It projects Gaussian centers with the calibrated source camera model and
performs an opaque z-buffer splat in Torch. The result is useful for checking
camera/extrinsic alignment and reconstruction coverage, but is not intended to
replace NuRec's production renderer.
"""

from __future__ import annotations

from pathlib import Path

import torch

from PIL import Image

from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.sensors.ray_gen import camera_rays_to_image_points


def _pixel_offsets(radius_px: int, device: torch.device) -> torch.Tensor:
    offsets = torch.arange(-radius_px, radius_px + 1, device=device, dtype=torch.long)
    offset_x, offset_y = torch.meshgrid(offsets, offsets, indexing="xy")
    return torch.stack([offset_x.reshape(-1), offset_y.reshape(-1)], dim=-1)


def _project_gaussian_chunk(
    positions: torch.Tensor,
    T_sensor_scene: torch.Tensor,
    camera_model_parameters: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project scene-space positions to a source image.

    ``T_sensor_scene`` maps camera coordinates into the rendering scene, as
    documented by :class:`RenderingData`. Row-vector points therefore use the
    transpose of its inverse rotation to enter camera coordinates.
    """
    rotation_sensor_scene = T_sensor_scene[:3, :3]
    translation_sensor_scene = T_sensor_scene[:3, 3]
    positions_camera = (positions - translation_sensor_scene) @ rotation_sensor_scene
    projection = camera_rays_to_image_points(camera_model_parameters, positions_camera)
    image_points = projection.image_points
    valid = projection.valid_flag & (positions_camera[:, 2] > 1.0e-4)
    return image_points, positions_camera[:, 2], valid


@torch.inference_mode()
def render_static_gaussians(
    primitive: KelvinInstantNuRecPrimitive,
    *,
    T_sensor_scene: torch.Tensor,
    camera_model_parameters: object,
    height: int,
    width: int,
    gaussian_chunk_size: int,
    splat_radius_px: int,
) -> torch.Tensor:
    """Render a static primitive into one RGB image using opaque z-buffer splats."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")

    static_layer = primitive.static_layer
    device = static_layer.positions.device
    image_size = height * width
    depth_buffer = torch.full((image_size,), torch.inf, device=device, dtype=torch.float32)
    offsets = _pixel_offsets(splat_radius_px, device)

    def iter_splats():
        for start in range(0, len(static_layer), gaussian_chunk_size):
            stop = min(start + gaussian_chunk_size, len(static_layer))
            image_points, depths, valid = _project_gaussian_chunk(
                static_layer.positions[start:stop],
                T_sensor_scene,
                camera_model_parameters,
            )
            pixels = torch.round(image_points).to(torch.long)
            pixels = pixels[:, None, :] + offsets[None, :, :]
            flat_x = pixels[..., 0].reshape(-1)
            flat_y = pixels[..., 1].reshape(-1)
            splat_depths = depths[:, None].expand(-1, len(offsets)).reshape(-1)
            splat_colors = (
                static_layer.rgb[start:stop, None, :].expand(-1, len(offsets), -1).reshape(-1, 3)
            )
            splat_valid = valid[:, None].expand(-1, len(offsets)).reshape(-1)
            splat_valid &= (flat_x >= 0) & (flat_x < width) & (flat_y >= 0) & (flat_y < height)
            if not torch.any(splat_valid):
                continue
            flat_indices = flat_y[splat_valid] * width + flat_x[splat_valid]
            yield flat_indices, splat_depths[splat_valid], splat_colors[splat_valid]

    for flat_indices, depths, _ in iter_splats():
        depth_buffer.scatter_reduce_(0, flat_indices, depths, reduce="amin", include_self=True)

    colors = torch.zeros((image_size, 3), device=device, dtype=torch.float32)
    for flat_indices, depths, splat_colors in iter_splats():
        is_nearest = depths == depth_buffer[flat_indices]
        if torch.any(is_nearest):
            colors.index_copy_(0, flat_indices[is_nearest], splat_colors[is_nearest])

    return colors.reshape(height, width, 3).clamp_(0.0, 1.0)


def _write_rgb_png(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(array, mode="RGB").save(path)


@torch.inference_mode()
def render_input_camera_frames(
    primitive: KelvinInstantNuRecPrimitive,
    context: DataAndRenderingBatch,
    *,
    output_dir: Path,
    camera_names_by_index: dict[int, str],
    gaussian_chunk_size: int,
    splat_radius_px: int,
) -> list[Path]:
    """Render every frame in ``context`` and write render/comparison PNGs.

    The primitive and camera poses must be in the same scene frame. This is
    true before primitive merging, which is why the predict system invokes this
    helper directly after each chunk's export preprocessing.
    """
    data_camera = unpack_optional(context.data.camera)
    rendering_camera = unpack_optional(unpack_optional(context.rendering).camera)
    source_images = unpack_optional(data_camera.labels.rgb)
    if source_images.shape[0] != rendering_camera.b:
        raise ValueError("Source-image and rendering-camera counts must match")

    output_paths: list[Path] = []
    for frame_idx, frame_meta in enumerate(data_camera.meta):
        source_image = source_images[frame_idx]
        height, width = source_image.shape[:2]
        T_sensor_scene = tquat_to_se3_matrix(
            rendering_camera.poses_tquat_startend[frame_idx, 1],
            unbatch=True,
        ).to(device=primitive.device(), dtype=torch.float32)
        rendered_image = render_static_gaussians(
            primitive,
            T_sensor_scene=T_sensor_scene,
            camera_model_parameters=rendering_camera.sensor_model_parameters[frame_idx],
            height=height,
            width=width,
            gaussian_chunk_size=gaussian_chunk_size,
            splat_radius_px=splat_radius_px,
        )
        timestamp_us = int(rendering_camera.timestamps_startend_us_cpu[frame_idx, 1].item())
        camera_name = camera_names_by_index.get(
            frame_meta.unique_sensor_idx,
            f"camera_{frame_meta.unique_sensor_idx}",
        ).replace("/", "_")
        frame_stem = f"frame_{frame_idx:03d}_{timestamp_us}"
        render_path = output_dir / camera_name / f"{frame_stem}_render.png"
        comparison_path = output_dir / camera_name / f"{frame_stem}_comparison.png"
        _write_rgb_png(rendered_image, render_path)
        _write_rgb_png(torch.cat([source_image, rendered_image], dim=1), comparison_path)
        output_paths.append(render_path)

    return output_paths
