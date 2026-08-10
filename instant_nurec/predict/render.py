# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-view 3D Gaussian rendering through the CUDA ``gsplat`` rasterizer."""

from __future__ import annotations

import inspect

from pathlib import Path
from typing import Callable

import torch

from PIL import Image

from instant_nurec.primitives.kelvin_primitive import KelvinInstantNuRecPrimitive
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional


def _require_gsplat() -> Callable:
    try:
        from gsplat import rasterization
    except ImportError as error:
        raise RuntimeError(
            "--render-input-cameras requires gsplat. Install the compatible renderer with "
            "`proxy pip install gsplat==1.5.3` in the active InstantNuRec environment."
        ) from error
    return rasterization


def _camera_vector(camera_model_parameters: object, name: str, size: int, device: torch.device) -> torch.Tensor:
    values = getattr(camera_model_parameters, name, None)
    if values is None:
        raise TypeError(
            "gsplat source-view rendering requires an OpenCV pinhole camera model; "
            f"{type(camera_model_parameters).__name__} does not define {name!r}."
        )
    tensor = torch.as_tensor(values, device=device, dtype=torch.float32).reshape(-1)
    if tensor.numel() != size:
        raise ValueError(f"Camera parameter {name!r} must contain {size} values, got {tensor.numel()}.")
    return tensor


def _pinhole_intrinsics(camera_model_parameters: object, device: torch.device) -> torch.Tensor:
    focal_length = _camera_vector(camera_model_parameters, "focal_length", 2, device)
    # NCore's camera projection code uses pixel-center image coordinates and
    # applies this half-pixel conversion when interpreting calibration values.
    principal_point = _camera_vector(camera_model_parameters, "principal_point", 2, device) + 0.5
    intrinsics = torch.eye(3, device=device, dtype=torch.float32)
    intrinsics[0, 0], intrinsics[1, 1] = focal_length
    intrinsics[0, 2], intrinsics[1, 2] = principal_point
    return intrinsics


def _distortion_kwargs(camera_model_parameters: object, rasterization: Callable, device: torch.device) -> dict:
    """Pass Waymo's OpenCV distortion only when the installed gsplat supports it."""
    signature = inspect.signature(rasterization)
    parameters = signature.parameters
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    requested = {
        "radial_coeffs": _camera_vector(camera_model_parameters, "radial_coeffs", 6, device),
        "tangential_coeffs": _camera_vector(camera_model_parameters, "tangential_coeffs", 2, device),
        "thin_prism_coeffs": _camera_vector(camera_model_parameters, "thin_prism_coeffs", 4, device),
    }
    has_distortion = any(torch.any(coefficients != 0.0).item() for coefficients in requested.values())
    if not has_distortion:
        return {}

    unsupported = [name for name in requested if name not in parameters and not accepts_kwargs]
    if unsupported:
        raise RuntimeError(
            "The installed gsplat rasterization API cannot apply this Waymo camera's distortion "
            f"({', '.join(unsupported)}). Install gsplat==1.5.3 or a newer compatible version."
        )
    return {name: values.unsqueeze(0) for name, values in requested.items()}


@torch.inference_mode()
def render_static_gaussians(
    primitive: KelvinInstantNuRecPrimitive,
    *,
    T_sensor_scene: torch.Tensor,
    camera_model_parameters: object,
    height: int,
    width: int,
) -> torch.Tensor:
    """Render learned anisotropic static Gaussians with alpha compositing."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    if type(camera_model_parameters).__name__ != "OpenCVPinholeCameraModelParameters":
        raise TypeError(
            "gsplat source-view rendering currently supports Waymo OpenCV pinhole cameras only; "
            f"got {type(camera_model_parameters).__name__}."
        )

    rasterization = _require_gsplat()
    static_layer = primitive.static_layer
    device = static_layer.positions.device
    if len(static_layer) == 0:
        return torch.zeros((height, width, 3), device=device, dtype=torch.float32)
    viewmat = torch.linalg.inv(T_sensor_scene.to(device=device, dtype=torch.float32)).unsqueeze(0).contiguous()
    intrinsics = _pinhole_intrinsics(camera_model_parameters, device).unsqueeze(0).contiguous()
    distortion_kwargs = _distortion_kwargs(camera_model_parameters, rasterization, device)
    with_ut = bool(distortion_kwargs)

    rendered, _, _ = rasterization(
        means=static_layer.positions.float().contiguous(),
        quats=static_layer.rotations.float().contiguous(),
        scales=static_layer.scales.float().contiguous(),
        opacities=static_layer.densities[:, 0].float().contiguous(),
        colors=static_layer.rgb.float().contiguous(),
        viewmats=viewmat,
        Ks=intrinsics,
        width=width,
        height=height,
        render_mode="RGB",
        camera_model="pinhole",
        with_ut=with_ut,
        packed=not with_ut,
        **distortion_kwargs,
    )
    return rendered[0, ..., :3].clamp_(0.0, 1.0)


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
) -> list[Path]:
    """Render every input frame and write 3DGS render/comparison PNGs.

    The primitive and camera poses must be in the same scene frame. The system
    calls this before primitive merging, while that invariant still holds.
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
