# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Eager inference wrapper that packages learned static and dynamic Gaussians."""

from __future__ import annotations

import logging
import math

import torch

from torch import nn

from instant_nurec.datasets.tracks import CuboidTracks
from instant_nurec.model.static_core import KelvinPointQueryStaticOutput, KelvinStaticCore
from instant_nurec.primitives.kelvin_primitive import (
    KelvinDynamicLayer,
    KelvinInstantNuRecPrimitive,
    KelvinSemanticClass,
    KelvinStaticLayer,
)
from instant_nurec.utils.batch import DataAndRenderingBatch
from instant_nurec.utils.geometry import tquat_to_se3_matrix
from instant_nurec.utils.misc import unpack_optional
from instant_nurec.utils.motion import TimeRemapping, warp_points_with_cuboid_tracks
from instant_nurec.utils.sensor import to_simple_pinhole_model_parameters
from instant_nurec.utils.types import TrackFlags


logger = logging.getLogger(__name__)


class KelvinInferenceModel(nn.Module):
    """Run the released heads and package static/dynamic Gaussian layers."""

    def __init__(
        self,
        static_core: KelvinStaticCore,
        *,
        scene_rescale: float,
        use_cuboid_motion_calibration: bool = True,
        expected_frames: int,
        expected_height: int,
        expected_width: int,
    ) -> None:
        super().__init__()
        self.static_core = static_core
        self.scene_rescale = scene_rescale
        self.use_cuboid_motion_calibration = use_cuboid_motion_calibration
        self.expected_b = 1
        self.expected_v = expected_frames
        self.expected_h = expected_height
        self.expected_w = expected_width
        self.register_buffer(
            "cuboids_dims_padding",
            static_core.decoder.cuboids_dims_padding.detach().clone(),
            persistent=False,
        )

    def _validate_input_shape(self, rgb: torch.Tensor) -> None:
        b, v, h, w, c = rgb.shape
        expected = (self.expected_b, self.expected_v, self.expected_h, self.expected_w, 3)
        if (b, v, h, w, c) != expected:
            raise ValueError(
                f"Input shape mismatch: got rgb {tuple(rgb.shape)}, expected {expected}. "
                f"Model expects {self.expected_v} input frames at {self.expected_h}x{self.expected_w}; "
                "check that len(context_camera_ids) * n_frames_per_sample equals the expected frame count."
            )

    def prepare_context(self, context: list[DataAndRenderingBatch]) -> list[DataAndRenderingBatch]:
        return context

    def _extract_tensors(self, batch: DataAndRenderingBatch):
        """Extract tensors and derive the three timestamps used by the motion head."""
        data = unpack_optional(batch.data.camera)
        rendering = unpack_optional(unpack_optional(batch.rendering).camera)

        rgb = unpack_optional(data.labels.rgb).unsqueeze(0)
        rays = rendering.rays.unsqueeze(0)
        distance_to_depth_scale = rendering.distance_to_depth_scale.unsqueeze(0)

        c2w = tquat_to_se3_matrix(rendering.poses_tquat_startend[:, 1, :], unbatch=False).clone()
        c2w[:, :3, 3] *= self.scene_rescale
        c2w = c2w.unsqueeze(0)

        pinhole_parameters = [
            to_simple_pinhole_model_parameters(rendering.sensor_model_parameters[frame_idx])
            for frame_idx in range(data.b)
        ]
        fov = torch.tensor(
            [
                [
                    2 * math.atan2(parameters.resolution[0] / 2, parameters.focal_length[0]),
                    2 * math.atan2(parameters.resolution[1] / 2, parameters.focal_length[1]),
                ]
                for parameters in pinhole_parameters
            ],
            dtype=torch.float32,
            device=rgb.device,
        ).unsqueeze(0)
        camera_idxs = torch.tensor(
            [meta.unique_sensor_idx for meta in data.meta], dtype=torch.int64, device=rgb.device
        ).unsqueeze(0)

        source_timestamps_us = unpack_optional(rendering.rays_timestamps_us).unsqueeze(0)
        time_remapping = TimeRemapping.from_timestamps_startend_us(
            rendering.timestamps_startend_us_cpu,
            camera_idxs[0].cpu(),
        )
        frame_gaps_us = time_remapping.frame_gap_timestamps_us.to(rgb.device)
        prev_target_timestamps_us = source_timestamps_us - frame_gaps_us[None, :, 0, None, None, None]
        next_target_timestamps_us = source_timestamps_us + frame_gaps_us[None, :, 1, None, None, None]

        return (
            rgb,
            c2w,
            fov,
            rays,
            distance_to_depth_scale,
            camera_idxs,
            source_timestamps_us,
            prev_target_timestamps_us,
            next_target_timestamps_us,
            [time_remapping],
        )

    @staticmethod
    def _semantic_movable_value() -> int:
        return KelvinSemanticClass.MOVABLE.value

    @staticmethod
    def _gather_source_pixels(values: torch.Tensor, source_indices: torch.Tensor | None) -> torch.Tensor:
        """Flatten a dense ``(B,V,H,W,C)`` field to the emitted Gaussian order."""
        flattened = values[0].reshape(-1, *values.shape[4:])
        if source_indices is None:
            return flattened
        return flattened[source_indices[0].reshape(-1).long()]

    def _empty_dynamic_layer(self, device: torch.device) -> KelvinDynamicLayer:
        return KelvinDynamicLayer(
            max_densities=torch.zeros(0, 1, device=device),
            keyframe_positions=torch.zeros(0, 3, 3, device=device),
            keyframe_timestamps_us=torch.zeros(0, 3, dtype=torch.int64, device=device),
            rotations=torch.zeros(0, 4, device=device),
            scales=torch.zeros(0, 3, device=device),
            rgb=torch.zeros(0, 3, device=device),
        )

    @staticmethod
    def _log_displacement_stats(
        *,
        chunk_index: int,
        label: str,
        positions: torch.Tensor,
        previous_positions: torch.Tensor,
        next_positions: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Log enough displacement statistics to distinguish a silent motion head from masking."""
        count = int(mask.sum().item())
        if count == 0:
            logger.warning("Motion diagnostics for chunk %d (%s): no selected Gaussians.", chunk_index, label)
            return

        previous_norm = torch.linalg.vector_norm(previous_positions[mask] - positions[mask], dim=-1).float()
        next_norm = torch.linalg.vector_norm(next_positions[mask] - positions[mask], dim=-1).float()

        def summarize(values: torch.Tensor) -> tuple[int, float, float, float, float]:
            finite_values = values[torch.isfinite(values)]
            if finite_values.numel() == 0:
                return 0, math.nan, math.nan, math.nan, math.nan
            return (
                int(finite_values.numel()),
                float(finite_values.min().item()),
                float(finite_values.median().item()),
                float(finite_values.mean().item()),
                float(finite_values.max().item()),
            )

        prev_finite, prev_min, prev_median, prev_mean, prev_max = summarize(previous_norm)
        next_finite, next_min, next_median, next_mean, next_max = summarize(next_norm)
        logger.info(
            "Motion diagnostics for chunk %d (%s, n=%d, meters): "
            "previous[finite=%d/%d min=%.6g median=%.6g mean=%.6g max=%.6g], "
            "next[finite=%d/%d min=%.6g median=%.6g mean=%.6g max=%.6g].",
            chunk_index,
            label,
            count,
            prev_finite,
            count,
            prev_min,
            prev_median,
            prev_mean,
            prev_max,
            next_finite,
            count,
            next_min,
            next_median,
            next_mean,
            next_max,
        )

    def _refine_dynamic_motion(
        self,
        *,
        positions: torch.Tensor,
        semantic_class: torch.Tensor,
        predicted_prev_positions: torch.Tensor,
        predicted_next_positions: torch.Tensor,
        rays: torch.Tensor,
        source_timestamps_us: torch.Tensor,
        prev_target_timestamps_us: torch.Tensor,
        next_target_timestamps_us: torch.Tensor,
        cuboid_tracks: CuboidTracks | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prefer cuboid-track motion, otherwise retain learned semantic/motion outputs."""
        learned_dynamic_mask = semantic_class == self._semantic_movable_value()
        if cuboid_tracks is None:
            return learned_dynamic_mask, predicted_prev_positions, predicted_next_positions

        dynamic_tracks = CuboidTracks.Ops.subset_from_mask(
            cuboid_tracks, cuboid_tracks.tracks_flags & TrackFlags.DYNAMIC != 0
        )
        if dynamic_tracks.n_tracks == 0:
            return learned_dynamic_mask, predicted_prev_positions, predicted_next_positions

        auxiliary_track_indices = torch.full_like(learned_dynamic_mask, -1, dtype=torch.int64)
        if learned_dynamic_mask.any():
            intersections = dynamic_tracks.ray_intersection(
                rays[learned_dynamic_mask, :3],
                rays[learned_dynamic_mask, 3:],
                source_timestamps_us[learned_dynamic_mask],
                max_intersections_per_ray=2,
            )
            track_indices = intersections.intersections_tracks_idx[..., 0]
            track_indices[intersections.intersections_cnt != 1] = -1
            auxiliary_track_indices[learned_dynamic_mask] = track_indices

        dynamic_mask, (previous_positions, next_positions) = warp_points_with_cuboid_tracks(
            points=positions,
            source_timestamps_us=source_timestamps_us,
            target_timestamps_us_list=[prev_target_timestamps_us, next_target_timestamps_us],
            dynamic_tracks=dynamic_tracks,
            aux_tracks_idx=auxiliary_track_indices,
            cuboids_dims_padding=self.cuboids_dims_padding,
        )
        return dynamic_mask, previous_positions, next_positions

    def reconstruct(
        self,
        context: list[DataAndRenderingBatch],
        cuboid_tracks: list[CuboidTracks] | None = None,
    ) -> list[KelvinInstantNuRecPrimitive]:
        primitives: list[KelvinInstantNuRecPrimitive] = []
        for batch_index, batch in enumerate(context):
            tensors = self._extract_tensors(batch)
            self._validate_input_shape(tensors[0])
            output = self.static_core(*tensors)
            source_indices = output.source_indices if isinstance(output, KelvinPointQueryStaticOutput) else None

            positions = output.positions[0].reshape(-1, 3)
            rotations = output.rotations[0].reshape(-1, 4)
            scales = output.scales[0].reshape(-1, 3)
            densities = output.densities[0].reshape(-1, 1)
            rgb = output.rgb[0].reshape(-1, 3)
            semantic_class = output.semantic_class[0].reshape(-1)
            normals = output.normals[0].reshape(-1, 3)
            prev_flow = self._gather_source_pixels(output.prev_flow, source_indices)
            next_flow = self._gather_source_pixels(output.next_flow, source_indices)
            source_timestamps_us = self._gather_source_pixels(output.source_timestamps_us, source_indices).squeeze(-1)
            prev_target_timestamps_us = self._gather_source_pixels(
                output.prev_target_timestamps_us, source_indices
            ).squeeze(-1)
            next_target_timestamps_us = self._gather_source_pixels(
                output.next_target_timestamps_us, source_indices
            ).squeeze(-1)
            rays = self._gather_source_pixels(tensors[3], source_indices)

            learned_dynamic_mask = semantic_class == self._semantic_movable_value()
            predicted_previous_positions = positions + prev_flow
            predicted_next_positions = positions + next_flow
            self._log_displacement_stats(
                chunk_index=batch_index,
                label="learned-motion-head/all-queries",
                positions=positions,
                previous_positions=predicted_previous_positions,
                next_positions=predicted_next_positions,
                mask=torch.ones_like(learned_dynamic_mask),
            )
            self._log_displacement_stats(
                chunk_index=batch_index,
                label="learned-motion-head/movable",
                positions=positions,
                previous_positions=predicted_previous_positions,
                next_positions=predicted_next_positions,
                mask=learned_dynamic_mask,
            )

            chunk_cuboid_tracks = cuboid_tracks[batch_index] if cuboid_tracks is not None else None
            if not self.use_cuboid_motion_calibration:
                chunk_cuboid_tracks = None
                logger.info("Cuboid motion calibration disabled for chunk %d; using learned motion only.", batch_index)

            dynamic_mask, previous_positions, next_positions = self._refine_dynamic_motion(
                positions=positions,
                semantic_class=semantic_class,
                predicted_prev_positions=predicted_previous_positions,
                predicted_next_positions=predicted_next_positions,
                rays=rays,
                source_timestamps_us=source_timestamps_us,
                prev_target_timestamps_us=prev_target_timestamps_us,
                next_target_timestamps_us=next_target_timestamps_us,
                cuboid_tracks=chunk_cuboid_tracks,
            )
            if chunk_cuboid_tracks is not None:
                self._log_displacement_stats(
                    chunk_index=batch_index,
                    label="cuboid-calibrated/final-dynamic",
                    positions=positions,
                    previous_positions=previous_positions,
                    next_positions=next_positions,
                    mask=dynamic_mask,
                )
            class_counts = torch.bincount(semantic_class, minlength=len(KelvinSemanticClass))
            logger.info(
                "Semantic Gaussians in chunk %d: others=%d ego=%d sky=%d road=%d movable=%d; dynamic=%d.",
                batch_index,
                class_counts[KelvinSemanticClass.OTHERS].item(),
                class_counts[KelvinSemanticClass.EGO].item(),
                class_counts[KelvinSemanticClass.SKY].item(),
                class_counts[KelvinSemanticClass.ROAD].item(),
                class_counts[KelvinSemanticClass.MOVABLE].item(),
                dynamic_mask.sum().item(),
            )

            static_indices = torch.where(~dynamic_mask)[0]
            dynamic_indices = torch.where(dynamic_mask)[0]
            static_layer = KelvinStaticLayer(
                positions=positions[static_indices],
                rotations=rotations[static_indices],
                scales=scales[static_indices],
                densities=densities[static_indices],
                rgb=rgb[static_indices],
                semantic_class=semantic_class[static_indices].unsqueeze(-1).to(torch.uint8),
                normals=normals[static_indices],
            )
            if dynamic_indices.numel() == 0:
                dynamic_layer = self._empty_dynamic_layer(static_layer.positions.device)
            else:
                dynamic_layer = KelvinDynamicLayer(
                    max_densities=densities[dynamic_indices],
                    keyframe_positions=torch.stack(
                        [
                            previous_positions[dynamic_indices],
                            positions[dynamic_indices],
                            next_positions[dynamic_indices],
                        ],
                        dim=1,
                    ),
                    keyframe_timestamps_us=torch.stack(
                        [
                            prev_target_timestamps_us[dynamic_indices],
                            source_timestamps_us[dynamic_indices],
                            next_target_timestamps_us[dynamic_indices],
                        ],
                        dim=1,
                    ),
                    rotations=rotations[dynamic_indices],
                    scales=scales[dynamic_indices],
                    rgb=rgb[dynamic_indices],
                ).ensure_minimum_density(0.75)

            primitives.append(
                KelvinInstantNuRecPrimitive(
                    static_layer=static_layer,
                    dynamic_layers=[dynamic_layer],
                    sky_cubemap=output.sky_cubemap[0],
                    sky_cubemap_mask=output.sky_cubemap_mask[0],
                    affine_matrix=output.affine_matrix[0],
                )
            )
        return primitives
