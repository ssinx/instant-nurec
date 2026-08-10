# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Waymo Open Dataset v2 Parquet input for InstantNuRec prediction."""

from __future__ import annotations

import io
import logging

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

from ncore.impl.data.types import OpenCVPinholeCameraModelParameters, ShutterType

from instant_nurec.config_schema.dataset import WaymoParquetInstantNuRecDatasetConfig
from instant_nurec.datasets.instantnurec_base import CameraSubsampler, InstantNuRecDataError
from instant_nurec.datasets.samplers import AdaptiveSequentialFrameBatchSampler
from instant_nurec.utils.batch import CameraFrameLabels, DataAndRenderingBatch, DataBatch, FrameMeta, InstantNuRecDataBatch
from instant_nurec.utils.geometry import se3_matrix_inverse
from instant_nurec.utils.misc import to_torch
from instant_nurec.utils.types import FrameConversion, HalfClosedInterval, RigTrajectories


logger = logging.getLogger(__name__)


WAYMO_CAMERA_NAME_TO_ID = {
    "FRONT": 1,
    "FRONT_LEFT": 2,
    "FRONT_RIGHT": 3,
    "SIDE_LEFT": 4,
    "SIDE_RIGHT": 5,
}


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise InstantNuRecDataError("Waymo v2 input requires pyarrow; install the project dependencies first.") from error
    return pa, pq


def _field(row: dict[str, Any], suffix: str) -> Any:
    matches = [name for name in row if name.endswith(suffix)]
    if len(matches) != 1:
        raise InstantNuRecDataError(
            f"Waymo Parquet field ending with {suffix!r} was not uniquely found; available fields: {sorted(row)}"
        )
    return row[matches[0]]


@dataclass(slots=True)
class _WaymoSegment:
    image_bytes: dict[int, dict[int, bytes]]
    timestamps_us: dict[int, np.ndarray]
    T_vehicle_worlds: dict[int, np.ndarray]
    camera_parameters: dict[int, OpenCVPinholeCameraModelParameters]
    T_camera_vehicle: dict[int, np.ndarray]


class WaymoParquetInstantNuRecDataset(Dataset[InstantNuRecDataBatch]):
    """Reads synchronized Waymo v2 images, calibration, and vehicle poses.

    Dynamic cuboid tracks are intentionally omitted in this first direct-Parquet
    path. The static reconstruction model does not require them.
    """

    def __init__(
        self,
        config: WaymoParquetInstantNuRecDatasetConfig,
        frame_width: int,
        frame_height: int,
        n_frames_per_sample: int,
    ) -> None:
        self.config = config
        self.root = Path(config.waymo_root).expanduser()
        self.camera_subsampler = CameraSubsampler(frame_width=frame_width, frame_height=frame_height)
        self.frame_batch_sampler = AdaptiveSequentialFrameBatchSampler(
            config.frame_batch_sampler,
            n_frames_per_sample=n_frames_per_sample,
        )
        if config.context_camera_ids != config.supervision_camera_ids:
            raise ValueError("Waymo predict requires context_camera_ids and supervision_camera_ids to have the same order.")
        invalid_camera_ids = [camera_id for camera_id in config.context_camera_ids if camera_id not in WAYMO_CAMERA_NAME_TO_ID]
        if invalid_camera_ids:
            raise ValueError(
                f"Unknown Waymo camera id(s): {invalid_camera_ids}. "
                f"Choose from {sorted(WAYMO_CAMERA_NAME_TO_ID)}."
            )
        self.camera_ids = list(config.context_camera_ids)
        self._segment_cache: dict[str, _WaymoSegment] = {}

    def __len__(self) -> int:
        return len(self.config.segment_ids) * self.config.frame_batch_sampler.n_samples_per_sequence

    def _component_paths(self, component: str, segment_id: str) -> list[Path]:
        component_dir = self.root / self.config.split / component
        if not component_dir.is_dir():
            raise InstantNuRecDataError(f"Waymo component directory does not exist: {component_dir}")
        paths = sorted(component_dir.glob(f"*_{component}_{segment_id}*.parquet"))
        if not paths:
            raise InstantNuRecDataError(
                f"No {component} Parquet files for segment {segment_id!r} under {component_dir}"
            )
        return paths

    def _read_component(self, component: str, segment_id: str, columns: list[str] | None = None):
        pa, pq = _require_pyarrow()
        tables = [pq.read_table(path, columns=columns) for path in self._component_paths(component, segment_id)]
        return pa.concat_tables(tables)

    @staticmethod
    def _camera_parameters(row: dict[str, Any]) -> OpenCVPinholeCameraModelParameters:
        focal_length = np.array([_field(row, "intrinsic.f_u"), _field(row, "intrinsic.f_v")], dtype=np.float32)
        principal_point = np.array([_field(row, "intrinsic.c_u"), _field(row, "intrinsic.c_v")], dtype=np.float32)
        radial_coeffs = np.array(
            [_field(row, "intrinsic.k1"), _field(row, "intrinsic.k2"), _field(row, "intrinsic.k3"), 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        tangential_coeffs = np.array([_field(row, "intrinsic.p1"), _field(row, "intrinsic.p2")], dtype=np.float32)
        return OpenCVPinholeCameraModelParameters(
            resolution=np.array([_field(row, ".width"), _field(row, ".height")], dtype=np.int32),
            shutter_type=ShutterType.GLOBAL,
            external_distortion_parameters=None,
            principal_point=principal_point,
            focal_length=focal_length,
            radial_coeffs=radial_coeffs,
            tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=np.zeros(4, dtype=np.float32),
        )

    def _load_segment(self, segment_id: str) -> _WaymoSegment:
        if segment_id in self._segment_cache:
            return self._segment_cache[segment_id]

        image_rows = self._read_component(
            "camera_image",
            segment_id,
            ["key.frame_timestamp_micros", "key.camera_name", "[CameraImageComponent].image"],
        ).to_pylist()
        calibration_rows = self._read_component("camera_calibration", segment_id).to_pylist()
        pose_rows = self._read_component("vehicle_pose", segment_id).to_pylist()

        image_bytes: dict[int, dict[int, bytes]] = {}
        for row in image_rows:
            camera_id = int(row["key.camera_name"])
            timestamp_us = int(row["key.frame_timestamp_micros"])
            image_bytes.setdefault(camera_id, {})[timestamp_us] = row["[CameraImageComponent].image"]

        camera_parameters: dict[int, OpenCVPinholeCameraModelParameters] = {}
        T_camera_vehicle: dict[int, np.ndarray] = {}
        for row in calibration_rows:
            camera_id = int(row["key.camera_name"])
            camera_parameters[camera_id] = self.camera_subsampler.apply_camera_parameters(self._camera_parameters(row))
            T_camera_vehicle[camera_id] = np.asarray(_field(row, "extrinsic.transform"), dtype=np.float64).reshape(4, 4)

        T_vehicle_worlds = {
            int(row["key.frame_timestamp_micros"]): np.asarray(
                _field(row, "world_from_vehicle.transform"), dtype=np.float64
            ).reshape(4, 4)
            for row in pose_rows
        }
        timestamps_us = {
            camera_id: np.asarray(sorted(frames), dtype=np.int64) for camera_id, frames in image_bytes.items()
        }
        segment = _WaymoSegment(
            image_bytes=image_bytes,
            timestamps_us=timestamps_us,
            T_vehicle_worlds=T_vehicle_worlds,
            camera_parameters=camera_parameters,
            T_camera_vehicle=T_camera_vehicle,
        )
        self._segment_cache[segment_id] = segment
        return segment

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    def __getitem__(self, batch_idx: int) -> InstantNuRecDataBatch:
        segment_idx = batch_idx // self.config.frame_batch_sampler.n_samples_per_sequence
        sample_idx = batch_idx % self.config.frame_batch_sampler.n_samples_per_sequence
        segment_id = self.config.segment_ids[segment_idx]
        segment = self._load_segment(segment_id)
        selected_camera_ids = [(name, WAYMO_CAMERA_NAME_TO_ID[name]) for name in self.camera_ids]

        missing = [
            name
            for name, camera_id in selected_camera_ids
            if camera_id not in segment.timestamps_us
            or camera_id not in segment.camera_parameters
            or camera_id not in segment.T_camera_vehicle
        ]
        if missing:
            raise InstantNuRecDataError(f"Segment {segment_id!r} is missing requested Waymo camera(s): {missing}")

        timestamps_by_camera = {name: segment.timestamps_us[camera_id] for name, camera_id in selected_camera_ids}
        start_us = max(int(timestamps.min()) for timestamps in timestamps_by_camera.values())
        end_us = min(int(timestamps.max()) for timestamps in timestamps_by_camera.values())
        frame_batch = self.frame_batch_sampler.sample_frame_batch(
            sample_idx,
            timestamps_by_camera,
            [HalfClosedInterval(start_us, end_us)],
        )
        if not frame_batch:
            return InstantNuRecDataBatch(context=[], context_rig=[], meta=[])

        sampled_timestamps = {
            name: timestamps_by_camera[name][frame_indices] for name, frame_indices in frame_batch.items()
        }
        missing_poses = sorted(
            {
                int(timestamp_us)
                for timestamps in sampled_timestamps.values()
                for timestamp_us in timestamps
                if int(timestamp_us) not in segment.T_vehicle_worlds
            }
        )
        if missing_poses:
            raise InstantNuRecDataError(
                f"vehicle_pose is missing {len(missing_poses)} sampled timestamp(s) for segment {segment_id!r}"
            )

        reference_timestamp_us = int(sampled_timestamps[self.camera_ids[0]][0])
        T_world_ref = se3_matrix_inverse(
            to_torch(segment.T_vehicle_worlds[reference_timestamp_us], device="cpu", dtype=torch.float64)
        )
        all_pose_timestamps_us = np.asarray(sorted(segment.T_vehicle_worlds), dtype=np.int64)
        T_vehicle_worlds = np.stack([segment.T_vehicle_worlds[int(timestamp)] for timestamp in all_pose_timestamps_us])

        camera_batch_list: list[DataBatch.Camera] = []
        camera_timestamps_startend: dict[str, torch.Tensor] = {}
        camera_calibrations: OrderedDict[str, RigTrajectories.CameraCalibration] = OrderedDict()
        unique_frame_idx = 0
        for unique_sensor_idx, (camera_name, camera_id) in enumerate(selected_camera_ids):
            timestamps = sampled_timestamps[camera_name]
            camera_timestamps_startend[camera_name] = torch.from_numpy(
                np.stack([timestamps, timestamps], axis=1)
            ).to(dtype=torch.int64)
            camera_calibrations[camera_name] = RigTrajectories.CameraCalibration(
                sequence_id=segment_id,
                unique_sensor_idx=unique_sensor_idx,
                T_sensor_rig=to_torch(segment.T_camera_vehicle[camera_id], device="cpu", dtype=torch.float64),
                camera_model_parameters=segment.camera_parameters[camera_id],
            )
            for timestamp_us in timestamps:
                image = self.camera_subsampler.apply_frame_data(
                    self._decode_image(segment.image_bytes[camera_id][int(timestamp_us)])
                )
                camera_batch_list.append(
                    DataBatch.Camera(
                        meta=[FrameMeta(unique_sensor_idx=unique_sensor_idx, unique_frame_idx=unique_frame_idx)],
                        labels=CameraFrameLabels(rgb=to_torch(image, device="cpu").unsqueeze(0)),
                    )
                )
                unique_frame_idx += 1

        rig = RigTrajectories(
            T_world_base=se3_matrix_inverse(T_world_ref),
            world_to_scene=FrameConversion(matrix=np.eye(4, dtype=np.float32)),
            rig_trajectories=[
                RigTrajectories.RigTrajectory(
                    sequence_id=segment_id,
                    cameras_frame_timestamps_us=camera_timestamps_startend,
                    T_rig_worlds=T_world_ref @ to_torch(T_vehicle_worlds, device="cpu", dtype=torch.float64),
                    T_rig_world_timestamps_us=to_torch(all_pose_timestamps_us, device="cpu", dtype=torch.int64),
                )
            ],
            camera_calibrations=camera_calibrations,
        )
        context = DataAndRenderingBatch(data=DataBatch(camera=DataBatch.Camera.collate_fn(camera_batch_list)))
        return InstantNuRecDataBatch(
            context=[context],
            context_rig=[rig],
            meta=[{"sequence_id": segment_id, "waymo_root": self.root}],
        )
