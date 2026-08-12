# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone argparse CLI for NCore V4 inputs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from instant_nurec.pretrained import (
    DEFAULT_MODEL_VARIANT,
    MODEL_VARIANTS,
    get_model_profile,
)

_DEFAULT_MAX_CHUNKS = 8
_DEFAULT_N_GAUSSIANS = 2_000_000
_WAYMO_NCORE_DEFAULT_CAMERA_IDS: dict[str, tuple[str, ...]] = {
    "pa-front": ("camera_front_50fov",),
    "pa-multiview": (
        "camera_front_50fov",
        "camera_front_left_50fov",
        "camera_front_right_50fov",
        "camera_side_left_50fov",
        "camera_side_right_50fov",
    ),
}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="instant_nurec",
        description="Standalone Kelvin predict-mode CLI.",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_VARIANTS,
        default=DEFAULT_MODEL_VARIANT,
        help=(
            "Released model profile. pa-front uses one 784x448 camera with 18 frames; "
            "pa-multiview uses three 504x280 cameras with 18 frames each by default; "
            "pq-front uses the selective point-query decoder with 18 front-camera "
            "frames. "
            f"Default: {DEFAULT_MODEL_VARIANT}."
        ),
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ncore-path",
        type=Path,
        help=(
            "ncorev4 input. Either a single sequence ``.json`` (NuRec-aligned) "
            "or a ``.lst`` manifest with one JSON path per line "
            "(absolute or relative-to-the-LST-file's directory; "
            "``#``-prefixed and blank lines skipped)."
        ),
    )
    input_group.add_argument(
        "--waymo-tfrecord",
        type=Path,
        help=(
            "One Waymo TFRecord segment. It is converted with NVIDIA NCore's "
            "official Bazel converter before prediction."
        ),
    )
    parser.add_argument(
        "--waymo-ncore",
        action="store_true",
        help=(
            "Treat --ncore-path as output from NVIDIA NCore's official Waymo "
            "TFRecord-to-NCore V4 converter. Without --camera-id, pa-front uses "
            "camera_front_50fov and pa-multiview uses all five NCore Waymo camera IDs."
        ),
    )
    parser.add_argument(
        "--ncore-repo",
        type=Path,
        help=(
            "Path to an NVIDIA/ncore source checkout. Required with "
            "--waymo-tfrecord."
        ),
    )
    parser.add_argument(
        "--waymo-conversion-dir",
        type=Path,
        help=(
            "Directory for NCore V4 data created from --waymo-tfrecord. "
            "Required with --waymo-tfrecord. Existing converted metadata is reused."
        ),
    )
    parser.add_argument(
        "--force-waymo-conversion",
        action="store_true",
        help="Re-run the official NCore converter even when its metadata JSON already exists.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for PLY and sky-output bundles.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "If set, merge per-chunk primitives into a single "
            "frustum-ownership PLY per sequence and run kl-optimal "
            "voxelization (target count controlled by --n-gaussians). "
            "Default (flag absent) writes per-chunk PLYs and skips "
            "voxelization entirely."
        ),
    )
    parser.add_argument(
        "--n-gaussians",
        type=int,
        default=_DEFAULT_N_GAUSSIANS,
        help=(
            f"Target number of static Gaussians after kl-optimal voxelization "
            f"(default: {_DEFAULT_N_GAUSSIANS}). The voxel size is searched "
            f"iteratively via bracketed binary search (starting at 0.1) until "
            f"the count lands in [0.9 * target, target]. Only consulted when "
            f"--merge is set (voxelization is bundled with merge); must be > 0."
        ),
    )
    parser.add_argument(
        "--camera-id",
        dest="camera_ids",
        metavar="CAMERA_ID",
        type=str,
        action="append",
        default=None,
        help=(
            "Override a profile's context camera. Repeat the flag to "
            "provide multiple cameras in canonical order. The selected profile "
            "validates its camera contract. pq-front is fixed to "
            "camera_front_wide_120fov. If omitted, the profile's default camera "
            "set is used."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=_DEFAULT_MAX_CHUNKS,
        help=(
            f"Maximum number of time-chunks processed per clip "
            f"(default: {_DEFAULT_MAX_CHUNKS}). One chunk spans up to 13.5 s. "
            f"Longer clips are "
            f"truncated and a WARNING is logged naming the dropped chunk "
            f"count and the --max-chunks value needed to cover the full clip."
        ),
    )
    parser.add_argument(
        "--render-input-cameras",
        action="store_true",
        help=(
            "Render every source-camera frame after reconstruction with gsplat. "
            "Outputs PNGs plus input/render comparisons under "
            "<output-dir>/<run-id>/render/."
        ),
    )
    parser.add_argument(
        "--render-preview",
        action="store_true",
        help=(
            "Render the first context frame of each output chunk as a PNG, "
            "using its calibrated NCore F-theta model and rolling-shutter poses and "
            "compositing the observation-derived sky behind the Gaussians. "
            "Install with `uv sync --extra render` first."
        ),
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help=(
            "Render every original frame from the first context camera as an H.264 MP4, "
            "using the source F-theta calibration, exposure trajectory, and sky. Requires "
            "one resolved sequence, --merge, enough --max-chunks for full coverage, "
            "`uv sync --extra render`, and ffmpeg with libx264."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level forwarded to logging.basicConfig.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    if args.render_video and not args.merge:
        parser.error("--render-video requires --merge so one complete scene is rendered")
    if args.max_chunks <= 0:
        parser.error("--max-chunks must be greater than zero")

    if args.render_preview or args.render_video:
        from instant_nurec.predict.render_preview import require_gsplat

        try:
            require_gsplat()
        except RuntimeError as exc:
            parser.error(str(exc))

    if args.render_video:
        from instant_nurec.predict.render_video import require_ffmpeg

        try:
            require_ffmpeg()
        except RuntimeError as exc:
            parser.error(str(exc))

    # Lazy imports keep argparse-only invocations (e.g. --help) cheap.
    from instant_nurec.config_schema.dataset import (
        AdaptiveSequentialFrameBatchSamplerConfig,
        InstantNuRecSplitsConfig,
        NCoreInstantNuRecCuboidTracksParamsConfig,
        NCoreInstantNuRecDatasetConfig,
    )
    from instant_nurec.config_schema.instantnurec import (
        GaussiansInstantNuRecSystemConfig,
        InstantNuRecConfig,
    )
    from instant_nurec.config_schema.models import (
        KelvinDPTDecoderConfig,
        KelvinModelConfig,
        KelvinPointQueryCADecoderConfig,
    )
    from instant_nurec.config_schema.predict import (
        InputCameraRenderConfig,
        PredictConfig,
        PrimitiveMergeConfig,
    )
    from instant_nurec.ncore_input import resolve_ncore_paths
    from instant_nurec.predict.run import run_predict
    from instant_nurec.waymo_conversion import convert_waymo_tfrecord_to_ncore

    profile = get_model_profile(args.model)
    is_waymo_ncore = args.waymo_ncore or args.waymo_tfrecord is not None
    if is_waymo_ncore and profile.decoder_kind == "point-query":
        raise ValueError(
            "pq-front requires the released PAI front-camera contract; use pa-front or pa-multiview for Waymo."
        )
    default_camera_ids = (
        _WAYMO_NCORE_DEFAULT_CAMERA_IDS[args.model] if is_waymo_ncore else profile.context_camera_ids
    )
    camera_ids = list(args.camera_ids or default_camera_ids)
    if args.waymo_tfrecord is not None:
        if args.ncore_repo is None or args.waymo_conversion_dir is None:
            raise ValueError("--ncore-repo and --waymo-conversion-dir are required with --waymo-tfrecord.")
        json_paths = [
            convert_waymo_tfrecord_to_ncore(
                args.waymo_tfrecord,
                args.ncore_repo,
                args.waymo_conversion_dir,
                force=args.force_waymo_conversion,
            )
        ]
    else:
        if args.ncore_repo is not None or args.waymo_conversion_dir is not None or args.force_waymo_conversion:
            raise ValueError(
                "--ncore-repo, --waymo-conversion-dir, and --force-waymo-conversion require --waymo-tfrecord."
            )
        json_paths = resolve_ncore_paths(args.ncore_path)
    if args.render_video and len(json_paths) != 1:
        parser.error(
            "--render-video currently requires exactly one resolved NCore sequence "
            "so all requested chunks are merged into one complete scene"
        )
    dataset_config_kwargs = {
        "ncore_json_paths": [str(p) for p in json_paths],
        "camera_subsampler": {
            "frame_width": profile.frame_width,
            "frame_height": profile.frame_height,
        },
        "context_camera_ids": camera_ids,
        "supervision_camera_ids": camera_ids,
        "frame_batch_sampler": AdaptiveSequentialFrameBatchSamplerConfig(
            n_frames_per_sample=profile.n_frames_per_sample,
            n_samples_per_sequence=args.max_chunks,
        ),
    }
    if is_waymo_ncore:
        dataset_config_kwargs["cuboid_tracks_params"] = NCoreInstantNuRecCuboidTracksParamsConfig(
            track_label_source="EXTERNAL"
        )
    dataset_config = NCoreInstantNuRecDatasetConfig(**dataset_config_kwargs)
    allowed_camera_counts = profile.supported_camera_counts
    if len(camera_ids) not in allowed_camera_counts:
        counts = ", ".join(str(count) for count in allowed_camera_counts)
        raise ValueError(
            f"{args.model} expects {counts} context camera(s); got {len(camera_ids)}. "
            "Repeat --camera-id once per camera."
        )

    decoder_config = (
        KelvinPointQueryCADecoderConfig(motion_depth=profile.motion_depth)
        if profile.decoder_kind == "point-query"
        else KelvinDPTDecoderConfig(motion_depth=profile.motion_depth)
    )

    config = InstantNuRecConfig(
        out_dir=str(args.output_dir),
        release_profile=args.model,
        system=(
            GaussiansInstantNuRecSystemConfig(predict_batch_size=args.max_chunks)
            if args.render_video
            else GaussiansInstantNuRecSystemConfig()
        ),
        model=KelvinModelConfig(
            decoder=decoder_config,
        ),
        dataset=InstantNuRecSplitsConfig(
            predict=dataset_config,
        ),
        predict=PredictConfig(
            primitive_merge=PrimitiveMergeConfig(
                enabled=args.merge,
                enable_voxelization=args.merge,
                target_n_gaussians=args.n_gaussians,
            ),
            input_camera_render=InputCameraRenderConfig(
                enabled=args.render_input_cameras,
            ),
            render_preview=args.render_preview,
            render_video=args.render_video,
        ),
    )
    run_predict(config)
    if args.merge:
        print(
            "Next: refine into USDZ with NuRec — "
            "https://docs.nvidia.com/nurec/nurec/reconstruct-av-scene.html",
            flush=True,
        )
    else:
        print(
            "Next: view your 3DGS PLY with SuperSplat "
            "(https://playcanvas.com/supersplat/editor) "
            "or ply_viewer (NuRec container).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
