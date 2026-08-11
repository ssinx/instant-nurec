# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert a single Waymo TFRecord through NVIDIA NCore's official converter."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile

from pathlib import Path


class WaymoConversionError(RuntimeError):
    """Raised when the official NCore Waymo converter cannot produce metadata."""


_METADATA_REGISTRY_FILENAME = ".instant_nurec_waymo_metadata.json"


def _metadata_registry_path(conversion_dir: Path) -> Path:
    return conversion_dir / _METADATA_REGISTRY_FILENAME


def _load_metadata_registry(conversion_dir: Path) -> dict[str, str]:
    registry_path = _metadata_registry_path(conversion_dir)
    if not registry_path.is_file():
        return {}
    try:
        registry = json.loads(registry_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        raise WaymoConversionError(f"Could not read Waymo conversion registry: {registry_path}") from error
    is_string_mapping = isinstance(registry, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in registry.items()
    )
    if not is_string_mapping:
        raise WaymoConversionError(f"Invalid Waymo conversion registry: {registry_path}")
    return registry


def _store_metadata_registry(conversion_dir: Path, registry: dict[str, str]) -> None:
    registry_path = _metadata_registry_path(conversion_dir)
    temporary_path = registry_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(registry_path)


def _metadata_paths(conversion_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in conversion_dir.glob("*/*.json")
        if path.stem == path.parent.name
    )


def convert_waymo_tfrecord_to_ncore(
    tfrecord_path: Path,
    ncore_repo: Path,
    conversion_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Convert one TFRecord and return its official NCore V4 metadata JSON.

    The upstream converter accepts an input directory and converts every
    TFRecord in it. A temporary directory containing only a symlink to the
    requested file keeps a prediction invocation scoped to one segment.
    """
    tfrecord_path = tfrecord_path.expanduser().resolve()
    ncore_repo = ncore_repo.expanduser().resolve()
    conversion_dir = conversion_dir.expanduser().resolve()

    if not tfrecord_path.is_file():
        raise WaymoConversionError(f"Waymo TFRecord does not exist: {tfrecord_path}")
    if tfrecord_path.suffix != ".tfrecord":
        raise WaymoConversionError(f"Expected a .tfrecord file, got: {tfrecord_path}")
    if not (ncore_repo / "tools" / "data_converter" / "waymo").is_dir():
        raise WaymoConversionError(
            "--ncore-repo must point to an NVIDIA/ncore checkout containing "
            f"tools/data_converter/waymo: {ncore_repo}"
        )

    conversion_dir.mkdir(parents=True, exist_ok=True)
    registry = _load_metadata_registry(conversion_dir)
    registry_key = str(tfrecord_path)
    if not force and (metadata_relative_path := registry.get(registry_key)) is not None:
        metadata_path = conversion_dir / metadata_relative_path
        if metadata_path.is_file():
            return metadata_path

    existing_metadata_mtimes = {path: path.stat().st_mtime_ns for path in _metadata_paths(conversion_dir)}
    bazel_command = shutil.which("bazel") or shutil.which("bazelisk")
    if bazel_command is None:
        raise WaymoConversionError(
            "NVIDIA NCore's official Waymo converter requires Bazel or Bazelisk on PATH. "
            "Install one, then rerun this command."
        )

    with tempfile.TemporaryDirectory(prefix="instant_nurec_waymo_", dir=conversion_dir) as input_dir:
        staged_tfrecord = Path(input_dir) / tfrecord_path.name
        try:
            staged_tfrecord.symlink_to(tfrecord_path)
        except OSError as error:
            raise WaymoConversionError(
                f"Could not stage TFRecord symlink {staged_tfrecord} -> {tfrecord_path}"
            ) from error

        command = [
            bazel_command,
            "run",
            "//tools/data_converter/waymo:convert",
            "--",
            "--root-dir",
            str(input_dir),
            "--output-dir",
            str(conversion_dir),
            "waymo-v4",
            "--profile",
            "separate-sensors",
            "--world-global-mode",
            "localized",
        ]
        try:
            subprocess.run(command, cwd=ncore_repo, check=True)
        except FileNotFoundError as error:
            raise WaymoConversionError("Bazel is required to run NVIDIA NCore's official Waymo converter.") from error
        except subprocess.CalledProcessError as error:
            raise WaymoConversionError(
                f"Official NCore Waymo conversion failed with exit code {error.returncode}."
            ) from error

    written_metadata_paths = [
        path
        for path in _metadata_paths(conversion_dir)
        if existing_metadata_mtimes.get(path) != path.stat().st_mtime_ns
    ]
    if len(written_metadata_paths) != 1:
        raise WaymoConversionError(
            "Official NCore conversion did not produce exactly one metadata JSON for the requested TFRecord. "
            f"Found: {[str(path) for path in written_metadata_paths]}"
        )
    metadata_path = written_metadata_paths[0]
    registry[registry_key] = str(metadata_path.relative_to(conversion_dir))
    _store_metadata_registry(conversion_dir, registry)
    return metadata_path
