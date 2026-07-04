# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Data parser for pre-prepared datasets for all cameras, with no additional processing needed
Optional fields - semantics, mask_filenames, cameras.distortion_params, cameras.times
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs, Semantics
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class BathyNerfDataParserConfig(DataParserConfig):
    """BathyNerf dataset config"""

    _target: Type = field(default_factory=lambda: BathyNerfDataParser)
    """target class to instantiate"""
    data: Path = Path("data/bathynerf/default")
    mask_dir: Optional[Path] = None
    """Directory containing per-image foreground masks (optional).

    Masks must share the same filename stem as the corresponding image (e.g., image
    `images/DJI_..._0503.JPG` matches `masks/DJI_..._0503.png|jpg|jpeg`).
    """
    medium_mask_dir: Optional[Path] = None
    """Directory containing per-image medium masks used by two-media models (optional).

    Medium masks follow the same naming rule as masks: same filename stem as the image.
    """
    mask_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
    """Recognized file extensions for classic masks."""
    medium_mask_extensions: Tuple[str, ...] = (".png", ".jpg", ".jpeg")
    """Recognized file extensions for medium masks."""
    scene_box_config: Optional[Dict[str, Any]] = None
    """Scene box metadata loaded from dataset files for bookkeeping (e.g., aabb_scale)."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. Expects images in images_{factor} (and masks_{factor}/medium_masks_{factor} if used)."""


def _resolve_mask_dir(base_dir: Path, provided: Optional[Path], default_subdir: str, kind: str) -> Optional[Path]:
    """Resolve a directory containing masks relative to the dataset root."""
    target: Optional[Path]
    if provided is None:
        target = base_dir / default_subdir
    else:
        target = provided if provided.is_absolute() else (base_dir / provided)

    if target is None or not target.exists():
        return None
    if not target.is_dir():
        CONSOLE.log(f"[yellow]BathyNerfDataParser: {kind} path '{target}' is not a directory; skipping.[/yellow]")
        return None
    return target


def _match_mask_filenames(
    image_paths: List[Path], mask_dir: Path, allowed_suffixes: Tuple[str, ...], kind: str
) -> Optional[List[Path]]:
    """Match mask files to image filenames by stem, returning ordered list or None."""
    allowed = tuple(ext.lower() for ext in allowed_suffixes)
    mask_files = [path for path in mask_dir.rglob("*") if path.is_file() and path.suffix.lower() in allowed]
    if len(mask_files) == 0:
        CONSOLE.log(f"[yellow]BathyNerfDataParser: no {kind} files found in '{mask_dir}', skipping.[/yellow]")
        return None

    lookup: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}
    for mask_path in mask_files:
        stem = mask_path.stem
        if stem in lookup:
            duplicates.setdefault(stem, [lookup[stem]]).append(mask_path)
        else:
            lookup[stem] = mask_path

    if duplicates:
        for stem, paths in duplicates.items():
            CONSOLE.log(
                f"[yellow]BathyNerfDataParser: duplicate {kind} candidates for '{stem}': {[str(p) for p in paths]}[/yellow]"
            )

    ordered_masks: List[Path] = []
    missing: List[Path] = []
    for image_path in image_paths:
        match = lookup.get(image_path.stem)
        if match is None:
            missing.append(image_path)
        else:
            ordered_masks.append(match)

    if missing:
        CONSOLE.log(
            f"[yellow]BathyNerfDataParser: missing {kind} files for {len(missing)} images; skipping attachment.[/yellow]"
        )
        for path in missing[:10]:
            CONSOLE.log(f"  - {path}")
        if len(missing) > 10:
            CONSOLE.log(f"  ... and {len(missing) - 10} more.")
        return None

    return ordered_masks


@dataclass
class BathyNerfDataParser(DataParser):
    """BathyNerf DatasetParser"""

    config: BathyNerfDataParserConfig

    def _resolve_split_filepath(self, split: str) -> Path:
        """Resolve the NPZ file for a split, falling back from test->val when needed."""
        filepath = self.config.data / f"{split}.npz"
        if filepath.exists():
            return filepath

        if split == "test":
            fallback = self.config.data / "val.npz"
            if fallback.exists():
                CONSOLE.log("[yellow]BathyNerfDataParser (test): test.npz not found; falling back to val.npz.[/yellow]")
                return fallback

        raise FileNotFoundError(f"BathyNerfDataParser ({split}): expected dataset split file at '{filepath}'.")

    def _generate_dataparser_outputs(self, split="train"):
        filepath = self._resolve_split_filepath(split)
        data = np.load(filepath, allow_pickle=True)
        dataset_root = filepath.parent

        def _resolve_dataset_path(path_value) -> Path:
            path_obj = Path(path_value)
            return path_obj if path_obj.is_absolute() else dataset_root / path_obj

        image_filenames = [_resolve_dataset_path(path) for path in data["image_filenames"].tolist()]
        mask_filenames = None
        if "mask_filenames" in data.keys():
            mask_filenames = [_resolve_dataset_path(path) for path in data["mask_filenames"].tolist()]

        metadata = {}
        if "metadata" in data.keys():
            metadata = data["metadata"].item()
            existing_medium_masks = metadata.get("medium_mask_filenames")
            if existing_medium_masks is not None:
                metadata["medium_mask_filenames"] = [
                    _resolve_dataset_path(path) for path in list(existing_medium_masks)
                ]
                CONSOLE.log(
                    f"[green]BathyNerfDataParser ({split}): Loaded {len(existing_medium_masks)} medium mask filenames from NPZ metadata[/green]"
                )

        if "semantics" in data.keys():
            semantics = data["semantics"].item()
            metadata["semantics"] = Semantics(
                filenames=[_resolve_dataset_path(path) for path in semantics["filenames"].tolist()],
                classes=semantics["classes"].tolist(),
                colors=torch.from_numpy(semantics["colors"]),
                mask_classes=semantics["mask_classes"].tolist(),
            )

        if mask_filenames is None:
            mask_dir = _resolve_mask_dir(dataset_root, self.config.mask_dir, "masks", kind="mask")
            if mask_dir is not None:
                matched_masks = _match_mask_filenames(
                    image_filenames, mask_dir, self.config.mask_extensions, kind="mask"
                )
                if matched_masks is not None:
                    mask_filenames = matched_masks

        if metadata.get("medium_mask_filenames") is None:
            medium_mask_dir = _resolve_mask_dir(
                dataset_root, self.config.medium_mask_dir, "medium_masks", kind="medium mask"
            )
            if medium_mask_dir is not None:
                CONSOLE.log(
                    f"[cyan]BathyNerfDataParser ({split}): Searching for medium masks in {medium_mask_dir}[/cyan]"
                )
                matched_medium_masks = _match_mask_filenames(
                    image_filenames, medium_mask_dir, self.config.medium_mask_extensions, kind="medium mask"
                )
                if matched_medium_masks is not None:
                    metadata["medium_mask_filenames"] = matched_medium_masks
                    CONSOLE.log(
                        f"[green]BathyNerfDataParser ({split}): Successfully matched {len(matched_medium_masks)} medium masks from filesystem[/green]"
                    )
                else:
                    CONSOLE.log(
                        f"[yellow]BathyNerfDataParser ({split}): WARNING: Medium mask directory found but no masks matched to images[/yellow]"
                    )
            else:
                CONSOLE.log(
                    f"[yellow]BathyNerfDataParser ({split}): No medium mask directory found (checked: {dataset_root / 'medium_masks'})[/yellow]"
                )

        downscale_factor = self.config.downscale_factor
        if downscale_factor is None:
            downscale_factor = 1
        if not isinstance(downscale_factor, int) or downscale_factor < 1:
            raise ValueError(f"BathyNerfDataParser: downscale_factor must be a positive int, got {downscale_factor}.")

        def _remap_downscale_path(path: Path, folder_name: str) -> Path:
            rel_path = path
            if path.is_absolute():
                try:
                    rel_path = path.relative_to(dataset_root)
                except ValueError:
                    rel_path = path

            parts = list(rel_path.parts)
            if folder_name in parts:
                idx = parts.index(folder_name)
                parts[idx] = f"{folder_name}_{downscale_factor}"
                new_rel = Path(*parts)
            else:
                parent = rel_path.parent
                if parent.name:
                    new_rel = parent.with_name(f"{parent.name}_{downscale_factor}") / rel_path.name
                else:
                    new_rel = Path(f"{folder_name}_{downscale_factor}") / rel_path.name
                CONSOLE.log(
                    f"[yellow]BathyNerfDataParser ({split}): expected '{folder_name}' in path '{rel_path}', "
                    f"remapping to '{new_rel}'.[/yellow]"
                )

            if path.is_absolute():
                try:
                    return dataset_root / new_rel if path.relative_to(dataset_root) else new_rel
                except ValueError:
                    return new_rel
            return dataset_root / new_rel

        def _remap_and_check(paths: List[Path], folder_name: str, kind: str) -> List[Path]:
            remapped: List[Path] = []
            missing: List[Path] = []
            for p in paths:
                new_p = _remap_downscale_path(p, folder_name)
                remapped.append(new_p)
                if not new_p.exists():
                    missing.append(new_p)
            if missing:
                preview = "\n".join(f"  - {p}" for p in missing[:10])
                more = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more."
                raise RuntimeError(
                    f"BathyNerfDataParser ({split}): missing downscaled {kind} files for factor {downscale_factor}.\n"
                    f"Expected under '{folder_name}_{downscale_factor}'.\n{preview}{more}"
                )
            return remapped

        if downscale_factor > 1:
            image_filenames = _remap_and_check(image_filenames, "images", "image")
            if mask_filenames is not None:
                mask_filenames = _remap_and_check(mask_filenames, "masks", "mask")
            if metadata.get("medium_mask_filenames") is not None:
                metadata["medium_mask_filenames"] = _remap_and_check(
                    list(metadata["medium_mask_filenames"]), "medium_masks", "medium mask"
                )

        scene_box_aabb = torch.from_numpy(data["scene_box"])
        if scene_box_aabb.shape == (3, 2):
            scene_box_aabb = scene_box_aabb.T
        elif scene_box_aabb.shape != (2, 3):
            raise ValueError(
                f"BathyNerfDataParser ({split}): scene_box has unexpected shape {tuple(scene_box_aabb.shape)}; "
                "expected (2, 3) or (3, 2)."
            )
        scene_box = SceneBox(aabb=scene_box_aabb)

        camera_np = data["cameras"].item()
        distortion_params = None
        if "distortion_params" in camera_np.keys():
            distortion_params = torch.from_numpy(camera_np["distortion_params"])
        cameras = Cameras(
            fx=torch.from_numpy(camera_np["fx"]),
            fy=torch.from_numpy(camera_np["fy"]),
            cx=torch.from_numpy(camera_np["cx"]),
            cy=torch.from_numpy(camera_np["cy"]),
            distortion_params=distortion_params,
            height=torch.from_numpy(camera_np["height"]),
            width=torch.from_numpy(camera_np["width"]),
            camera_to_worlds=torch.from_numpy(camera_np["camera_to_worlds"])[:, :3, :4],
            camera_type=torch.from_numpy(camera_np["camera_type"]),
            times=torch.from_numpy(camera_np["times"]) if "times" in camera_np.keys() else None,
        )
        if downscale_factor > 1:
            cameras.rescale_output_resolution(scaling_factor=1.0 / downscale_factor)

        applied_scale = 1.0
        applied_transform = torch.eye(4, dtype=torch.float32)[:3, :]
        if "applied_scale" in data.keys():
            applied_scale = float(data["applied_scale"])
        if "applied_transform" in data.keys():
            applied_transform = data["applied_transform"].astype(np.float32)
            assert applied_transform.shape == (3, 4)

        # Load normalization parameters for reversible export
        # These allow converting normalized coordinates back to original Metashape coordinates
        if "normalization_rotation" in data.keys():
            metadata["normalization_rotation"] = data["normalization_rotation"].astype(np.float32)
        if "normalization_center" in data.keys():
            metadata["normalization_center"] = data["normalization_center"].astype(np.float32)
        if "normalization_scale" in data.keys():
            metadata["normalization_scale"] = float(data["normalization_scale"])
        if "original_water_normal" in data.keys():
            metadata["original_water_normal"] = data["original_water_normal"].astype(np.float32)
        if "original_water_d" in data.keys():
            metadata["original_water_d"] = float(data["original_water_d"])

        # Load Metashape chunk transform for full global coordinate reversibility
        if "chunk_rotation" in data.keys():
            metadata["chunk_rotation"] = data["chunk_rotation"].astype(np.float32)
        if "chunk_translation" in data.keys():
            metadata["chunk_translation"] = data["chunk_translation"].astype(np.float32)

        # eff_scale: scene-units-to-global conversion factor used by downstream geometry helpers.
        metadata["eff_scale"] = float(metadata.get("normalization_scale", 1.0)) * float(applied_scale)

        marker_positions = None
        if "marker_positions" in data.keys():
            marker_positions = torch.from_numpy(data["marker_positions"]).float()
            metadata["marker_positions"] = marker_positions

        # Summary logging for medium masks
        if metadata.get("medium_mask_filenames") is not None:
            num_masks = len(metadata["medium_mask_filenames"])
            num_images = len(image_filenames)
            if num_masks == num_images:
                CONSOLE.log(
                    f"[bold green]✓ BathyNerfDataParser ({split}): Medium masks ready: {num_masks}/{num_images} images[/bold green]"
                )
            else:
                CONSOLE.log(
                    f"[bold yellow]⚠ BathyNerfDataParser ({split}): Medium mask count mismatch: {num_masks} masks for {num_images} images[/bold yellow]"
                )
        else:
            CONSOLE.log(
                f"[bold cyan]ℹ BathyNerfDataParser ({split}): No medium masks loaded (training will use all pixels as interface hits)[/bold cyan]"
            )

        metadata["data_dir"] = str(self.config.data)

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames,
            dataparser_transform=applied_transform,
            dataparser_scale=applied_scale,
            metadata=metadata,
        )
        return dataparser_outputs
