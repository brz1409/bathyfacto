# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# Copyright 2026 Markus Brezovsky, TU Wien
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
"""Dataparser for BathyFacto NPZ datasets.

Adds to upstream's minimal parser: the water-plane metadata, medium masks matched by filename
stem, pre-downscaled ``<split>_<factor>x.npz`` splits, and the normalization metadata the
exporter needs to map points back to the global frame.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs, Semantics
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class BathyNerfDataParserConfig(DataParserConfig):
    """Configuration of the BathyFacto NPZ dataparser."""

    _target: Type = field(default_factory=lambda: BathyNerfDataParser)
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
    use_alpha_as_sampling_mask: bool = False
    """When True, an RGBA image's alpha channel additionally becomes the ray-sampling mask.

    Upstream nerfstudio never does this (``data["mask"]`` comes only from ``mask_filenames``),
    so the default is off. Alpha is used for ``image_valid_mask`` and folded into
    ``medium_mask`` either way. This flag only controls whether it also restricts where rays
    are sampled.
    """


def _as_float32(value: np.ndarray) -> np.ndarray:
    return value.astype(np.float32)


_NORMALIZATION_METADATA: Tuple[Tuple[str, Callable[[Any], Any]], ...] = (
    ("normalization_rotation", _as_float32),
    ("normalization_center", _as_float32),
    ("normalization_scale", float),
    ("original_water_normal", _as_float32),
    ("original_water_d", float),
    ("chunk_rotation", _as_float32),
    ("chunk_translation", _as_float32),
)
"""NPZ keys copied into the dataparser metadata, each with the cast applied on load."""


def _water_plane_and_scale_suffix(metadata: Dict[str, Any]) -> str:
    """Format the water plane height and metre scale for the train summary line.

    Reads ``metadata["water_surface"]["plane_model"]`` (scene-frame plane n.p + d = 0) and
    ``metadata["eff_scale"]``. Display only. A value that is missing, or a plane that is not
    (near-)horizontal, is left out of the suffix.
    """
    parts: List[str] = []
    plane_model = (metadata.get("water_surface") or {}).get("plane_model") or {}
    normal = plane_model.get("normal")
    d_value = plane_model.get("d")
    if normal is not None and d_value is not None and abs(float(normal[2])) > 1e-6:
        water_z = -float(d_value) / float(normal[2])
        parts.append(f"water plane z={water_z:.3f} (scene)")
    eff_scale = metadata.get("eff_scale")
    if eff_scale is not None:
        parts.append(f"{float(eff_scale):.2f} m per scene unit")
    return "".join(f", {part}" for part in parts)


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
        CONSOLE.log(f"[yellow]bathyfacto: {kind} path '{target}' is not a directory, skipping it.[/yellow]")
        return None
    return target


def _match_mask_filenames(
    image_paths: List[Path],
    mask_dir: Path,
    allowed_suffixes: Tuple[str, ...],
    kind: str,
    required_for_split: Optional[str] = None,
) -> Optional[List[Path]]:
    """Match mask files to image filenames by stem, returning ordered list or None.

    Args:
        required_for_split: When set, every image must have a mask. A missing one raises
            instead of returning None, and the error names this split.

    Raises:
        ValueError: If ``required_for_split`` is set and an image has no mask with its stem.
    """
    allowed = tuple(ext.lower() for ext in allowed_suffixes)
    mask_files = [path for path in mask_dir.rglob("*") if path.is_file() and path.suffix.lower() in allowed]
    if len(mask_files) == 0 and required_for_split is None:
        CONSOLE.log(f"[yellow]bathyfacto: no {kind} files found in '{mask_dir}', skipping.[/yellow]")
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
                f"[yellow]bathyfacto: duplicate {kind} candidates for '{stem}': {[str(p) for p in paths]}[/yellow]"
            )

    ordered_masks: List[Path] = []
    missing: List[Path] = []
    for image_path in image_paths:
        match = lookup.get(image_path.stem)
        if match is None:
            missing.append(image_path)
        else:
            ordered_masks.append(match)

    if missing and required_for_split is not None:
        listed = ", ".join(path.name for path in missing[:10])
        if len(missing) > 10:
            listed += f" and {len(missing) - 10} more"
        raise ValueError(
            f"bathyfacto  {required_for_split}: {len(missing)} of {len(image_paths)} images have no {kind} "
            f"in '{mask_dir}'. A mask must have the same filename stem as its image, letter case included. "
            f"Unmatched images: {listed}. Add the missing masks, or remove the directory to take the "
            "water from the geometric plane."
        )

    if missing:
        CONSOLE.log(
            f"[yellow]bathyfacto: missing {kind} files for {len(missing)} images, skipping attachment.[/yellow]"
        )
        for path in missing[:10]:
            CONSOLE.log(f"  - {path}")
        if len(missing) > 10:
            CONSOLE.log(f"  ... and {len(missing) - 10} more.")
        return None

    return ordered_masks


@dataclass
class BathyNerfDataParser(DataParser):
    """Loads the train/val/test/data NPZ splits written by ``bathyfacto-build-dataset``."""

    config: BathyNerfDataParserConfig

    def _split_candidates(self, split: str) -> List[Path]:
        """Candidate NPZ paths for a split, most specific first.

        With ``downscale_factor > 1`` the downscaled split (``train_2x.npz``) is preferred,
        matching what ``bathyfacto-build-dataset`` writes into the dataset root. The
        un-suffixed name stays as a fallback so a dataset built before the naming was
        unified still loads, and so ``downscale_factor`` 1 or None is unchanged.
        """
        factor = self.config.downscale_factor
        candidates: List[Path] = []
        if factor is not None and factor > 1:
            candidates.append(self.config.data / f"{split}_{factor}x.npz")
        candidates.append(self.config.data / f"{split}.npz")
        return candidates

    def _resolve_split_filepath(self, split: str) -> Tuple[Path, bool]:
        """Resolve the NPZ file for a split, falling back from test->val when needed.

        Returns the path and whether it is a pre-downscaled ``<split>_<factor>x.npz`` (the first
        candidate when ``downscale_factor > 1``), whose paths and intrinsics are already scaled.
        """
        candidates = self._split_candidates(split)
        for filepath in candidates:
            if filepath.exists():
                return filepath, filepath is not candidates[-1]

        if split == "test":
            fallbacks = self._split_candidates("val")
            for fallback in fallbacks:
                if fallback.exists():
                    CONSOLE.log(f"[yellow]bathyfacto  test: no test split, using {fallback.name}[/yellow]")
                    return fallback, fallback is not fallbacks[-1]

        tried = " or ".join(f"'{p}'" for p in candidates)
        raise FileNotFoundError(f"BathyNerfDataParser ({split}): expected dataset split file at {tried}.")

    def _generate_dataparser_outputs(self, split="train"):
        filepath, is_prescaled_split = self._resolve_split_filepath(split)
        with np.load(filepath, allow_pickle=True) as npz:
            data = {key: npz[key] for key in npz.files}
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

        # Surfaced to InputDataset, which decides whether an image's alpha channel also
        # becomes the ray-sampling mask (see BathyNerfDataParserConfig for the rationale).
        metadata["use_alpha_as_sampling_mask"] = self.config.use_alpha_as_sampling_mask

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

        # Set only when no medium_masks directory exists at all, which the summary line reports.
        no_medium_mask_dir: Optional[Path] = None
        if metadata.get("medium_mask_filenames") is None:
            medium_mask_dir = _resolve_mask_dir(
                dataset_root, self.config.medium_mask_dir, "medium_masks", kind="medium mask"
            )
            configured = self.config.medium_mask_dir
            if configured is not None and medium_mask_dir is None:
                configured_path = configured if configured.is_absolute() else dataset_root / configured
                raise ValueError(
                    f"bathyfacto  {split}: the configured medium_mask_dir '{configured_path}' is not an "
                    "existing directory. Fix the path, or drop the setting to use medium_masks/ in the "
                    "dataset or, without it, the geometric water plane."
                )
            if medium_mask_dir is not None:
                metadata["medium_mask_filenames"] = _match_mask_filenames(
                    image_filenames,
                    medium_mask_dir,
                    self.config.medium_mask_extensions,
                    kind="medium mask",
                    required_for_split=split,
                )
            else:
                no_medium_mask_dir = dataset_root / "medium_masks"

        downscale_factor = self.config.downscale_factor
        if downscale_factor is None:
            downscale_factor = 1
        if not isinstance(downscale_factor, int) or downscale_factor < 1:
            raise ValueError(f"BathyNerfDataParser: downscale_factor must be a positive int, got {downscale_factor}.")

        # bathyfacto.build_dataset writes a dedicated f"{split}_{factor}x.npz" per scale
        # (preferred by _split_candidates above), whose image_filenames already point into
        # images_{factor}/... and whose camera intrinsics are already scaled down. Remapping
        # paths and rescaling intrinsics again below would apply both corrections twice
        # (images_{factor}_{factor}, intrinsics divided by factor^2). Only the historical
        # fallback -- a plain f"{split}.npz" at full resolution, with images_{factor}/
        # written separately by an old dataset -- still needs the remap and rescale done here.
        needs_remap = downscale_factor > 1 and not is_prescaled_split

        def _remap_downscale_path(path: Path, folder_name: str) -> Path:
            # Paths inside the dataset root were joined onto it by _resolve_dataset_path, which
            # leaves them relative when --data is relative. Strip the root either way so it is
            # prefixed exactly once below. Absolute paths outside the root stay absolute.
            try:
                rel_path = path.relative_to(dataset_root)
                inside_root = True
            except ValueError:
                rel_path = path
                inside_root = False

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
                    f"[yellow]bathyfacto  {split}: expected '{folder_name}' in path '{rel_path}', "
                    f"remapping to '{new_rel}'.[/yellow]"
                )

            return dataset_root / new_rel if inside_root else new_rel

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

        if needs_remap:
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
                f"BathyNerfDataParser ({split}): scene_box has unexpected shape "
                f"{tuple(scene_box_aabb.shape)}, expected (2, 3) or (3, 2)."
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
        if needs_remap:
            cameras.rescale_output_resolution(scaling_factor=1.0 / downscale_factor)

        applied_scale = 1.0
        applied_transform = torch.eye(4, dtype=torch.float32)[:3, :]
        if "applied_scale" in data.keys():
            applied_scale = float(data["applied_scale"])
        if "applied_transform" in data.keys():
            applied_transform = data["applied_transform"].astype(np.float32)
            assert applied_transform.shape == (3, 4)

        # Normalization and Metashape chunk transform: the exporter maps points back to the
        # original global frame with these.
        for key, cast in _NORMALIZATION_METADATA:
            if key in data.keys():
                metadata[key] = cast(data[key])

        # Metres per scene unit, shown in the train summary line.
        metadata["eff_scale"] = float(metadata.get("normalization_scale", 1.0)) * float(applied_scale)

        marker_positions = None
        if "marker_positions" in data.keys():
            marker_positions = torch.from_numpy(data["marker_positions"]).float()
            metadata["marker_positions"] = marker_positions

        # One summary line per split.
        num_images = len(image_filenames)
        medium_mask_filenames = metadata.get("medium_mask_filenames")
        if medium_mask_filenames is not None:
            num_masks = len(medium_mask_filenames)
            if num_masks != num_images:
                CONSOLE.log(
                    f"[yellow]bathyfacto  {split}: medium mask count mismatch, "
                    f"{num_masks} masks for {num_images} images.[/yellow]"
                )
            summary = f"bathyfacto  {split}: {num_images} images, {num_masks}/{num_images} medium masks"
            if split == "train":
                summary += _water_plane_and_scale_suffix(metadata)
            CONSOLE.log(summary)
        elif no_medium_mask_dir is not None:
            CONSOLE.log(
                f"[yellow]bathyfacto  {split}: {num_images} images, no medium masks in "
                f"{no_medium_mask_dir}, water comes from the geometric plane.[/yellow]"
            )
        else:
            CONSOLE.log(f"[yellow]bathyfacto  {split}: no medium masks, water comes from the geometric plane.[/yellow]")

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
