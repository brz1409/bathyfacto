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

"""The markers.xml alignment filter must survive downscaling.

``downscale_images_and_masks`` derives its image list from ``markers.xml``: sorted by camera
label and restricted to cameras that carry a ``<transform>``, i.e. the ones Metashape aligned.
``extract_camera_transforms`` uses the same contract to build the pose array. An unfiltered
list (``sorted(image_dir.glob("*"))``) would keep the image of an unaligned camera, which has no
pose, so the two parallel arrays would go out of step and every later image would be paired with
the wrong pose.

The fixture places the unaligned camera in the middle of the label order, so a lost filter
shifts entries instead of appending a spare one.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from bathyfacto.metashape_xml import downscale_images_and_masks, extract_camera_transforms, parse_metashape_xml

# Label order (alphabetical, which is how both readers sort):
#   a_aligned, b_unaligned, c_aligned, d_aligned
# b_unaligned carries no <transform> -> it has no pose and must be dropped from the image list.
_ALL_STEMS = ["a_aligned", "b_unaligned", "c_aligned", "d_aligned"]
_ALIGNED = ["a_aligned", "c_aligned", "d_aligned"]
_UNALIGNED = "b_unaligned"

_MARKERS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<document>
  <chunk>
    <cameras>
      <camera id="0" label="a_aligned"><transform>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</transform></camera>
      <camera id="1" label="b_unaligned"/>
      <camera id="2" label="c_aligned"><transform>1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1</transform></camera>
      <camera id="3" label="d_aligned"><transform>1 0 0 2 0 1 0 0 0 0 1 0 0 0 0 1</transform></camera>
    </cameras>
  </chunk>
</document>
"""


def _build_fixture(root: Path, *, with_masks: bool = True) -> tuple[Path, Path, Path]:
    """Write markers.xml, 4 images and (optionally) 4 medium masks. Returns dirs."""
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for stem in _ALL_STEMS:
        Image.fromarray(np.full((8, 8, 3), 100, dtype=np.uint8), mode="RGB").save(image_dir / f"{stem}.png")

    mask_dir = root / "medium_masks"
    if with_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
        for stem in _ALL_STEMS:
            Image.fromarray(np.full((8, 8), 255, dtype=np.uint8), mode="L").save(mask_dir / f"{stem}.png")

    (root / "markers.xml").write_text(_MARKERS_XML)
    return image_dir, mask_dir, root


def _stems(paths: list) -> list:
    return [Path(p).stem for p in paths]


def test_pose_reader_drops_the_unaligned_camera(tmp_path):
    """Baseline: this is the list the image list has to agree with."""
    _image_dir, _mask_dir, base = _build_fixture(tmp_path)

    transforms, pose_image_names = extract_camera_transforms(parse_metashape_xml(base / "markers.xml"))

    assert transforms.shape[0] == 3
    assert _stems(pose_image_names) == _ALIGNED


def test_downscale_branch_matches_the_pose_list(tmp_path):
    image_dir, mask_dir, base = _build_fixture(tmp_path)
    _transforms, pose_image_names = extract_camera_transforms(parse_metashape_xml(base / "markers.xml"))

    images, masks, dims = downscale_images_and_masks(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_base_dir=base,
        num_downscales=1,
        verbose=False,
        xml_root=parse_metashape_xml(base / "markers.xml"),
    )

    assert _stems(images[2]) == _stems(pose_image_names)
    assert _UNALIGNED not in _stems(images[2])
    assert masks[2] is not None
    assert _stems(masks[2]) == _ALIGNED, "downscaled masks must stay index-aligned with images"
    assert dims[2] == (4, 4)


def test_masks_are_matched_by_stem_not_by_sort_position(tmp_path):
    """Masks are a parallel array of the images; the mask set is not filtered the same way."""
    image_dir, mask_dir, base = _build_fixture(tmp_path)

    images, masks, _dims = downscale_images_and_masks(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_base_dir=base,
        num_downscales=1,
        verbose=False,
        xml_root=parse_metashape_xml(base / "markers.xml"),
    )

    assert masks[2] is not None
    assert len(masks[2]) == len(images[2])
    assert _stems(masks[2]) == _stems(images[2]), "each mask must belong to the image at the same index"


def test_missing_mask_yields_none_not_a_shifted_list(tmp_path):
    """A gap in the mask set must produce a None slot, never a shorter (shifted) list."""
    image_dir, mask_dir, base = _build_fixture(tmp_path)
    (mask_dir / "c_aligned.png").unlink()  # c_aligned is index 1 among the aligned cameras

    images, masks, _dims = downscale_images_and_masks(
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_base_dir=base,
        num_downscales=1,
        verbose=False,
        xml_root=parse_metashape_xml(base / "markers.xml"),
    )

    assert masks[2] is not None
    assert len(masks[2]) == len(images[2]) == 3
    assert masks[2][1] is None, "the image with no mask must get a None slot at its own index"
    assert Path(masks[2][0]).stem == "a_aligned"
    assert Path(masks[2][2]).stem == "d_aligned", "the last entry must not have shifted up"


def test_zero_downscales_is_rejected(tmp_path):
    """The builder only downscales for ``--num-downscales >= 1``; zero levels is a caller error."""
    import pytest

    image_dir, mask_dir, base = _build_fixture(tmp_path)

    with pytest.raises(ValueError, match="num_downscales"):
        downscale_images_and_masks(
            image_dir=image_dir, mask_dir=mask_dir, output_base_dir=base, num_downscales=0, xml_root=None
        )
