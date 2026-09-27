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

"""How BathyNerfDataParser pairs medium masks with images when the NPZ lists none.

A mask belongs to the image with exactly the same filename stem. The extension does not take
part in the match, and the extension check itself ignores case. The stem comparison is case
sensitive. Masks are searched recursively below ``medium_masks/`` in the dataset root, or below
``medium_mask_dir`` when the config sets one (relative paths resolve against the dataset root).

Once a mask directory exists, every image of the split needs its mask. If one or more are
missing, loading aborts with an error that names the split and the unmatched images (at most ten,
plus a count of the rest). Without any mask directory the model falls back to the geometric water
plane, which ``test_dataparser_messages.py`` covers.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import torch
from PIL import Image

import bathyfacto.bathyfacto_dataparser as dataparser_module
from bathyfacto.bathyfacto_datamanager import BathyInputDataset
from bathyfacto.bathyfacto_dataparser import BathyNerfDataParser, BathyNerfDataParserConfig

_H, _W = 4, 6
_IMAGE_NAMES = ["cam_a.jpg", "cam_b.jpg", "cam_c.jpg"]


@pytest.fixture
def recorder(record_console):
    return record_console(dataparser_module)


def _water_rows(index: int) -> np.ndarray:
    """A mask unique to image ``index``: the bottom ``index + 1`` rows are water."""
    mask = np.zeros((_H, _W), dtype=np.uint8)
    mask[_H - (index + 1) :, :] = 255
    return mask


def _write_dataset(root: Path, image_names: List[str]) -> None:
    """Images plus a ``train.npz`` that lists no medium masks, so the parser has to match them."""
    (root / "images").mkdir()
    for name in image_names:
        Image.fromarray(np.full((_H, _W, 3), 100, dtype=np.uint8), mode="RGB").save(root / "images" / name)

    n = len(image_names)
    cameras: Dict[str, np.ndarray] = {
        "fx": np.full((n, 1), 10.0, dtype=np.float32),
        "fy": np.full((n, 1), 10.0, dtype=np.float32),
        "cx": np.full((n, 1), _W / 2, dtype=np.float32),
        "cy": np.full((n, 1), _H / 2, dtype=np.float32),
        "height": np.full((n, 1), _H, dtype=np.int64),
        "width": np.full((n, 1), _W, dtype=np.int64),
        "camera_to_worlds": np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        "camera_type": np.ones((n, 1), dtype=np.int64),
    }
    np.savez(
        root / "train.npz",
        image_filenames=np.array([f"images/{name}" for name in image_names]),
        cameras=np.array(cameras, dtype=object),
        scene_box=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        metadata=np.array({}, dtype=object),
    )


def _write_mask(path: Path, index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_water_rows(index), mode="L").save(path, format="PNG")


def _parse(root: Path, **config):
    return BathyNerfDataParser(BathyNerfDataParserConfig(data=root, **config))._generate_dataparser_outputs("train")


def _assert_masks_follow_images(outputs, expected_masks: List[Path]) -> None:
    assert outputs.metadata["medium_mask_filenames"] == expected_masks
    dataset = BathyInputDataset(outputs)
    for index in range(len(expected_masks)):
        medium_mask = dataset.get_data(index)["medium_mask"]
        expected = torch.from_numpy(_water_rows(index) != 0).unsqueeze(-1)
        assert torch.equal(medium_mask, expected), f"image {index} got the wrong medium mask"


def test_png_masks_match_jpg_images_by_stem_in_image_order(tmp_path, recorder):
    """Mask extension differs from the image extension and the masks sit in a subfolder."""
    _write_dataset(tmp_path, _IMAGE_NAMES)
    masks = [tmp_path / "medium_masks" / "nested" / f"cam_{c}.png" for c in "abc"]
    # Written in reverse order, so directory order cannot produce the right pairing by accident.
    for index in reversed(range(3)):
        _write_mask(masks[index], index)

    outputs = _parse(tmp_path)

    _assert_masks_follow_images(outputs, masks)
    assert recorder.messages == ["bathyfacto  train: 3 images, 3/3 medium masks, 1.00 m per scene unit"]


def test_uppercase_mask_extension_is_recognised(tmp_path, recorder):
    _write_dataset(tmp_path, _IMAGE_NAMES)
    masks = [tmp_path / "medium_masks" / f"cam_{c}.PNG" for c in "abc"]
    for index, path in enumerate(masks):
        _write_mask(path, index)

    _assert_masks_follow_images(_parse(tmp_path), masks)


def test_one_missing_mask_aborts_loading_and_names_the_image(tmp_path, recorder):
    _write_dataset(tmp_path, _IMAGE_NAMES)
    _write_mask(tmp_path / "medium_masks" / "cam_a.png", 0)
    _write_mask(tmp_path / "medium_masks" / "cam_c.png", 2)

    with pytest.raises(ValueError) as excinfo:
        _parse(tmp_path)

    message = str(excinfo.value)
    assert message.startswith("bathyfacto  train: 1 of 3 images have no medium mask")
    assert "cam_b.jpg" in message
    assert "cam_a.jpg" not in message and "cam_c.jpg" not in message


def test_stem_match_is_case_sensitive(tmp_path, recorder):
    """``CAM_B.png`` is not the mask of ``cam_b.jpg``: it counts as missing and loading aborts."""
    _write_dataset(tmp_path, _IMAGE_NAMES)
    _write_mask(tmp_path / "medium_masks" / "cam_a.png", 0)
    _write_mask(tmp_path / "medium_masks" / "CAM_B.png", 1)
    _write_mask(tmp_path / "medium_masks" / "cam_c.png", 2)

    with pytest.raises(ValueError, match="bathyfacto  train: 1 of 3 images") as excinfo:
        _parse(tmp_path)

    assert "cam_b.jpg" in str(excinfo.value)


def test_the_list_of_unmatched_images_stops_at_ten_plus_a_count(tmp_path, recorder):
    names = [f"img_{i:02d}.jpg" for i in range(13)]
    _write_dataset(tmp_path, names)
    _write_mask(tmp_path / "medium_masks" / "img_00.png", 0)

    with pytest.raises(ValueError) as excinfo:
        _parse(tmp_path)

    message = str(excinfo.value)
    assert message.startswith("bathyfacto  train: 12 of 13 images have no medium mask")
    listed = [name for name in names if name in message]
    assert listed == names[1:11]
    assert "and 2 more" in message


def test_an_empty_mask_directory_aborts_loading(tmp_path, recorder):
    _write_dataset(tmp_path, _IMAGE_NAMES)
    (tmp_path / "medium_masks").mkdir()

    with pytest.raises(ValueError, match="bathyfacto  train: 3 of 3 images have no medium mask"):
        _parse(tmp_path)


def test_mask_directory_from_the_config_resolves_against_the_dataset_root(tmp_path, recorder):
    """``medium_mask_dir`` replaces ``medium_masks/``, whose masks are then ignored."""
    _write_dataset(tmp_path, _IMAGE_NAMES)
    masks = [tmp_path / "water" / f"cam_{c}.png" for c in "abc"]
    for index, path in enumerate(masks):
        _write_mask(path, index)
    # A decoy in the default location, paired the other way round.
    for index, c in enumerate("abc"):
        _write_mask(tmp_path / "medium_masks" / f"cam_{c}.png", 2 - index)

    _assert_masks_follow_images(_parse(tmp_path, medium_mask_dir=Path("water")), masks)
    _assert_masks_follow_images(_parse(tmp_path, medium_mask_dir=tmp_path / "water"), masks)


def test_a_configured_mask_directory_that_does_not_exist_aborts_loading(tmp_path, recorder):
    """An explicit ``medium_mask_dir`` must not fall back to the plane, even if ``medium_masks/`` exists."""
    _write_dataset(tmp_path, _IMAGE_NAMES)
    for index, c in enumerate("abc"):
        _write_mask(tmp_path / "medium_masks" / f"cam_{c}.png", index)

    with pytest.raises(ValueError, match="bathyfacto  train: ") as excinfo:
        _parse(tmp_path, medium_mask_dir=Path("water"))

    assert str(tmp_path / "water") in str(excinfo.value)
