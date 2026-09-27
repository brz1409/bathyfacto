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

"""CPU smoke test of the whole ``bathyfacto`` method: train, evaluate, export.

The registered ``bathyfacto`` config is used with only the sizes shrunk and the torch backend
selected, so that it runs without tiny-cuda-nn. Twenty training iterations run through the
pipeline, the pipeline's all-images evaluation scores the val split, and the point cloud export
renders the ``data`` split.

The synthetic scene is four cameras near the edge of the scene box looking straight down on a
water plane at z=-0.2, with the lower half of every image marked as water. Rays leaving the box
early accumulate little density, so both sides of the export's accumulation threshold are
populated.

The images are 160x120. A second, untrained pipeline on 64x48 images checks that the colour bar
on the eval images does not break the evaluation of images smaller than the bar.
"""

import copy
import dataclasses
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from PIL import Image

from bathyfacto.bathyfacto_config import BathyFactoMethod
from bathyfacto.bathyfacto_datamanager import MEDIUM_MASK_KEY, BathyDataManagerConfig, attach_medium_mask
from bathyfacto.bathyfacto_model import BathyFactoModelConfig
from bathyfacto.export import generate_bathy_point_cloud, get_bathy_export_dataset

_H, _W = 120, 160
_N_IMAGES = 4
_SPLITS = {"train": [0, 1, 2], "val": [3], "data": [0, 1, 2, 3]}
_TRAIN_STEPS = 20
_ACCUMULATION_THRESHOLD = 0.5
"""The threshold the published cloud-to-mesh numbers were measured with."""


def _write_dataset(root: Path, height: int = _H, width: int = _W) -> None:
    (root / "images").mkdir()
    (root / "medium_masks").mkdir()
    rng = np.random.default_rng(0)
    camera_to_worlds = []
    for i in range(_N_IMAGES):
        image = np.zeros((height, width, 3), dtype=np.int64)
        image[: height // 2] = (90, 140, 60)
        image[height // 2 :] = (40, 90, 150)
        image = np.clip(image + rng.integers(-10, 10, image.shape), 0, 255).astype(np.uint8)
        Image.fromarray(image).save(root / "images" / f"{i:04d}.png")
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 2 :] = 255
        Image.fromarray(mask, mode="L").save(root / "medium_masks" / f"{i:04d}.png")
        pose = np.eye(4, dtype=np.float32)  # identity rotation: the camera looks along -z
        pose[:3, 3] = [0.7 + 0.05 * i, 0.05 * i, 0.3]
        camera_to_worlds.append(pose)
    camera_to_worlds = np.stack(camera_to_worlds)

    metadata = {"water_surface": {"plane_model": {"normal": [0.0, 0.0, 1.0], "d": 0.2}}}
    for split, indices in _SPLITS.items():
        k = len(indices)
        cameras = {
            "fx": np.full((k, 1), 60.0, dtype=np.float32),
            "fy": np.full((k, 1), 60.0, dtype=np.float32),
            "cx": np.full((k, 1), width / 2, dtype=np.float32),
            "cy": np.full((k, 1), height / 2, dtype=np.float32),
            "height": np.full((k, 1), height, dtype=np.int64),
            "width": np.full((k, 1), width, dtype=np.int64),
            "camera_to_worlds": camera_to_worlds[indices],
            "camera_type": np.ones((k, 1), dtype=np.int64),
        }
        np.savez(
            root / f"{split}.npz",
            image_filenames=np.array([f"images/{i:04d}.png" for i in indices]),
            cameras=np.array(cameras, dtype=object),
            scene_box=np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            metadata=np.array(metadata, dtype=object),
            normalization_rotation=np.eye(3, dtype=np.float32),
            normalization_center=np.zeros(3, dtype=np.float32),
            normalization_scale=np.float32(1.0),
            chunk_rotation=np.eye(3, dtype=np.float32),
            chunk_translation=np.zeros(3, dtype=np.float32),
            applied_scale=np.float32(1.0),
        )


def _setup_pipeline(root: Path):
    """The registered ``bathyfacto`` pipeline on the dataset at ``root``, shrunk for the CPU."""
    config = copy.deepcopy(BathyFactoMethod.config)
    datamanager = config.pipeline.datamanager
    model = config.pipeline.model
    assert isinstance(datamanager, BathyDataManagerConfig) and isinstance(model, BathyFactoModelConfig)
    # dataclasses.replace rejects unknown field names, so a typo here fails instead of being ignored.
    config.pipeline.datamanager = dataclasses.replace(
        datamanager, data=root, train_num_rays_per_batch=256, eval_num_rays_per_batch=256
    )
    config.pipeline.model = dataclasses.replace(
        model,
        implementation="torch",
        average_init_density=1.0,
        eval_num_rays_per_chunk=1 << 12,
        num_levels=4,
        max_res=128,
        log2_hashmap_size=12,
        hidden_dim=16,
        hidden_dim_color=16,
        num_proposal_iterations=1,
        proposal_net_args_list=[
            {"hidden_dim": 16, "log2_hashmap_size": 12, "num_levels": 4, "max_res": 64, "use_linear": False}
        ],
        num_proposal_samples_per_ray=(32,),
        num_nerf_samples_per_ray=16,
    )

    return config, config.pipeline.setup(device="cpu", test_mode="val")


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Dict[str, Any]:
    """The ``bathyfacto`` pipeline after a few CPU training iterations, plus the losses seen."""
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("smoke_dataset")
    _write_dataset(root)
    config, pipeline = _setup_pipeline(root)
    optimizers = Optimizers(config.optimizers, pipeline.get_param_groups())
    callbacks = pipeline.get_training_callbacks(
        TrainingCallbackAttributes(optimizers=optimizers, grad_scaler=None, pipeline=pipeline, trainer=None)
    )
    losses: List[float] = []
    for step in range(_TRAIN_STEPS):
        for callback in callbacks:
            callback.run_callback_at_location(step, TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
        optimizers.zero_grad_all()
        _, loss_dict, _ = pipeline.get_train_loss_dict(step)
        loss = torch.stack(list(loss_dict.values())).sum()
        loss.backward()
        optimizers.optimizer_step_all()
        for callback in callbacks:
            callback.run_callback_at_location(step, TrainingCallbackLocation.AFTER_TRAIN_ITERATION)
        losses.append(float(loss))
    return {"pipeline": pipeline, "losses": losses}


def test_training_runs_and_the_loss_goes_down(trained):
    losses = trained["losses"]
    assert all(np.isfinite(losses))
    assert np.mean(losses[-5:]) < np.mean(losses[:5])


def test_all_images_evaluation_gives_finite_image_metrics(trained):
    pipeline = trained["pipeline"]
    metrics = pipeline.get_average_image_metrics(pipeline.datamanager.fixed_indices_eval_dataloader, "eval")

    for key in ("psnr", "ssim", "lpips"):
        assert key in metrics and np.isfinite(metrics[key]), f"{key}: {metrics.get(key)}"


def test_all_images_evaluation_runs_on_images_smaller_than_the_colour_bar(tmp_path):
    """At 64x48 the colour bar box does not fit into the image. The metrics must still come out."""
    torch.manual_seed(0)
    _write_dataset(tmp_path, height=48, width=64)
    _, pipeline = _setup_pipeline(tmp_path)

    metrics = pipeline.get_average_image_metrics(pipeline.datamanager.fixed_indices_eval_dataloader, "eval")

    for key in ("psnr", "ssim", "lpips"):
        assert key in metrics and np.isfinite(metrics[key]), f"{key}: {metrics.get(key)}"


def test_export_keeps_exactly_the_rays_above_the_accumulation_threshold(trained):
    """One point per rendered ray with accumulation above 0.5, none for the others.

    The reference count renders the same rays independently, in eval mode like the export.
    """
    pipeline = trained["pipeline"]
    pipeline.eval()

    dataset, _ = get_bathy_export_dataset(pipeline)
    above = below = 0
    for index in range(len(dataset)):
        rays = dataset.cameras.generate_rays(camera_indices=index, keep_shape=True)
        attach_medium_mask(rays, dataset[index].get(MEDIUM_MASK_KEY))
        with torch.no_grad():
            accumulation = pipeline.model.get_outputs_for_camera_ray_bundle(rays)["accumulation"].reshape(-1)
        above += int((accumulation > _ACCUMULATION_THRESHOLD).sum())
        below += int((accumulation <= _ACCUMULATION_THRESHOLD).sum())
    assert above > 0 and below > 0, "the scene must populate both sides of the threshold"

    pcd = generate_bathy_point_cloud(pipeline, num_points=above + below)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    assert points.shape == (above, 3)
    assert np.isfinite(points).all()
    assert colors.shape == (above, 3) and ((colors >= 0.0) & (colors <= 1.0)).all()
    assert (points[:, 2] < -0.2).any(), "the water half of the images must yield points below the water plane"


def test_export_after_an_evaluation_matches_the_eval_mode_export_and_restores_the_mode(trained):
    """``get_average_image_metrics`` hands the pipeline back in train mode. The export must still
    render in eval mode, give the same points as on an eval-mode pipeline, and leave the mode it
    found."""
    pipeline = trained["pipeline"]
    pipeline.eval()
    reference = np.asarray(generate_bathy_point_cloud(pipeline, num_points=500).points)
    assert not pipeline.training

    pipeline.get_average_image_metrics(pipeline.datamanager.fixed_indices_eval_dataloader, "eval")
    assert pipeline.training, "precondition: the evaluation leaves the pipeline in train mode"

    points = np.asarray(generate_bathy_point_cloud(pipeline, num_points=500).points)

    assert pipeline.training and pipeline.model.training
    np.testing.assert_array_equal(points, reference)
