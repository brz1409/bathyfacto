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

"""Training-time diagnostics for BathyFacto: collapse guard, depth monitor, per-camera plots, W&B logging."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.model_components.two_media_geometry import water_plane_intersection
from nerfstudio.utils.rich_utils import CONSOLE


def metric_to_float(metric: Any) -> float:
    """Normalize torchmetrics-style outputs to a scalar float."""
    if isinstance(metric, tuple):
        metric = metric[0]
    if isinstance(metric, torch.Tensor):
        return float(metric.item())
    return float(metric)


def make_collapse_guard_callback(
    training_callback_attributes: TrainingCallbackAttributes,
    collapse_guard_enabled: bool,
    collapse_guard_warmup: int,
    collapse_guard_window: int,
    collapse_guard_spike_factor: float,
    collapse_guard_patience: int,
) -> Optional[TrainingCallback]:
    """Stop training when loss collapses (NaN or sustained spike above running median).

    Maintains a 'best model' checkpoint (``best-model.ckpt``) that survives
    ``save_only_latest_checkpoint`` cleanup.  When a collapse is detected the
    training is stopped and the best checkpoint path is printed so it can be
    used for export.

    Args:
        training_callback_attributes: passed in from get_training_callbacks.
        collapse_guard_enabled: if False, returns None immediately.
        collapse_guard_warmup: steps before detection activates.
        collapse_guard_window: rolling window size for median baseline.
        collapse_guard_spike_factor: spike multiplier threshold.
        collapse_guard_patience: consecutive spike steps before stopping.

    Returns:
        TrainingCallback or None if disabled/trainer unavailable.
    """
    if not collapse_guard_enabled:
        return None

    trainer = training_callback_attributes.trainer
    if trainer is None:
        return None

    warmup = collapse_guard_warmup
    window = collapse_guard_window
    spike_factor = collapse_guard_spike_factor
    patience = collapse_guard_patience

    loss_history: deque[float] = deque(maxlen=window)
    spike_counter = 0
    best_step = 0
    best_loss = float("inf")
    last_best_save_step = 0
    best_save_interval = 2000

    def _save_best(step: int) -> None:
        """Save a separate best-model checkpoint that won't be overwritten."""
        import shutil

        ckpt_dir = trainer.checkpoint_dir
        if not ckpt_dir.exists():
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best-model.ckpt"
        tmp_path = ckpt_dir / f"step-{step:09d}.ckpt"
        if tmp_path.exists():
            shutil.copy2(tmp_path, best_path)
        else:
            torch.save(
                {
                    "step": step,
                    "pipeline": trainer.pipeline.module.state_dict()
                    if hasattr(trainer.pipeline, "module")
                    else trainer.pipeline.state_dict(),
                    "optimizers": {k: v.state_dict() for (k, v) in trainer.optimizers.optimizers.items()},
                    "schedulers": {k: v.state_dict() for (k, v) in trainer.optimizers.schedulers.items()},
                    "scalers": trainer.grad_scaler.state_dict(),
                },
                best_path,
            )
        CONSOLE.print(
            f"[bold green][CollapseGuard] New best model saved at step {step} "
            f"(loss={best_loss:.6f}) → {best_path}[/bold green]"
        )

    def _check_collapse(step: int) -> None:
        nonlocal spike_counter, best_step, best_loss, last_best_save_step

        loss_val = getattr(trainer, "_collapse_guard_last_loss", None)
        if loss_val is None:
            return

        loss_f = float(loss_val)

        if np.isnan(loss_f):
            CONSOLE.print(f"[bold red][CollapseGuard] NaN loss at step {step}. Stopping training.[/bold red]")
            CONSOLE.print(
                f"[bold red][CollapseGuard] Best model: step {best_step}, loss {best_loss:.6f}. "
                f"Checkpoint: {trainer.checkpoint_dir / 'best-model.ckpt'}[/bold red]"
            )
            trainer.stop_training = True
            return

        if loss_f < best_loss:
            best_loss = loss_f
            best_step = step
            if step >= warmup and (step - last_best_save_step) >= best_save_interval:
                _save_best(step)
                last_best_save_step = step

        if step < warmup:
            loss_history.append(loss_f)
            return

        if len(loss_history) < 10:
            loss_history.append(loss_f)
            return

        sorted_hist = sorted(loss_history)
        median = sorted_hist[len(sorted_hist) // 2]

        if median > 0 and loss_f > spike_factor * median:
            spike_counter += 1
            if spike_counter >= patience:
                CONSOLE.print(
                    f"[bold red][CollapseGuard] Loss collapse detected at step {step}! "
                    f"Loss={loss_f:.6f}, median={median:.6f} ({loss_f / median:.1f}x). "
                    f"Stopping training.[/bold red]"
                )
                CONSOLE.print(
                    f"[bold red][CollapseGuard] Best model: step {best_step}, loss {best_loss:.6f}. "
                    f"Checkpoint: {trainer.checkpoint_dir / 'best-model.ckpt'}[/bold red]"
                )
                trainer.stop_training = True
                return
            elif spike_counter == 1:
                CONSOLE.print(
                    f"[bold yellow][CollapseGuard] Spike at step {step}: "
                    f"loss={loss_f:.6f} > {spike_factor}x median={median:.6f}. "
                    f"Patience {spike_counter}/{patience}.[/bold yellow]"
                )
        else:
            if spike_counter > 0:
                CONSOLE.print(
                    f"[green][CollapseGuard] Loss recovered at step {step}: "
                    f"loss={loss_f:.6f}. Spike counter reset.[/green]"
                )
            spike_counter = 0

        loss_history.append(loss_f)

    return TrainingCallback(
        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
        update_every_num_iters=1,
        func=_check_collapse,
    )


def make_depth_monitor_callback(
    training_callback_attributes: TrainingCallbackAttributes,
    model: Any,
) -> Optional[TrainingCallback]:
    """Periodic callback that logs depth error against a GT reference mesh.

    Args:
        training_callback_attributes: passed in from get_training_callbacks.
        model: the BathyFactoModel instance (typed as Any to avoid circular import).

    Returns:
        TrainingCallback firing every 1000 steps, or None if pipeline unavailable.
    """
    pipeline = training_callback_attributes.pipeline
    _st: Dict[str, Any] = {}

    @torch.no_grad()
    def _cb(step: int) -> None:
        if _st.get("off"):
            return
        if "rb" not in _st:
            try:
                import open3d as o3d

                from nerfstudio.exporter.bathy_pointcloud_utils import (
                    get_bathy_export_dataset,
                    transform_points_to_bathy_global_frame,
                )

                data_dir = pipeline.datamanager.dataparser.config.data
                if model.config.depth_monitor_mesh_name is not None:
                    mesh_path = data_dir / model.config.depth_monitor_mesh_name
                    if not mesh_path.exists():
                        print(f"[DepthMon] mesh not found: {mesh_path}, disabled")
                        _st["off"] = True
                        return
                else:
                    candidates = sorted(data_dir.glob("reference_mesh*.ply"))
                    if len(candidates) == 1:
                        mesh_path = candidates[0]
                    else:
                        print(
                            f"[DepthMon] expected 1 reference_mesh*.ply in {data_dir}, found {len(candidates)};"
                            " set depth_monitor_mesh_name explicitly — disabled"
                        )
                        _st["off"] = True
                        return

                mesh = o3d.io.read_triangle_mesh(str(mesh_path))
                ms = o3d.t.geometry.RaycastingScene()
                ms.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

                dataset, dp_out = get_bathy_export_dataset(pipeline)
                dev = model.device

                all_o, all_d, all_pa, all_ci, all_nr, all_fr = [], [], [], [], [], []
                all_gt, all_ip, all_sc = [], [], []

                for img_i in [0, len(dataset) // 2]:
                    cam = dataset.cameras[img_i : img_i + 1].to(dev)
                    batch = dataset[img_i]
                    rb = cam.generate_rays(camera_indices=0, keep_shape=False).to(dev)
                    if model.config.medium_mask_key in batch:
                        mm = batch[model.config.medium_mask_key]
                        if not isinstance(mm, Tensor):
                            mm = torch.as_tensor(mm)
                        rb.metadata = rb.metadata or {}
                        rb.metadata[model.config.medium_mask_key] = mm.reshape(-1, 1).float().to(dev)

                    rb = model.collider(rb).flatten()
                    n = len(rb)
                    o = rb.origins
                    d = F.normalize(rb.directions, dim=-1)
                    nr = rb.nears[..., 0]
                    fr = rb.fars[..., 0]

                    ip, _ = water_plane_intersection(o, d, model.water_plane_normal, model.water_plane_d)
                    # Helper returns +inf for parallel rays; clamp to far+1 to preserve mask behavior.
                    ip = torch.where(torch.isinf(ip), fr + 1.0, ip)
                    hits = (ip > nr) & (ip < fr)
                    wm = model._get_water_pixel_mask(rb, n, device=dev)
                    if wm is not None:
                        hits = hits & wm

                    ref_d = model._compute_refraction(
                        d, model.config.air_refractive_index, model.config.water_refractive_index
                    )
                    entry = o + d * ip.unsqueeze(-1)

                    vi = hits.nonzero(as_tuple=False).squeeze(-1)
                    if vi.shape[0] > 2500:
                        vi = vi[torch.randperm(vi.shape[0], device=dev)[:2500]]

                    eg = transform_points_to_bathy_global_frame(entry[vi], dp_out)
                    eg2 = transform_points_to_bathy_global_frame(entry[vi] + ref_d[vi], dp_out)
                    rdg = F.normalize(eg2 - eg, dim=-1)

                    co = (eg + 1e-4 * rdg).cpu().numpy()
                    cd = rdg.cpu().numpy()
                    rays_np = np.concatenate([co, cd], axis=1).astype(np.float32)
                    res = ms.cast_rays(o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32))
                    th = res["t_hit"].numpy() + 1e-4
                    hok = np.isfinite(th)
                    if not hok.any():
                        continue

                    hm = torch.from_numpy(hok).to(dev)
                    fi = vi[hm]
                    scale = torch.linalg.norm(eg2[hm] - eg[hm], dim=-1)

                    all_o.append(rb.origins[fi])
                    all_d.append(rb.directions[fi])
                    all_pa.append(rb.pixel_area[fi])
                    if rb.camera_indices is not None:
                        all_ci.append(rb.camera_indices[fi])
                    all_nr.append(rb.nears[fi])
                    all_fr.append(rb.fars[fi])
                    all_gt.append(torch.from_numpy(th[hok]).float().to(dev))
                    all_ip.append(ip[fi])
                    all_sc.append(scale)

                if not all_o:
                    _st["off"] = True
                    return

                _st["rb"] = RayBundle(
                    origins=torch.cat(all_o),
                    directions=torch.cat(all_d),
                    pixel_area=torch.cat(all_pa),
                    camera_indices=torch.cat(all_ci) if all_ci else None,
                    nears=torch.cat(all_nr),
                    fars=torch.cat(all_fr),
                    metadata={model.config.medium_mask_key: torch.ones(sum(x.shape[0] for x in all_o), 1, device=dev)},
                    times=None,
                )
                _st["gt"] = torch.cat(all_gt)
                _st["ip"] = torch.cat(all_ip)
                _st["sc"] = torch.cat(all_sc)
                print(f"[DepthMon] {len(_st['gt'])} rays ready")
            except Exception as e:
                print(f"[DepthMon] init failed: {e}")
                _st["off"] = True
                return

        was_training = model.training
        model.eval()
        out = model.get_outputs(_st["rb"])
        model.train(was_training)

        depth = out["depth"].squeeze()
        acc = out["accumulation"].squeeze()
        t_water = (depth - _st["ip"]).clamp_min(0)
        pred_global = t_water * _st["sc"]
        ok = acc > 0.5
        if ok.any():
            err = (pred_global[ok] - _st["gt"][ok]).cpu().numpy()
            med = float(np.median(err))
            mn = float(np.mean(err))
            print(f"[Step {step:6d}] depth->GT: median={med:+.3f}m, mean={mn:+.3f}m ({ok.sum()}/{len(ok)} rays)")
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({"depth_gt_median_err_m": med, "depth_gt_mean_err_m": mn}, step=step)
            except Exception:
                pass
        else:
            print(f"[Step {step:6d}] depth->GT: no rays with acc>0.5")

    return TrainingCallback(
        where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
        update_every_num_iters=1000,
        func=_cb,
    )


def add_colorbar_to_tensor(
    tensor_img: Tensor,
    title: str,
    min_val: float,
    max_val: float,
    cmap: int = cv2.COLORMAP_TURBO,
) -> Tensor:
    """Overlay a labeled colorbar (title, min/max) onto an image tensor.

    Renders a horizontal gradient bar with a semi-transparent dark background
    box behind the title and min/max numeric labels, then returns the composited
    image as a float tensor in ``[0, 1]`` on the same device as the input.

    Args:
        tensor_img: ``(H, W, 3)`` image tensor with values in ``[0, 1]``.
        title: Text drawn centered above the colorbar.
        min_val: Numeric value drawn at the left edge of the bar.
        max_val: Numeric value drawn at the right edge of the bar.
        cmap: OpenCV colormap constant applied to the gradient (default TURBO).

    Returns:
        A float tensor on the same device as ``tensor_img`` with values in
        ``[0, 1]`` and the same shape as the input.
    """
    # 1. Move to CPU and convert to 8-bit image
    img_np = (tensor_img.detach().cpu().numpy() * 255.0).astype(np.uint8)
    H, W, _ = img_np.shape

    # 2. Create the gradient bar
    bar_w = int(W * 0.5)
    bar_h = max(30, int(H * 0.025))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_title = 1.2
    font_scale_nums = 1
    thickness = 2

    box_width = bar_w + 60
    box_height = bar_h + 90
    box_x = (W - box_width) // 2
    box_y = H - box_height - 40

    gradient = np.linspace(0, 255, bar_w, dtype=np.uint8)
    gradient = np.tile(gradient, (bar_h, 1))
    colored_bar = cv2.applyColorMap(gradient, cmap)
    colored_bar = cv2.cvtColor(colored_bar, cv2.COLOR_BGR2RGB)

    # 3. Draw a dark semi-transparent background box so text is readable
    overlay = img_np.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img_np, 0.4, 0, img_np)  # 60% opacity

    # 4. Paste the colorbar onto the image
    bar_x = box_x + 30
    bar_y = box_y + 45
    img_np[bar_y : bar_y + bar_h, bar_x : bar_x + bar_w] = colored_bar

    # 5. Draw Title and Min/Max Text
    text_color = (255, 255, 255)

    # Center the Title above the bar
    title_size = cv2.getTextSize(title, font, font_scale_title, thickness)[0]
    title_x = box_x + (box_width - title_size[0]) // 2
    cv2.putText(img_np, title, (title_x, box_y + 35), font, font_scale_title, text_color, thickness, cv2.LINE_AA)

    # Left-align the Min value exactly with the left edge of the bar
    text_y = bar_y + bar_h + 35
    min_text = f"{min_val:.2f}"
    cv2.putText(img_np, min_text, (bar_x, text_y), font, font_scale_nums, text_color, thickness, cv2.LINE_AA)

    # Right-align the Max value exactly with the right edge of the bar
    max_text = f"{max_val:.2f}"
    max_size = cv2.getTextSize(max_text, font, font_scale_nums, thickness)[0]
    max_x = bar_x + bar_w - max_size[0]
    cv2.putText(img_np, max_text, (max_x, text_y), font, font_scale_nums, text_color, thickness, cv2.LINE_AA)

    # 6. Return as PyTorch Tensor
    return torch.from_numpy(img_np).float().to(tensor_img.device) / 255.0
