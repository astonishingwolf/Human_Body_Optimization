"""Image-driven optimization example: SAM-3D init -> MHR gaussians -> mask optimization.
Uses DifferentiableMHRParameters for joint offsets ([J, 3] per-joint translation),
same as example_optimization.py. Requires MHR from pymomentum (do not pass --mhr-model-path .pt).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gaussian_splatting import set_gaussian_type, splat_gaussians
from camera import GSCamera, intrinsics_to_fov
from lib.sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
from lib.sam_3d_body.models.modules.parameter_optimizer import DifferentiableMHRParameters
from lib.sam_3d_body.utils.optimize_mhr_params import create_optimizer_for_params
from mhr_animatable_gaussians import MHRAnimatableGaussians
from ops import image2torch


def _load_image_mask(
    image_path: Path,
    mask_path: Path,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image_size
    image = Image.open(image_path).convert("RGB").resize((w, h), Image.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize((w, h), Image.NEAREST)

    image_t = image2torch(np.asarray(image)).to(device=device, dtype=torch.float32)  # [1,3,H,W]
    mask_t = torch.from_numpy(np.asarray(mask)).to(device=device, dtype=torch.float32) / 255.0
    mask_t = (mask_t > 0.5).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return image_t, mask_t


def _focal_to_fov(focal: float, size_px: int) -> float:
    return float(2.0 * np.arctan(size_px / (2.0 * max(focal, 1e-6))))


def _get_fov_from_sam_output(
    sam_out: dict,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[float, float]:
    if "cam_int" in sam_out and sam_out["cam_int"] is not None:
        cam_arr = np.asarray(sam_out["cam_int"], dtype=np.float32)
        try:
            if cam_arr.ndim == 1 and cam_arr.size == 9:
                cam_arr = cam_arr.reshape(3, 3)
            elif cam_arr.ndim == 3 and cam_arr.shape[0] == 1:
                cam_arr = cam_arr[0]

            if cam_arr.shape == (3, 3):
                cam_int = torch.from_numpy(cam_arr).to(device=device, dtype=torch.float32).unsqueeze(0)
                fov_x, fov_y = intrinsics_to_fov(cam_int)
                return float(fov_x[0].item()), float(fov_y[0].item())
        except Exception:
            # Fall back to focal-based estimate below.
            pass

    h, w = image_size
    focal_length = sam_out["focal_length"]
    if np.isscalar(focal_length):
        fx = float(focal_length)
        fy = float(focal_length)
    else:
        arr = np.asarray(focal_length).reshape(-1)
        if arr.size >= 2:
            fx, fy = float(arr[0]), float(arr[1])
        else:
            fx = fy = float(arr[0])
    return _focal_to_fov(fx, w), _focal_to_fov(fy, h)


def _build_camera_world2cam(
    cam_t: torch.Tensor,
    fov_x: float,
    fov_y: float,
    image_size: tuple[int, int],
    device: torch.device,
) -> GSCamera:
    h, w = image_size
    world2cam = torch.eye(4, device=device, dtype=torch.float32)
    world2cam[:3, 3] = cam_t.float().view(3)
    return GSCamera.from_matrix(
        world2cam,
        fov_x=fov_x,
        fov_y=fov_y,
        image_width=w,
        image_height=h,
    ).to(device)


def _save_vis(
    target_rgb: torch.Tensor,
    target_mask: torch.Tensor,
    pred_rgb: torch.Tensor,
    pred_mask: torch.Tensor,
    output_path: Path,
) -> None:
    tgt = (target_rgb[0].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    prm = (pred_rgb[0].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    tgt_m = (target_mask[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
    prm_m = (pred_mask[0, 0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)

    row1 = np.concatenate([tgt, prm], axis=1)
    row2 = np.concatenate([np.stack([tgt_m] * 3, axis=-1), np.stack([prm_m] * 3, axis=-1)], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    Image.fromarray(grid).save(output_path)


def _save_rgb_tensor(rgb: torch.Tensor, output_path: Path) -> None:
    rgb_np = (rgb[0].permute(1, 2, 0).detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(rgb_np).save(output_path)


def _save_mask_tensor(mask: torch.Tensor, output_path: Path) -> None:
    mask_np = (mask[0, 0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(mask_np).save(output_path)


def _make_mask_overlay(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> np.ndarray:
    pred = pred_mask[0, 0].detach().cpu().clamp(0, 1).numpy()
    gt = gt_mask[0, 0].detach().cpu().clamp(0, 1).numpy()
    overlay = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
    # Red: prediction, Green: GT, Yellow: overlap
    overlay[..., 0] = pred
    overlay[..., 1] = gt
    return (overlay * 255).astype(np.uint8)


def _save_before_after_gt_artifacts(
    gt_image: torch.Tensor,
    gt_mask: torch.Tensor,
    before_mask: torch.Tensor,
    after_mask: torch.Tensor,
    output_dir: Path,
) -> None:
    _save_mask_tensor(gt_mask, output_dir / "gt_mask.png")
    _save_mask_tensor(before_mask, output_dir / "before_mask.png")
    _save_mask_tensor(after_mask, output_dir / "after_mask.png")

    # Intersections show how much predicted mask overlaps GT.
    before_intersection = (before_mask * gt_mask).clamp(0, 1)
    after_intersection = (after_mask * gt_mask).clamp(0, 1)
    _save_mask_tensor(before_intersection, output_dir / "before_gt_intersection.png")
    _save_mask_tensor(after_intersection, output_dir / "after_gt_intersection.png")

    # Partial masks combine prediction and GT to highlight coverage drift/improvement.
    before_partial = (0.5 * before_mask + 0.5 * gt_mask).clamp(0, 1)
    after_partial = (0.5 * after_mask + 0.5 * gt_mask).clamp(0, 1)
    _save_mask_tensor(before_partial, output_dir / "before_gt_partial_mask.png")
    _save_mask_tensor(after_partial, output_dir / "after_gt_partial_mask.png")

    # Save color overlays for qualitative comparison.
    Image.fromarray(_make_mask_overlay(before_mask, gt_mask)).save(output_dir / "before_gt_overlay.png")
    Image.fromarray(_make_mask_overlay(after_mask, gt_mask)).save(output_dir / "after_gt_overlay.png")

    # Apply masks to GT image to compare visible regions.
    _save_rgb_tensor(gt_image * gt_mask, output_dir / "gt_image_gt_masked.png")
    _save_rgb_tensor(gt_image * before_mask, output_dir / "gt_image_before_masked.png")
    _save_rgb_tensor(gt_image * after_mask, output_dir / "gt_image_after_masked.png")
    _save_rgb_tensor(gt_image * before_partial, output_dir / "gt_image_before_gt_partial.png")
    _save_rgb_tensor(gt_image * after_partial, output_dir / "gt_image_after_gt_partial.png")


def _tensor_stats(x: torch.Tensor) -> str:
    y = x.detach().float().reshape(-1)
    if y.numel() == 0:
        return "empty"
    return (
        f"mean={y.mean().item():.6f} std={y.std(unbiased=False).item():.6f} "
        f"min={y.min().item():.6f} max={y.max().item():.6f} norm={y.norm().item():.6f}"
    )


def _print_param_block(
    title: str,
    shape_t: torch.Tensor,
    pose_t: torch.Tensor,
    joint_offsets_t: torch.Tensor,
    scale_t: torch.Tensor,
) -> None:
    print(title)
    print(f"  shape: {_tensor_stats(shape_t)}")
    print(f"  pose : {_tensor_stats(pose_t)}")
    print(f"  joint_offsets: {_tensor_stats(joint_offsets_t)}")
    print(f"  scale: {_tensor_stats(scale_t)}")


def _prepare_gaussians_for_render(gaussians):
    """
    Convert gaussian tensor shapes to renderer-compatible format.
    Expected by renderer: xyz in [B, N, 3] or [N, 3].
    """
    if gaussians.xyz.ndim == 4 and gaussians.xyz.shape[1] == 1:
        gaussians = gaussians.clone().squeeze(dim=1)  # [B, 1, N, ...] -> [B, N, ...]
    elif gaussians.xyz.ndim != 3:
        raise ValueError(f"Unsupported gaussian xyz shape for rendering: {gaussians.xyz.shape}")
    return gaussians


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_gaussian_type("3d")

    image_path = Path(args.image_path)
    mask_path = Path(args.mask_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = (args.height, args.width)
    gt_image, gt_mask = _load_image_mask(image_path, mask_path, image_size, device)

    # 1) Infer SAM-3D parameters from the image.
    sam_model, sam_cfg = load_sam_3d_body(
        checkpoint_path=args.sam_ckpt,
        device=str(device),
        mhr_path=args.sam_mhr_path,
    )
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=sam_model,
        model_cfg=sam_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    sam_outputs = estimator.process_one_image(
        img=str(image_path),
        bboxes=None,
        masks=None,
        inference_type="body",
    )
    if not sam_outputs:
        raise RuntimeError("SAM-3D did not return any human prediction.")
    sam_out = sam_outputs[0]

    # 2) Initialize MHR animatable gaussians with SAM-3D parameters.
    mhr_gs = MHRAnimatableGaussians(
        device=device,
        lod=args.lod,
        mhr_model_path=args.mhr_model_path if args.mhr_model_path else None,
    )
    # Match SAM3DGS initialization to avoid oversized splats that saturate the mask.
    with torch.no_grad():
        mhr_gs.scale.fill_(1e-2)
        mhr_gs.opacity.fill_(0.5)
        mhr_gs.sh.fill_(0.5)

    init_shape = torch.from_numpy(np.asarray(sam_out["shape_params"])).to(device=device, dtype=torch.float32).unsqueeze(0)
    init_model = torch.from_numpy(np.asarray(sam_out["mhr_model_params"])).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    init_expr = torch.from_numpy(np.asarray(sam_out["expr_params"])).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    init_cam_t = torch.from_numpy(np.asarray(sam_out["pred_cam_t"])).to(device=device, dtype=torch.float32).view(1, 3)

    fov_x, fov_y = _get_fov_from_sam_output(sam_out, image_size, device)

    if getattr(mhr_gs, "use_torchscript", True):
        raise RuntimeError(
            "DifferentiableMHRParameters requires MHR from pymomentum, not TorchScript. "
            "Do not pass --mhr-model-path (or pass a path that does not exist) so that MHR.from_files() is used."
        )
    if not hasattr(mhr_gs.mhr, "character_torch") or not hasattr(mhr_gs.mhr.character_torch, "parameter_transform"):
        raise RuntimeError("MHR character has no parameter_transform; cannot use DifferentiableMHRParameters.")

    character_torch = mhr_gs.mhr.character_torch
    pad_size = getattr(mhr_gs, "num_identity_blendshapes", 45) + getattr(
        mhr_gs, "num_expr_blendshapes", 72
    )
    init_model_flat = init_model.squeeze(1)  # [1, 204]
    padding = torch.zeros(1, pad_size, device=device, dtype=init_model_flat.dtype)
    model_with_pad = torch.cat([init_model_flat, padding], dim=1)
    joint_params_flat = character_torch.model_parameters_to_joint_parameters(model_with_pad)
    J = character_torch.skeleton.joint_translation_offsets.shape[0]
    joint_params = joint_params_flat.reshape(1, J, 7)
    static_offsets = character_torch.skeleton.joint_translation_offsets
    full_translation = (joint_params[0, :, :3] + static_offsets).clone()
    diff_params = DifferentiableMHRParameters(
        character_torch, batch_size=1, device=device, learnable_offsets=True
    )
    with torch.no_grad():
        diff_params.joint_offsets.data = full_translation
        diff_params.pose_params.data = joint_params[:, :, 3:6].clone()
        diff_params.scale_params.data = joint_params[:, :, 6:7].clone()
        diff_params.shape_params.data = init_shape.clone()
    init_joint_offsets_diff = full_translation.clone()
    init_pose_diff = joint_params[:, :, 3:6].clone()
    init_scale_diff = joint_params[:, :, 6:7].clone()

    learning_rates = {
        "joint_offsets": args.lr_joint_offsets,
        "pose": args.lr_pose,
        "shape": args.lr_shape,
        "scale": args.lr_scale,
    }
    optimizer = create_optimizer_for_params(diff_params, learning_rates)
    _print_param_block(
        "Initial parameters (DifferentiableMHRParameters):",
        diff_params.shape_params.detach(),
        diff_params.pose_params.detach(),
        diff_params.joint_offsets.detach(),
        diff_params.scale_params.detach(),
    )
    print(f"  camera(fixed): {_tensor_stats(init_cam_t)}")

    bg_color = torch.ones(3, device=device, dtype=torch.float32)
    gt_image_masked = gt_image * gt_mask + bg_color.view(1, 3, 1, 1) * (1.0 - gt_mask)

    with torch.no_grad():
        mhr_gs.set_animate_params(init_shape, init_model, init_expr)
        init_gs = mhr_gs.to_gaussian_parameters(pose_indices=[0], to_colmap=True)
        init_gs = _prepare_gaussians_for_render(init_gs)
        camera = _build_camera_world2cam(init_cam_t[0], fov_x, fov_y, image_size, device)
        init_render = splat_gaussians(init_gs, camera, bg_color=bg_color)
        init_img = init_render["image"]
        init_mask = init_render["mask"].clamp(0, 1)
        _save_vis(gt_image_masked, gt_mask, init_img, init_mask, output_dir / "init_vs_target.png")

    for step in range(args.num_iters):
        optimizer.zero_grad()

        joint_params_flat = diff_params.get_joint_parameters()  # [B, J*7]
        mhr_gs.set_animate_params_joint_params(diff_params.shape_params, joint_params_flat, init_expr)

        gs = mhr_gs.to_gaussian_parameters(pose_indices=[0], to_colmap=True)
        gs = _prepare_gaussians_for_render(gs)
        camera = _build_camera_world2cam(init_cam_t[0], fov_x, fov_y, image_size, device)

        render_out = splat_gaussians(gs, camera, bg_color=bg_color)
        pred_image = render_out["image"]
        pred_mask = render_out["mask"].clamp(1e-6, 1.0 - 1e-6)

        loss_l2 = F.mse_loss(pred_mask, gt_mask)
        loss = args.w_l2 * loss_l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(diff_params.parameters(), max_norm=1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == args.num_iters - 1:
            print(
                f"iter={step:04d} total={loss.item():.6f} "
                f"l2={loss_l2.item():.6f}"
            )
            _save_vis(
                gt_image_masked,
                gt_mask,
                pred_image,
                pred_mask,
                output_dir / f"iter_{step:04d}.png",
            )

    with torch.no_grad():
        joint_params_flat = diff_params.get_joint_parameters()  # [B, J*7]
        mhr_gs.set_animate_params_joint_params(diff_params.shape_params, joint_params_flat, init_expr)
        final_gs = mhr_gs.to_gaussian_parameters(pose_indices=[0], to_colmap=True)
        final_gs = _prepare_gaussians_for_render(final_gs)
        camera = _build_camera_world2cam(init_cam_t[0], fov_x, fov_y, image_size, device)
        final_render = splat_gaussians(final_gs, camera, bg_color=bg_color)
        final_mask = final_render["mask"].clamp(0, 1)

        _save_vis(
            gt_image_masked,
            gt_mask,
            final_render["image"],
            final_mask,
            output_dir / "final_vs_target.png",
        )
        _save_before_after_gt_artifacts(
            gt_image=gt_image,
            gt_mask=gt_mask,
            before_mask=init_mask,
            after_mask=final_mask,
            output_dir=output_dir,
        )

        pd = diff_params.get_parameter_dict()
        torch.save(
            {
                "shape_init": init_shape.detach().cpu(),
                "model_init": init_model.detach().cpu(),
                "shape_final": pd["shape_params"].cpu(),
                "joint_offsets_init": init_joint_offsets_diff.cpu(),
                "joint_offsets_final": pd["joint_offsets"].cpu(),
                "pose_final": pd["pose_params"].cpu(),
                "scale_final": pd["scale_params"].cpu(),
                "expr_fixed": init_expr.detach().cpu(),
                "cam_t_fixed": init_cam_t.detach().cpu(),
            },
            output_dir / "optimized_params.pt",
        )
    pd = diff_params.get_parameter_dict()
    _print_param_block(
        "Final optimized parameters (DifferentiableMHRParameters):",
        pd["shape_params"],
        pd["pose_params"],
        pd["joint_offsets"],
        pd["scale_params"],
    )
    init_jo = init_joint_offsets_diff.to(device)
    print(
        "Delta norms: "
        f"shape={torch.norm(pd['shape_params'].to(device) - init_shape).item():.6f}, "
        f"pose={torch.norm(pd['pose_params'].to(device) - init_pose_diff.to(device)).item():.6f}, "
        f"joint_offsets={torch.norm(pd['joint_offsets'].to(device) - init_jo).item():.6f}, "
        f"scale={torch.norm(pd['scale_params'].to(device) - init_scale_diff.to(device)).item():.6f}"
    )
    print(f"Saved outputs to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, default="data/0000.png")
    parser.add_argument("--mask-path", type=str, default="data/0000_mask.png")
    parser.add_argument("--output-dir", type=str, default="outputs/mhr_image_optimization")

    parser.add_argument("--sam-ckpt", type=str, default="/data/users/soham/baselines/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt")
    parser.add_argument("--sam-mhr-path", type=str, default="/data/users/soham/baselines/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
    parser.add_argument(
        "--mhr-model-path",
        type=str,
        default=None,
        help="Must be unset so MHR loads from pymomentum (DifferentiableMHRParameters). Do not pass a .pt TorchScript path.",
    )
    parser.add_argument("--lod", type=int, default=1)

    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)

    parser.add_argument("--lr-pose", type=float, default=1e-3)
    parser.add_argument("--lr-joint-offsets", type=float, default=1e-3)
    parser.add_argument("--lr-scale", type=float, default=1e-3)
    parser.add_argument("--lr-shape", type=float, default=1e-3)

    parser.add_argument("--w-l2", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
