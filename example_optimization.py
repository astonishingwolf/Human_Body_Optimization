"""Compact example: optimize random-initialized MHR params to match zero-params mesh."""

from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.sam_3d_body.models.modules.parameter_optimizer import DifferentiableMHRParameters
from lib.sam_3d_body.utils.optimize_mhr_params import create_optimizer_for_params
from pymomentum.torch.character import Character
from thirdparty.MHR.mhr.io import get_default_asset_folder
from thirdparty.MHR.mhr.mhr import MHR


def _print_param_stats(label: str, params_dict: dict[str, torch.Tensor]) -> None:
    print(f"\n{label}")
    for key, value in params_dict.items():
        tensor = value.detach().float()
        print(
            f"  {key}: shape={tuple(tensor.shape)} "
            f"mean={tensor.mean().item():.6f} std={tensor.std().item():.6f} "
            f"norm={tensor.norm().item():.6f}"
        )


def _save_obj(vertices: torch.Tensor, faces: torch.Tensor, file_path: Path) -> None:
    verts = vertices.detach().cpu()
    if verts.ndim == 3:
        verts = verts[0]
    tris = faces.detach().cpu().long()
    if tris.ndim == 3:
        tris = tris[0]

    with file_path.open("w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0].item():.6f} {v[1].item():.6f} {v[2].item():.6f}\n")
        for tri in tris:
            # OBJ uses 1-based indexing.
            f.write(
                f"f {tri[0].item() + 1} {tri[1].item() + 1} {tri[2].item() + 1}\n"
            )


def _save_mesh_views_png(
    target_vertices: torch.Tensor,
    initial_vertices: torch.Tensor,
    final_vertices: torch.Tensor,
    faces: torch.Tensor,
    file_path: Path,
) -> None:
    verts_target = target_vertices.detach().cpu()[0]
    verts_initial = initial_vertices.detach().cpu()[0]
    verts_final = final_vertices.detach().cpu()[0]
    tris = faces.detach().cpu().long()
    if tris.ndim == 3:
        tris = tris[0]

    all_verts = torch.cat([verts_target, verts_initial, verts_final], dim=0)
    mins = all_verts.min(dim=0).values
    maxs = all_verts.max(dim=0).values
    center = (mins + maxs) / 2.0
    radius = ((maxs - mins).max() / 2.0).item()
    if radius <= 0:
        radius = 1.0

    mesh_data = [
        ("Target (Zero Pose)", verts_target),
        ("Initial (Angular Offset)", verts_initial),
        ("Final (Optimized)", verts_final),
    ]

    views = [
        ("Front", 15, -70),
        ("Side", 15, 20),
        ("Top", 90, -90),
    ]

    fig = plt.figure(figsize=(18, 16))
    for row_idx, (mesh_title, verts) in enumerate(mesh_data):
        for col_idx, (view_name, elev, azim) in enumerate(views):
            ax = fig.add_subplot(3, 3, row_idx * 3 + col_idx + 1, projection="3d")
            ax.plot_trisurf(
                verts[:, 0].numpy(),
                verts[:, 1].numpy(),
                verts[:, 2].numpy(),
                triangles=tris.numpy(),
                color="#88c0d0",
                edgecolor="none",
                linewidth=0.0,
                antialiased=False,
                shade=True,
                alpha=1.0,
            )
            ax.set_title(f"{mesh_title} | {view_name}", fontsize=10)
            ax.set_xlim(center[0].item() - radius, center[0].item() + radius)
            ax.set_ylim(center[1].item() - radius, center[1].item() + radius)
            ax.set_zlim(center[2].item() - radius, center[2].item() + radius)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    fig.savefig(file_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def chamfer_distance(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, None]:
    """
    Torch-only bidirectional Chamfer distance.

    Args:
        x: Point cloud of shape [B, N, 3]
        y: Point cloud of shape [B, M, 3]

    Returns:
        (loss, None) to match the common chamfer API shape.
    """
    pairwise_dist = torch.cdist(x, y, p=2)
    x_to_y = pairwise_dist.min(dim=2).values
    y_to_x = pairwise_dist.min(dim=1).values
    loss = x_to_y.mean() + y_to_x.mean()
    return loss, None


def initialize_mhr_model(
    mhr_folder: Path | None = None,
    device: torch.device | None = None,
) -> tuple[MHR, Character]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mhr_folder is None:
        mhr_folder = get_default_asset_folder()

    mhr_model = MHR.from_files(
        folder=mhr_folder,
        device=device,
        lod=1,
        wants_pose_correctives=True,
    )
    return mhr_model, mhr_model.character_torch


def create_mesh_with_zero_parameters(
    mhr_model: MHR,
    character_torch: Character,
    device: torch.device,
) -> tuple[torch.Tensor, DifferentiableMHRParameters]:
    params_zero = DifferentiableMHRParameters(
        character_torch=character_torch,
        batch_size=1,
        device=device,
        learnable_offsets=False,
    )
    with torch.no_grad():
        params_zero.pose_params.zero_()
        params_zero.shape_params.zero_()
        params_zero.scale_params.zero_()
    with torch.no_grad():
        vertices, _ = params_zero.forward_mhr(mhr_model, character_torch)
    return vertices, params_zero


def create_mesh_with_random_parameters(
    mhr_model: MHR,
    character_torch: Character,
    device: torch.device,
    seed: int = 42,
    pose_displacement_std: float = 0.12,
) -> tuple[torch.Tensor, DifferentiableMHRParameters]:
    torch.manual_seed(seed)
    params_random = DifferentiableMHRParameters(
        character_torch=character_torch,
        batch_size=1,
        device=device,
        learnable_offsets=True,
    )
    with torch.no_grad():
        # Start from zero-pose/zero-shape/zero-scale, then add angular displacement only.
        params_random.pose_params.zero_()
        params_random.shape_params.zero_()
        params_random.scale_params.zero_()
        params_random.pose_params.add_(
            torch.randn_like(params_random.pose_params) * pose_displacement_std
        )

    with torch.no_grad():
        vertices, _ = params_random.forward_mhr(mhr_model, character_torch)
    return vertices, params_random


def optimize_with_chamfer_distance(
    target_vertices: torch.Tensor,
    initial_params: DifferentiableMHRParameters,
    mhr_model: MHR,
    character_torch: Character,
    num_iterations: int = 200,
    learning_rates: dict | None = None,
    log_every: int = 20,
) -> DifferentiableMHRParameters:
    if learning_rates is None:
        learning_rates = {
            "joint_offsets": 1e-3,
            "pose": 1e-2,
            "shape": 1e-3,
            "scale": 1e-2,
        }

    params = initial_params
    target_vertices = target_vertices.detach()
    optimizer = create_optimizer_for_params(params, learning_rates)

    for step in range(num_iterations):
        optimizer.zero_grad()
        pred_vertices, _ = params.forward_mhr(mhr_model, character_torch)
        chamfer_loss, _ = chamfer_distance(pred_vertices, target_vertices)
        reg_loss = 1e-3 * (
            torch.norm(params.pose_params)
            + torch.norm(params.shape_params)
            + torch.norm(params.scale_params)
        )
        loss = chamfer_loss + reg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=1.0)
        optimizer.step()

        if log_every > 0 and (step % log_every == 0 or step == num_iterations - 1):
            print(f"iter={step:04d} loss={loss.item():.6f} chamfer={chamfer_loss.item():.6f}")

    return params


def main_example() -> DifferentiableMHRParameters:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mhr_model, character_torch = initialize_mhr_model(device=device)
    output_dir = Path(__file__).resolve().parent / "outputs" / "mhr_optimization"
    output_dir.mkdir(parents=True, exist_ok=True)

    target_vertices, _ = create_mesh_with_zero_parameters(mhr_model, character_torch, device)
    initial_vertices, params_random = create_mesh_with_random_parameters(
        mhr_model, character_torch, device, seed=42
    )
    faces = character_torch.mesh.faces

    params_before = params_random.get_parameter_dict()
    _print_param_stats("Parameters before optimization:", params_before)
    torch.save(params_before, output_dir / "params_before.pt")
    torch.save(initial_vertices.detach().cpu(), output_dir / "initial_vertices.pt")
    _save_obj(initial_vertices, faces, output_dir / "initial_mesh.obj")

    initial_chamfer, _ = chamfer_distance(initial_vertices, target_vertices)
    optimized_params = optimize_with_chamfer_distance(
        target_vertices=target_vertices,
        initial_params=params_random,
        mhr_model=mhr_model,
        character_torch=character_torch,
        num_iterations=200,
    )
    final_vertices, _ = optimized_params.forward_mhr(mhr_model, character_torch)
    final_chamfer, _ = chamfer_distance(final_vertices, target_vertices)
    params_after = optimized_params.get_parameter_dict()

    _print_param_stats("Parameters after optimization:", params_after)
    torch.save(params_after, output_dir / "params_after.pt")
    torch.save(final_vertices.detach().cpu(), output_dir / "final_vertices.pt")
    _save_obj(final_vertices, faces, output_dir / "final_mesh.obj")
    torch.save(target_vertices.detach().cpu(), output_dir / "target_vertices.pt")
    _save_obj(target_vertices, faces, output_dir / "target_mesh.obj")
    _save_mesh_views_png(
        target_vertices=target_vertices,
        initial_vertices=initial_vertices,
        final_vertices=final_vertices,
        faces=faces,
        file_path=output_dir / "mesh_comparison.png",
    )

    print(f"initial_chamfer={initial_chamfer.item():.6f}")
    print(f"final_chamfer={final_chamfer.item():.6f}")
    print(f"saved_outputs={output_dir}")
    return optimized_params


if __name__ == "__main__":
    main_example()
