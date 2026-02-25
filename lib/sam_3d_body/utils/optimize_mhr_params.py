# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.modules.parameter_optimizer import DifferentiableMHRParameters
from thirdparty.MHR.mhr.mhr import MHR
from pymomentum.torch.character import Character


def optimize_mhr_parameters(
    target_vertices: torch.Tensor,  # [B, V, 3]
    mhr_model: MHR,
    character_torch: Character,
    initial_params: Optional[DifferentiableMHRParameters] = None,
    num_iterations: int = 100,
    learning_rates: Optional[Dict[str, float]] = None,
    freeze_params: Optional[List[str]] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> DifferentiableMHRParameters:
    """
    Optimize MHR parameters to fit target vertices.
    
    Args:
        target_vertices: Target vertex positions [B, V, 3]
        mhr_model: MHR model instance
        character_torch: Character model
        initial_params: Initial parameter values (optional)
        num_iterations: Number of optimization iterations
        learning_rates: Dict with lr for each parameter type
            e.g., {'joint_offsets': 0.001, 'pose': 0.01, 'shape': 0.001, 'scale': 0.01}
        freeze_params: List of parameter types to freeze
            e.g., ['shape'] to freeze shape parameters
        loss_weights: Weights for different loss components
            e.g., {'vertex': 1.0, 'regularization': 0.01}
        verbose: Whether to print progress
    
    Returns:
        Optimized DifferentiableMHRParameters
    """
    # Initialize parameters
    if initial_params is None:
        params = DifferentiableMHRParameters(
            character_torch,
            batch_size=target_vertices.shape[0],
            device=target_vertices.device,
        )
    else:
        params = initial_params

    # Freeze specified parameters
    if freeze_params:
        params.freeze_parameters(freeze_params)

    # Set up optimizer with different learning rates
    if learning_rates is None:
        learning_rates = {
            "joint_offsets": 0.001,
            "pose": 0.01,
            "shape": 0.001,
            "scale": 0.01,
        }

    # Collect parameters with their learning rates
    optimizer_params = []
    if params.joint_offsets.requires_grad:
        optimizer_params.append(
            {"params": params.joint_offsets, "lr": learning_rates["joint_offsets"]}
        )
    if params.pose_params.requires_grad:
        optimizer_params.append(
            {"params": params.pose_params, "lr": learning_rates["pose"]}
        )
    if params.shape_params.requires_grad:
        optimizer_params.append(
            {"params": params.shape_params, "lr": learning_rates["shape"]}
        )
    if params.scale_params.requires_grad:
        optimizer_params.append(
            {"params": params.scale_params, "lr": learning_rates["scale"]}
        )

    if len(optimizer_params) == 0:
        raise ValueError("All parameters are frozen! Nothing to optimize.")

    optimizer = torch.optim.Adam(optimizer_params)

    # Loss weights
    if loss_weights is None:
        loss_weights = {"vertex": 1.0, "regularization": 0.01}

    # Optimization loop
    for iteration in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass
        pred_vertices, _ = params.forward_mhr(mhr_model, character_torch)

        # Compute loss
        vertex_loss = F.mse_loss(pred_vertices, target_vertices)

        # Regularization (optional)
        reg_loss = torch.tensor(0.0, device=pred_vertices.device)
        if loss_weights.get("regularization", 0) > 0:
            if params.pose_params.requires_grad:
                reg_loss = reg_loss + 0.01 * torch.norm(params.pose_params)
            if params.shape_params.requires_grad:
                reg_loss = reg_loss + 0.01 * torch.norm(params.shape_params)
            if params.scale_params.requires_grad:
                reg_loss = reg_loss + 0.01 * torch.norm(params.scale_params)

        total_loss = (
            loss_weights["vertex"] * vertex_loss
            + loss_weights["regularization"] * reg_loss
        )

        # Backward pass
        total_loss.backward()

        # Optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=1.0)

        # Update
        optimizer.step()

        if verbose and (iteration % 10 == 0 or iteration == num_iterations - 1):
            print(
                f"Iter {iteration:4d}: Total Loss = {total_loss.item():.6f}, "
                f"Vertex Loss = {vertex_loss.item():.6f}, "
                f"Reg Loss = {reg_loss.item():.6f}"
            )

    return params


def create_optimizer_for_params(
    params: DifferentiableMHRParameters,
    learning_rates: Optional[Dict[str, float]] = None,
) -> torch.optim.Optimizer:
    """
    Create an optimizer for DifferentiableMHRParameters with independent learning rates.
    
    Args:
        params: DifferentiableMHRParameters instance
        learning_rates: Dict with lr for each parameter type
    
    Returns:
        Optimizer instance
    """
    if learning_rates is None:
        learning_rates = {
            "joint_offsets": 0.001,
            "pose": 0.01,
            "shape": 0.001,
            "scale": 0.01,
        }

    optimizer_params = []
    if params.joint_offsets.requires_grad:
        optimizer_params.append(
            {"params": params.joint_offsets, "lr": learning_rates["joint_offsets"]}
        )
    if params.pose_params.requires_grad:
        optimizer_params.append(
            {"params": params.pose_params, "lr": learning_rates["pose"]}
        )
    if params.shape_params.requires_grad:
        optimizer_params.append(
            {"params": params.shape_params, "lr": learning_rates["shape"]}
        )
    if params.scale_params.requires_grad:
        optimizer_params.append(
            {"params": params.scale_params, "lr": learning_rates["scale"]}
        )

    return torch.optim.Adam(optimizer_params)


def compute_vertex_loss(
    pred_vertices: torch.Tensor,
    target_vertices: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    """
    Compute vertex loss between predicted and target vertices.
    
    Args:
        pred_vertices: Predicted vertices [B, V, 3]
        target_vertices: Target vertices [B, V, 3]
        loss_type: Type of loss ('mse', 'l1', 'chamfer')
    
    Returns:
        Loss value
    """
    if loss_type == "mse":
        return F.mse_loss(pred_vertices, target_vertices)
    elif loss_type == "l1":
        return F.l1_loss(pred_vertices, target_vertices)
    elif loss_type == "chamfer":
        # Simple chamfer distance (not the full version)
        dist1 = torch.cdist(pred_vertices, target_vertices)
        dist2 = torch.cdist(target_vertices, pred_vertices)
        return dist1.min(dim=-1)[0].mean() + dist2.min(dim=-1)[0].mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
