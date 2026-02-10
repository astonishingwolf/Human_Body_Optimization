# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from pymomentum.torch.character import Character
from thirdparty.MHR.mhr.mhr import MHR

from .differentiable_skeleton import DifferentiableSkeleton


class DifferentiableMHRParameters(nn.Module):
    """
    Manages all optimizable MHR parameters as independent nn.Parameters.
    
    This allows independent optimization of:
    - Joint offsets: [J, 3] translation offsets per joint
    - Pose parameters: [B, J, 3] Euler XYZ rotations per joint
    - Shape parameters: [B, 45] blendshape coefficients
    - Scale parameters: [B, J, 1] log2 scale per joint
    """

    def __init__(
        self,
        character_torch: Character,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        init_from_existing: Optional[Dict] = None,
        learnable_offsets: bool = True,
    ):
        super().__init__()

        num_joints = len(character_torch.skeleton.joint_names)

        if device is None:
            # Try to get device from character_torch parameters or buffers
            try:
                device = next(character_torch.parameters()).device
            except StopIteration:
                # If no parameters, try buffers
                try:
                    device = next(character_torch.buffers()).device
                except StopIteration:
                    # Default to CPU if no parameters or buffers
                    device = torch.device("cpu")

        # 1. Joint Offsets [J, 3] - learnable translation offsets
        if learnable_offsets:
            self.joint_offsets = nn.Parameter(
                character_torch.skeleton.joint_translation_offsets.clone().to(device)
            )
        else:
            self.register_buffer(
                "joint_offsets",
                character_torch.skeleton.joint_translation_offsets.clone().to(device)
            )

        # 2. Pose Parameters [B, J, 3] - Euler XYZ rotations
        self.pose_params = nn.Parameter(
            torch.zeros(batch_size, num_joints, 3, device=device)
        )

        # 3. Shape Parameters [B, 45] - blendshape coefficients
        self.shape_params = nn.Parameter(
            torch.zeros(batch_size, 45, device=device)
        )

        # 4. Scale Parameters [B, J, 1] - log2 scale
        self.scale_params = nn.Parameter(
            torch.zeros(batch_size, num_joints, 1, device=device)
        )

        # Store character for conversions
        self.character_torch = character_torch
        self.num_joints = num_joints
        self.batch_size = batch_size

        # Initialize from existing if provided
        if init_from_existing:
            self.load_from_dict(init_from_existing)

    def get_joint_parameters(self) -> torch.Tensor:
        """
        Convert independent parameters to joint_parameters format [B, J*7].
        
        Format per joint: [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, log2_scale]
        Translation is computed as: joint_offsets (no additional translation params for now)
        """
        batch_size = self.pose_params.shape[0]

        # Translation: use joint offsets (expand to batch)
        translation = self.joint_offsets[None, :, :].expand(batch_size, -1, -1)

        # Rotation: pose_params (Euler XYZ)
        rotation = self.pose_params

        # Scale: scale_params (already in log2 space)
        scale = self.scale_params

        # Concatenate: [trans, rot, scale] -> [B, J, 7]
        joint_params = torch.cat([translation, rotation, scale], dim=-1)

        # Flatten to [B, J*7]
        return joint_params.flatten(1, 2)

    def get_shape_coefficients(self) -> torch.Tensor:
        """Return shape parameters [B, 45]"""
        return self.shape_params


    def forward_mhr(
        self,
        mhr_model: MHR,
        character_torch: Character,
        expr_params: Optional[torch.Tensor] = None,
        global_trans: Optional[torch.Tensor] = None,
        global_rot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MHR model using independent parameters.
        
        This method works directly with joint parameters, bypassing the model_parameters
        conversion for optimization efficiency. It replicates the logic from MHR.forward()
        but uses our independent parameter format.
        
        Args:
            mhr_model: MHR model instance
            character_torch: Character model (with differentiable skeleton if needed)
            expr_params: Optional expression parameters [B, 72]
            global_trans: Optional global translation [B, 3] (not used currently)
            global_rot: Optional global rotation [B, 3] (not used currently)
        
        Returns:
            vertices: [B, V, 3] skinned vertices
            skeleton_state: [B, J, 8] skeleton state
        """
        batch_size = self.pose_params.shape[0]
        device = self.pose_params.device

        # Get shape coefficients (identity blendshapes)
        shape_coeffs = self.get_shape_coefficients()

        # Get joint parameters from independent parameters
        joint_params = self.get_joint_parameters()

        # Convert joint parameters to skeleton state (global)
        skel_state = character_torch.joint_parameters_to_skeleton_state(joint_params)

        # Get expression parameters
        if expr_params is None:
            expr_params = torch.zeros(
                batch_size,
                mhr_model.get_num_face_expression_blendshapes(),
                device=device,
            )

        # Combine identity and expression blendshapes
        coeffs = torch.cat([shape_coeffs, expr_params], dim=1)

        # Compute rest pose vertices from blendshapes
        rest_pose = character_torch.blend_shape.forward(coeffs)

        # Apply pose correctives if available (non-linear pose-dependent deformations)
        if (
            hasattr(mhr_model, "pose_correctives_model")
            and mhr_model.pose_correctives_model is not None
        ):
            pose_correctives = mhr_model.pose_correctives_model.forward(
                joint_parameters=joint_params
            )
            rest_pose = rest_pose + pose_correctives

        # Apply linear blend skinning
        vertices = character_torch.skin_points(
            skel_state=skel_state, rest_vertex_positions=rest_pose
        )

        return vertices, skel_state

    def load_from_dict(self, params_dict: Dict):
        """Load parameters from dictionary"""
        if "joint_offsets" in params_dict:
            with torch.no_grad():
                self.joint_offsets.data = params_dict["joint_offsets"].to(
                    self.joint_offsets.device
                )
        if "pose_params" in params_dict:
            with torch.no_grad():
                self.pose_params.data = params_dict["pose_params"].to(
                    self.pose_params.device
                )
        if "shape_params" in params_dict:
            with torch.no_grad():
                self.shape_params.data = params_dict["shape_params"].to(
                    self.shape_params.device
                )
        if "scale_params" in params_dict:
            with torch.no_grad():
                self.scale_params.data = params_dict["scale_params"].to(
                    self.scale_params.device
                )

    def get_parameter_dict(self) -> Dict:
        """Get all parameters as dictionary"""
        return {
            "joint_offsets": self.joint_offsets.data.clone(),
            "pose_params": self.pose_params.data.clone(),
            "shape_params": self.shape_params.data.clone(),
            "scale_params": self.scale_params.data.clone(),
        }

    def freeze_parameters(self, param_names: list):
        """Freeze specified parameters"""
        for param_name in param_names:
            if param_name == "joint_offsets":
                self.joint_offsets.requires_grad = False
            elif param_name == "pose":
                self.pose_params.requires_grad = False
            elif param_name == "shape":
                self.shape_params.requires_grad = False
            elif param_name == "scale":
                self.scale_params.requires_grad = False

    def unfreeze_parameters(self, param_names: list):
        """Unfreeze specified parameters"""
        for param_name in param_names:
            if param_name == "joint_offsets":
                self.joint_offsets.requires_grad = True
            elif param_name == "pose":
                self.pose_params.requires_grad = True
            elif param_name == "shape":
                self.shape_params.requires_grad = True
            elif param_name == "scale":
                self.scale_params.requires_grad = True


class DifferentiableMHRWrapper(nn.Module):
    """
    Wrapper that makes MHR model work with DifferentiableMHRParameters.
    """

    def __init__(
        self,
        mhr_model: MHR,
        character_torch: Character,
        learnable_offsets: bool = True,
    ):
        super().__init__()
        self.mhr_model = mhr_model
        self.character_torch = character_torch

        # Replace skeleton with differentiable version
        # Note: This modifies character_torch in place. If you need the original
        # skeleton preserved, create a copy of character_torch first.
        if learnable_offsets:
            if not isinstance(character_torch.skeleton, DifferentiableSkeleton):
                original_skeleton = character_torch.skeleton
                character_torch.skeleton = DifferentiableSkeleton(
                    original_skeleton, learnable_offsets=True
                )

    def forward(
        self,
        params: DifferentiableMHRParameters,
        expr_params: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using DifferentiableMHRParameters"""
        return params.forward_mhr(
            self.mhr_model, self.character_torch, expr_params
        )
