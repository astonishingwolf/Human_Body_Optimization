# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import torch.nn as nn

from pymomentum.torch.character import Skeleton
from pymomentum.torch.utility import _unsqueeze_joint_params
import pymomentum.quaternion as pym_quaternion
from pymomentum.backend import trs_backend


class DifferentiableSkeleton(nn.Module):
    """
    Wrapper around Skeleton that makes joint_translation_offsets learnable.
    
    This allows joint offsets to be optimized independently of other parameters.
    All other functionality remains the same as the original Skeleton class.
    """

    def __init__(self, skeleton: Skeleton, learnable_offsets: bool = True):
        super().__init__()
        self.skeleton = skeleton
        self.learnable_offsets = learnable_offsets

        if learnable_offsets:
            # Convert buffer to learnable parameter
            self.joint_translation_offsets = nn.Parameter(
                skeleton.joint_translation_offsets.clone()
            )
        else:
            # Keep as frozen buffer
            self.register_buffer(
                "joint_translation_offsets",
                skeleton.joint_translation_offsets
            )

        # Copy other frozen buffers (these remain frozen)
        self.joint_prerotations = skeleton.joint_prerotations
        self.pmi = skeleton.pmi
        self._pmi_buffer_sizes = skeleton._pmi_buffer_sizes
        self.joint_parents = skeleton.joint_parents
        self.joint_names = skeleton.joint_names

    def joint_parameters_to_local_skeleton_state(
        self, joint_parameters: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert joint parameters to local skeleton state.
        Uses learnable joint_translation_offsets instead of frozen buffer.
        """
        if not hasattr(self, "joint_prerotations"):
            raise RuntimeError("Character has no skeleton")

        joint_parameters = _unsqueeze_joint_params(joint_parameters)

        joint_translation_offsets = self.joint_translation_offsets
        while joint_translation_offsets.ndim < joint_parameters.ndim:
            joint_translation_offsets = joint_translation_offsets.unsqueeze(0)

        joint_prerotations = self.joint_prerotations
        while joint_prerotations.ndim < joint_parameters.ndim:
            joint_prerotations = joint_prerotations.unsqueeze(0)

        # Use learnable offsets here
        local_state_t = (
            joint_parameters[..., :3] + self.joint_translation_offsets[None, :]
        )
        local_state_q = pym_quaternion.euler_xyz_to_quaternion(
            joint_parameters[..., 3:6]
        )
        local_state_q = pym_quaternion.multiply_assume_normalized(
            self.joint_prerotations[None],
            local_state_q,
        )
        # exp2 is not supported by all of our export formats, so we have to implement
        # it using exp and natural log instead. The constant here is ln(2.0)
        local_state_s = torch.exp(0.6931471824645996 * joint_parameters[..., 6:])
        return torch.cat([local_state_t, local_state_q, local_state_s], dim=-1)

    def local_skeleton_state_to_joint_parameters(
        self,
        local_skel_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert local skeleton state to joint parameters.
        Accounts for learnable joint_translation_offsets.
        """
        joint_translation_offsets = self.joint_translation_offsets
        joint_prerotations = self.joint_prerotations

        # Compute translation joint parameters (subtract learnable offsets)
        translation_params = local_skel_state[..., :3] - joint_translation_offsets[None]

        # Invert out the pre-rotations:
        local_rotations = local_skel_state[..., 3:7]
        adjusted_rotations = pym_quaternion.multiply(
            pym_quaternion.inverse(joint_prerotations[None]),
            local_rotations,
        )

        # Use the pymomentum.quaternion implementation instead of real_lbs_quaternion's implementation
        # because it's more numerically stable. This is important for backpropagating through this
        # function.
        rotation_joint_params = pym_quaternion.quaternion_to_xyz_euler(
            adjusted_rotations, eps=1e-6
        )

        # Compute scale joint parameters
        scale_joint_params = torch.log2(local_skel_state[..., 7:8])

        return torch.cat(
            [translation_params, rotation_joint_params, scale_joint_params], dim=-1
        ).flatten(-2, -1)

    def joint_parameters_to_local_trs(self, joint_parameters: torch.Tensor):
        """
        Convert joint parameters to local TRS using learnable offsets.
        Override to use our learnable joint_translation_offsets.
        """

        if not hasattr(self, "joint_prerotations"):
            raise RuntimeError("Character has no skeleton")

        joint_parameters = _unsqueeze_joint_params(joint_parameters)

        # Convert joint_prerotations from quaternions to rotation matrices
        joint_rotation_matrices = pym_quaternion.to_rotation_matrix(
            self.joint_prerotations
        )

        # Use the TRS backend with our learnable offsets
        return trs_backend.get_local_state_from_joint_params(
            joint_params=joint_parameters,
            joint_offset=self.joint_translation_offsets,  # Use learnable offsets
            joint_rotation=joint_rotation_matrices,
        )

    def joint_parameters_to_trs(self, joint_parameters: torch.Tensor):
        """
        Convert joint parameters to global TRS using learnable offsets.
        Override to use our learnable joint_translation_offsets.
        """
        # Get local TRS state using our overridden method
        local_trs = self.joint_parameters_to_local_trs(joint_parameters)
        local_state_t, local_state_r, local_state_s = local_trs

        # Convert local TRS to global TRS using forward kinematics
        return trs_backend.global_trs_state_from_local_trs_state(
            local_state_t=local_state_t,
            local_state_r=local_state_r,
            local_state_s=local_state_s,
            prefix_mul_indices=list(
                self.pmi.split(
                    split_size=self._pmi_buffer_sizes,
                    dim=1,
                )
            ),
        )

    def local_trs_to_global_trs(self, local_trs):
        """Delegate to skeleton's method."""
        return self.skeleton.local_trs_to_global_trs(local_trs)

    def global_trs_to_local_trs(self, global_trs):
        """Delegate to skeleton's method."""
        return self.skeleton.global_trs_to_local_trs(global_trs)

    def local_skeleton_state_to_skeleton_state(
        self, local_skel_state: torch.Tensor
    ) -> torch.Tensor:
        """Delegate to skeleton's method."""
        return self.skeleton.local_skeleton_state_to_skeleton_state(local_skel_state)

    def skeleton_state_to_local_skeleton_state(
        self, skel_state: torch.Tensor
    ) -> torch.Tensor:
        """Delegate to skeleton's method."""
        return self.skeleton.skeleton_state_to_local_skeleton_state(skel_state)

    def skeleton_state_to_joint_parameters(
        self, skel_state: torch.Tensor
    ) -> torch.Tensor:
        """Convert skeleton state to joint parameters using learnable offsets."""
        return self.local_skeleton_state_to_joint_parameters(
            self.skeleton_state_to_local_skeleton_state(skel_state)
        )

    def joint_parameters_to_skeleton_state(
        self, joint_parameters: torch.Tensor
    ) -> torch.Tensor:
        """Convert joint parameters to skeleton state using learnable offsets."""
        local_skel_state = self.joint_parameters_to_local_skeleton_state(
            joint_parameters
        )
        return self.local_skeleton_state_to_skeleton_state(local_skel_state)

    def forward(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        """Forward pass through skeleton using learnable offsets."""
        if joint_parameters.ndim == 1:
            joint_parameters = joint_parameters[None, :]
        return self.joint_parameters_to_skeleton_state(joint_parameters)
