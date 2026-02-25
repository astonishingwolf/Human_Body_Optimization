# SAM-3D-Body Integration

This directory contains the integration of the **SAM-3D-Body** foundation model, which serves as the backbone for F4DHuman's avatar reconstruction pipeline.

## Overview

SAM-3D-Body provides robust 2D feature extraction and 3D human pose/shape estimation from monocular images. In F4DHuman, it is used to:
1.  Extract high-quality image features (via DINOv2/DINOv3 backbones).
2.  Estimate SMPL-X parameters (Pose, Shape, Expression) to guide the 3D Gaussian animation.
3.  Provide camera parameter estimates.

## Key Files

| File | Description |
| :--- | :--- |
| `sam_3d_body_estimator.py` | High-level wrapper (`SAM3DBodyEstimator`) that orchestrates detection, segmentation, and model inference for a single image. |
| `models/meta_arch/sam3db_body_base.py` | Defines the `SAM3DBodyBase` class, the core PyTorch module implementing the architecture (Backbone, Head, Decoder). `SAM3DAvatar` inherits from this class. |
| `utils/config.py` | Configuration utilities for loading and merging SAM-3D-Body configs (YACS/OmegaConf). |
| `build_models.py` | Factory functions to load pretrained SAM-3D-Body models from checkpoints or Hugging Face. |

## Integration with F4DHuman

The `SAM3DAvatar` class (in `lib/model/sam_3d_avatar.py`) extends `SAM3DBodyBase`. It reuses the pre-trained backbone and pose estimation heads while adding a new `AvatarBranch` for predicting 3D Gaussian Splatting parameters.

### Data Flow
1.  **Input:** RGB Image.
2.  **Backbone:** Extracts multi-scale features.
3.  **Pose Branch (Frozen/Finetuned):** Estimates SMPL-X parameters ($ \theta, \beta, \psi $).
4.  **Avatar Branch (New):**
    -   Takes backbone features and estimated pose.
    -   Predicts canonical 3D Gaussians (offsets from T-pose).
    -   Animates Gaussians to the estimated pose using MHR (Meta Human Rig).
