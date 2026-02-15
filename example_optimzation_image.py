import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Local imports
from mhr_animatable_gaussians import MHRAnimatableGaussians
from gaussian_splatting import splat_gaussians
from camera import GSCamera
from ops import image2torch

def load_image_and_mask(image_path, mask_path, device, size=(512, 512)):
    # Load image
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    img_tensor = image2torch(np.array(img)).to(device) # [1, 3, H, W]

    # Load mask
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(size, Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(mask)).float().to(device) / 255.0
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    
    # Binarize mask
    mask_tensor = (mask_tensor > 0.5).float()

    return img_tensor, mask_tensor

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    data_dir = Path("data")
    image_path = data_dir / "0000.png"
    mask_path = data_dir / "0000_mask.png"
    output_dir = Path("outputs/image_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.exists() or not mask_path.exists():
        print(f"Error: Data files not found in {data_dir}")
        return

    # 1. Load Data
    H, W = 512, 512
    target_image, target_mask = load_image_and_mask(image_path, mask_path, device, size=(H, W))
    
    # Apply mask to target image (white background)
    bg_color = torch.ones(3, device=device)
    target_image_masked = target_image * target_mask + bg_color.view(1, 3, 1, 1) * (1 - target_mask)

    # 2. Initialize Model
    # Note: Ensure you have the MHR assets or a path to them if not using the default
    try:
        mhr_gaussians = MHRAnimatableGaussians(device=device, lod=1)
    except Exception as e:
        print(f"Failed to initialize MHR model: {e}")
        print("Please ensure pymomentum and MHR assets are correctly set up.")
        return

    # 3. Setup Camera
    # Assume a standard full-body view
    # Position: (0, 0.8, 2.5) looking at (0, 0.8, 0) is a reasonable default for a standing human
    camera_pos = torch.tensor([0.0, 0.8, 2.5], device=device)
    look_at = torch.tensor([0.0, 0.8, 0.0], device=device)
    fov = 0.6 # radians, approx 35 degrees
    
    cam = GSCamera.from_look_at(
        cam_pos=camera_pos,
        look_at=look_at,
        fov_x=fov,
        fov_y=fov,
        image_width=W,
        image_height=H
    ).to(device)

    # 4. Optimization Setup
    # We want to optimize pose, shape, and maybe camera params (simplified here to just model params)
    
    # Initialize parameters
    # Batch size 1
    B = 1
    # MHR shapes: Identity (45), Model/Pose (204), Expression (72)
    # We'll optimize model_params (pose) and shape_params
    
    # Start with mean pose/shape
    shape_params = torch.zeros(B, 45, device=device, requires_grad=True)
    model_params = torch.zeros(B, 204, device=device, requires_grad=True)
    # Initialize with a slight T-pose or A-pose if needed, but 0 is standard rest
    
    optimizer = optim.Adam([
        {'params': [shape_params], 'lr': 0.01},
        {'params': [model_params], 'lr': 0.01}
    ])

    print("Starting optimization...")
    pbar = tqdm(range(200))
    
    for step in pbar:
        optimizer.zero_grad()

        # Update model parameters
        mhr_gaussians.set_animate_params(shape_params, model_params)
        
        # Generate Gaussians
        # This returns a GaussianParameters object
        gs_params = mhr_gaussians.animate_gaussians(
            tpose_gs_dic={
                "offset_xyz": mhr_gaussians.offset_xyz,
                "scale": mhr_gaussians.scale,
                "rotation": mhr_gaussians.rotation,
                "opacity": mhr_gaussians.opacity,
                "sh": mhr_gaussians.sh,
            },
            to_colmap=True # Important for camera consistency
        )

        # Render
        # splat_gaussians returns a TensorDict with 'image', 'mask', etc.
        render_dict = splat_gaussians(
            gaussians=gs_params,
            cameras=cam,
            bg_color=bg_color,
            render_rasterization=False
        )
        
        pred_image = render_dict['image'] # [1, 3, H, W]
        pred_mask = render_dict['mask']   # [1, 1, H, W]

        # Loss
        # 1. Mask Loss (IOU-like or MSE)
        mask_loss = torch.nn.functional.mse_loss(pred_mask, target_mask)
        
        # 2. RGB Loss (only inside mask)
        # Simple L1 loss
        rgb_loss = torch.abs(pred_image - target_image_masked).mean()

        loss = mask_loss + rgb_loss

        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss: {loss.item():.4f} (Mask: {mask_loss.item():.4f}, RGB: {rgb_loss.item():.4f})")

        if step % 50 == 0:
            # Save visualization
            with torch.no_grad():
                viz = torch.cat([target_image_masked, pred_image], dim=3) # Concat horizontally
                viz_np = (viz[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(viz_np).save(output_dir / f"step_{step:04d}.png")

    print(f"Optimization finished. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
