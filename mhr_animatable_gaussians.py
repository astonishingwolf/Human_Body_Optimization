"""
Animatable Gaussian pipeline using MHR class directly (from mhr.mhr.MHR).

This version uses the MHR class from the momentum/pymomentum package directly
instead of relying on a torchscript model. This provides better flexibility
and access to the full MHR API.

Key differences from mhr_animatable_gaussians.py:
- Uses MHR.from_files() to load the model directly
- Supports configurable LOD (level of detail)
- No torchscript fallback (requires momentum package)
"""
import os
import warnings
from typing import Optional, Dict, Tuple, List, Union

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
import roma
from PIL import Image, ImageOps

MOMENTUM_ENABLED = os.environ.get("MOMENTUM_ENABLED") is None
try:
    if MOMENTUM_ENABLED:
        from mhr.mhr import MHR
        MOMENTUM_ENABLED = True
    else:
        raise ImportError
except:
    MOMENTUM_ENABLED = False

if not MOMENTUM_ENABLED:
    MHR = object

# Try to import GaussianParameters, fallback to simple dict-like class if unavailable
from gaussian_splatting import GaussianParameters, get_gaussian_type

from ops import matrix_to_quaternion, quaternion_multiply



def _project_points(pts, fx, fy, cx, cy):
    """
    Projects 3D points in camera space to 2D image coordinates.
    
    Args:
        pts: [N, 3] tensor of 3D points.
        fx, fy: focal lengths (scalars).
        cx, cy: principal point (scalars).
    Returns:
        u, v: [N] tensors of image coordinates.
    """
    z = pts[:, 2]
    x = pts[:, 0]
    y = pts[:, 1]
    z = torch.clamp(z, min=1e-3)
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy
    return u, v


def _compute_face_bbox(u, v, img_w, img_h, expand_ratio=0.1):
    """
    Compute a square bounding box for face vertices, expanded by a ratio.
    """
    u_min, u_max = u.min().item(), u.max().item()
    v_min, v_max = v.min().item(), v.max().item()
    
    # Expand
    w_box = u_max - u_min
    h_box = v_max - v_min
    pad_w = w_box * expand_ratio
    pad_h = h_box * expand_ratio
    
    u_min -= pad_w
    u_max += pad_w
    v_min -= pad_h
    v_max += pad_h
    
    # Make square
    w_box = u_max - u_min
    h_box = v_max - v_min
    
    if w_box > h_box:
        diff = w_box - h_box
        v_min -= diff / 2
        v_max += diff / 2
    else:
        diff = h_box - w_box
        u_min -= diff / 2
        u_max += diff / 2
        
    return int(u_min), int(v_min), int(u_max), int(v_max)


def _crop_pil_with_padding(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_size: Tuple[int, int] = (512, 512),
) -> torch.Tensor:
    """Crop PIL image with padding for out-of-bounds regions."""
    x1, y1, x2, y2 = bbox
    width, height = img.size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    x1_c, y1_c = max(0, x1), max(0, y1)
    x2_c, y2_c = min(width, x2), min(height, y2)

    if x2_c <= x1_c or y2_c <= y1_c:
        return torch.zeros(3, *target_size)

    crop = img.crop((x1_c, y1_c, x2_c, y2_c))

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = ImageOps.expand(crop, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

    crop = crop.resize((target_size[1], target_size[0]), Image.LANCZOS)

    return TF.to_tensor(crop)


def mhr_to_face_crop(batch, pose_output, gs_posed, face_indices):
    xyz = gs_posed.xyz
    assert gs_posed.xyz.ndim == 4, f'gs_posed.xyz should be in 4D shape (B, V, L, 3), but get {gs_posed.xyz}'
    assert gs_posed.xyz.shape[1] == 1, 'Currently only support V=1'
    xyz = xyz.squeeze(1)
    face_xyz = xyz[:, face_indices, :] # [B, N_face, 3]

    # Add camera translation to move to Camera Space
    pred_cam_t = pose_output['pred_cam_t']
    face_xyz = face_xyz + pred_cam_t.unsqueeze(1)

    images = batch['img'] # (B, V, C, H, W)
    fov_x_list = batch['fov_x']
    fov_y_list = batch['fov_y']
    raw_images = batch['raw_img']  # (B, V, C, H_raw, W_raw) or List[PIL.Image]
    raw_sizes = batch['raw_sizes']  # (B, V, 2) with (H, W), record the original size before resizing
    crop_bboxes = batch['crop_bboxes']  # (B, V, 4) with (x_min, y_min, x_max, y_max)

    assert images.ndim == 5, f"images must be 5D (B, V, C, H, W), but get {images.shape}"
    B, V, C, H, W = images.shape
    BV = B * V
    images = images.view(BV, *images.shape[2:])
    raw_sizes = raw_sizes.view(BV, 2)
    fov_x_list = fov_x_list.view(BV)
    fov_y_list = fov_y_list.view(BV)
    crop_bboxes = crop_bboxes.view(BV, 4)
    device = images.device

    if isinstance(raw_images, torch.Tensor):
        assert raw_images.ndim == 5, f"raw_images must be 5D (B, V, C, raw_H, raw_W), but get {raw_images.shape}"
        raw_images_flat: Union[torch.Tensor, List[Image.Image]] = raw_images.view(BV, *raw_images.shape[2:])
        raw_H, raw_W = raw_images_flat.shape[-2:]
    else:
        if len(raw_images) == B and raw_images and isinstance(raw_images[0], list):
            raw_images_flat = [raw_images[b][v] for b in range(B) for v in range(V)]
        else:
            raw_images_flat = raw_images
        if len(raw_images_flat) != BV:
            raise ValueError(f"Expected {BV} raw images, got {len(raw_images_flat)}")
        raw_H, raw_W = None, None

    face_crops, debug_viz_data = [], []

    for i in range(BV):
        # Compute Intrinsics for Processed Image
        fov_x = fov_x_list[i]
        fov_y = fov_y_list[i]

        # Assuming fov is in radians
        fx = W / (2 * torch.tan(fov_x / 2))
        fy = H / (2 * torch.tan(fov_y / 2))
        cx = W / 2.0
        cy = H / 2.0

        # Project face vertices to processed image space
        pts = face_xyz[i]  # [N_face, 3]
        u, v = _project_points(pts, fx, fy, cx, cy)

        # 3. Compute BBox in processed image space
        x1_proc, y1_proc, x2_proc, y2_proc = _compute_face_bbox(u, v, W, H)

        # 4. Convert face bbox from processed image space to raw image space
        # crop_bbox = (x_min_raw, y_min_raw, x_max_raw, y_max_raw)
        crop_x_min = crop_bboxes[i, 0].item()
        crop_y_min = crop_bboxes[i, 1].item()
        crop_x_max = crop_bboxes[i, 2].item()
        crop_y_max = crop_bboxes[i, 3].item()

        crop_w = crop_x_max - crop_x_min
        crop_h = crop_y_max - crop_y_min

        # Scale factors from processed image to raw crop region
        scale_x = crop_w / W
        scale_y = crop_h / H

        # Transform face bbox to raw image space
        x1_raw = x1_proc * scale_x + crop_x_min
        y1_raw = y1_proc * scale_y + crop_y_min
        x2_raw = x2_proc * scale_x + crop_x_min
        y2_raw = y2_proc * scale_y + crop_y_min

        # Convert to integer
        x1_raw_int = int(x1_raw)
        y1_raw_int = int(y1_raw)
        x2_raw_int = int(x2_raw)
        y2_raw_int = int(y2_raw)

        orig_raw_H = int(raw_sizes[i, 0].item())
        orig_raw_W = int(raw_sizes[i, 1].item())

        # Store debug viz data (in original raw image coordinates)
        debug_viz_data.append({
            'proc_bbox': (x1_proc, y1_proc, x2_proc, y2_proc),
            'raw_bbox': (x1_raw_int, y1_raw_int, x2_raw_int, y2_raw_int),
            'crop_bbox': (int(crop_x_min), int(crop_y_min), int(crop_x_max), int(crop_y_max)),
            'orig_raw_size': (int(orig_raw_H), int(orig_raw_W)),
        })

        if isinstance(raw_images_flat, torch.Tensor):
            raw_img = raw_images_flat[i]  # [C, raw_img_H, raw_img_W]

            # If raw_images was resized for batching, we need to scale coordinates
            # from original raw space to current tensor space
            if abs(raw_H - orig_raw_H) > 1 or abs(raw_W - orig_raw_W) > 1:
                assert False, "This branch is not tested"
                tensor_scale_x = raw_W / orig_raw_W
                tensor_scale_y = raw_H / orig_raw_H
                x1_tensor = int(x1_raw * tensor_scale_x)
                y1_tensor = int(y1_raw * tensor_scale_y)
                x2_tensor = int(x2_raw * tensor_scale_x)
                y2_tensor = int(y2_raw * tensor_scale_y)
                current_W, current_H = raw_W, raw_H
            else:
                x1_tensor, y1_tensor = x1_raw_int, y1_raw_int
                x2_tensor, y2_tensor = x2_raw_int, y2_raw_int
                current_W, current_H = int(orig_raw_W), int(orig_raw_H)

            pad_left = max(0, -x1_tensor)
            pad_top = max(0, -y1_tensor)
            pad_right = max(0, x2_tensor - current_W)
            pad_bottom = max(0, y2_tensor - current_H)

            x1_c = max(0, x1_tensor)
            y1_c = max(0, y1_tensor)
            x2_c = min(current_W, x2_tensor)
            y2_c = min(current_H, y2_tensor)

            if x2_c <= x1_c or y2_c <= y1_c:
                assert False, f"Zero cropping area: [{y1_c} : {y2_c}] [{x1_c} : {x2_c}]"
                crop = torch.zeros((C, 512, 512), device=raw_img.device, dtype=raw_img.dtype)
            else:
                crop = raw_img[:, y1_c:y2_c, x1_c:x2_c]

                if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                    crop = TF.pad(crop, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

                crop = TF.resize(crop, [512, 512], antialias=True)
        else:
            raw_img_pil = raw_images_flat[i]
            crop = _crop_pil_with_padding(
                raw_img_pil,
                (x1_raw_int, y1_raw_int, x2_raw_int, y2_raw_int),
                target_size=(512, 512),
            ).to(device=device, dtype=images.dtype)

        face_crops.append(crop)
    
    # Visualization for debugging (set to True to enable)
    if False:
        _visualize_face_bboxes(
            images, raw_images, debug_viz_data,
            output_dir="./"
        )

    face_crops = torch.stack(face_crops) # [B, C, 512, 512]

    return face_crops, debug_viz_data


def _visualize_face_bboxes(
    images: torch.Tensor,
    raw_images,
    debug_viz_data: List[Dict],
    output_dir: str = "./",
) -> None:
    """
    Visualize face bounding boxes on both processed and raw images for debugging.
    
    Args:
        images: [BV, C, H, W] processed images tensor
        raw_images: Either [B, V, C, raw_H, raw_W] tensor or list of PIL images
        debug_viz_data: List of dicts with bbox info:
            - 'proc_bbox': (x1, y1, x2, y2) in processed image space
            - 'raw_bbox': (x1, y1, x2, y2) in raw image space
            - 'crop_bbox': (x1, y1, x2, y2) crop region in raw image
            - 'orig_raw_size': (H, W) original raw image size
        output_dir: Directory to save visualization images
    """
    from PIL import ImageDraw
    
    os.makedirs(output_dir, exist_ok=True)
    
    BV = images.shape[0]
    
    for i in range(BV):
        viz_data = debug_viz_data[i]
        proc_bbox = viz_data['proc_bbox']
        raw_bbox = viz_data['raw_bbox']
        crop_bbox = viz_data['crop_bbox']
        
        # Visualize processed image with face bbox
        proc_img = images[i].detach().cpu()  # [C, H, W]
        proc_img = (proc_img.clamp(0, 1) * 255).to(torch.uint8)
        proc_img_pil = TF.to_pil_image(proc_img)
        draw_proc = ImageDraw.Draw(proc_img_pil)
        
        # Draw face bbox (green) on processed image
        x1, y1, x2, y2 = proc_bbox
        draw_proc.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        
        proc_img_pil.save(os.path.join(output_dir, f"face_bbox_proc_{i}.png"))
        
        # Visualize raw image with face bbox and crop bbox
        if isinstance(raw_images, torch.Tensor):
            # raw_images is [B, V, C, H, W] or flattened to [BV, C, H, W]
            if raw_images.ndim == 5:
                B = raw_images.shape[0]
                V = raw_images.shape[1]
                b_idx = i // V
                v_idx = i % V
                raw_img = raw_images[b_idx, v_idx].detach().cpu()
            else:
                raw_img = raw_images[i].detach().cpu()
            raw_img = (raw_img.clamp(0, 1) * 255).to(torch.uint8)
            raw_img_pil = TF.to_pil_image(raw_img)
        else:
            # raw_images is a list of PIL images
            if len(raw_images) > 0 and isinstance(raw_images[0], list):
                # Nested list [B][V]
                B = len(raw_images)
                V = len(raw_images[0]) if raw_images else 1
                b_idx = i // V
                v_idx = i % V
                raw_img_pil = raw_images[b_idx][v_idx].copy()
            else:
                raw_img_pil = raw_images[i].copy()
        
        draw_raw = ImageDraw.Draw(raw_img_pil)
        
        # Draw crop bbox (blue) on raw image
        cx1, cy1, cx2, cy2 = crop_bbox
        draw_raw.rectangle([cx1, cy1, cx2, cy2], outline=(0, 0, 255), width=3)
        
        # Draw face bbox (green) on raw image
        fx1, fy1, fx2, fy2 = raw_bbox
        draw_raw.rectangle([fx1, fy1, fx2, fy2], outline=(0, 255, 0), width=2)
        
        raw_img_pil.save(os.path.join(output_dir, f"face_bbox_raw_{i}.png"))


def _trs_to_matrix(trs: torch.Tensor, to_meters: bool = True) -> torch.Tensor:
    """
    Convert packed (x, y, z, qx, qy, qz, qw, s) to 4x4 matrix.
    MHR stores translations in centimeters; to_meters divides by 100.
    """
    t = trs[..., :3]
    if to_meters:
        t = t / 100.0
    q = trs[..., 3:7]
    s = trs[..., 7:8]
    rot = roma.unitquat_to_rotmat(q)
    rot_scaled = rot * s.unsqueeze(-1)

    mat = torch.zeros(*trs.shape[:-1], 4, 4, device=trs.device, dtype=trs.dtype)
    mat[..., :3, :3] = rot_scaled
    mat[..., :3, 3] = t
    mat[..., 3, 3] = 1.0
    return mat


class MHRAnimatableGaussians:
    """
    Animatable 3D Gaussians driven by the Meta Human Rig (MHR) model.

    This class uses the MHR class directly from the momentum package.

    Key Features:
    -   **Backend:** Uses MHR class from mhr.mhr module
    -   **LOD Support:** Configurable level of detail (0-6, where 0 is highest detail)
    -   **Animation:** Accepts shape, model (pose), and expression parameters
    -   **Gaussian Anchoring:** Manages 3D Gaussians anchored to MHR mesh vertices

    Workflow:
    1.  Initialize with desired LOD
    2.  `set_animate_params()`: Feed shape/model/expression params
    3.  `animate_gaussians()`: Transform T-pose Gaussians to world space
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        lod: int = 1,
        mhr_model: Optional[MHR] = None,
        mhr_model_path: Optional[str] = None,
    ):
        """
        Args:
            device: torch device.
            lod: Level of detail for MHR model (0-6). Default is 1 (matches TScript behavior).
            mhr_model: Optional pre-loaded MHR model instance.
            mhr_model_path: Optional path to MHR assets folder or torchscript model (.pt).
        """
        self.device = device
        self.lod = lod

        print(f"MHRAnimatableGaussians LOD={lod}")

        # Load MHR model
        self.use_torchscript = False
        if mhr_model is not None:
            self.mhr = mhr_model.to(device)
        elif lod == 1 or not MOMENTUM_ENABLED:
            if mhr_model_path is not None and os.path.exists(mhr_model_path):
                self.mhr = torch.jit.load(mhr_model_path, map_location=device)
                self.use_torchscript = True
            else:
                if MOMENTUM_ENABLED:
                    self.mhr = MHR.from_files(device=device, lod=lod)
                else:
                    raise ValueError(f"MHR model path {mhr_model_path} does not exist and pymomentum is not available.")
        else:
            self.mhr = MHR.from_files(device=device, lod=lod)

        self.mhr.eval()
        for param in self.mhr.parameters():
            param.requires_grad = False

        # Blendshape counts (torchscript models may omit expression blendshapes)
        self.num_identity_blendshapes, self.num_expr_blendshapes = self._get_blendshape_counts()

        # Static buffers from MHR
        ct = self.mhr.character_torch
        self.rest_vertices = ct.mesh.rest_vertices.to(device) / 100.0  # meters
        self.faces = ct.mesh.faces.long()
        self.inverse_bind_pose = ct.linear_blend_skinning.inverse_bind_pose.to(device)
        self.skin_indices_flattened = ct.linear_blend_skinning.skin_indices_flattened.to(device).long()
        self.skin_weights_flattened = ct.linear_blend_skinning.skin_weights_flattened.to(device)
        self.vert_indices_flattened = ct.linear_blend_skinning.vert_indices_flattened.to(device).long()
        self.joint_parents = ct.skeleton.joint_parents.to(device).long()

        self.num_joints = self.inverse_bind_pose.shape[0]
        self.num_verts = self.rest_vertices.shape[0]

        gaussian_type = get_gaussian_type()
        scale_dim = 3 if gaussian_type == "3d" else 2

        # Animation state
        self.offset_xyz = torch.zeros((1, 1, self.num_verts, 3), device=device)
        self.scale = torch.ones((1, 1, self.num_verts, scale_dim), device=device)
        self.rotation = torch.tensor([1, 0, 0, 0], device=device, dtype=torch.float32)[None, None, None, :].repeat(1, 1, self.num_verts, 1)
        self.opacity = torch.ones((1, 1, self.num_verts, 1), device=device)
        self.sh = torch.zeros((1, 1, self.num_verts, 1, 3), device=device)

        self.shape_params = None
        self.model_params = None
        self.expr_params = None
        self.apply_pose_correctives = True

        rest_pose = self._compute_rest_pose(
            torch.zeros(1, self.num_identity_blendshapes, device=device)
        )
        # Store as [V, 3] to match downstream anchor-point expectations.
        self.query_points = rest_pose[0] if rest_pose.ndim == 3 else rest_pose

    def to(self, device):
        """Move buffers and model to device."""
        self.device = device
        self.mhr = self.mhr.to(device)
        for attr in ["rest_vertices", "inverse_bind_pose", "skin_indices_flattened",
                     "skin_weights_flattened", "vert_indices_flattened", "joint_parents",
                     "offset_xyz", "scale", "rotation", "opacity", "sh", "query_points"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ----------- Core MHR helpers -----------
    def _compute_joint_parameters(self, model_params: torch.Tensor) -> torch.Tensor:
        """
        model_params: [B, 204]
        identity: [B, 45] (unused, kept for API compatibility)
        returns: joint_parameters [B, J, 8] from parameter transform
        """

        pad_size = self.num_identity_blendshapes

        if not self.use_torchscript:
            pad_size += self.num_expr_blendshapes
        padding = torch.zeros(model_params.shape[0], pad_size).to(model_params).requires_grad_(False)
        joint_parameters = torch.cat([model_params, padding], dim=1)
        return self.mhr.character_torch.model_parameters_to_joint_parameters(joint_parameters)

    def _compute_rest_pose(self, identity: torch.Tensor) -> torch.Tensor:
        """
        Returns rest pose vertices after blendshapes (before pose correctives), in meters.

        Args:
            identity: [B, 45] identity blendshape coefficients
            expr: [B, 72] face expression coefficients
        """
        # Ensure identity and expr have compatible batch dimensions
        if identity.ndim == 1:
            identity = identity.unsqueeze(0)
        
        # a mismatch between torchscript and pymomentum MHR model
        if not self.use_torchscript:
            pad = torch.zeros(identity.shape[0], self.num_expr_blendshapes, device=identity.device)
            identity = torch.cat([identity, pad], dim=1)

        rest_pose = self.mhr.character_torch.blend_shape.forward(identity)
        return rest_pose / 100.0  # Convert from cm to meters

    def _apply_pose_correctives(self, joint_parameters: torch.Tensor, rest_pose: torch.Tensor) -> torch.Tensor:
        if self.apply_pose_correctives and self.mhr.pose_correctives_model is not None:
            corr = self.mhr.pose_correctives_model(joint_parameters)
            rest_pose = rest_pose + corr / 100.0  # corr is in cm
        return rest_pose

    def _compute_skeleton_state(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        return self.mhr.character_torch.joint_parameters_to_skeleton_state(joint_parameters)

    def _blend_transforms(self, joint_mats: torch.Tensor) -> torch.Tensor:
        """
        joint_mats: [BN, J, 4, 4]
        returns per-vertex transform matrices [BN, V, 4, 4]
        """
        ibp = _trs_to_matrix(self.inverse_bind_pose, to_meters=True)  # [J,4,4]
        comp = torch.matmul(joint_mats.index_select(1, self.skin_indices_flattened), ibp.index_select(0, self.skin_indices_flattened))
        weights = self.skin_weights_flattened.view(1, -1, 1, 1)
        comp = comp * weights  # [BN, K, 4, 4]
        per_vertex = torch.zeros(joint_mats.shape[0], self.num_verts, 4, 4, device=joint_mats.device, dtype=joint_mats.dtype)
        per_vertex.index_add_(1, self.vert_indices_flattened, comp)
        return per_vertex

    # ----------- Public API -----------
    def set_animate_params(
        self,
        shape_params: torch.Tensor,
        model_params: torch.Tensor,
        expr_params: Optional[torch.Tensor] = None,
        apply_pose_correctives: bool = True,
    ):
        """
        Store animation parameters.
        Args:
            shape_params: [B, 45]
            model_params: [B, N, 204] or [B, 204]
            expr_params: [B, N, 72] or None (defaults to zeros)
        """
        if model_params.ndim == 2:
            model_params = model_params[:, None, :]
            if expr_params is not None and expr_params.ndim == 2:
                expr_params = expr_params[:, None, :]

        B, N, _ = model_params.shape
        if expr_params is None:
            expr_params = torch.zeros(B, N, self.num_expr_blendshapes, device=model_params.device)
        else:
            expected_expr = self.num_expr_blendshapes
            if expected_expr == 0:
                expr_params = expr_params[..., :0]
            elif expr_params.shape[-1] != expected_expr:
                if expr_params.shape[-1] > expected_expr:
                    expr_params = expr_params[..., :expected_expr]
                else:
                    pad = expected_expr - expr_params.shape[-1]
                    expr_params = F.pad(expr_params, (0, pad))
        self.shape_params = shape_params.to(self.device).float()
        self.model_params = model_params.to(self.device).float()
        self.expr_params = expr_params.to(self.device).float()
        self.apply_pose_correctives = apply_pose_correctives

    def _get_blendshape_counts(self) -> tuple[int, int]:
        """Infer identity/expression blendshape counts, guarding torchscript models."""
        ct = self.mhr.character_torch
        total = None
        if hasattr(ct, "blend_shape") and hasattr(ct.blend_shape, "shape_vectors"):
            try:
                total = int(ct.blend_shape.shape_vectors.shape[0])
            except Exception:
                total = None

        identity = None
        expr = None
        if hasattr(self.mhr, "get_num_identity_blendshapes"):
            try:
                identity = int(self.mhr.get_num_identity_blendshapes())
            except Exception:
                identity = None
        if hasattr(self.mhr, "get_num_face_expression_blendshapes"):
            try:
                expr = int(self.mhr.get_num_face_expression_blendshapes())
            except Exception:
                expr = None

        if identity is None and total is not None:
            identity = total
        if expr is None:
            expr = 0

        return identity or 0, expr or 0

    def _prepare_pose(self, pose_indices):
        assert self.shape_params is not None, "Call set_animate_params first."
        B = self.shape_params.shape[0]
        N = len(pose_indices)
        shape = self.shape_params[:, None, :].expand(B, N, -1).reshape(B * N, -1)
        model = self.model_params[:, pose_indices].reshape(B * N, -1)
        expr = self.expr_params[:, pose_indices].reshape(B * N, -1)
        return shape, model, expr, B, N

    def _compute_pose_transforms(self, shape, model, expr):
        joint_params = self._compute_joint_parameters(model)
        skel_state = self._compute_skeleton_state(joint_params)
        rest_pose = self._compute_rest_pose(shape)
        rest_pose = self._apply_pose_correctives(joint_params, rest_pose)
        joint_mats = _trs_to_matrix(skel_state, to_meters=True)  # [BN, J, 4,4]
        per_vertex_tf = self._blend_transforms(joint_mats)  # [BN, V, 4,4]
        return rest_pose, per_vertex_tf, joint_mats

    def animate_gaussians(
        self,
        tpose_gs_dic,
        pose_indices=None,
        to_colmap=True,
    ) -> GaussianParameters:
        """
        Animate T-pose Gaussians using MHR transforms.
        Args:
            tpose_gs_dic: TensorDict with keys offset_xyz, scale, rotation, opacity, sh.
                Shapes [B, 1, L, ...]; L must equal num_verts if no anchor mapping is provided.
            pose_indices: list of pose indices to animate; defaults to all.
            to_colmap: apply COLMAP coordinate flip.
        Returns:
            GaussianParameters with shapes [B, N, L, ...]
        """
        if pose_indices is None:
            pose_indices = list(range(self.model_params.shape[1]))
        shape, model, expr, B, N = self._prepare_pose(pose_indices)

        # Validate anchors
        offset_xyz = tpose_gs_dic["offset_xyz"]
        scale_in = tpose_gs_dic["scale"]
        rotation_in = tpose_gs_dic["rotation"]
        opacity_in = tpose_gs_dic["opacity"]
        sh_in = tpose_gs_dic["sh"]

        L = offset_xyz.shape[2]
        if L != self.num_verts:
            raise ValueError(f"L={L} must equal number of verts {self.num_verts}")

        rest_pose, per_vertex_tf, _ = self._compute_pose_transforms(shape, model, expr)
        per_vertex_tf = rearrange(per_vertex_tf, "(b n) ... -> b n ...", b=B, n=N)
        rest_pose = rearrange(rest_pose, "(b n) v c -> b n v c", b=B, n=N)

        # Base points are rest pose vertices
        base_points = rest_pose  # [B,N,L,3]

        gaussian_type = get_gaussian_type()
        scale_dim = 3 if gaussian_type == "3d" else 2

        # Expand T-pose GS params across poses
        offset_xyz = offset_xyz.expand(B, N, L, 3)
        scale = scale_in.expand(B, N, L, scale_dim)
        rotation = rotation_in.expand(B, N, L, 4)
        opacity = opacity_in.expand(B, N, L, 1)
        sh = sh_in.expand(B, N, L, 1, 3)

        base_mean = base_points + offset_xyz
        base_h = torch.cat([base_mean, torch.ones_like(base_mean[..., :1])], dim=-1)[..., None]  # [B,N,L,4,1]

        # Apply transforms
        tf_selected = per_vertex_tf  # [B,N,L,4,4]
        posed_h = torch.matmul(tf_selected, base_h).squeeze(-1)
        posed_xyz = posed_h[..., :3]

        rot_mat = tf_selected[..., :3, :3]
        rigid_quat = F.normalize(matrix_to_quaternion(rot_mat).float(), dim=-1, eps=1e-5).to(rot_mat.dtype)
        posed_quat = quaternion_multiply(rigid_quat, rotation)

        gs_param = GaussianParameters(
            xyz=posed_xyz,
            scale=scale,
            rotation=posed_quat,
            opacity=opacity,
            sh=sh,
            active_sh_degree=0,  
        )

        if to_colmap:
            sign = torch.tensor([1, 1, -1, -1], device=gs_param.xyz.device, dtype=gs_param.xyz.dtype)
            gs_param.xyz *= sign[1:].view(1, 1, 1, -1)
            gs_param.rotation *= sign.view(1, 1, 1, -1)
        return gs_param

    # Convenience: mimic slim_animatable_gaussians naming
    def to_gaussian_parameters(self, pose_indices=None, to_colmap=True):
        return self.animate_gaussians(
            tpose_gs_dic={
                "offset_xyz": self.offset_xyz,
                "scale": self.scale,
                "rotation": self.rotation,
                "opacity": self.opacity,
                "sh": self.sh,
            },
            pose_indices=pose_indices,
            to_colmap=to_colmap,
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        keys = ["offset_xyz", "scale", "rotation", "opacity", "sh"]
        return {k: getattr(self, k) for k in keys}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        for k, v in state_dict.items():
            if hasattr(self, k):
                setattr(self, k, v.to(self.device))
        return self

    def forward_mhr(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: Optional[torch.Tensor] = None,
        apply_correctives: bool = True,
    ):
        """
        Direct forward pass through MHR model.

        Args:
            identity_coeffs: [B, 45] identity blendshape coefficients
            model_parameters: [B, 204] model parameters
            face_expr_coeffs: [B, 72] face expression coefficients (optional)
            apply_correctives: whether to apply pose correctives

        Returns:
            verts: [B, V, 3] vertices in centimeters
            skel_state: [B, J, 8] skeleton state
        """
        return self.mhr(identity_coeffs, model_parameters, face_expr_coeffs, apply_correctives)
