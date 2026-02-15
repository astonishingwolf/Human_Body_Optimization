"""3D Gaussian splatting rendering."""
import math
import torch
import diff_gaussian_rasterization_da as diff_gaussian_rasterization

from typing import List, Union
from torch.cuda.amp import autocast

from basetype import TensorDict
from camera import GSCamera
from ops import build_scaling_rotation


class GaussianParameters(TensorDict):
    """
    The unified data structure for 3D Gaussian parameters.

    This class holds all learnable parameters required for 3D Gaussian Splatting:
    - xyz: [N, 3] Centers
    - scale: [N, 3] Scaling factors (activated)
    - rotation: [N, 4] Quaternions (activated)
    - opacity: [N, 1] Opacity values (activated)
    - sh: [N, (deg+1)**2, 3] Spherical Harmonics coefficients

    It inherits from TensorDict to support batch operations (split, stack) and device movement.
    It also provides methods to compute the covariance matrix (`calc_covariance`) required 
    by the rasterizer.
    """

    DEFAULT_KEYS = [
        'xyz',              # spacial center of Gaussian. (N, 3) or (B, N, 3).
        'scale',            # scale of Gaussian in xyz direction. (N, 3) or (B, N, 3). Already activated.
        'rotation',         # rotation of Gaussian. (N, 4) or (B, N, 4). Already activated.
        'covariance',       # covariance matrix of Gaussian. (N, 6) or (B, N, 6) in packed format.
        'opacity',          # opacity of Gaussian. (N, 1) or (B, N, 1). In [0, 1], already activated.
        'feature',          # feature of Gaussian. Optional. (N, C) or (B, N, C)
        'sh',               # spherical harmonics. (N, D, C), or (B, N, D, C). D = (deg + 1) ** 2)
        'precomp_color',    # precomputed color. (N, 3) or (B, N, 3).
        'active_sh_degree', # degree of SH.
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.check_default_parameters()

    def check_default_parameters(self):
        """Check the default parameters."""
        for k in self.DEFAULT_KEYS:
            if k not in self:
                self[k] = None
                if k == 'active_sh_degree':
                    self[k] = 0

    def calc_covariance(self, scaling_modifier=1.0):
        """Compute 3DGS covariance as 6D packed format.
        
        Returns 6D packed covariance [N, 6] for 3DGS rasterizer.
        """
        with torch.amp.autocast(device_type='cuda', enabled=False):
            if self.rotation.ndim == 3:
                scale = self.scale.reshape(-1, 3)
                rotation = self.rotation.reshape(-1, 4)
            else:
                scale = self.scale
                rotation = self.rotation
            # Build scaling-rotation matrix: RS (3x3)
            RS = build_scaling_rotation(scale * scaling_modifier, rotation.float()).permute(0, 2, 1)
            
            # Compute covariance matrix: RS @ RS^T
            cov3D = RS @ RS.transpose(-1, -2)
            
            # Pack to 6D format: [xx, xy, xz, yy, yz, zz]
            cov3D_packed = torch.zeros((cov3D.shape[0], 6), dtype=cov3D.dtype, device=cov3D.device)
            cov3D_packed[:, 0] = cov3D[:, 0, 0]  # xx
            cov3D_packed[:, 1] = cov3D[:, 0, 1]  # xy
            cov3D_packed[:, 2] = cov3D[:, 0, 2]  # xz
            cov3D_packed[:, 3] = cov3D[:, 1, 1]  # yy
            cov3D_packed[:, 4] = cov3D[:, 1, 2]  # yz
            cov3D_packed[:, 5] = cov3D[:, 2, 2]  # zz
            
            if self.rotation.ndim == 3:
                cov3D_packed = cov3D_packed.reshape(self.rotation.shape[0], -1, 6)
            
            self.covariance = cov3D_packed

            if torch.isnan(self.covariance).any():
                print("NAN FOUND in covariance")
                print(f"rotation: {self.rotation.shape}, {self.rotation.min():.3f}, {self.rotation.max():.3f} {self.rotation.dtype}")
                print(f"scale: {self.scale.shape}, {self.scale.min():.3f}, {self.scale.max():.3f} {self.scale.dtype}")
                assert False
            return cov3D_packed

    @staticmethod
    def from_point_cloud(points, colors, radius=1e-3):
        device, dtype = points.device, points.dtype
        if colors.shape[1] == 4:
            opacity = colors[:, 3:4]
        else:
            opacity = torch.ones_like(points[:, :1])
        # 3DGS covariance: 6D packed format
        # Initialize as identity-like covariance
        cov = torch.zeros((points.shape[0], 6), dtype=dtype, device=device)
        cov[:, 0] = radius ** 2  # xx
        cov[:, 3] = radius ** 2  # yy
        cov[:, 5] = radius ** 2  # zz
        return GaussianParameters(
            xyz=points,
            covariance=cov,
            opacity=opacity,
            sh=RGB2SH(colors[:, :3]).reshape(-1, 1, 3),
            active_sh_degree=0)

##### Spherical Harmonics #####


C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0, :]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1, :] +
                C1 * z * sh[..., 2, :] -
                C1 * x * sh[..., 3, :])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4, :] +
                    C2[1] * yz * sh[..., 5, :] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6, :] +
                    C2[3] * xz * sh[..., 7, :] +
                    C2[4] * (xx - yy) * sh[..., 8, :])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


def RGB2SH(rgb):
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    return sh * C0 + 0.5


def sh2rgb(G, camera, shs):
    shs = shs.transpose(1, 2).reshape(-1, 3, (G.max_sh_degree + 1) ** 2)
    dir_pp_normalized = None
    if G.active_sh_degree > 0:
        dir_pp = (G.xyz - camera.camera_center).repeat(shs.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(G.active_sh_degree, shs, dir_pp_normalized)

    return sh2rgb + 0.5 #torch.clamp_min(sh2rgb + 0.5, 0.0)


def calc_vd_color(G: GaussianParameters, camera: GSCamera):
    """Calculate view dependent color.
    G.sh: (N, deg, 3)
    """
    sh = G.sh

    assert sh.ndim == 3, f"sh.ndim must be (N, deg, 3), but got {sh.shape}"
    
    if sh.ndim == 2:
        sh = sh.unsqueeze(1) 
    elif sh.ndim == 3:
        if sh.shape[-2] == 3 and sh.shape[-1] != 3:
            sh = sh.transpose(-1, -2) 
            
    dir_pp_normalized = None
    if G.active_sh_degree > 0:
        dir_pp = (G.xyz - camera.camera_center).repeat(sh.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(G.active_sh_degree, sh, dir_pp_normalized)

    return sh2rgb + 0.5 #torch.clamp_min(sh2rgb + 0.5, 0.0)


def splat_gaussians(
    gaussians: Union[GaussianParameters, List[GaussianParameters]],
    cameras: Union[GSCamera, List[GSCamera]],
    bg_color: torch.Tensor,
    render_feature=False,
    render_rasterization=False,
    opacity_key='opacity',
    use_gen_size=True,
    debug: bool=False):
    """Render gaussians on a given camera.
    Args:
        gaussians: A single or a list of GaussianParameters.
        cameras: A single or a list of GSCamera.
        bg_color: The background color.
        render_feature: Whether to render feature.
        debug: Whether to debug.
    Returns:
        A dictionary containing the rendered images, radii, viewspace points, visibility filter, depth, and mask.
    """

    if not isinstance(gaussians, list):
        if gaussians.xyz.ndim == 3:
            gaussians = gaussians.split(1)
        elif gaussians.xyz.ndim == 2:
            gaussians = [gaussians]
        else:
            raise ValueError(f'Invalid shape of gaussians: {gaussians.xyz.shape}')
    if not isinstance(cameras, list):
        cameras = [cameras]

    # Create zero tensor. We will use it to make pytorch return gradients 
    # of the 2D (screen-space) means. We use list because Gaussians might
    # have different number of points in the list
    screen_point_grad = [torch.zeros_like(gaussians[0].xyz).requires_grad_(True)
                         for _ in cameras]

    dic = {
        'image': [],
        'radii': [],
        'depth': [],
        'mask': [],
        'pointindice': [],
        'pointcontrib': [],
        'viewspace_points': [],
        'visibility_filter': [],
        'feature': []}

    for i, camera in enumerate(cameras):
        G = gaussians[0] if len(gaussians) == 1 else gaussians[i]
        if camera.camera_center.device != G.xyz.device:
            camera.to(G.xyz.device)

        raster_res = _rasterize(
            G, camera, screen_point_grad[i],
            opacity_key=opacity_key,
            bg_color=bg_color,
            render_feature=render_feature,
            render_rasterization=render_rasterization,
            use_gen_size=use_gen_size,
            debug=debug)

        dic['image'].append(raster_res[0])
        dic['radii'].append(raster_res[1])
        dic['depth'].append(raster_res[2])
        dic['mask'].append(raster_res[3])
        if render_rasterization:
            dic['pointindice'].append(raster_res[4])
            dic['pointcontrib'].append(raster_res[5])
        if render_feature:
            dic['feature'].append(raster_res[4])
        dic['viewspace_points'].append(screen_point_grad[i])
        dic['visibility_filter'].append(raster_res[1] > 0)

    stack_keys = ['image', 'depth', 'mask']
    if render_feature:
        stack_keys.append('feature')
    if render_rasterization:
        stack_keys.extend(['pointindice', 'pointcontrib'])
    for k in stack_keys:
        dic[k] = torch.stack(dic[k])
    
    return TensorDict(dic)


def _rasterize(
        G: GaussianParameters,
        camera: GSCamera,
        screen_point_grad: torch.Tensor,
        bg_color: torch.Tensor,
        opacity_key='opacity',
        render_feature=False,
        render_rasterization=False,
        use_gen_size=True,
        debug=False):
    """Rasterize a single gaussian using 3DGS rasterizer.
    Args:
        G: The gaussian to be rasterized.
        screenspace_points: The screenspace points to be rasterized.
    """
    mod = diff_gaussian_rasterization

    image_height, image_width = camera.image_height, camera.image_width
    scale_modifier = 1.0
    if use_gen_size and 'gen_size' in G:
        image_height, image_width = G.gen_size

    raster_settings = mod.GaussianRasterizationSettings(
        image_height=image_height,
        image_width=image_width,
        tanfovx=math.tan(camera.fov_x / 2),
        tanfovy=math.tan(camera.fov_y / 2),
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform.float(),
        projmatrix=camera.full_proj_transform.float(),
        sh_degree=G.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=debug)
    rasterizer = mod.GaussianRasterizer(raster_settings=raster_settings)

    G.squeeze() # in case any batch dimension remains

    # Compute covariance for 3DGS
    G.calc_covariance(scale_modifier)
    cov3D_precomp = G.covariance

    extra_dic = {}
    if render_feature:
        extra_dic = {'semantic_feature': G.feature}

    if G.precomp_color is not None:
        color = G.precomp_color
    else:
        color = calc_vd_color(G, camera)
    
    for k, v in G.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                print(f'NAN FOUND {k}: {v.shape}, {v.min():.3f}, {v.max():.3f} {v.dtype}')
                assert False

    # 3DGS rasterizer returns (rendered_image, radii, depth, mask, [pointindice/feature], [pointcontrib])
    rendered_image, radii, depth, mask = rasterizer(
        means3D=G.xyz,
        means2D=screen_point_grad,
        shs=None, colors_precomp=color,
        opacities=G[opacity_key],
        cov3D_precomp=cov3D_precomp,
        **extra_dic)

    if render_rasterization:
        return (rendered_image, radii, depth, mask, None, None)
    elif render_feature:
        return (rendered_image, radii, depth, mask, rendered_image)
    else:
        return (rendered_image, radii, depth, mask)
