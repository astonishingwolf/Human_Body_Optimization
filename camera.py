"""Camera and related transformation.
"""
import math
import torch
import numpy as np
from PIL import Image
from typing import NamedTuple, Union, Optional, List, Tuple, Dict, Any
from scipy.spatial.transform import Rotation
from einops import rearrange, einsum
from numpy.typing import NDArray

#from lib.ops import matrix_to_quaternion

######################################
##### Gaussian Splatting Cameras #####
######################################


def dynamic_orbit_angles(n_orbits: int, n_steps_per_orbit: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sample parameters for dynamic orbits.
    
    Args:
        n_orbits: Number of different orbits to sample
        n_steps_per_orbit: Number of camera positions per orbit
        
    Returns:
        elevs: Array of elevation angles for each orbit [n_orbits, n_steps_per_orbit]
        azims: Array of azimuth angles for each orbit [n_orbits, n_steps_per_orbit]
    """
    # Initialize elevation angles
    elevs = np.zeros((n_orbits, n_steps_per_orbit))
    azims = np.zeros((n_orbits, n_steps_per_orbit))
    
    # Sample number of sinusoids for each orbit (1-3)
    n_sinusoids = np.array([4]*n_orbits) #np.random.randint(1, 4, size=n_orbits)
    t = np.repeat(np.linspace(0, 1, n_steps_per_orbit)[None], n_orbits, axis=0)
    
    # For each orbit, create the elevation pattern
    for i in range(n_orbits):
        # Create vectorized sinusoids for this orbit
        periods = np.arange(1, n_sinusoids[i] + 1)
        amplitudes = np.random.uniform(10, 30, size=n_sinusoids[i]) * np.pi / 180
        amplitudes *= 2.0 ** np.arange(0, -4, -1)

        # we let both the phases and azimuth to be a bit random from the center
        phases = np.random.uniform(-np.pi / 12, np.pi / 12, size=n_sinusoids[i])
        azim_0 = np.random.uniform(-np.pi / 12, np.pi / 12, (1,))
        # we let the phases to stochastically add a pi
        random_signs = np.random.choice([-1, 1], size=n_sinusoids[i])
        periods = random_signs * periods

        # Compute all sinusoids at once and sum them
        period_matrix = np.repeat(2 * np.pi * periods[:, None], n_steps_per_orbit, axis=1)
        phase_matrix = np.repeat(phases[:, np.newaxis], n_steps_per_orbit, axis=1)
        amplitude_matrix = np.repeat(amplitudes[:, np.newaxis], n_steps_per_orbit, axis=1)
        
        # Calculate sinusoids and sum them
        sinusoids = amplitude_matrix * np.sin(t * period_matrix + phase_matrix)
        elevs[i] = np.sum(sinusoids, axis=0)

        # Clamp elevation angles to be between -89 and 89 degrees (in radians)
        max_angle = np.pi / 2 - 0.05
        elevs[i] = np.clip(elevs[i], -max_angle, max_angle)

        delta_elevs = np.roll(elevs[i], -1) - elevs[i]
        max_de = np.max(np.abs(delta_elevs)) + 0.01# + np.pi * 2 / n_steps_per_orbit
        delta_azims = max_de - np.abs(delta_elevs)
        delta_azims = delta_azims / delta_azims.sum() * np.pi * 2

        # we randomly decide the rotation direction
        if np.random.rand() > 0.5:
            delta_azims = delta_azims[::-1]
        azims[i] = np.cumsum(np.concatenate([azim_0, delta_azims]))[:-1]
    #print(azims)
    return azims, elevs


def horizontal_turntable_angles(
        n_steps: int = 120,
        n_rot: int = 1,
        elev_low: float = -math.pi/4,
        elev_high: float = math.pi/4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the elevation and azimus angle of 360 rotation.
    
    Args:
        n_steps: Number of steps for the turntable rotation
        n_rot: Number of rotations to complete
        elev_low: Lower bound for elevation angle in radians
        elev_high: Upper bound for elevation angle in radians
        
    Returns:
        tuple: (azims, elevs) where azims and elevs are numpy arrays of azimuth and elevation angles
    """
    half_steps = math.ceil(n_steps / 2)
    rot_steps = n_steps // n_rot
    elevs = np.linspace(elev_low, elev_high, half_steps)
    elevs = np.concatenate([elevs, elevs[::-1]]) if n_steps % 2 == 0 \
        else np.concatenate([elevs[:-1], elevs[::-1]])
    elevs = np.concatenate([elevs[half_steps//2:], elevs[:half_steps//2]])
    azims = np.linspace(0, 2 * np.pi, rot_steps)
    return torch.from_numpy(azims).float(), torch.from_numpy(elevs).float()


def frontal_nearview_angles(
        n_steps: int = 120,
        n_rot: int = 3,
        azim_low: float = -math.pi/4,
        azim_high: float = math.pi/4,
        elev_low: float = -math.pi/4,
        elev_high: float = math.pi/4) -> Tuple[np.ndarray, np.ndarray]:
    """Return the elevation and azimus angle of 360 rotation.
    """
    rot_steps = n_steps // n_rot // 2
    extra_steps = n_steps - rot_steps * n_rot * 2
    elevs = np.linspace(elev_low, elev_high, rot_steps)
    elevs = np.concatenate([elevs, elevs[::-1]]) # low -> high -> low
    # shift to 0 -> high -> low -> 0
    elevs = np.concatenate([elevs[rot_steps//2:], elevs[:rot_steps//2], elevs[:extra_steps]])

    azims = np.linspace(azim_low, azim_high, rot_steps)
    azims = np.concatenate([azims, azims[::-1], azims[:extra_steps]])

    return azims, elevs


def fibonacci_sphere_angles(n_frame: int = 120) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the elevation and azimus angle of 360 rotation.
    """

    indices = torch.arange(n_frame * 2)
    elev_fibonacci = torch.pi / 2 - torch.arccos(1 - indices / n_frame) 
    azim_fibonacci = torch.pi * (1 + 5**0.5) * indices

    # remove a small room around the vertical line
    valid_mask = (elev_fibonacci < torch.pi / 2 - torch.pi / 6) & (elev_fibonacci > - torch.pi / 2 + torch.pi / 6)

    return azim_fibonacci[valid_mask], elev_fibonacci[valid_mask]


def fibonacci_camera_with_noise(
    look_at_base: torch.Tensor,
    radius_min: float = 1.1, 
    radius_max: float = 2.1, 
    fov_max: float = 35.0,
    n_views: int = 60,
    noise_std: float = 0.04) -> tuple[torch.Tensor, float, float]:
    """Sample camera with uniform distribution of radius and fov and uniformly on the sphere using fibonacci sequence.
    """

    radius = np.random.uniform(radius_min, radius_max)
    fov = 2 * math.atan2(radius_max * math.tan(math.radians(fov_max / 2)), radius) * 180 / math.pi

    # first sample 360 views
    azims, elevs = fibonacci_sphere_angles(n_views)
    azims += torch.randn_like(azims) * noise_std 
    elevs += torch.randn_like(elevs) * noise_std

    cam_poses = pos_from_angle(azims, elevs, radius)
    cam_poses = look_at_base[None] + cam_poses.to(look_at_base)
    return cam_poses, radius, fov


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    fov_x: float
    fov_y: float
    image: np.ndarray
    image_path: str
    depth_path: str
    mask_path: str
    width: int
    height: int


class GSCamera:
    """
    Camera class compatible with the Gaussian Splatting rasterizer.
    
    This class encapsulates camera extrinsics (world-to-camera), intrinsics (projection),
    and image metadata. It handles the conversion between different coordinate systems 
    (e.g., Colmap/OpenCV to the coordinate system expected by the rasterizer).

    Attributes:
        world_view_transform (Tensor): [4, 4] World-to-Camera matrix (transposed for rasterizer).
        full_proj_transform (Tensor): [4, 4] World-to-Clip matrix (transposed).
        camera_center (Tensor): [3] Camera position in world space.
        fov_x, fov_y (float): Field of view in radians.
        image_width, image_height (int): Resolution.
    """

    def __init__(self,
                 world_view_transform: Optional[torch.Tensor] = None, # world to camera
                 full_proj_transform: Optional[torch.Tensor] = None, # world to screen
                 image: Optional[torch.Tensor] = None, 
                 mask: Optional[torch.Tensor] = None, 
                 depth: Optional[torch.Tensor] = None, 
                 normal: Optional[torch.Tensor] = None,
                 image_path: Optional[str] = None, 
                 mask_path: Optional[str] = None,
                 depth_path: Optional[str] = None, 
                 normal_path: Optional[str] = None,
                 image_width: int = 512, 
                 image_height: int = 512,
                 fov_x: float = 0.2443, 
                 fov_y: float = 0.2443,
                 znear: float = 0.01, 
                 zfar: float = 100,
                 kao_cam: Optional[Any] = None):
        self.image, self.mask, self.depth, self.normal = image, mask, depth, normal
        self.image_width, self.image_height = image_width, image_height
        self.image_path = image_path
        self.mask_path, self.depth_path = mask_path, depth_path
        self.normal_path = normal_path
        self.fov_x, self.fov_y = fov_x, fov_y
        self.znear, self.zfar = znear, zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.kao_cam = kao_cam
        self.camera_center: Optional[torch.Tensor] = None
        if world_view_transform is not None:
            orig_type = world_view_transform.dtype
            view_inv = torch.inverse(world_view_transform.float()).to(orig_type)
            self.camera_center = view_inv[3][:3]

    def __repr__(self) -> str:
        viz_fn = lambda x : '[' + ', '.join([f'{t:.3f}' for t in x]) + ']' \
            if isinstance(x, torch.Tensor) and x.ndim == 1 else f'{x:.3f}'

        #viz_fn = lambda x : x
        return f'GSCamera(FoVx={viz_fn(self.fov_x)} FoVy={viz_fn(self.fov_y)} world2cam={self.world_view_transform.shape if self.world_view_transform is not None else None} proj={self.full_proj_transform.shape if self.full_proj_transform is not None else None} device={self.world_view_transform.device if self.world_view_transform is not None else "None"} image_path={self.image_path} depth_path={self.depth_path} mask_path={self.mask_path}\n'

    def to(self, device: Union[str, torch.device]) -> 'GSCamera':
        """Move to device."""
        if self.image is not None:
            self.image = self.image.to(device)
        if self.world_view_transform is not None:
            self.world_view_transform = self.world_view_transform.to(device)
        if self.full_proj_transform is not None:
            self.full_proj_transform = self.full_proj_transform.to(device)
        if self.camera_center is not None:
            self.camera_center = self.camera_center.to(device)
        return self

    def as_dict(self) -> Dict[str, Any]:
        """Return a dictionary of attributes."""
        attrs = ['image_path', 'mask_path', 'depth_path', 'normal_path',
                 'image', 'mask', 'depth', 'normal',
                 'image_height', 'image_width',
                 'fov_x', 'fov_y', 'camera_center',
                 'world_view_transform', 'full_proj_transform']
        return {f'_camdata_{k}': getattr(self, k) for k in attrs \
                if getattr(self, k) is not None}

    #@property
    #def quat(self):
    #    cam2world = torch.linalg.inv(self.world_view_transform.T)
    #    return matrix_to_quaternion(cam2world[:3, :3])

    @staticmethod
    def from_dict(dic: Dict[str, Any]) -> Union['GSCamera', List['GSCamera']]:
        """Load the attributes from a dictionary. Support a batch of cameras."""
        idx = len('_camdata_')
        if not isinstance(dic['_camdata_fov_x'], float):
            n_cams = len(dic['_camdata_fov_x'])
            cams = []
            for i in range(n_cams):
                cam = GSCamera()
                for k, v in dic.items():
                    setattr(cam, k[idx:], v[i])
                cams.append(cam)
            return cams
        else:
            cam = GSCamera()
            for k, v in dic.items():
                if isinstance(v, list) and isinstance(v[0], list):
                    v = v[0]
                setattr(cam, k[idx:], v)
            return cam

    def load_image(self) -> Optional[torch.Tensor]:
        """Deprecated. Load the image only when requested."""
        if self.image is not None:
            return self.image
        if self.image_path is None:
            return None
        image = np.array(Image.open(self.image_path))
        image = torch.from_numpy(image) / 255.0
        self.image = image.permute(2, 0, 1)
        self.image_width = self.image.shape[2]
        self.image_height = self.image.shape[1]
        return self.image

    def load_depth(self) -> Optional[torch.Tensor]:
        """Load the depth only when requested."""
        if self.depth_path is None:
            return None
        if self.depth is not None:
            return self.depth
        depth = np.array(Image.open(self.depth_path))
        self.depth = torch.from_numpy(depth).float()
        return self.depth

    def load_mask(self) -> Optional[torch.Tensor]:
        """Load the mask only when requested."""
        if self.mask_path is None:
            return None
        if self.mask is not None:
            return self.mask
        mask = np.array(Image.open(self.mask_path))
        self.mask = torch.from_numpy(mask).float()
        return self.mask

    @staticmethod
    def from_compact(c: torch.Tensor, **kwargs) -> List['GSCamera']:
        """
        Args:
            c: [N, 25]. Extrinsics + Intrinsics.
        """
        cam2world_colmap = c[:, :16].reshape(-1, 4, 4)
        world2cam_colmap = torch.linalg.inv(cam2world_colmap)
        intrinsics = c[:, 16:].reshape(-1, 3, 3)
        fov_xs = 2 * torch.atan(1 / intrinsics[:, 0, 0])
        fov_ys = 2 * torch.atan(1 / intrinsics[:, 1, 1])
        #print(world2cam_colmap @ torch.Tensor([0, 0, 0, 1]).to(world2cam_colmap))
        return [GSCamera.from_matrix(w2c, float(fov_x), float(fov_y), **kwargs)
                for w2c, fov_x, fov_y in \
                zip(world2cam_colmap, fov_xs, fov_ys)]

    def to_compact(self) -> torch.Tensor:
        """
        Args:
            c: [N, 25]. Extrinsics + Intrinsics.
        """
        if self.world_view_transform is None:
            raise ValueError("world_view_transform is None")
        cam2world_colmap = torch.linalg.inv(self.world_view_transform.T)
        intrinsics = intrinsics_from_fov(self.fov_x, self.fov_y).to(cam2world_colmap)
        return torch.cat([cam2world_colmap.reshape(-1), intrinsics.view(-1)])

    @staticmethod
    def to_extrinsics_intrinsics(cameras: List['GSCamera'], device: Optional[Union[str, torch.device]] = None) -> Dict[str, torch.Tensor]:
        """
        Returns:
            {
                image: [N, 1, C, H, W], batch view channel H W
                extrinsics: [N, 1, 4, 4],
                intrinsics: [N, 1, 3, 3],
                near: [N, 1],
                far: [N, 1],
            }
        """
        dic = {}
        device = device if device is not None else \
            cameras[0].world_view_transform.device
        #ones = torch.ones((len(cameras), 1)).to(device)
        def intrinsics_fn(x: float) -> torch.Tensor:
            return torch.Tensor([
                [0.5 / math.tan(x/2), 0, 0.5],
                [0, 0.5 / math.tan(x / 2), 0.5],
                [0, 0, 1]])
        
        dic['image'] = None if cameras[0].image is None else \
            torch.stack([c.image for c in cameras]).to(device)
        E = torch.stack([
            torch.linalg.inv(c.world_view_transform.T)
            for c in cameras]).to(device)
        E[..., :3, :3] /= torch.det(E[..., :3, :3])[..., None, None] ** (1/3)
        dic['extrinsics'] = E
        dic['intrinsics'] = torch.stack([
            intrinsics_fn(c.fov_x) for c in cameras]).to(device)
        return dic

    @staticmethod
    def from_extrinsics_intrinsics(extrinsics: torch.Tensor, intrinsics: torch.Tensor, **kwargs) -> 'GSCamera':
        fov_x, fov_y = intrinsics_to_fov(intrinsics)
        world_view_transform = torch.linalg.inv(extrinsics)
        return GSCamera.from_matrix(world_view_transform, fov_x, fov_y, **kwargs)

    @staticmethod
    def from_look_at(cam_pos: torch.Tensor, fov_x: float, fov_y: float, look_at: Optional[torch.Tensor] = None, **kwargs) -> 'GSCamera':
        """Creates a Camera object from a look-at point and camera position."""
        look_at = torch.zeros(3).to(cam_pos) if look_at is None else look_at
        world2cam = extrinsics_from_lookat(cam_pos, look_at)[0]
        #cam2world = kaolin2colmap_cam2world(torch.linalg.inv(world2cam))
        #world2cam = torch.linalg.inv(cam2world)
        return GSCamera.from_matrix(world2cam, fov_x, fov_y, **kwargs)

    @staticmethod
    def from_info(cam_info: CameraInfo) -> 'GSCamera':
        """Creates a Camera object from a CameraInfo."""

        return GSCamera.from_matrix(
            world2view_from_rt(cam_info.R, cam_info.T),
            fov_x=cam_info.fov_x, fov_y=cam_info.fov_y,
            image_path=cam_info.image_path,
            depth_path=cam_info.depth_path,
            mask_path=cam_info.mask_path)

    def reset_full_proj_transform(self) -> None:
        """Reset the full projection transform."""
        world2cam = self.world_view_transform.T
        proj_matrix = perspective_matrix_colmap(0.01, 100, self.fov_x, self.fov_y)
        self.full_proj_transform = world2cam.T @ proj_matrix.to(world2cam).T

    @staticmethod
    def from_matrix(world2cam: torch.Tensor, fov_x: float, fov_y: float, **kwargs) -> 'GSCamera':
        """Creates a Camera object from a CameraInfo.
        Args:
            FoV: Field of view in radian.
        """

        proj_matrix = perspective_matrix_colmap(0.01, 100, fov_x, fov_y)
        full_proj_transform = world2cam.T @ proj_matrix.to(world2cam).T
        return GSCamera(
            world2cam.T.float(), full_proj_transform.float(),
            fov_x=fov_x, fov_y=fov_y, **kwargs)


def to_homogenized(points: torch.Tensor) -> torch.Tensor:
    """Convert points to homogenized coordinates."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def from_homogenized(points: torch.Tensor) -> torch.Tensor:
    """Convert points from homogenized coordinates."""
    return points[..., :-1] / points[..., -1:]


def world2cam_projection(w2c: torch.Tensor, intrinsics: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Convert world to camera space and project to screen space.
    Args:
        w2c: torch.Tensor, [B, 4, 4], world to camera matrix. (Already transposed!)
        intrinsics: torch.Tensor, [B, 3, 3], intrinsics matrix.
        points: torch.Tensor, [B, N, 3], world space points.
    Returns:
        torch.Tensor, [B, N, 2], screen space points.
    """
    cam_points = from_homogenized(to_homogenized(points) @ w2c)
    cam_points = cam_points @ intrinsics.permute(0, 2, 1)
    return from_homogenized(cam_points)


def unproject(
    coordinates: torch.Tensor, # (..., 3), :2 are x, y, 2 is z
    intrinsics: torch.Tensor, # (..., 3, 3)
    extrinsics: torch.Tensor, # (..., 4, 4)
) -> torch.Tensor:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    z = coordinates[..., 2:].clone() # (L,)
    coordinates[..., 2] = 1
    ray_directions = einsum(
        intrinsics.inverse(), coordinates,
        "... i j, ... j -> ... i") # (..., 3)
    coordinates[..., 2:] = z
    # Apply the supplied depth values.
    points = ray_directions * z

    # Apply extrinsics transformation
    return einsum(extrinsics[..., :3, :3], points, '... i j, ... j -> ... i') \
            + extrinsics[..., :3, 3]


def matrix_to_angle(R: torch.Tensor) -> NDArray[np.floating]:
    """Convert rotation matrix to angles."""
    # [yaw, pitch, roll]
    return Rotation.from_matrix(R.cpu().numpy()).as_euler('yxz')


def world2view_from_rt(R: Union[torch.Tensor, np.ndarray], t: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Get world to view matrix from rotation and translation.
    Args:
        R: torch.Tensor, [3, 3], rotation matrix.
        t: torch.Tensor, [3, ], translation vector.
    """
    if isinstance(R, np.ndarray):
        Rt = np.zeros((4, 4), dtype=np.float32)
        Rt[:3, :3] = R.T
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        return Rt
    else:
        Rt = torch.zeros((4, 4), dtype=torch.float32)
        Rt[:3, :3] = R.T
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        return Rt


def perspective_matrix_colmap(znear: float, zfar: float, fov_x: float, fov_y: float) -> torch.Tensor:
    """In colmap coordinate system, convert a frustum to NDC space in [-1, 1].
    """
    # Handle both scalar and tensor inputs
    if isinstance(fov_y, torch.Tensor):
        tanHalfFovY = torch.tan(fov_y / 2).item()
    else:
        tanHalfFovY = math.tan(fov_y / 2)

    if isinstance(fov_x, torch.Tensor):
        tanHalfFovX = torch.tan(fov_x / 2).item()
    else:
        tanHalfFovX = math.tan(fov_x / 2)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    # normalize world space x to NDC x, scale x by (znear / z)
    P[0, 0] = 2.0 * znear / (right - left)
    # shift x to the center
    P[0, 2] = (right + left) / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def full_perspective_matrix(K: torch.Tensor, znear: float = 0.01, zfar: float = 100) -> torch.Tensor:
    """In colmap coordinate system, convert a frustum to NDC space in [-1, 1].
    """

    fx = K[0, 0] * 2
    fy = K[1, 1] * 2
    x0 = K[0, 2] - 0.5
    y0 = K[1, 2] - 0.5

    tx = ty = 0 # the center of NDC space, 0
    lx = ly = 2 # the boundary length, -1 to 1 is 2
    z_sign = 1.0
    U = z_sign * zfar / (zfar - znear)
    V = -(zfar * znear) / (zfar - znear)

    return torch.Tensor([
        [2 * fx / lx, 0, -2 * x0 / lx + tx, 0],
        [0, 2 * fy / ly, -2 * y0 / ly + ty, 0],
        [0, 0, U, V],
        [0, 0, z_sign, 0]
    ])


def fov2focal(fov: float, pixels: int) -> float:
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal: float, pixels: int) -> float:
    return 2 * math.atan(pixels / (2 * focal))


def intrinsics_to_fov(intrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert intrinsics to FoV.
    Args:
        intrinsics: a tensor of shape [batch, 3, 3]
    Returns:
        FoVx, FoVy: a tensor of shape [batch], in radians.
    """
    fx = intrinsics[..., 0, 0]
    fy = intrinsics[..., 1, 1]
    cx = intrinsics[..., 0, 2]
    cy = intrinsics[..., 1, 2]
    return 2 * torch.atan2(cx, fx), 2 * torch.atan2(cy, fy)


def intrinsics_from_fov(
        fov_x: float,
        fov_y: float,
        H: float = 1.0, 
        W: float = 1.0) -> torch.Tensor:
    """Get the camera intrinsics matrix from FoV.
    Notice that this transforms points into screen space [0, size]^2 rather than NDC.
    .----> x
    |
    |
    v y
    Args:
        fov_x: Field of View in x-axis (radians).
        fov_y: Field of View in y-axis (degrees).
        znear: near plane.
        zfar: far plane.
    """
    #S = min(H, W)
    fx = 0.5 * H / math.tan(fov_y / 2)
    fy = 0.5 * W / math.tan(fov_x / 2)

    return torch.Tensor([
        [fx, 0, 0.5 * H],
        [0, fy, 0.5 * W],
        [0, 0, 1]])


def kaolin2colmap_cam2world(kaolin_cam2world: torch.Tensor) -> torch.Tensor:
    """Convert a Kaolin camera to world matrix to a Colmap matrix.
    COLMAP (OpenCV)'s cooredinate system uses -y and -z than Kaolin.
    """
    sign = torch.Tensor([1, -1, -1, 1]).expand_as(kaolin_cam2world)
    return kaolin_cam2world * sign.to(kaolin_cam2world)


def extrinsics_from_lookat(
    cam_pos: torch.Tensor,  # [3] or [B, 3]
    look_at: torch.Tensor,  # [3] or [B, 3]
    up: Optional[torch.Tensor] = None) -> torch.Tensor:  # [3] or [B, 3]
    """Create rotation and translation matrices from camera position and look-at point.
    
    Args:
    cam_pos: Camera position in world coordinates
    look_at: Point to look at in world coordinates 
    up: Up vector, defaults to [0,1,0]
    
    Returns:
    R: Rotation matrix [3,3] or [B,3,3]
    T: Translation vector [3] or [B,3]
    """

    up = torch.tensor([[0.0, 1.0, 0.0]], device=cam_pos.device) if up is None else up
    
    # Handle both batched and unbatched inputs
    if cam_pos.dim() == 1:
        cam_pos = cam_pos.unsqueeze(0)
        look_at = look_at.unsqueeze(0)
    if up.dim() == 1:
        up = up.unsqueeze(0)

    # Create camera coordinate system
    forward = look_at - cam_pos  # Forward direction
    forward = forward / torch.norm(forward.float(), dim=-1, keepdim=True).to(forward.dtype)
    
    # right = torch.cross(up, forward)
    right = torch.cross(up, forward, dim=-1)  # Right direction
    right = right / torch.norm(right.float(), dim=-1, keepdim=True).to(right.dtype)
    
    # up = torch.cross(forward, right)
    up = torch.cross(forward, right, dim=-1)  # Recompute up for orthogonality
    
    # Stack vectors to create rotation matrix
    # point transform: x' = Rx; coordinate transform: x' = R^T x
    # we are doing coordinate transform, so the rotation matrix is transposed
    R = torch.stack([right, up, forward], dim=-1).permute(0, 2, 1)
    
    # Translation is negative of rotated camera position
    T = -torch.einsum('...ij,...j->...i', R, cam_pos)
    
    E = torch.eye(4, device=cam_pos.device)[None].repeat(cam_pos.shape[0], 1, 1)
    E[..., :3, :3] = R
    E[..., :3, 3] = T
    return E


def unproject_depth(depth: torch.Tensor, cam2world_matrix: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Args:
        depth: [N, H, W], the depth map
        cam2world_matrix: [N, 4, 4], the camera-to-world matrix
        intrinsics: [N, 3, 3], the intrinsics matrix. The intrinsics should be in the unit of pixel
    Returns:
        world_points: [N, H, W, 3], the world points
    """
    H, W = depth.shape[-2:]
    N, M = depth.shape[0], H * W
    # ij indexing: first column expansion, second row expansion
    uv = torch.stack(torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=cam2world_matrix.device), 
        torch.arange(W, dtype=torch.float32, device=cam2world_matrix.device), 
        indexing='ij')) + 0.5 # pixel center
    uv = uv.view(2, -1)[None].repeat(N, 1, 1) # shape: (N, 2, M)

    K_inv = torch.linalg.inv(intrinsics)
    uv_ = torch.concat([uv, torch.ones(N, 1, M).to(intrinsics.device)], dim=1)
    xyz = K_inv @ uv_ # [N, 3, 3] x [N, 3, M] = [N, 3, M]

    #xyz = torch.linalg.solve(intrinsics, torch.concat((uv, 
    #        torch.ones(N, M, 1).to(intrinsics.device)), dim=-1).permute(0, 2, 1)) 

    cam_rel_points = torch.concat((xyz * depth.view(N, 1, M), torch.ones(N, 1, M).to(intrinsics.device)), dim=1) # shape: (N, 4, M)

    world_points = torch.bmm(cam2world_matrix, cam_rel_points).permute(0, 2, 1)[:, :, :3]

    return world_points


def angle2matrix(angles: torch.Tensor) -> torch.Tensor:
    """Convert angles to rotation matrices.
    Args:
        angles: (N, 3), [yaw, pitch, roll]
    Returns:
        R: (N, 3, 3), rotation matrices
    """
    return torch.from_numpy(Rotation.from_euler('yxz', angles.cpu().numpy()).as_matrix()).float().to(angles.device)


def make_colmap_camera(angles: torch.Tensor, radius: float, fov: float) -> List[GSCamera]:
    """
    Args:
        angles: (N, 3), [yaw, pitch, roll]
    """
    R = angle2matrix(angles)
    # build world2cam matrix
    world2cams = torch.eye(4)[None].repeat(angles.shape[0], 1, 1)
    world2cams[:, :3, :3] = R
    world2cams[:, 2, 3] = -radius
    cam2worlds = torch.linalg.inv(world2cams)
    cam2worlds[..., 1] *= -1
    cam2worlds[..., 2] *= -1
    world2cams = torch.linalg.inv(cam2worlds)
    return [GSCamera.from_matrix(w2c, fov, fov) for w2c in world2cams]


def pos_from_angle(
        azim: torch.Tensor,
        elev: torch.Tensor,
        radius: Union[torch.Tensor, float]) -> torch.Tensor:
    """Create point from angles and radius.
        Kaolin coordinate system. (X -> Right, Y -> Up, Z -> Back).
        azim=0 elev=0 -> Z.
    Args:
        azim: (N,) or scalar, azimuthal angle in radians
        elev: (N,) or scalar, polar angle in radians
        radius: (N,) or scalar, radius in meters
    Returns:
        (N, 3), camera position in meters
    """

    cos_elev = torch.cos(elev)
    x = cos_elev * torch.sin(azim)
    z = cos_elev * torch.cos(azim)
    y = torch.sin(elev)

    if isinstance(radius, torch.Tensor):
        return radius.unsqueeze(-1) * torch.stack([x, y, z], -1)
    else:
        return radius * torch.stack([x, y, z], -1)


def sample_delta_angle(
        azim_std=0.,
        elev_std=0.,
        roll_std=0.,
        n_sample=1,
        device='cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a delta angle from a Gaussian or uniform distribution.
    Args:
        azim: azimuthal angle (rotation around y axis) in radians
        elev: polar angle (angle from the y axis) in radians
        azim_std: standard deviation of azimuthal angle in radians
        elev_std: standard deviation of polar angle in radians
        n_sample: number of samples to return
        device: device to put output on
        noise: 'gaussian' or 'uniform'
    """
    dh = torch.rand((n_sample,), device=device) * azim_std if azim_std > 0 \
        else torch.zeros((n_sample,), device=device)
    dv = torch.randn((n_sample,), device=device) * elev_std if elev_std > 0 \
        else torch.zeros((n_sample,), device=device)
    dr = torch.randn((n_sample,), device=device) * roll_std if roll_std > 0 \
        else torch.zeros((n_sample,), device=device)
    return dh, dv, dr


def get_lookat_camera(
        look_at: Optional[torch.Tensor] = None,
        azims: Optional[torch.Tensor] = None,
        elevs: Optional[torch.Tensor] = None,
        fov=0.2443,
        radius=2.7,
        H=512, W=512,
        device='cpu') -> List[GSCamera]:
    """
    Sample a camera pose looking at a point with a Gaussian or uniform distribution.
    Args:
        azim: azimuthal angle (rotation around y axis) in radians
        elev: polar angle (angle from the y axis) in radians
        look_at: 3-vector, point to look at
        azim_std: standard deviation of azimuthal angle in radians
        elev_std: standard deviation of polar angle in radians
        fov: field of view in radians
        radius: distance from camera to look_at point
        n_sample: number of samples to return
        resolution: image resolution of the camera
        device: device to put output on
        noise: 'gaussian' or 'uniform'
    """

    if azims is None or elevs is None:
        raise ValueError("azims and elevs must be provided")
    
    if isinstance(azims, np.ndarray):
        azims = torch.from_numpy(azims).float()
        elevs = torch.from_numpy(elevs).float()
    look_at = torch.zeros(3).float() if look_at is None else look_at
    azims = azims.to(device)
    elevs = elevs.to(device)
    look_at = look_at.to(device)
    
    common_kwargs = dict(
        look_at=look_at,
        fov_x=fov, fov_y=fov,
        image_width=W, image_height=H)
    cam_positions = pos_from_angle(azims, elevs, radius).to(device)
    
    cams = [GSCamera.from_look_at(cam_pos=x, **common_kwargs)
            for x in cam_positions]
    return cams
