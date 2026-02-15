"""Gaussian splatting rendering wrapper.

This module automatically selects between 3DGS and 2DGS implementations based on configuration.
Set GAUSSIAN_TYPE environment variable or use set_gaussian_type() to choose:
- "3d" or "3dgs" for 3D Gaussian Splatting
- "2d" or "2dgs" for 2D Gaussian Splatting (default)
"""
import os
import importlib
import sys

# Global variable to track which implementation to use
_GAUSSIAN_TYPE = os.environ.get("GAUSSIAN_TYPE", "3d").lower()
_MODULE = None

def set_gaussian_type(gaussian_type: str):
    """Set the Gaussian type to use.
    
    Args:
        gaussian_type: "3d", "3dgs", "2d", or "2dgs"
    """
    global _GAUSSIAN_TYPE, _MODULE
    gaussian_type = gaussian_type.lower()
    if gaussian_type in ["3d", "3dgs"]:
        print("GAUSSIAN_TYPE: 3DGS selected")
        _GAUSSIAN_TYPE = "3d"
        _MODULE = importlib.import_module("gaussian_splatting_3d")
    elif gaussian_type in ["2d", "2dgs"]:
        print("GAUSSIAN_TYPE: 2DGS selected")
        _GAUSSIAN_TYPE = "2d"
        _MODULE = importlib.import_module("gaussian_splatting_2d")
    else:
        raise ValueError(f"Invalid gaussian_type: {gaussian_type}. Must be '3d', '3dgs', '2d', or '2dgs'")
    
    _reload_exports()

def get_gaussian_type() -> str:
    """Get the current Gaussian type."""
    return _GAUSSIAN_TYPE

def _get_module():
    """Get the current implementation module."""
    global _MODULE
    if _MODULE is None:
        # Initialize based on current _GAUSSIAN_TYPE
        if _GAUSSIAN_TYPE in ["3d", "3dgs"]:
            _MODULE = importlib.import_module("gaussian_splatting_3d")
        else:
            _MODULE = importlib.import_module("gaussian_splatting_2d")
    return _MODULE

def _reload_exports():
    """Reload exports from the current module."""
    global GaussianParameters, splat_gaussians, RGB2SH, SH2RGB, eval_sh, sh2rgb, calc_vd_color
    mod_name_3d = "gaussian_splatting_3d"
    mod_name_2d = "gaussian_splatting_2d"
    if mod_name_3d in sys.modules:
        importlib.reload(sys.modules[mod_name_3d])
    if mod_name_2d in sys.modules:
        importlib.reload(sys.modules[mod_name_2d])
    
    mod = _get_module()
    GaussianParameters = mod.GaussianParameters
    splat_gaussians = mod.splat_gaussians
    RGB2SH = mod.RGB2SH
    SH2RGB = mod.SH2RGB
    eval_sh = mod.eval_sh
    sh2rgb = mod.sh2rgb
    calc_vd_color = mod.calc_vd_color
    
    current_module = sys.modules[__name__]
    current_module.GaussianParameters = GaussianParameters
    current_module.splat_gaussians = splat_gaussians
    current_module.RGB2SH = RGB2SH
    current_module.SH2RGB = SH2RGB
    current_module.eval_sh = eval_sh
    current_module.sh2rgb = sh2rgb
    current_module.calc_vd_color = calc_vd_color

# Initialize module based on environment variable or default
_get_module()
_reload_exports()

__all__ = [
    "GaussianParameters",
    "splat_gaussians",
    "RGB2SH",
    "SH2RGB",
    "eval_sh",
    "sh2rgb",
    "calc_vd_color",
    "set_gaussian_type",
    "get_gaussian_type",
]
