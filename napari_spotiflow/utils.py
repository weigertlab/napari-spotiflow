import numpy as np
from typing import Literal

def _validate_axes(img: np.ndarray, axes: Literal["YX", "YXC", "CYX", "TYX", "TYXC", "TCYX", "ZYX", "ZYXC", "CZYX", "ZTYX", "ZTYXC", "ZTCYX"]) -> None:
    assert img.ndim == len(axes), f"Image has {img.ndim} dimensions, but axes has {len(axes)} dimensions"
    return

def _prepare_input(img: np.ndarray, axes: Literal["YX", "YXC", "CYX", "TYX", "TYXC", "TCYX", "ZYX", "ZYXC", "CZYX", "ZTYX", "TZYXC", "TCZYX"]) -> np.ndarray:
    """Reshape input for Spotiflow's API compatibility. If `axes` contains "Z", then assumes `img` is a volumetric (3D) image.

    Args:
        img (np.ndarray): input image to be reformatted
        axes (Literal["YX", "YXC", "CYX", "TYX", "TYXC", "TCYX", "ZYX", "ZYXC", "ZCYX", "ZTYX", "ZTYXC", "ZTCYX"]): given axes

    Raises:
        ValueError: thrown if axis is not valid

    Returns:
        np.ndarray: reshaped NumPy array compatible with Spotiflow's `predict` API
    """
    _validate_axes(img, axes)
    if axes == "YX" or axes == "ZYX":
        return img[..., None]
    elif axes == "YXC" or axes == "ZYXC":
        return img
    elif axes == "CYX":
        return img.transpose(1,2,0)
    elif axes == "CZYX":
        return img.transpose(1,2,3,0)
    elif axes == "TYX" or axes == "TZYX":
        return img[..., None]
    elif axes == "TYXC" or axes == "TZYXC":
        return img
    elif axes == "TCYX":
        return img.transpose(0,2,3,1)
    elif axes == "TCZYX":
        return img.transpose(0,2,3,4,0)
    else:
        raise ValueError(f"Invalid axes: {axes}")
