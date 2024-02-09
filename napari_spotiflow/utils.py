import numpy as np
from typing import Literal

def _validate_axes(img: np.ndarray, axes: Literal["YX", "YXC", "CYX", "TYX", "TYXC", "TCYX"]):
    assert img.ndim == len(axes), f"Image has {img.ndim} dimensions, but axes has {len(axes)} dimensions"
    return

def _prepare_input(img: np.ndarray, axes: Literal["YX", "YXC", "CYX", "TYX", "TYXC", "TCYX"]):
    _validate_axes(img, axes)
    if axes == "YX":
        return img[..., None]
    elif axes == "YXC":
        return img
    elif axes == "CYX":
        return img.transpose(1, 2, 0)
    elif axes == "TYX":
        return img[..., None]
    elif axes == "TYXC":
        return img
    elif axes == "TCYX":
        return img.transpose(0, 2, 3, 1)
    else:
        raise ValueError(f"Invalid axes: {axes}")
