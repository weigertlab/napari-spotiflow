from typing import Literal

import numpy as np
from napari.utils import progress as napari_progress

# fmt: off
SUPPORTED_AXES_LAYOUTS = Literal[
      "YX",   "YXC",   "CYX",
     "TYX",  "TYXC",  "TCYX",
     "ZYX",  "ZYXC",  "ZCYX",  "CZYX",
    "TZYX", "TZYXC", "TZCYX", "TCZYX",
]  # No "ZT*", etc.
# fmt: on

_point_layer2d_default_kwargs = dict(
    size=8,
    symbol="ring",
    opacity=1,
    face_color=[1.0, 0.5, 0.2],
    border_color=[1.0, 0.5, 0.2],
)

_point_layer3d_default_kwargs = dict(
    size=8,
    symbol="ring",
    opacity=1,
    face_color=[1.0, 0.5, 0.2],
    border_color=[1.0, 0.5, 0.2],
    out_of_slice_display=True,
)


def _validate_axes(img: np.ndarray, axes: SUPPORTED_AXES_LAYOUTS) -> None:
    """Validate that the number of dimensions in the image matches the given axes string."""
    if img.ndim != len(axes):
        raise ValueError(
            f"Image has {img.ndim} dimensions, but axes has {len(axes)} dimensions"
        )


def _prepare_input(img: np.ndarray, axes: SUPPORTED_AXES_LAYOUTS) -> np.ndarray:
    """Reshape input for Spotiflow's API compatibility based on axes notation.

    Args:
        img (np.ndarray): Input image array.
        axes (str): Axes representation of the image.

    Returns:
        np.ndarray: Image reshaped into Spotiflow-compatible format.
    """
    _validate_axes(img, axes)

    if axes in {"YX", "ZYX", "TYX", "TZYX"}:
        return img[..., None]
    elif axes in {"YXC", "ZYXC", "TYXC", "TZYXC"}:
        return img
    elif axes == "CYX":
        return img.transpose(1, 2, 0)
    elif axes == "CZYX":
        return img.transpose(1, 2, 3, 0)
    elif axes == "ZCYX":
        return img.transpose(0, 2, 3, 1)
    elif axes == "TCYX":
        return img.transpose(0, 2, 3, 1)
    elif axes == "TZCYX":
        return img.transpose(0, 1, 3, 4, 2)
    elif axes == "TCZYX":
        return img.transpose(0, 2, 3, 4, 1)
    else:
        raise ValueError(f"Invalid axes: {axes}")


def _patched_progbar(desc):
    def _progbar(*args, **kwargs):
        return napari_progress(*args, desc=desc, **kwargs)

    return _progbar
