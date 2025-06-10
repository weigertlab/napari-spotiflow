import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari


import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    FloatSlider,
    LiteralEvalLineEdit,
    PushButton,
    RadioButtons,
    SpinBox,
    create_widget,
)
from napari.utils import progress as napari_progress
from spotiflow.model import Spotiflow
from spotiflow.model.pretrained import _REGISTERED, list_registered
from spotiflow.utils import normalize

from .utils import (
    _patched_progbar,
    _point_layer2d_default_kwargs,
    _point_layer3d_default_kwargs,
    _prepare_input,
    _validate_axes,
)

LOGO = Path(__file__).parent / "resources" / "spotiflow_transp_small.png"

BASE_IMAGE_AXES_CHOICES = ["YX", "YXC", "CYX", "TYX", "TYXC", "TCYX"]
BASE_IMAGE_AXES_CHOICES_3D = ["CZYX", "TCZYX"]+[
    f"Z{axes}" if "T" not in axes else f"TZ{axes[1:]}"
    for axes in BASE_IMAGE_AXES_CHOICES
]

MODELS_REG_2D = sorted([r for r in list_registered() if not _REGISTERED[r].is_3d])
MODELS_REG_3D = sorted([r for r in list_registered() if _REGISTERED[r].is_3d])

# Default chosen models
if "general" in MODELS_REG_2D:
    MODELS_REG_2D.remove("general")
    MODELS_REG_2D.insert(0, "general")
if "synth_3d" in MODELS_REG_3D:
    MODELS_REG_3D.remove("synth_3d")
    MODELS_REG_3D.insert(0, "synth_3d")


class SpotiflowDetectionWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()

        self._curr_image_axes_choices = ("",)

        self._viewer = viewer
        self._label_head = create_widget(
            widget_type="Label",
            label=f'<h1><img src="{LOGO}"></h1>',
        )
        self._image = create_widget(
            label="Input",
            annotation="napari.layers.Image",
        )
        self._image_axes = RadioButtons(
            label="Axes order",
            orientation="horizontal",
            choices=lambda _: tuple(self._curr_image_axes_choices),
            value=self._curr_image_axes_choices[0],
        )
        self._label_nn = create_widget(
            widget_type="Label",
            label="<br><b>Model:</b>",
        )
        self._mode = RadioButtons(
            label="Mode",
            orientation="horizontal",
            choices=["2D", "3D"],
            value="2D",
        )
        self._model_type = RadioButtons(
            label="Model type",
            orientation="horizontal",
            choices=["Pre-trained", "Custom"],
            value="Pre-trained",
        )
        self._model_2d = ComboBox(
            visible=True,
            label="Pre-trained model (2D)",
            choices=MODELS_REG_2D,
            value=MODELS_REG_2D[0],
        )
        self._model_3d = ComboBox(
            visible=False,
            label="Pre-trained model (3D)",
            choices=MODELS_REG_3D,
            value=MODELS_REG_3D[0],
        )
        self._model_folder = FileEdit(
            visible=False,
            label="Custom model",
            mode="d",
        )
        self._norm_image = CheckBox(
            label="Normalize image",
            value=True,
        )
        self._perc_low = FloatSlider(
            label="Normalization percentile low",
            value=1.0,
            min=0.0,
            max=100.0,
            step=0.1,
            visible=True,
        )
        self._perc_high = FloatSlider(
            label="Normalization percentile high",
            value=99.8,
            min=0.0,
            max=100.0,
            step=0.1,
            visible=True,
        )
        self._subpix = CheckBox(
            label="Subpixel prediction",
            value=True,
        )
        self._label_postproc = create_widget(
            widget_type="Label",
            label="<br><b>Postprocessing:</b>",
        )
        self._opt_thresh = CheckBox(
            label="Use optimized probability threshold",
            value=True,
        )
        self._prob_thresh = FloatSlider(
            label="Probability threshold",
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            visible=False,
        )
        self._min_distance = SpinBox(
            label="Minimum distance",
            min=1,
            max=51,
            step=1,
            value=2,
        )
        self._label_adv = create_widget(
            widget_type="Label",
            label="<br><b>Advanced options:</b>",
        )
        self._auto_n_tiles = CheckBox(
            label="Automatically infer tiling",
            value=True,
        )
        self._n_tiles = LiteralEvalLineEdit(
            label="Number of tiles",
            value="1,1",
            visible=False,
        )
        self._cnn_output = CheckBox(
            label="Show CNN output",
            value=False,
        )
        self._detect_button = PushButton(label="Detect spots")

        # Prettify labels (https://doc.qt.io/qt-5/qsizepolicy.html#Policy-enum)
        for w in (
            self._label_head,
            self._label_nn,
            self._label_postproc,
            self._label_adv,
        ):
            w.native.setSizePolicy(1 | 2, 0)

        self._opt_thresh.changed.connect(self._on_opt_thresh_changed)
        self._auto_n_tiles.changed.connect(self._on_auto_ntiles_changed)
        self._norm_image.changed.connect(self._on_norm_image_changed)
        self._model_type.changed.connect(self._on_model_type_changed)
        self._mode.changed.connect(self._on_mode_changed)
        self._image.changed.connect(self._on_image_changed)
        self._image_axes.changed.connect(self._on_image_axes_update)
        self._detect_button.changed.connect(self._safe_detect_wrapper)

        self.extend(
            [
                self._label_head,
                self._image,
                self._image_axes,
                self._label_nn,
                self._mode,
                self._model_type,
                self._model_2d,
                self._model_3d,
                self._model_folder,
                self._norm_image,
                self._perc_low,
                self._perc_high,
                self._subpix,
                self._label_postproc,
                self._opt_thresh,
                self._prob_thresh,
                self._min_distance,
                self._label_adv,
                self._auto_n_tiles,
                self._n_tiles,
                self._cnn_output,
                self._detect_button,
            ]
        )

        self._model = None
        self._last_model_combination = ("",)

    def _on_opt_thresh_changed(self, event: bool):
        if not event:
            self._prob_thresh.show()
        else:
            self._prob_thresh.hide()

    def _on_auto_ntiles_changed(self, event: bool):
        if not event:
            self._n_tiles.show()
        else:
            self._n_tiles.hide()

    def _on_norm_image_changed(self, event: bool):
        if event:
            self._perc_low.show()
            self._perc_high.show()
        else:
            self._perc_low.hide()
            self._perc_high.hide()

    def _on_model_type_changed(self, event: str):
        if event == "Pre-trained":
            if self._mode.value == "2D":
                self._model_2d.show()
                self._model_3d.hide()
                self._model_folder.hide()
            elif self._mode.value == "3D":
                self._model_2d.hide()
                self._model_3d.show()
                self._model_folder.hide()
            else:
                raise ValueError(f"Invalid mode: {self._mode.value}")
        elif event == "Custom":
            self._model_2d.hide()
            self._model_3d.hide()
            self._model_folder.show()

    def _on_mode_changed(self, event: str):
        if event == "2D" and self._model_type.value == "Pre-trained":
            self._model_2d.show()
            self._model_3d.hide()
        elif event == "3D" and self._model_type.value == "Pre-trained":
            self._model_2d.hide()
            self._model_3d.show()
        self._axes_choice_update()

    def _on_image_changed(self, event: "napari.layers.Image"):
        if event is not None:
            self._axes_choice_update()
        else:
            self._detect_button.enabled = False

    def _on_image_axes_update(self, event: str):
        if event == "":
            self._detect_button.enabled = False
        else:
            self._detect_button.enabled = True

    def _axes_choice_update(self):
        if self._image.value is None:
            self._curr_image_axes_choices = ("",)
        else:
            if self._mode.value == "2D":
                _relevant_axis_choices = BASE_IMAGE_AXES_CHOICES
            elif self._mode.value == "3D":
                _relevant_axis_choices = BASE_IMAGE_AXES_CHOICES_3D
            else:
                raise ValueError(f"Invalid mode: {self._mode.value}")
            _relevant_axis_choices = [
                axes
                for axes in _relevant_axis_choices
                if len(axes) == self._image.value.data.ndim
            ]
            self._curr_image_axes_choices = (
                deepcopy(tuple(_relevant_axis_choices))
                if len(_relevant_axis_choices) > 0
                else ("",)
            )
        self._image_axes.reset_choices()
        self._image_axes.value = self._curr_image_axes_choices[0]

    def _toggle_activity_dock(self, show=True):
        """
        (un)toggle activity dock to display progress bar
        """
        with warnings.catch_warnings():
            # FIXME: this is a temporary patch to avoid deprec warnings from napari
            # from popping up to the user.
            # Should switch to the public method when available
            # (https://github.com/napari/napari/issues/4598)
            warnings.simplefilter("ignore")
            self._viewer.window._status_bar._toggle_activity_dock(show)

    def _load_model(self, event=None):
        model_comb = (
            self._model_type.value,
            self._mode.value,
            self._model_2d.value,
            self._model_3d.value,
            self._model_folder.value,
        )
        if (
            self._model is not None
            and len(self._last_model_combination) == 5
            and all(
                (
                    curr == prev
                    for curr, prev in zip(model_comb, self._last_model_combination)
                )
            )
        ):
            # Do not reload is demanded model is already loaded
            # This assumes that the model is not changed outside of this widget
            # TODO: keep this or reload anyway?
            return
        if self._model_type.value == "Pre-trained":
            if self._mode.value == "2D":
                self._model = Spotiflow.from_pretrained(self._model_2d.value)
            elif self._mode.value == "3D":
                self._model = Spotiflow.from_pretrained(self._model_3d.value)
            else:
                raise ValueError(f"Invalid mode: {self._mode.value}")
        elif self._model_type.value == "Custom":
            self._model = Spotiflow.from_folder(self._model_folder.value)
        else:
            raise ValueError(f"Invalid model type: {self._model_type.value}")
        self._last_model_combination = (
            deepcopy(self._model_type.value),
            deepcopy(self._mode.value),
            deepcopy(self._model_2d.value),
            deepcopy(self._model_3d.value),
            deepcopy(self._model_folder.value),
        )

    def _get_image_data(self):
        if self._image.value is None:
            raise ValueError("No image layer to process")
        arr = (
            self._image.value.data[0]
            if self._image.value.multiscale
            else self._image.value.data
        )
        return np.asarray(arr)

    def _detect(self):
        self._load_model()
        if self._image.value is None:
            raise ValueError("No image layer to process")
        if self._mode == "3D" and not self._model.config.is_3d:
            raise ValueError("3D mode specified, but the loaded model is 2D")
        if self._mode == "2D" and self._model.config.is_3d:
            raise ValueError("2D mode specified, but the loaded model is 3D")

        img = self._get_image_data()
        if self._norm_image.value:
            img = normalize(img, self._perc_low.value, self._perc_high.value)
        if self._subpix.value and not self._model.config.compute_flow:
            warnings.warn(
                "Subpixel prediction is requested, but the model does not support it."
            )
            subpix = False
        else:
            subpix = self._subpix.value
        _validate_axes(img, self._image_axes.value)
        img = _prepare_input(img, self._image_axes.value)

        spotiflow_predict_kwargs = {
            "prob_thresh": self._prob_thresh.value
            if not self._opt_thresh.value
            else None,
            "min_distance": self._min_distance.value,
            "n_tiles": self._n_tiles.value if not self._auto_n_tiles.value else None,
            "verbose": True,
            "subpix": subpix,
            "normalizer": None,
        }

        if "T" not in self._image_axes.value:
            spots, details = self._model.predict(
                img,
                progress_bar_wrapper=_patched_progbar(desc="Tiled detection"),
                **spotiflow_predict_kwargs,
            )
            if self._cnn_output.value:
                details_prob_heatmap = details.heatmap
                if subpix:
                    details_flow = details.flow
        else:
            spots_t, details_t = tuple(
                zip(
                    *tuple(
                        self._model.predict(
                            _x,
                            progress_bar_wrapper=_patched_progbar(
                                desc="Single-frame tiled detection"
                            ),
                            **spotiflow_predict_kwargs,
                        )
                        for _x in napari_progress(
                            img, desc="Time-lapse detection", total=img.shape[0]
                        )
                    )
                )
            )
            spots = np.concatenate(
                [np.column_stack((np.full((spots.shape[0], 1), i), spots)) for i, spots in enumerate(spots_t)],
                axis=0
            )
            if self._cnn_output.value:
                details_prob_heatmap = np.stack(
                    [det.heatmap for det in details_t], axis=0
                )
                if subpix:
                    details_flow = np.stack([det.flow for det in details_t], axis=0)

        if self._cnn_output.value:
            if subpix:
                self._viewer.add_image(
                    0.5 * (1 + details_flow),
                    name=f"Stereographic flow ({self._image.value.name})",
                    scale=self._model.config.grid,
                )
            self._viewer.add_image(
                details_prob_heatmap,
                name=f"Probability heatmap ({self._image.value.name})",
                colormap="magma",
                scale=self._model.config.grid,
            )

        pts_layer_kwargs = (
            _point_layer2d_default_kwargs
            if self._mode.value == "2D"
            else _point_layer3d_default_kwargs
        )
        self._viewer.add_points(
            spots,
            name=f"Spots ({self._image.value.name})",
            **pts_layer_kwargs,
        )
        return

    def _safe_detect_wrapper(self, event=None):
        self._detect_button.enabled = False
        self._toggle_activity_dock(True)
        try:
            self._detect()
        except Exception as e:
            raise e
        finally:
            self._toggle_activity_dock(False)
            self._detect_button.enabled = True
