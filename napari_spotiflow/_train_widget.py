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
from napari.utils.notifications import show_error, show_info
from spotiflow.cli.train import get_data
from spotiflow.model import Spotiflow, SpotiflowModelConfig
from spotiflow.model.pretrained import _REGISTERED, list_registered

from datetime import datetime

LOGO = Path(__file__).parent / "resources" / "spotiflow_transp_small.png"

MODELS_REG_2D = sorted([r for r in list_registered() if not _REGISTERED[r].is_3d])
MODELS_REG_3D = sorted([r for r in list_registered() if _REGISTERED[r].is_3d])

# Default pre-trained models
if "general" in MODELS_REG_2D:
    MODELS_REG_2D.remove("general")
    MODELS_REG_2D.insert(0, "general")
if "synth_3d" in MODELS_REG_3D:
    MODELS_REG_3D.remove("synth_3d")
    MODELS_REG_3D.insert(0, "synth_3d")


class SpotiflowTrainingWidget(Container):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer
        self._label_head = create_widget(
            widget_type="Label",
            label=f'<h1><img src="{LOGO}"></h1>',
        )
        self._training_mode = RadioButtons(
            label="Training mode",
            orientation="horizontal",
            choices=["Fine-tune", "Train"],
            value="Fine-tune",
        )
        self._finetune_from = RadioButtons(
            label="Fine-tune from",
            orientation="horizontal",
            choices=["Pre-trained", "Custom"],
            value="Pre-trained",
        )
        self._finetune_pretrained_2d = ComboBox(
            visible=True,
            label="Pre-trained model",
            choices=MODELS_REG_2D,
            value=MODELS_REG_2D[0],
        )
        self._finetune_pretrained_3d = ComboBox(
            visible=False,
            label="Pre-trained model",
            choices=MODELS_REG_3D,
            value=MODELS_REG_3D[0],
        )
        self._finetune_custom = FileEdit(
            visible=False,
            label="Custom model",
            mode="d",
        )
        self._label_data = create_widget(
            widget_type="Label",
            label="<br><b>Data:</b>",
        )
        self._data_dir = FileEdit(
            label="Data source",
            mode="d",
        )
        self._label_model_config = create_widget(
            widget_type="Label",
            label="<br><b>Model configuration:</b>",
        )
        self._model_mode = RadioButtons(
            label="Mode",
            orientation="horizontal",
            choices=["2D", "3D"],
            value="2D",
        )
        self._in_channels = SpinBox(
            label="Input channels",
            value=1,
            min=1,
            max=7,
        )
        self._sigma = FloatSlider(
            label="Sigma",
            value=1.0,
            min=1.0,
            max=15.0,
        )
        self._grid = SpinBox(
            label="Grid",
            value=2,
            min=1,
            max=2,
            visible=False,
        )
        self._label_train_config = create_widget(
            widget_type="Label",
            label="<br><b>Training configuration:</b>",
        )
        self._crop_size = LiteralEvalLineEdit(
            label="Crop size (YX)",
            value="512",
        )
        self._crop_size_z = LiteralEvalLineEdit(
            label="Crop size (Z)",
            value="16",
            visible=False,
        )
        self._smart_crop = CheckBox(
            label="Prioritize crops with spots",
            value=False,
        )
        self._num_epochs = SpinBox(
            label="Number of epochs",
            value=30,
            min=5,
            max=1000,
        )
        self._batch_size = SpinBox(
            label="Batch size",
            value=4,
            min=1,
            max=32,
        )
        self._label_output = create_widget(
            widget_type="Label",
            label="<br><b>Output:</b>",
        )
        self._save_dir = FileEdit(
            label="Save directory",
            mode="d",
            value=Path(os.getcwd()).resolve() / "spotiflow_models",
        )
        self._predict_test = CheckBox(
            label="Display predictions on test set",
            value=False,
            visible=False, # TODO: implement
        )
        
        self._train_button = PushButton(label="Launch training")

        # Prettify labels (https://doc.qt.io/qt-5/qsizepolicy.html#Policy-enum)
        for w in (
            self._label_head,
            self._label_data,
            self._label_model_config,
            self._label_train_config,
            self._label_output,
        ):
            w.native.setSizePolicy(1 | 2, 0)


        self._finetune_from.changed.connect(self._on_finetune_from_changed)
        self._model_mode.changed.connect(self._on_model_mode_changed)
        self._training_mode.changed.connect(self._on_training_mode_changed)
        self._train_button.changed.connect(self._safe_train_wrapper)

        self.extend(
            [
                self._label_head,
                self._training_mode,
                self._finetune_from,
                self._finetune_pretrained_2d,
                self._finetune_pretrained_3d,
                self._finetune_custom,
                self._label_data,
                self._data_dir,
                self._label_model_config,
                self._model_mode,
                self._in_channels,
                self._sigma,
                self._grid,
                self._label_train_config,
                self._crop_size,
                self._crop_size_z,
                self._smart_crop,
                self._num_epochs,
                self._batch_size,
                self._label_output,
                self._save_dir,
                self._predict_test,
                self._train_button,
            ]
        )

    def _on_finetune_from_changed(self, event: str):
        if event == "Pre-trained":
            if self._model_mode.value == "2D":
                self._finetune_pretrained_2d.show()
                self._finetune_pretrained_3d.hide()
                self._finetune_custom.hide()
            elif self._model_mode.value == "3D":
                self._finetune_pretrained_2d.hide()
                self._finetune_pretrained_3d.show()
                self._finetune_custom.hide()
            else:
                raise ValueError(f"Invalid model mode: {self._model_mode.value}")
        elif event == "Custom":
            self._finetune_pretrained_2d.hide()
            self._finetune_pretrained_3d.hide()
            self._finetune_custom.show()
        else:
            raise ValueError(f"Invalid finetune mode: {event}")


    def _on_model_mode_changed(self, event: str):
        if event == "2D" and self._finetune_from.value == "Pre-trained":
            self._finetune_pretrained_2d.show()
            self._finetune_pretrained_3d.hide()
        elif event == "3D" and self._finetune_from.value == "Pre-trained":
            self._finetune_pretrained_2d.hide()
            self._finetune_pretrained_3d.show()
        
        if event == "3D":
            self._crop_size_z.show()
            self._grid.show()
        elif event == "2D":
            self._crop_size_z.hide()
            self._grid.hide()
    
    def _on_training_mode_changed(self, event: str):
        if event == "Train":
            self._num_epochs.value = 200
            self._finetune_pretrained_2d.hide()
            self._finetune_pretrained_3d.hide()
            self._finetune_custom.hide()
            self._finetune_from.hide()
        elif event == "Fine-tune":
            self._num_epochs.value = 30
            self._finetune_from.show()
            if self._finetune_from.value == "Custom":
                self._finetune_pretrained_2d.hide()
                self._finetune_pretrained_3d.hide()
                self._finetune_custom.show()
            elif self._model_mode.value == "2D":
                self._finetune_pretrained_2d.show()
                self._finetune_pretrained_3d.hide()
                self._finetune_custom.hide()
            elif self._model_mode.value == "3D":
                self._finetune_pretrained_2d.hide()
                self._finetune_pretrained_3d.show()
                self._finetune_custom.hide() 

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

    def _train(self):
        if self._predict_test.value:
            # TODO: implement
            raise NotImplementedError("Prediction on test set is not implemented yet. Please disable it.")
        print("Training launched")
        if not Path(self._data_dir.value).is_dir():
            show_error(f"Invalid data directory: {self._data_dir.value}")
            return
        save_subdir = Path(self._save_dir.value)/datetime.now().strftime("%Y%m%d_%H%M%S")
        save_subdir.mkdir(parents=True, exist_ok=False)
        if self._training_mode.value == "Train":
            model = Spotiflow(
                SpotiflowModelConfig(
                    sigma=self._sigma.value,
                    is_3d=self._model_mode.value == "3D",
                    in_channels=self._in_channels.value,
                    grid=3*(self._grid.value,) if self._model_mode.value == "3D" else (1,1),
                )
            )
            finetuned_from = None
        elif self._training_mode.value == "Fine-tune":
            if self._finetune_from.value == "Pre-trained":
                model = Spotiflow.from_pretrained(
                    self._finetune_pretrained_2d.value if self._model_mode.value == "2D" else self._finetune_pretrained_3d.value,
                    inference_mode=False,
                    verbose=True
                )
                finetuned_from = self._finetune_pretrained_2d.value if self._model_mode.value == "2D" else self._finetune_pretrained_3d.value
            elif self._finetune_from.value == "Custom":
                model = Spotiflow.from_folder(
                    self._finetune_custom.value,
                    inference_mode=False,
                    verbose=True
                )
                finetuned_from = self._finetune_custom.value

        train_images, train_spots = get_data(Path(self._data_dir.value)/"train", is_3d=self._model_mode.value == "3D")
        if len(train_images) != len(train_spots):
            show_error(f"Number of images and spots in {(Path(self._data_dir.value)/'train').resolve()} do not match.")
            return
        if len(train_images) == 0:
            show_error(f"No images were found in the folder \"{(Path(self._data_dir.value)/'train').resolve()}\".")
            return
        print(f"Training data loaded (N={len(train_images)}).")
        val_images, val_spots = get_data(Path(self._data_dir.value)/"val", is_3d=self._model_mode.value == "3D")
        if len(val_images) != len(val_spots):
            show_error(f"Number of images and spots in {(Path(self._data_dir.value)/'val').resolve()} folder do not match.")
        if len(val_images) == 0:
            show_error(f"No images were found in the {(Path(self._data_dir.value)/'val').resolve()}.")
        print(f"Validation data loaded (N={len(val_images)}).")
        model.fit(
            train_images,
            train_spots,
            val_images,
            val_spots,
            save_dir=save_subdir,
            logger="tensorboard",
            augment_train=True,
            train_config={
                "batch_size": self._batch_size.value,
                "num_epochs": self._num_epochs.value,
                "crop_size": self._crop_size.value,
                "crop_size_z": self._crop_size_z.value,
                "smart_crop": self._smart_crop.value,
                "finetuned_from": finetuned_from,
            }
        )
        show_info(f"Training completed. Model was saved to \"{save_subdir}\"")
        return

    def _safe_train_wrapper(self, event=None):
        self._train_button.enabled = False
        try:
            self._train()
        except Exception as e:
            raise e
        finally:
            self._train_button.enabled = True
