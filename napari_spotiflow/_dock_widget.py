import functools
import os
from copy import deepcopy
from typing import List, Union
from warnings import warn
import logging

import napari
import numpy as np
from magicgui import magicgui
from magicgui import widgets as mw
from magicgui.application import use_app

from .utils import _prepare_input, _validate_axes

logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)

BASE_IMAGE_AXES_CHOICES = ["YX", "YXC", "CYX", "TYX", "TYXC", "TCYX"]
BASE_IMAGE_AXES_CHOICES_3D = [f"Z{axes}" if "T" not in axes else f"TZ{axes[1:]}" for axes in BASE_IMAGE_AXES_CHOICES]

CURR_IMAGE_AXES_CHOICES = deepcopy(BASE_IMAGE_AXES_CHOICES)
IS_3D = False

def abspath(root, relpath):
    from pathlib import Path
    root = Path(root)
    if root.is_dir():
        path = root/relpath
    else:
        path = root.parent/relpath
    return str(path.absolute())


def change_handler(*widgets, init=True):
    """Implementation from https://github.com/stardist/stardist-napari/blob/main/stardist_napari/_dock_widget.py
    """
    def decorator_change_handler(handler):
        @functools.wraps(handler)
        def wrapper(*args):
            return handler(*args)

        for widget in widgets:
            widget.changed.connect(wrapper)
            if init:
                widget.changed(widget.value)
        return wrapper

    return decorator_change_handler


def plugin_wrapper():
    # delay imports until plugin is requested by user
    import torch
    from spotiflow.model import Spotiflow
    from spotiflow.model.pretrained import list_registered, _REGISTERED
    from spotiflow.utils import normalize

    from napari_spotiflow import _point_layer2d_default_kwargs

    def get_data(image):
        image = image.data[0] if image.multiscale else image.data
        return np.asarray(image)

    models_reg_2d = [r for r in list_registered() if not _REGISTERED[r].is_3d]
    models_reg_3d = [r for r in list_registered() if _REGISTERED[r].is_3d]

    if 'general' in models_reg_2d:
        models_reg_2d = ['general'] + sorted([m for m in models_reg_2d if m != 'general'])
    else:
        models_reg_2d = sorted(models_reg_2d)
    
    if 'synth_3d' in models_reg_3d:
        models_reg_3d = ['synth_3d'] + sorted([m for m in models_reg_3d if m != 'synth_3d'])
    else:
        models_reg_3d = sorted(models_reg_3d)

    CUSTOM_MODEL = 'CUSTOM_MODEL'
    model_type_choices = [('Pre-trained', Spotiflow), ('Custom', CUSTOM_MODEL)]
    peak_mode_choices = ["fast", "skimage"]
    global CURR_IMAGE_AXES_CHOICES

    image_layers = [l for l in napari.current_viewer().layers if isinstance(l, napari.layers.Image)]
    if len(image_layers) > 0:
        ndim_first = image_layers[0].data.ndim
        choice_potential = BASE_IMAGE_AXES_CHOICES if not IS_3D else BASE_IMAGE_AXES_CHOICES_3D
        CURR_IMAGE_AXES_CHOICES = [c for c in choice_potential if len(c) == ndim_first]
        if len(CURR_IMAGE_AXES_CHOICES) == 0:
            CURR_IMAGE_AXES_CHOICES = [""]
    else:
        CURR_IMAGE_AXES_CHOICES = [""]



    @functools.lru_cache(maxsize=None)
    def get_model(model_type, model, device):
        kwargs = dict(inference_mode=True, map_location=device)
        
        if model_type == CUSTOM_MODEL:
            return Spotiflow.from_folder(model, **kwargs)
        else:
            return model_type.from_pretrained(model, **kwargs)
        
    # -------------------------------------------------------------------------


    DEFAULTS = dict (
        mode           = '2D',
        model_type     = Spotiflow,
        model2d        = 'general',
        model3d        = 'synth_3d',
        norm_image     = True,
        perc_low       =  1.0,
        perc_high      = 99.8,
        use_optimized  = True,
        prob_thresh    = 0.5,
        n_tiles        = '1,1',
        cnn_output     = False,
        peak_mode      = 'fast',
        exclude_border = False,
        scale          = 1.0,
        min_distance   = 2,
        auto_n_tiles   = True,
        subpix         = True,
    )

    # -------------------------------------------------------------------------
    logo = abspath(__file__, 'resources/spotiflow_transp_small.png')
    @magicgui (
        label_head      = dict(widget_type='Label', label=f'<h1><img src="{logo}"></h1>'),
        image           = dict(label='Input Image'),
        image_axes      = dict(widget_type='RadioButtons', label='Image axes order', orientation='horizontal', choices=CURR_IMAGE_AXES_CHOICES, value=CURR_IMAGE_AXES_CHOICES[0]),
        label_nn        = dict(widget_type='Label', label='<br><b>Neural Network Prediction:</b>'),
        mode            = dict(widget_type='RadioButtons', label='Mode', orientation='horizontal', choices=['2D', '3D'], value=DEFAULTS["mode"]),
        model_type      = dict(widget_type='RadioButtons', label='Model Type', orientation='horizontal', choices=model_type_choices, value=DEFAULTS['model_type']),
        model2d         = dict(widget_type='ComboBox', visible=True, label='Pre-trained Model (2D)', choices=models_reg_2d, value=DEFAULTS['model2d']),
        model3d         = dict(widget_type='ComboBox', visible=True, label='Pre-trained Model (3D)', choices=models_reg_3d, value=DEFAULTS['model3d']),
        model_folder    = dict(widget_type='FileEdit', visible=True, label='Custom Model', mode='d'),
        norm_image      = dict(widget_type='CheckBox', text='Normalize Image', value=DEFAULTS['norm_image']),
        scale           = dict(widget_type='FloatSpinBox', label='Scale factor',                min=0.5, max=2, step=0.1,  value=DEFAULTS['scale']),
        subpix          = dict(widget_type='CheckBox', text='Subpixel prediction', value=DEFAULTS['subpix']),
        label_nms       = dict(widget_type='Label', label='<br><b>NMS Postprocessing:</b>'),
        perc_low        = dict(widget_type='FloatSpinBox', label='Percentile low',              min=0.0, max=100.0, step=0.1,  value=DEFAULTS['perc_low']),
        perc_high       = dict(widget_type='FloatSpinBox', label='Percentile high',             min=0.0, max=100.0, step=0.1,  value=DEFAULTS['perc_high']),
        use_optimized   = dict(widget_type='CheckBox', text='Use optimized probability threshold', value=DEFAULTS['use_optimized']),
        prob_thresh     = dict(widget_type='FloatSpinBox', label='Probability/Score Threshold', min=0.0, max=  1.0, step=0.05, value=DEFAULTS['prob_thresh']),
        peak_mode       = dict(widget_type='RadioButtons', label='Peak extraction mode', orientation='horizontal', choices=peak_mode_choices, value=DEFAULTS['peak_mode']),
        exclude_border  = dict(widget_type='CheckBox', text='Exclude border', value=DEFAULTS['exclude_border']),
        min_distance    = dict(widget_type='SpinBox', label='Minimum distance', min=1, max=5, step=1, value=DEFAULTS['min_distance']),
        auto_n_tiles    = dict(widget_type='Checkbox', text='Automatically infer tiling', value=DEFAULTS['auto_n_tiles']),
        n_tiles         = dict(widget_type='LiteralEvalLineEdit', label='Number of Tiles', value=DEFAULTS['n_tiles']),
        label_adv       = dict(widget_type='Label', label='<br><b>Advanced Options:</b>'),
        cnn_output      = dict(widget_type='CheckBox', text='Show CNN Output', value=DEFAULTS['cnn_output']),
        progress_bar    = dict(label=' ', min=0, max=0, visible=False),
        layout          = 'vertical',
        persist         = True,
        call_button     = True,
    )
    def plugin (
        viewer: napari.Viewer,
        label_head,
        image: napari.layers.Image,
        image_axes: str,
        label_nn,
        mode,
        model_type,
        model2d,
        model3d,
        model_folder,
        norm_image,
        perc_low,
        perc_high,
        scale,
        subpix,
        label_nms,
        use_optimized,
        prob_thresh,
        peak_mode,
        exclude_border,
        min_distance,
        label_adv,
        auto_n_tiles,
        n_tiles,
        cnn_output,
        progress_bar: mw.ProgressBar,
    ) -> list[napari.types.LayerDataTuple]:
        if image_axes == "":
            raise RuntimeError("Invalid axes order. If your input is 2D, please set the 2D mode. If your input is 3D, please set the 3D mode.")
        should_use_mps = torch.backends.mps.is_available() and (not IS_3D or (os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") is not None and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0"))
        DEVICE_STR = "cuda" if torch.cuda.is_available() else "mps" if should_use_mps and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") else "cpu"

        model = get_model(
            model_type,
            {
                Spotiflow: model2d if not IS_3D else model3d,
                CUSTOM_MODEL: model_folder,
            }[model_type],
            DEVICE_STR
        )

        # TODO: improve errors - display them in the GUI
        if IS_3D:
            assert model.config.is_3d, "Expected folder containing a 3D model. Are you sure the given folder is correct?"
        else:
            assert not model.config.is_3d, "Expected folder containing a 2D model. Are you sure the given folder is correct?"

        if subpix and not model.config.compute_flow:
            warn("Model was not trained to predict the stereographic flow. Will disable subpixel prediction.")
            subpix = False
           
        model.to(torch.device(DEVICE_STR))
        try:
            model = torch.compile(model)
        except RuntimeError as _:
            warn("Could not compile the model module. Will run without compiling, which can be slightly slower.")
        except Exception as e:
            raise e

        layers = []
    
        assert image is not None, "Please add an image layer to the viewer!"
        x = get_data(image)

        _validate_axes(x, image_axes)
        x = _prepare_input(x, image_axes)

        if "T" not in image_axes:
            if len(n_tiles)==2+int(IS_3D):
                n_tiles = n_tiles + (1,)

            if norm_image:
                print("Normalizing image...")
                x = normalize(x, perc_low, perc_high)

        else:
            if x.ndim==(4+int(IS_3D)) and len(n_tiles)==(2+int(IS_3D)):
                n_tiles = n_tiles + (1,)
            if norm_image:
                print("Normalizing frames...")
                x = np.stack([normalize(_x, perc_low, perc_high) for _x in x])

        app = use_app()
        def progress(size):
            def _progress(it, **kwargs):
                progress_bar.label = 'Spotiflow Prediction'
                progress_bar.range = (0, size)
                progress_bar.value = 0
                progress_bar.show()
                app.process_events()
                for item in it:
                    yield item
                    progress_bar.increment(1)
                    app.process_events()
                app.process_events()
            return _progress
        actual_prob_thresh = prob_thresh if not use_optimized else None
        if "T" not in image_axes:
            actual_n_tiles = tuple(max(1,s//(1024 if not IS_3D else 128)) for s in x.shape) if auto_n_tiles else n_tiles
            pred_points, details = model.predict(x,
                                                prob_thresh=actual_prob_thresh,
                                                n_tiles=actual_n_tiles,
                                                peak_mode=peak_mode,
                                                exclude_border=exclude_border,
                                                min_distance=min_distance,
                                                scale=scale,
                                                verbose=True,
                                                progress_bar_wrapper=progress(np.prod(actual_n_tiles)),
                                                device=DEVICE_STR,
                                                subpix=subpix,
                                                )

            if cnn_output:
                details_prob_heatmap = details.heatmap
                if subpix:
                    details_flow = details.flow

        else:
            # Predict frames
            actual_n_tiles = tuple(max(1,s//(1024 if not IS_3D else 128)) for s in x.shape[1:]) if auto_n_tiles else n_tiles
            pred_points_t, details_t = tuple(zip(*tuple(model.predict(_x,
                                                prob_thresh=actual_prob_thresh,
                                                n_tiles=actual_n_tiles,
                                                peak_mode=peak_mode,
                                                exclude_border=exclude_border,
                                                min_distance=min_distance,
                                                scale=scale,
                                                verbose=True,
                                                device=DEVICE_STR,
                                                subpix=subpix,
                                                ) for _x in progress(x.shape[0])(x))))

            pred_points = tuple(np.concatenate([[i], p])
                                for i,ps in enumerate(pred_points_t) for p in ps)
            if cnn_output:
                details_prob_heatmap = np.stack([det.heatmap for det in details_t], axis=0)
                if subpix:
                    details_flow = np.stack([det.flow for det in details_t], axis=0)

        if cnn_output:
            if subpix:
                layers.append((.5*(1+details_flow), dict(name=f'Stereographic flow ({image.name})', scale=model.config.grid,
                                               ), 'image'))
            layers.append((details_prob_heatmap, dict(name=f'Gaussian heatmap ({image.name})',
                                           colormap='magma', scale=model.config.grid), 'image'))
        points_layer_name = f'Spots ({image.name})'
        for l in viewer.layers:
            if l.name == points_layer_name:
                viewer.layers.remove(l)

        layers.append((pred_points, dict(name=f'Spots ({image.name})',
                                             **_point_layer2d_default_kwargs), 'points'))

        
        progress_bar.hide()

        return layers

    # # -------------------------------------------------------------------------
    
    plugin.n_tiles.value = DEFAULTS['n_tiles']
    plugin.label_head.value = '<small></small>'

    # make labels prettier (https://doc.qt.io/qt-5/qsizepolicy.html#Policy-enum)
    for w in (plugin.label_head, plugin.label_nn, plugin.label_nms, plugin.label_adv):
        w.native.setSizePolicy(1|2, 0)

    # -------------------------------------------------------------------------

    widget_for_modeltype = {
        Spotiflow: "pretrained",
        CUSTOM_MODEL: plugin.model_folder,
    }

    def widgets_inactive(*widgets, active):
        for widget in widgets:
            widget.visible = active

    # allow some widgets to shrink because their size depends on user input
    plugin.image.native.setMinimumWidth(240)
    plugin.model2d.native.setMinimumWidth(240)

    plugin.label_head.native.setOpenExternalLinks(True)

    layout = plugin.native.layout()
    layout.insertStretch(layout.count()-2)

    @change_handler(plugin.use_optimized)
    def _thr_change(active: bool):
        widgets_inactive(
            plugin.prob_thresh,
            active=not active
        )

    @change_handler(plugin.image, init=True)
    def _image_update(image: napari.layers.Image):
        global CURR_IMAGE_AXES_CHOICES
        global IS_3D
        possible_axes_choices = BASE_IMAGE_AXES_CHOICES if not IS_3D else BASE_IMAGE_AXES_CHOICES_3D
        if image is not None:
            inp_ndim = get_data(image).ndim
            # if not IS_3D:
            #     assert inp_ndim in (2,3,4), f"Invalid input dimension: {inp_ndim}. Should be 2, 3, or 4."
            # else:
            #     assert inp_ndim in (3,4,5), f"Invalid input dimension: {inp_ndim}. Should be 3, 4, or 5."
            # Update the choices for image_axes

            CURR_IMAGE_AXES_CHOICES = [c for c in possible_axes_choices if len(c) == inp_ndim]
        else:
            CURR_IMAGE_AXES_CHOICES = [""]

        if len(CURR_IMAGE_AXES_CHOICES) == 0:
            CURR_IMAGE_AXES_CHOICES = [""]
        # Trigger event to update the choices and value of image_axes
        plugin.image_axes.changed(CURR_IMAGE_AXES_CHOICES)
        plugin.image_axes.value = CURR_IMAGE_AXES_CHOICES[0]

    @change_handler(plugin.image_axes, init=False)
    def _image_axes_update(choices: List[str]):
        global CURR_IMAGE_AXES_CHOICES
        with plugin.image_axes.changed.blocked():
            plugin.image_axes.choices = CURR_IMAGE_AXES_CHOICES
        if plugin.image_axes.value not in choices:
            plugin.image_axes.value = CURR_IMAGE_AXES_CHOICES[0]
        if plugin.image_axes.value == "":
            plugin.call_button.enabled = False
        else:
            plugin.call_button.enabled = True

    @change_handler(plugin.norm_image)
    def _norm_image_change(active: bool):
        widgets_inactive(
            plugin.perc_low, plugin.perc_high, active=active
        )
    
    @change_handler(plugin.auto_n_tiles)
    def _auto_n_tiles_change(active: bool):
        widgets_inactive(
            plugin.n_tiles,
            active=not active
        )

    @change_handler(plugin.model_type, init=False)
    def _model_type_change(model_type: Union[str, type]):
        selected = widget_for_modeltype[model_type]
        if selected == "pretrained":
            selected = plugin.model2d if not IS_3D else plugin.model3d
        for w in set((plugin.model2d, plugin.model3d, plugin.model_folder)) - {selected}:
            w.hide()
        selected.show()
        selected.changed(selected.value)

    @change_handler(plugin.mode, init=True)
    def _model_mode_change(mode: str):
        global IS_3D
        IS_3D = mode == '3D'
        selected = plugin.model3d if IS_3D else plugin.model2d
        for w in set((plugin.model2d, plugin.model3d)) - {selected}:
            w.hide()
        selected.show()
        selected.changed(selected.value)
        _image_update(plugin.image.value)

    return plugin
