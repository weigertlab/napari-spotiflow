import napari
import numpy as np
from spotiflow.sample_data import test_image_hybiss_2d
from napari_spotiflow._predict_widget import SpotiflowDetectionWidget


def example_2d():
    x = test_image_hybiss_2d()
    viewer = napari.Viewer()
    viewer.add_image(x)
    widget = SpotiflowDetectionWidget(viewer)
    viewer.window.add_dock_widget(widget)


def example_2d_time():
    x = test_image_hybiss_2d()
    x = np.tile(x, (4, 1, 1))
    viewer = napari.Viewer()
    viewer.add_image(x)
    widget = SpotiflowDetectionWidget(viewer)
    viewer.window.add_dock_widget(widget)

if __name__ == "__main__":
    viewer = example_2d()
    napari.run()
