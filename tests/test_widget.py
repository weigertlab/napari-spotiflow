import napari
from napari_spotiflow._predict_widget import SpotiflowDetectionWidget

if __name__ == "__main__":
    v = napari.Viewer()
    dock = SpotiflowDetectionWidget(v)
    v.window.add_dock_widget(dock, area="right")
    napari.run()
