import napari
from napari_spotiflow._dock_widget import plugin_wrapper

if __name__ == "__main__":
    dock = plugin_wrapper()
    
    v = napari.Viewer()
    v.window.add_dock_widget(dock, area="right")
    napari.run()