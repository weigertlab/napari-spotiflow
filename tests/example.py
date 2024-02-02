import sys
import numpy as np
import napari

from csbdeep.utils import normalize
import numpy as np
from tqdm import tqdm 
import argparse
from spotiflow.model import SpotNet
from spotiflow.data import hybiss_data_2d

def example_2d():
    x =  hybiss_data_2d()
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('napari-spotiflow')


def example_2d_time():
    x =  hybiss_data_2d()
    x = np.tile(x, (4,1,1))
    viewer =  napari.Viewer()
    viewer.add_image(x)
    viewer.window.add_plugin_dock_widget('napari-spotiflow')
    
if __name__ == '__main__':

    viewer = example_2d()

    napari.run()
    
