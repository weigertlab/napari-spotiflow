"""

Simple csv reader populating a custom points layer 

"""
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from napari_builtins.io import napari_get_reader as default_napari_get_reader

COLUMNS = ('z', 'y', 'x')


COLUMNS_NAME_MAP_2D = {
    'axis-0' : 'y',
    'axis-1' : 'x',
}

COLUMNS_NAME_MAP_3D = {
    'axis-0' : 'z',
    'axis-1' : 'y',
    'axis-2' : 'x',
}


def _load_and_parse_csv(path, **kwargs):
    df = pd.read_csv(path, **kwargs)
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.strip()
    if 'axis-2' in df.columns:
        df = df.rename(columns = lambda n: COLUMNS_NAME_MAP_3D.get(n,n))
    else:
        df = df.rename(columns = lambda n: COLUMNS_NAME_MAP_2D.get(n,n))

    return df

def _validate_dataframe(df):
    return set(COLUMNS[-2:]).issubset(set(df.columns))

def _validate_path(path: Union[str, Path]):
    """ checks whether path is a valid csv """
    if isinstance(path, str):
        path = Path(path)
    check = isinstance(path, Path) and \
        path.suffix == ".csv" and \
        _validate_dataframe(_load_and_parse_csv(path))
    
    if not check:
        logging.warn(f'napari-spotiflow: failed to validate {path}')

    return check 

        


def napari_get_reader(path):
    print(f"opening {path} with napari-spotiflow")
    if _validate_path(path):
        return reader_function
    else:
        return default_napari_get_reader


def reader_function(path):
    from napari_spotiflow import _point_layer2d_default_kwargs
        
    if not _validate_path(path):
        return None 

    df = _load_and_parse_csv(path)

    # if 3d
    if set(COLUMNS).issubset(set(df.columns)):
        # data = df[list(columns)].to_numpy()
        data = df[['z','y','x']].to_numpy()
    else:
        # data = df[list(columns[-2:])].to_numpy()
        data = df[['y','x']].to_numpy()
        
    kwargs = dict(_point_layer2d_default_kwargs)

    return [(data, kwargs, 'points')]



def napari_write_points(path, data, meta):
    if data.shape[-1]==2:
        df = pd.DataFrame(data[:,::-1], columns=['x','y'])
    elif data.shape[-1]==3:
        df = pd.DataFrame(data[:,::-1], columns=['x','y','z'])
    else:
        return None    
    df.to_csv(path, index=False)
    return path
