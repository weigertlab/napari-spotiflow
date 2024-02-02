import pytest
from pathlib import Path
from napari_spotiflow._io_hooks import reader_function


def test_reader():
    paths = sorted(Path('data').glob('*.csv'))
    layers = tuple(reader_function(path) for path in paths)

    assert not any([lay is None for lay in layers])
    return layers


if __name__ == '__main__':

    layers = test_reader()
