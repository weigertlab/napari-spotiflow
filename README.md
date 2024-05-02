[![License: BSD-3](https://img.shields.io/badge/License-BSD3-blue.svg)](https://www.gnu.org/licenses/bsd3)
[![PyPI](https://img.shields.io/pypi/v/napari-spotiflow.svg?color=green)](https://pypi.org/project/napari-spotiflow)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-spotiflow.svg?color=green)](https://python.org)
[![tests](https://github.com/weigertlab/napari-spotiflow/workflows/tests/badge.svg)](https://github.com/weigertlab/napari-spotiflow/actions)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/napari-spotiflow)](https://pypistats.org/packages/napari-spotiflow)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-spotiflow)](https://napari-hub.org/plugins/napari-spotiflow)

![Logo](https://github.com/weigertlab/napari-spotiflow/raw/main/artwork/spotiflow_logo.png)
---

# Spotiflow: napari plugin

Napari plugin for *Spotiflow*, a deep learning-based, threshold-agnostic, and subpixel-accurate spot detection method for fluorescence microscopy. The plugin allows using several pre-trained models as well as user-trained ones. For the main repository, see [here](https://github.com/weigertlab/spotiflow). 

https://github.com/weigertlab/napari-spotiflow/assets/11042162/02940480-daa9-4a21-8cf5-ad73c26c9838

If you use this plugin for your research, please [cite us](https://github.com/weigertlab/spotiflow#how-to-cite).

----------------------------------

## Installation

The plugin can be installed directly from PyPi (make sure you use a conda environment with `napari` and `spotiflow` installed):

```
pip install napari-spotiflow
```

## Usage 

1. Open the image (or open one of our samples, _e.g._ `File > Open Sample > napari-spotiflow > HybISS`)
2. Start the plugin `Plugins > napari-spotiflow`
3. Select model (pre-trained or custom trained) and optionally adjust any other parameters
4. Click `Run`

## Supported input formats
- 2D (YX, YXC or CYX)
- 2D+t (TYX, TYXC or TCYX)

## How to cite
See the [main repository's _How to cite_ section](https://github.com/weigertlab/spotiflow?tab=readme-ov-file#how-to-cite).
