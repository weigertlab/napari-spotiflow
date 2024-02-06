[![License: BSD-3](https://img.shields.io/badge/License-BSD3-blue.svg)](https://www.gnu.org/licenses/bsd3)
[![PyPI](https://img.shields.io/pypi/v/napari-spotiflow.svg?color=green)](https://pypi.org/project/napari-spotiflow)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-spotiflow.svg?color=green)](https://python.org)
[![tests](https://github.com/weigertlab/napari-spotiflow/workflows/tests/badge.svg)](https://github.com/weigertlab/napari-spotiflow/actions)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-spotiflow)](https://napari-hub.org/plugins/napari-spotiflow)

![Logo](artwork/spotiflow_logo.png)
---

# napari-spotiflow

Napari plugin for *Spotiflow*, a deep learning-based, threshold-agnostic, and subpixel-accurate spot detection method for fluorescence microscopy. For the main repo, see [here](https://github.com/weigertlab/spotiflow). 

  
https://github.com/weigertlab/napari-spotiflow/assets/11042162/02940480-daa9-4a21-8cf5-ad73c26c9838


----------------------------------

# Usage 

1. Open 2d raw image (or open one of our samples eg `File > Open Sample > napari-spotiflow > HybISS`)
2. Start Plugin `Plugins > napari-spotiflow`
3. Select model (pretrained or custom trained) and optionally adjust other parameter
4. Click `run`

## Supported input formats
- [x] 2D (YX or YXC)
- [x] 2D+t (TYX or TYXC)

## Installation

Clone the repo and install it in an environment with `napari` and `spotiflow`.

```
git clone git@github.com:weigertlab/napari-spotiflow.git
pip install -e napari-spotiflow
```
