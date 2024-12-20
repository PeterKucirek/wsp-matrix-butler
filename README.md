# Matrix Butler (wsp-matrix-butler)

[![Conda Latest Release](https://anaconda.org/wsp_sap/wsp-matrix-butler/badges/version.svg)](https://anaconda.org/wsp_sap/wsp-matrix-butler)
[![Conda Last Updated](https://anaconda.org/wsp_sap/wsp-matrix-butler/badges/latest_release_date.svg)](https://anaconda.org/wsp_sap/wsp-matrix-butler)
[![Platforms](https://anaconda.org/wsp_sap/wsp-matrix-butler/badges/platforms.svg)](https://anaconda.org/wsp_sap/wsp-matrix-butler)
[![License](https://anaconda.org/wsp_sap/wsp-matrix-butler/badges/license.svg)](https://github.com/wsp-sag/wsp-matrix-butler/blob/master/LICENSE)

A SQLite-based mini-file system for organizing matrix files for MTO's Greater Golden Horseshoe Model

## Installation

Matrix Butler can be installed with conda by running:

```batch
conda install -c wsp_sap wsp-matrix-butler
```

## Usage

> [!IMPORTANT]
> As of v2.0, this package is imported using `wsp_matrix_butler` instead of `matrix_butler`

```python
from wsp_matrix_butler import MatrixButler

# Connect to an existing GGHM cache folder
butler = MatrixButler.connect("path/to/cache")

# List the contents (matrices) available in the MatrixButler instance
butler.list_matrices()
```

## Development

Development of the MatrixButler uses [pixi](https://pixi.sh/) for Python package management and [rattler-build](https://rattler.build/) for package building.

