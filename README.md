# Python implementation of "Three-dimensional Wadell roundness for particle angularity characterization of granular soils".
This package contains a python implementation of the algorithm by Junxing Zheng, Hantao He and Hossein Alimohammadi [1], which provides a method to calculate the 3D Wadell roundness.

It is based on the package _2D Wadell roundness and sphericity in Python_ which can be found at https://github.com/PaPieta/wadell_rs/tree/main.
A modified version of that package is included in this package.

## Installation
``` python
pip install git+https://github.com/pwrasmuss/3d_roundness.git
```

## User guide
See the notebook named test_spheres.ipynb for a simple example of how the algorithm works.
The model requires a binary segmentation in the form of a numpy array where the background has the value 0 and the objects have the value 1.
In the notebook, this array is loaded from a `.tiff`-file, with the dimensions (ZXY).

## Requirements
The repo requires the following packages: `numpy`, `scipy`, `skimage`, `scikit-learn`, `scikit-image` and `edt`.


## A Faster alternative
The algorithm presented in [1] is not suited for large volumes with many objects and in these cases alternatives are better. The GitHub repo https://github.com/PaPieta/fast_rs contains code for the paper by Pawel et al. (2025) [2].
The algorithm approximates 3D (or 2D) roundness and sphericity by calculating the local thickness of objects.

> [1]. Zheng, J., He, H. & Alimohammadi, H. Three-dimensional Wadell roundness for particle angularity characterization of granular soils. Acta Geotech. 16, 133–149 (2021). https://doi.org/10.1007/s11440-020-01004-9.
> [2]. Paper accepted to the CVPR workshop CVMI and will appear at https://cvmi-workshop.github.io/accepted.html.