# Sensitivity of glacier volume change estimation to DEM void interpolation

This is the code used to generate and interpolate DEM voids, as described in:

McNabb, R., C. Nuth, A. Kääb, and L. Girod (2019). Sensitivity of glacier volume change estimation to DEM void interpolation. *The Cryosphere*, doi: 10.5194/tc-13-895-2019

The following files are included in the repository:
* ddem_plottying.ipynb - A jupyter notebook with code for most of the figures in the paper (and some others that didn't make it).
* dem_processing.py - A script to generate/interpolate voids in a pair of DEMs.
* outlines - The glacier outlines used, modified from the Randolph Glacier Inventory v6.0 dataset.
* volume_change_results.csv - Volume change estimates for each glacier, for each interpolation method, and each correlation threshold.
