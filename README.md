diffvirial
----------

This repository contains the code and most of the data required to reproduce the analysis presented in Krumholz, Lada, &amp; Forbrich (2025), and provides instructions for obtaining the remaining data.

What's included -- code
=======================

This repository contains five Python routines. The with names of the form `plot_*.py` are plotting routines to produce the figures in the paper. Specifically:

* `plot_polytropes.py` produces figures 1 and 2, which show virial diagrams computed for polytropes
* `plot_avir_sim.py` produces figures 3 and 4, which show an example contour on a simulation and virial diagrams derived from that contour anlaysis
* `plot_m31_obs.py` produces figures 5, 6, and 7, which show virial diagrams for the giant molecular clouds of M31 computed using 12CO and 13CO data, and a comparison between the two.

The remaining two Python files, `avir_sim_tools.py` and `polytrope_tools.py`, containg calculation routines used by the plotting files.

What's included and what's not included -- data
===============================================

The scripts `plot_avir_sim.py` and `plot_m31_obs.py` require data from a simulation and from observations of M31, respectively. The M31 data are included in the repository, in the file `m31gmcs.csv`. For details on how these data are derived and an explanation of the content and form of this file, see Lada, Forbrich, Krumholz, & Keto (2025, submitted).

The simulation data required are not included in this repository due to their large size, but can be downloaded from the [Catalogue of Astrophysical Turbulence Simulations (CATS)](https://www.mhdturbulence.com/). The required files are four HDF5 files containing the outputs of an enzo simulation by [Collins et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012ApJ...750...13C/abstract). The files required are the MHD Gravo-turbulence Simulations for the case `Beta0.2`, and have names of the form `C12_Beta0.2_256_NNNN.h5`, where `NNNN` is a four-digit number. The files must be placed in the same directory as `plot_avir_sim.py` for the script to function properly.

Dependencies
============

The Python code provided has the following dependencies:

* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [Matplotlib](https://matplotlib.org/)
* [AstroPy](https://www.astropy.org/)
* [scikit-image](https://scikit-image.org/)

The figures in the paper were generated using NumPy 2.1.1, SciPy 1.14.1, Matplotlib 3.9.2, AstroPy 7.0.0, and scikit-image 0.25.0. Earlier versions may work, but have not been tested.
