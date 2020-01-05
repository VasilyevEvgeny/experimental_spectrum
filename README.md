# What is it?

Program allows to process the results of experiments on the measurement of frequency-angular spectra and integrated frequency spectra
of laser pulses after propagation in a nonlinear medium.

# Typical plots

**Frequency-angular spectrum (FAS)** - measurement of the pulse beam spectrum at different angles of its propagation

| `FAS for Gaussian beam` | `FAS for vortex beam` |
| :----------------------: | :---------------------: |
| ![gauss](https://github.com/VasilyevEvgeny/experimental_spectrum/blob/master/resources/spectra/fas_gauss.png) | ![vortex](https://github.com/VasilyevEvgeny/experimental_spectrum/blob/master/resources/spectra/fas_vortex.png) |

| `Frequency spectrum at constant angle` | `Angular spectrum at constant wavelength` |
| :----------------------: | :---------------------: |
| ![frequency_spectrum](https://github.com/VasilyevEvgeny/experimental_spectrum/blob/master/resources/spectra/frequency_spectrum.png) | ![angular_spectrum](https://github.com/VasilyevEvgeny/experimental_spectrum/blob/master/resources/spectra/angular_spectrum.png) |

**Integrated frequency spectrum (IFS)** - focusing all spatial garmonics and its spectrum measurement 

| `IFS in log scale` | `IFS in linear scale` |
| :----------------------: | :---------------------: |
| ![log_scale](https://github.com/VasilyevEvgeny/experimental_spectrum/blob/master/resources/spectra/ifs_log.png) | ![linear_scale](https://github.com/VasilyevEvgeny/experimental_spectrum/blob/master/resources/spectra/ifs_linear.png) |

# Principal design of experiments

| `FAS`  | `IFS`  |
| :------------------------------: | :------------------------------: |
| ![fas_experiment](https://github.com/VasilyevEvgeny/experimental_spectrum/blob/master/resources/experiment/fas_experiment.png) | ![ifs_experiment](https://github.com/VasilyevEvgeny/experimental_spectrum/blob/master/resources/experiment/ifs_experiment.png) |

# Installation

* **Windows**:
```pwsh
virtualenv .venv
cd .venv/Scripts
activate
pip install -r <path_to_project>/requirements.txt
```

* **Linux**
```bash
virtualenv .venv -p python3
cd .venv/bin
source ./activate
pip install -r <path_to_project>/requirements.txt
```

# Usage

To build the *frequency-angular / intergrated frequency spectrum*, you must create an object of class 
**``ProcessorFAS``** / **``ProcessorIFS``**:
```python
fas = ProcessorFAS(<path_to_experimental_data>)
ifs = ProcessorIFS(<path_to_experimental_data>)
```
There are 2 command line arguments:
* **global_res_dir** - path to the directory where a set of program results will be stored
* **res_dir_name** - name of the directory where a set of program results will be stored
At the next start of the program it is checked whether the directory **global_res_dir/res_dir_name** exists, and if not, 
a new empty one is created.

Classes **``ProcessorFAS``** and **``ProcessorIFS``** are derivative for class **``BaseProcessor``**, which has the next 
additional parameters:
* **sigma_lambda** *(default = 10 nm)* - dispersion of gaussian filter along lambda coordinate
* **log_scale** *(default = True)* - use logarithmic scale or not
* **log_power** *(default = -2)* - if *log_scale = True*, the minimum degree of the logarithm used to plot the spectra

Additional parameters for class **``ProcessorFAS``**:
* **micron_per_step** *(default = 10)* - number of microns in one step 
* **deg_per_micron** *(default = 5 / 3500)* - number of degrees in one micron 
* **sigma_angle** (default = 0 rad) - dispersion of gaussian filter along angle coordinate
* **steps_overlap** (default = 4) - number of overlapping steps in fas measurements
* **lambda_dn** - number of points along lambda coordinate through which the angular spectrum is plotted

There are no any additional parameters for class **``ProcessorIFS``**.

# Input

Typical input data for FAS processing ([1](https://github.com/VasilyevEvgeny/experimental_spectrum/tree/master/scripts/example_data/gauss_fas), [2](https://github.com/VasilyevEvgeny/experimental_spectrum/tree/master/scripts/example_data/vortex_fas))
and IFS processing ([1](https://github.com/VasilyevEvgeny/experimental_spectrum/tree/master/scripts/example_data/gauss_ifs), [2](https://github.com/VasilyevEvgeny/experimental_spectrum/tree/master/scripts/example_data/vortex_ifs)) is a set of files 
where wavelengths and the corresponding spectral harmonics intensities are recorded in arbitrary units.
