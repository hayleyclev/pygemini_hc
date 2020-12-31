# PyGemini

![ci](https://github.com/gemini3d/pygemini/workflows/ci/badge.svg)
![ci](https://github.com/gemini3d/pygemini/workflows/ci_mac/badge.svg)
![ci](https://github.com/gemini3d/pygemini/workflows/ci_windows/badge.svg)
![ci](https://github.com/gemini3d/pygemini/workflows/ci_prereq/badge.svg)

A Python interface to [Gemini3D](https://github.com/gemini3d/gemini).

## Setup

Setup PyGemini by:

```sh
git clone https://github.com/gemini3d/pygemini

pip install -e pygemini
```

### build

Not all users need to run Gemini3D on the same device where PyGemini is installed.
PyGemini uses the "build on run" method developed by Michael Hirsch, which allows complex multi-language Python packages to install reliably across operating systems (MacOS, Linux, Windows).
Upon the first `gemini3d.run()`, the underlying Gemini3D code is built, including all necessary libraries.

```sh
python -m gemini3d.prereqs
```

allows manually installing those libraries to save rebuild time, but this is optional as Gemini3D automatically downloads and builds missing libraries.

### Developers

For those working with GEMINI Fortran code itself or to work with non-release versions of GEMINI Fortran code:

1. install PyGemini in development mode as above
2. set environment variable GEMINI_ROOT to the Gemini3D Fortran code directory, otherwise PyGemini will Git clone a new copy.

## Run simulation

1. make a [config.nml](https://github.com/gemini3d/gemini/docs/Readme_input.md) with desired parameters for an equilibrium sim.
2. run the equilibrium sim:

    ```sh
    python -m gemini3d.run /path_to/config_eq.nml /path_to/sim_eq/
    ```
3. create a new config.nml for the actual simulation and run

    ```sh
    python -m gemini3d.run /path_to/config.nml /path_to/sim_out/
    ```

## Plots

An important part of any simulation is viewing the output.
Because of the large data involved, most plotting functions automatically save PNG stacks to disk for quick flipping through with your preferred image viewing program.

### Grid

Help ensure the simulation grid is what you intended by the following, which can be used before or after running the simulation.

```python
import gemini3d.plot

gemini3d.plot.grid("path/to/sim")
```

### simulation outputs

These commands create plots and save to disk under the "plots/" directory under the specified data directory.

command line:

```sh
python -m gemini3d.plot path/to/data -save png
```

or from within Python:

```python
import gemini3d.plot as plot

plot.frame("path/to/data", datetime(2020, 1, 2, 1, 2, 3), saveplot_fmt="png")

# or

plot.plot_all("path/to/data", saveplot_fmt="png")
```

## Convert data files to HDF5

There is a a script to convert data to HDF5, and another to convert grids to HDF5.
The scripts convert from {raw, Matlab, NetCDF4} to HDF5.
The benefits of doing this are especially significant for raw data, and HDF5 may compress by 50% or more, and make the data self-describing.

```sh
python scripts/convert_data.py h5 ~/mysim
```

```sh
python scripts/convert_grid.py h5 ~/mysim/inputs/simgrid.dat
```
