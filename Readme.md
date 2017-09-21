# **PYED**: Exact diagonalization for finite quantum systems

Copyright (C) 2017, H. U.R. Strand, Ya.V. Zhumagulov

The python module `pyed` implements exact diagonalization for finite fermionic many-body quantum systems, together with calculations of several response functions in imagianary time.

The many-body system is defined using `pytriqs` second-quantized operators and the response functions are stored in `pytriqs` Green's function containters.

The original purpose of `pyed` is to provide exact solutions to small finite systems, to be used as benchmarks and tests for stochastic many-body solvers.

Cluster pertrubation theory [1] addition to pyed allow calculate bandstructure and Fermi surface of several models. 

[1]  https://www.physique.usherbrooke.ca/pages/sites/default/files/senechal/publis/Senechal2011vn.pdf

## Installation

To do: Add `setup_utils` install script

There is currently no formal installation scripts packed with `pyed`. To use and develop the module simply ammend your `PYTHON_PATH` with the `./pyed/` folder, e.g., add the follwing

```
export PYTHON_PATH=${HOME}/path/to/pyed:$PYTHON_PATH
```

in your `.bashrc`, `.bash_profile`, or `.profile` file.

## Documentation

For documentation and usage examples please see the hands on [jupyter notebook](doc/Documentation.ipynb)

For documentation and usage of CPT addition examples please see the hands on [jupyter notebook](doc/Documentation_CPT_2D.ipynb)

## License

This application is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version (see <http://www.gnu.org/licenses/>).

It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
