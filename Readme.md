# **PYED**: Exact diagonalization for finite quantum systems

Copyright (C) 2018, H. U.R. Strand, Ya.V. Zhumagulov

The python module `pyed` implements exact diagonalization for finite fermionic many-body quantum systems, together with calculations of several response functions in imagianary time.

The many-body system is defined using `pytriqs` second-quantized operators and the response functions are stored in `pytriqs` Green's function containters.

The original purpose of `pyed` is to provide exact solutions to small finite systems, to be used as benchmarks and tests for stochastic many-body solvers.


## Installation

```
pip install git+https://github.com/yaros72/pyed
```


## Documentation

For documentation and usage examples please see the hands on [jupyter notebook](doc/Documentation.ipynb)

## License

This application is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version (see <http://www.gnu.org/licenses/>).

It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
