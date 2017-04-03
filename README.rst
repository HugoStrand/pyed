**PYED**: Simple exact diagonalization routines for finite quantum systems
==========================================================================

Copyright (C) 2017, H. U.R. Strand

Installation
============

Todo

Documentation
=============

.. math::
   G_{\alpha \beta}(\tau_1, \tau_2) =
   - \langle \mathcal{T} \, c_\alpha(\tau_1) c_\beta^\dagger(\tau_2) \rangle
   =
   \left\{
   \begin{array}{lr}
   - \frac{1}{\mathcal{Z}} \text{Tr}
     \left[ e^{-\beta H} c_\alpha(\tau_1) c_\beta^\dagger(\tau_2) \right]
     & \quad,\beta \ge \tau_1 > \tau_2 \ge 0 \\
   - \xi \frac{1}{\mathcal{Z}} \text{Tr}
     \left[ e^{-\beta H} c_\beta^\dagger(\tau_2) c_\alpha(\tau_1) \right]
     & \quad, \beta \ge \tau_2 > \tau_1 \ge 0
   \end{array}\right.

where :math:`\xi = \pm 1` for bosons and fermions respectively.

Assuming that the times are ordered :math:`\beta \ge \tau_1 > \tau_2 \ge 0` we can insert two equations of unity :math:`1 = \sum_n | n \rangle \langle n|` where :math:`|n\rangle` is the eigen basis of the Hamiltonian :math:`H|n\rangle = E_n | n \rangle`. For the Green's function this gives the expression
     
.. math::
   G_{\alpha \beta}(\tau_1, \tau_2) = -\frac{1}{\mathcal{Z}}
   \sum_{nm} \exp\left[(-\beta + \tau_1 - \tau_2)E_n\right]
             \exp\left[ (\tau_2 - \tau_1)E_m \right]
	     \langle n | c_\alpha | m \rangle \langle m | c_\beta^\dagger | n \rangle
   
Version
=======

Todo

License
=======

This application is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version (see <http://www.gnu.org/licenses/>).

It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
