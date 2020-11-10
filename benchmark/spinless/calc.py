  
""" Test calculation for Hubbard atom with two bath sites.

Using parameters of cthyb benchmark "spinless" with hybridization of spins...

Author: Hugo U.R. Strand (2017) hugo.strand@gmail.com

 """ 

# ----------------------------------------------------------------------

import itertools
import numpy as np

# ----------------------------------------------------------------------

from triqs.gf import Gf
from triqs.gf import MeshImTime, MeshProduct
from triqs.gf import GfImTime, GfImFreq

from triqs.operators import c, c_dag
from triqs.archive import HDFArchive

# ----------------------------------------------------------------------

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    # ------------------------------------------------------------------
    # -- Hubbard atom with two bath sites, Hamiltonian
    
    beta = 10.0
    V1 = 1.0
    V2 = 1.0
    epsilon1 = +2.30
    epsilon2 = -2.30
    t = 0.1
    mu = 1.0
    U = 2.0

    up, do = 0, 1
    docc = c_dag(up,0) * c(up,0) * c_dag(do,0) * c(do,0)
    nA = c_dag(up,0) * c(up,0) + c_dag(do,0) * c(do,0)
    nB = c_dag(up,1) * c(up,1) + c_dag(do,1) * c(do,1)
    nC = c_dag(up,2) * c(up,2) + c_dag(do,2) * c(do,2)

    hopA = c_dag(up,0) * c(do,0) + c_dag(do,0) * c(up,0)
    hopB = c_dag(up,1) * c(do,1) + c_dag(do,1) * c(up,1)
    hopC = c_dag(up,2) * c(do,2) + c_dag(do,2) * c(up,2)

    H = -mu * nA + epsilon1 * nB + epsilon2 * nC + U * docc + \
        V1 * (c_dag(up,0)*c(up,1) + c_dag(up,1)*c(up,0) + \
              c_dag(do,0)*c(do,1) + c_dag(do,1)*c(do,0) ) + \
        V2 * (c_dag(up,0)*c(up,2) + c_dag(up,2)*c(up,0) + \
              c_dag(do,0)*c(do,2) + c_dag(do,2)*c(do,0) ) + \
        -t * (hopA + hopB + hopC)
    
    # ------------------------------------------------------------------
    # -- Exact diagonalization

    fundamental_operators = [
        c(up,0), c(do,0), c(up,1), c(do,1), c(up,2), c(do,2)]
    
    ed = TriqsExactDiagonalization(H, fundamental_operators, beta)

    # ------------------------------------------------------------------
    # -- Single-particle Green's functions

    g_tau = GfImTime(name=r'$g$', beta=beta,
                     statistic='Fermion', n_points=500,
                     indices=['A', 'B'])

    for (i1, s1), (i2, s2) in itertools.product([('A', up), ('B', do)], repeat=2):
        print i1, s1, i2, s2
        ed.set_g2_tau(g_tau[i1, i2], c(s1,0), c_dag(s2,0))

    # ------------------------------------------------------------------
    # -- Store to hdf5
    
    with HDFArchive('data_ed.h5','w') as res:
        res['tot'] = g_tau
        
# ----------------------------------------------------------------------
