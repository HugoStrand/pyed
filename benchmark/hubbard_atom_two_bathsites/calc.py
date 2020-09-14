  
""" Test calculation for Hubbard atom with two bath sites.

Author: Hugo U.R. Strand (2017) hugo.strand@gmail.com

 """ 

# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------

from triqs.gf import Gf
from triqs.gf import MeshImTime, MeshProduct
from triqs.gf import GfImTime, GfImFreq

from triqs.operators import c, c_dag
from h5 import HDFArchive

# ----------------------------------------------------------------------

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    # ------------------------------------------------------------------
    # -- Hubbard atom with two bath sites, Hamiltonian
    
    beta = 2.0
    V1 = 2.0
    V2 = 5.0
    epsilon1 = 0.00
    epsilon2 = 4.00
    mu = 2.0
    U = 0.0

    up, do = 0, 1
    docc = c_dag(up,0) * c(up,0) * c_dag(do,0) * c(do,0)
    nA = c_dag(up,0) * c(up,0) + c_dag(do,0) * c(do,0)
    nB = c_dag(up,1) * c(up,1) + c_dag(do,1) * c(do,1)
    nC = c_dag(up,2) * c(up,2) + c_dag(do,2) * c(do,2)

    H = -mu * nA + epsilon1 * nB + epsilon2 * nC + U * docc + \
        V1 * (c_dag(up,0)*c(up,1) + c_dag(up,1)*c(up,0) + \
              c_dag(do,0)*c(do,1) + c_dag(do,1)*c(do,0) ) + \
        V2 * (c_dag(up,0)*c(up,2) + c_dag(up,2)*c(up,0) + \
              c_dag(do,0)*c(do,2) + c_dag(do,2)*c(do,0) )
    
    # ------------------------------------------------------------------
    # -- Exact diagonalization

    fundamental_operators = [
        c(up,0), c(do,0), c(up,1), c(do,1), c(up,2), c(do,2)]
    
    ed = TriqsExactDiagonalization(H, fundamental_operators, beta)

    # ------------------------------------------------------------------
    # -- Single-particle Green's functions

    g_tau = GfImTime(name=r'$g$', beta=beta,
                     statistic='Fermion', n_points=50,
                     target_shape=(1,1))

    g_iwn = GfImFreq(name='$g$', beta=beta,
                     statistic='Fermion', n_points=10,
                     target_shape=(1,1))
    
    ed.set_g2_tau(g_tau, c(up,0), c_dag(up,0))
    ed.set_g2_iwn(g_iwn, c(up,0), c_dag(up,0))

    # ------------------------------------------------------------------
    # -- Two particle Green's functions
    
    ntau = 20
    imtime = MeshImTime(beta, 'Fermion', ntau)
    prodmesh = MeshProduct(imtime, imtime, imtime)

    g40_tau = Gf(name='g40_tau', mesh=prodmesh, target_shape=[1, 1, 1, 1])
    g4_tau = Gf(name='g4_tau', mesh=prodmesh, target_shape=[1, 1, 1, 1])

    ed.set_g40_tau(g40_tau, g_tau)
    ed.set_g4_tau(g4_tau, c(up,0), c_dag(up,0), c(up,0), c_dag(up,0))

    # ------------------------------------------------------------------
    # -- Two particle Green's functions (equal times)

    prodmesh = MeshProduct(imtime, imtime)
    g3pp_tau = Gf(name='g4_tau', mesh=prodmesh, target_shape=[1, 1, 1, 1])
    ed.set_g3_tau(g3pp_tau, c(up,0), c_dag(up,0), c(up,0)*c_dag(up,0))

    # ------------------------------------------------------------------
    # -- Store to hdf5
    
    with HDFArchive('data_ed.h5','w') as res:
                
        res["G_tau"] = g_tau
        res["G_iw"] = g_iwn
        
        res["G20_tau"] = g40_tau
        res["G2_tau"] = g4_tau

        res["G3pp_tau"] = g3pp_tau
        
# ----------------------------------------------------------------------
