
"""
Solve a non-interacting three site problem and calculate
its two-particle Green's function with both ED and Wicks theorem. 

Author: Hugo U. R. Strand (2017), hugo.strand@gmail.com
"""

#----------------------------------------------------------------------

import numpy as np

#----------------------------------------------------------------------

from triqs.gf import Gf, GfImTime
from triqs.gf import MeshImTime, MeshProduct

from triqs.operators import c, c_dag

#----------------------------------------------------------------------

from pyed.CubeTetras import zero_outer_planes_and_equal_times
from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

#----------------------------------------------------------------------
def test_two_particle_greens_function():

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
    # -- single particle Green's functions

    g_tau = GfImTime(name=r'$g$', beta=beta,
                     statistic='Fermion', n_points=100,
                     target_shape=(1,1))
    
    ed.set_g2_tau(g_tau[0, 0], c(up,0), c_dag(up,0))
    
    # ------------------------------------------------------------------
    # -- Two particle Green's functions

    ntau = 10
    imtime = MeshImTime(beta, 'Fermion', ntau)
    prodmesh = MeshProduct(imtime, imtime, imtime)

    g40_tau = Gf(name='g40_tau', mesh=prodmesh, target_shape=(1,1,1,1))
    g4_tau = Gf(name='g4_tau', mesh=prodmesh, target_shape=(1,1,1,1))

    ed.set_g40_tau_matrix(g40_tau, g_tau)
    ed.set_g4_tau(g4_tau[0, 0, 0, 0], c(up,0), c_dag(up,0), c(up,0), c_dag(up,0))

    # ------------------------------------------------------------------
    # -- compare

    zero_outer_planes_and_equal_times(g4_tau)
    zero_outer_planes_and_equal_times(g40_tau)
    np.testing.assert_array_almost_equal(g4_tau.data, g40_tau.data)
    
#----------------------------------------------------------------------
if __name__ == '__main__':

    test_two_particle_greens_function()
