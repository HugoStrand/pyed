  
""" Test calculation for Hubbard atom with two bath sites.

Author: Hugo U.R. Strand (2017) hugo.strand@gmail.com

 """ 

# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------

from pytriqs.gf import Gf
from pytriqs.gf import MeshImTime, MeshImFreq

from pytriqs.gf import GfImTime, GfImFreq, TailGf
from pytriqs.operators import c, c_dag

from pytriqs.gf import inverse, iOmega_n, InverseFourier

# ----------------------------------------------------------------------

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization

# ----------------------------------------------------------------------
def test_cf_G_tau_and_G_iw_nonint(verbose=False):

    beta = 3.22
    eps = 1.234

    niw = 64
    ntau = 2 * niw + 1
    
    H = eps * c_dag(0,0) * c(0,0)

    fundamental_operators = [c(0,0)]

    ed = TriqsExactDiagonalization(H, fundamental_operators, beta)

    # ------------------------------------------------------------------
    # -- Single-particle Green's functions

    G_tau = GfImTime(beta=beta, statistic='Fermion', n_points=ntau, target_shape=(1,1))
    G_iw = GfImFreq(beta=beta, statistic='Fermion', n_points=niw, target_shape=(1,1))

    G_iw << inverse( iOmega_n - eps )
    G_tau << InverseFourier(G_iw)

    G_tau_ed = GfImTime(beta=beta, statistic='Fermion', n_points=ntau, target_shape=(1,1))
    G_iw_ed = GfImFreq(beta=beta, statistic='Fermion', n_points=niw, target_shape=(1,1))

    ed.set_g2_tau(G_tau_ed[0, 0], c(0,0), c_dag(0,0))
    ed.set_g2_iwn(G_iw_ed[0, 0], c(0,0), c_dag(0,0))

    # ------------------------------------------------------------------
    # -- Compare gfs

    from pytriqs.utility.comparison_tests import assert_gfs_are_close
    
    assert_gfs_are_close(G_tau, G_tau_ed)
    assert_gfs_are_close(G_iw, G_iw_ed)
    
    # ------------------------------------------------------------------
    # -- Plotting
    
    if verbose:
        from pytriqs.plot.mpl_interface import oplot, plt
        subp = [3, 1, 1]
        plt.subplot(*subp); subp[-1] += 1
        oplot(G_tau.real)
        oplot(G_tau_ed.real)

        plt.subplot(*subp); subp[-1] += 1
        diff = G_tau - G_tau_ed
        oplot(diff.real)
        
        plt.subplot(*subp); subp[-1] += 1
        oplot(G_iw)
        oplot(G_iw_ed)
        
        plt.show()

# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    test_cf_G_tau_and_G_iw_nonint(verbose=True)
