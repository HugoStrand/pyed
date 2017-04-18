# ----------------------------------------------------------------------

import glob
import cPickle
import itertools
import numpy as np

# ----------------------------------------------------------------------

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------

from pytriqs.gf import Gf
from pytriqs.archive import HDFArchive
from pytriqs.plot.mpl_interface import oplot

# ----------------------------------------------------------------------

from pyed.CubeTetras import zero_outer_planes_and_equal_times
            
# ----------------------------------------------------------------------
def plot_2d_g(ax, g2_tau, **kwargs):

    data = g2_tau.data[:, :, 0, 0]
    x = np.arange(data.shape[-1])
    X, Y = np.meshgrid(x, x)
    ax.plot_wireframe(X, Y, data.real, **kwargs)
    
# ----------------------------------------------------------------------
if __name__ == '__main__':
    
    A = HDFArchive('data_ed.h5', 'r')

    g_tau = A['G_tau']
    tau = np.array([tau for tau in g_tau.mesh])

    plt.figure(figsize=(3, 2))
    oplot(g_tau)
    plt.tight_layout()
    
    # -- Tow time gf
    g4_tau = A['G2_tau']
    g40_tau = A['G20_tau']

    zero_outer_planes_and_equal_times(g4_tau)
    zero_outer_planes_and_equal_times(g40_tau)
    np.testing.assert_array_almost_equal(g4_tau.data, g40_tau.data)

    fig = plt.figure(figsize=(12, 5))
    subp = [1, 3, 1]
    
    # -- All slice planes
    for i1, i2 in itertools.combinations(range(3), 2):

        cut_idx = 10
        plane_slice = [cut_idx]*3
        plane_slice[i1], plane_slice[i2] = all, all

        ax = fig.add_subplot(*subp, projection='3d')
        subp[-1] += 1

        plot_2d_g(ax, g4_tau[plane_slice],
                  label='g4', alpha=0.5, color='b')

        plot_2d_g(ax, g40_tau[plane_slice],
                  label='g40', alpha=0.5, color='g')


    plt.legend()
    plt.tight_layout()
    plt.show()
    
# ----------------------------------------------------------------------
