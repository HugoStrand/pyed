# ----------------------------------------------------------------------

import itertools
import numpy as np

# ----------------------------------------------------------------------

import matplotlib.pyplot as plt

# ----------------------------------------------------------------------

from triqs.gf import Gf
from h5 import HDFArchive
from triqs.plot.mpl_interface import oplotr

# ----------------------------------------------------------------------
if __name__ == '__main__':

    with HDFArchive('data_ed.h5', 'r') as res:
        oplotr(res['tot'], name='pyed')

    with HDFArchive('spinless.ed.h5', 'r') as res:
        oplotr(res['tot'], name='ed')

    plt.savefig('figure_ed_vs_pyed.pdf')
    plt.show()
