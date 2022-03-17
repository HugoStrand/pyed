
"""
Helper routines for the equal time imaginary time cube and
its sub tetrahedrons.

Author: Hugo U. R. Strand (2017), hugo.strand@gmail.com
"""

# ----------------------------------------------------------------------

import itertools
import numpy as np

# ----------------------------------------------------------------------


def Idxs(integer_index_list):
    from triqs.gf import Idx
    return tuple(Idx(i) for i in integer_index_list)

# ----------------------------------------------------------------------


def zero_outer_planes_and_equal_times(g4_tau):

    from triqs.gf import Idx
    beta = g4_tau.mesh.components[0].beta

    for idxs, (t1, t2, t3) in enumerate_tau3(g4_tau):
        if t1 == t2 or t2 == t3 or t1 == t3 or \
           t1 == 0 or t1 == beta or \
           t2 == 0 or t2 == beta or \
           t3 == 0 or t3 == beta:
            g4_tau[Idxs(idxs)] = 0.0

# ----------------------------------------------------------------------


def enumerate_tau3(g4_tau, make_real=True, beta=None):

    from triqs.gf import MeshImTime, MeshProduct

    assert(isinstance(g4_tau.mesh, MeshProduct))

    for mesh in g4_tau.mesh.components:
        assert(isinstance(mesh, MeshImTime))
        if beta is not None:
            assert(mesh.beta == beta)

    for (i1, t1), (i2, t2), (i3, t3) in itertools.product(*[
            enumerate(mesh) for mesh in g4_tau.mesh.components]):
        if make_real:
            yield (i1, i2, i3), (t1.real, t2.real, t3.real)
        else:
            yield (i1, i2, i3), (t1, t2, t3)

# ----------------------------------------------------------------------


class CubeTetrasBase(object):

    """ Base class with definition of the equal time tetrahedrons
    in three fermionic imaginary times. """

    def get_tetra_list(self):

        tetra_list = [
            (lambda x, y, z: x >= y and y >= z, [0, 1, 2], +1),
            (lambda x, y, z: y >= x and x >= z, [1, 0, 2], -1),
            (lambda x, y, z: y >= z and z >= x, [1, 2, 0], +1),
            (lambda x, y, z: z >= y and y >= x, [2, 1, 0], -1),
            (lambda x, y, z: x >= z and z >= y, [0, 2, 1], -1),
            (lambda x, y, z: z >= x and x >= y, [2, 0, 1], +1),
        ]

        return tetra_list

# ----------------------------------------------------------------------


class CubeTetras(CubeTetrasBase):

    """ Helper class for two-particle Green's function.

    Looping over all tetrahedrons in the imaginary time cube.
    \tau_1, \tau_2, \tau_3 \\in [0, \beta) """

    # ------------------------------------------------------------------
    def __init__(self, tau):

        self.tau = tau
        self.ntau = len(tau)
        self.tetra_list = self.get_tetra_list()

    # ------------------------------------------------------------------
    def __iter__(self):

        for tidx in range(6):

            func, perm, perm_sign = self.tetra_list[tidx]

            index = []
            for n1, n2, n3 in itertools.product(
                    list(range(self.ntau)), repeat=3):
                if func(n1, n2, n3):
                    index.append((n1, n2, n3))

            index = np.array(index).T

            i1, i2, i3 = index
            t1, t2, t3 = self.tau[i1], self.tau[i2], self.tau[i3]

            taus = np.vstack([t1, t2, t3])

            yield list(index), taus, perm, perm_sign

# ----------------------------------------------------------------------


class CubeTetrasMesh(CubeTetrasBase):

    """ Helper class for Triqs two-particle Green's function
    in imaginary time.

    Looping over all tetrahedrons in the imaginary time cube.
    \tau_1, \tau_2, \tau_3 \\in [0, \beta) """

    # ------------------------------------------------------------------
    def __init__(self, g4_tau):

        self.g4_tau = g4_tau
        self.tetra_list = self.get_tetra_list()

    # ------------------------------------------------------------------
    def __iter__(self):
        """ for triqs three time greens functions """

        tetra_idx = [[] for n in range(6)]
        tetra_tau = [[] for n in range(6)]

        for idxs, taus in enumerate_tau3(self.g4_tau):

            for tidx, tetra in enumerate(self.tetra_list):
                func, perm, perm_sign = tetra

                if func(*taus):
                    tetra_idx[tidx] += [idxs]
                    tetra_tau[tidx] += [taus]
                    break

        for tidx in range(6):
            func, perm, perm_sign = self.tetra_list[tidx]

            yield tetra_idx[tidx], tetra_tau[tidx], perm, perm_sign

# ----------------------------------------------------------------------
