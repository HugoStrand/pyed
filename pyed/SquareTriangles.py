
""" 
Helper routines for the equal time imaginary time square and
its sub triangles.

Author: Hugo U. R. Strand (2017), hugo.strand@gmail.com
"""

# ----------------------------------------------------------------------

import itertools
import numpy as np

# ----------------------------------------------------------------------
def zero_outer_planes_and_equal_times(g3_tau):

    beta = g3_tau.mesh.components[0].beta
    
    for idxs, (t1, t2) in enumerate_tau2(g3_tau):
        if t1 == t2 or \
           t1 == 0 or t1 == beta or \
           t2 == 0 or t2 == beta:
            g3_tau[list(idxs)][:] = 0.0

# ----------------------------------------------------------------------
def enumerate_tau2(g3_tau, make_real=True, beta=None):

    from pytriqs.gf import MeshImTime, MeshProduct
    
    assert( type(g3_tau.mesh) == MeshProduct )

    for mesh in g3_tau.mesh.components:
        assert( type(mesh) == MeshImTime )
        if beta is not None: assert( mesh.beta == beta )

    for (i1, t1), (i2, t2) in itertools.product(*[
            enumerate(mesh) for mesh in g3_tau.mesh.components]):
        if make_real:
            yield (i1, i2), (t1.real, t2.real)
        else:
            yield (i1, i2), (t1, t2)
            
# ----------------------------------------------------------------------
class SquareTrianglesBase(object):

    """ Base class with definition of the equal time tetrahedrons
    in three fermionic imaginary times. """
    
    def get_triangle_list(self):

        triangle_list = [
            (lambda x,y : x >= y, [0, 1], +1),
            (lambda x,y : x <  y, [1, 0], -1),
            ]
        
        return triangle_list

# ----------------------------------------------------------------------
class SuqareTraingles(SquareTrianglesBase):

    """ Helper class for two imaginary time Green's functions.
    
    Looping of the triangles on the imaginary time square.
    \tau_1, \tau_2 \in [0, \beta) """
    
    # ------------------------------------------------------------------
    def __init__(self, tau):

        self.tau = tau
        self.ntau = len(tau)
        self.triangle_list = self.get_triangle_list()
        self.N = len(self.triangle_list)

    # ------------------------------------------------------------------
    def __iter__(self):

        for tidx in range(self.N):
            
            func, perm, perm_sign = self.triangle_list[tidx]
    
            index = []
            for n1, n2 in itertools.product(
                    list(range(self.ntau)), repeat=2):
                if func(n1, n2): index.append((n1, n2))

            index = np.array(index).T
            
            i1, i2 = index
            t1, t2 = self.tau[i1], self.tau[i2]

            taus = np.vstack([t1, t2])
        
            yield list(index), taus, perm, perm_sign

# ----------------------------------------------------------------------
class SquareTrianglesMesh(SquareTrianglesBase):

    """ Helper class for Triqs three imaginary time Green's functions.
    
    Looping of the triangles on the imaginary time square.
    \tau_1, \tau_2 \in [0, \beta) """
    
    # ------------------------------------------------------------------
    def __init__(self, g3_tau):

        self.g3_tau = g3_tau
        self.triangle_list = self.get_triangle_list()
        self.N = len(self.triangle_list)
        
    # ------------------------------------------------------------------
    def __iter__(self):

        """ for pytriqs three time greens functions """

        triangle_idx = [ [] for n in range(self.N) ]
        triangle_tau = [ [] for n in range(self.N) ]

        for idxs, taus in enumerate_tau2(self.g3_tau):
            
            for tidx, triangle in enumerate(self.triangle_list):
                func, perm, perm_sign = triangle

                if func(*taus):
                    triangle_idx[tidx] += [ idxs ]
                    triangle_tau[tidx] += [ taus ]
                    break

        for tidx in range(self.N):
            func, perm, perm_sign = self.triangle_list[tidx]

            yield triangle_idx[tidx], triangle_tau[tidx], perm, perm_sign
            
# ----------------------------------------------------------------------
