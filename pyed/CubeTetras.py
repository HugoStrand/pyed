# ----------------------------------------------------------------------

""" Helper routines for the equal time imaginary time cube and
its sub tetrahedrons.

Author: Hugo U. R. Strand (2017), hugo.strand@gmail.com """

# ----------------------------------------------------------------------

import itertools
import numpy as np

# ----------------------------------------------------------------------
class CubeTetras(object):

    """ Helper class for two-particle Green's function.
    
    Looping over all tetrahedrons in the imaginary time cube.
    \tau_1, \tau_2, \tau_3 \in [0, \beta) 

    """
    
    # ------------------------------------------------------------------
    def __init__(self, tau):

        self.tau = tau
        self.ntau = len(tau)
        
        # (points in tetra, index order permut, permut sign)
        self.tetra_list = [
            (lambda x,y,z : x > y and y > z, [0, 1, 2], +1),
            (lambda x,y,z : y > x and x > z, [1, 0, 2], -1),
            (lambda x,y,z : y > z and z > x, [1, 2, 0], +1),
            (lambda x,y,z : z > y and y > x, [2, 1, 0], -1),
            (lambda x,y,z : x > z and z > y, [0, 2, 1], -1),
            (lambda x,y,z : z > x and x > y, [2, 0, 1], +1),
            ]
        
    # ------------------------------------------------------------------
    def __iter__(self):

        for tidx in xrange(6):
            
            func, perm, perm_sign = self.tetra_list[tidx]
    
            index = []
            for n1, n2, n3 in itertools.product(
                    range(self.ntau), repeat=3):
                if func(n1, n2, n3): index.append((n1, n2, n3))

            index = np.array(index).T
            
            i1, i2, i3 = index
            t1, t2, t3 = self.tau[i1], self.tau[i2], self.tau[i3]

            taus = np.vstack([t1, t2, t3])
        
            yield list(index), taus, perm, perm_sign

# ----------------------------------------------------------------------
def tetra_index(N, tetra_index=0):

    """ Get indices for one of the six tetrahedra. """
    
    tetra_list = [
        (lambda x,y,z : x > y and y > z, [0, 1, 2], +1),
        (lambda x,y,z : y > x and x > z, [1, 0, 2], -1),
        #
        (lambda x,y,z : y > z and z > x, [1, 2, 0], +1),
        (lambda x,y,z : z > y and y > x, [2, 1, 0], -1),
        #
        (lambda x,y,z : x > z and z > y, [0, 2, 1], -1),
        (lambda x,y,z : z > x and x > y, [2, 0, 1], +1),
        #
        #(lambda x,y,z : x >= y and y >= z, [0, 1, 2], +1),
        #(lambda x,y,z : y >= x and x >= z, [1, 0, 2], -1),
        #
        #(lambda x,y,z : y >= z and z >= x, [1, 2, 0], +1),
        #(lambda x,y,z : z >= y and y >= x, [2, 1, 0], -1),
        #
        #(lambda x,y,z : x >= z and z >= y, [0, 2, 1], -1),
        #(lambda x,y,z : z >= x and x >= y, [2, 0, 1], +1),
        ]

    func, perm, sign = tetra_list[tetra_index]
    
    index = []
    for n1, n2, n3 in itertools.product(range(N), repeat=3):
        if func(n1, n2, n3): index.append((n1, n2, n3))

    return np.array(index).T, perm, sign
