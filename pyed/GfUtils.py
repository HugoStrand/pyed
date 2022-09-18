
"""
Triqs Green's function utilities

Author: Hugo U. R. Strand (2018), hugo.strand@gmail.com
"""
# ----------------------------------------------------------------------

import numpy as np

# ----------------------------------------------------------------------


def g2_single_particle_transform(g2, T):
    """ Transform the single-particle Green's function according to the
    give unitary single particle transform matrix T """

    g2t = g2.copy()
    g2t.from_L_G_R(T, g2, T.T.conjugate())

    return g2t

# ----------------------------------------------------------------------


def g4_single_particle_transform(g4, T):
    """ Transform the two-particle Green's function according to the
    give unitary single particle transform matrix T """

    g4t = g4.copy()
    g4t.data[:] = np.einsum('ai,bj,ck,dl,PQRijkl->PQRabcd',
                            T, T.conjugate(), T, T.conjugate(), g4.data)

    return g4t
