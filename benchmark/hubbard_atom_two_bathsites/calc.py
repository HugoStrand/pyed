  
""" Test calculation for Hubbard atom with two bath sites.

Author: Hugo U.R. Strand (2017) hugo.strand@gmail.com

 """ 

# ----------------------------------------------------------------------

import itertools
import numpy as np

# ----------------------------------------------------------------------

#from pyed.FockStates import ManyBodyOperatorFactory

from pyed.ExactDiagonalization import ExactDiagonalization
from pyed.SparseMatrixFockStates import SparseMatrixRepresentation

from pyCollocation.Matsubara import matsubara_frequencies

# ----------------------------------------------------------------------

from pytriqs.operators import c, c_dag

# ----------------------------------------------------------------------

from pytriqs.gf import MeshImTime, MeshProduct

# ----------------------------------------------------------------------
class TriqsED(object):
    
    """ Exact diagonalization wrapper for Triqs operator expressions. """

    # ------------------------------------------------------------------
    def __init__(self, H, fundamental_operators, beta):

        self.beta = beta
        self.rep = SparseMatrixRepresentation(fundamental_operators)
        self.ed = ExactDiagonalization(self.rep.sparse_matrix(H), beta)

    # ------------------------------------------------------------------
    def set_g2_tau(self, g_tau, op1, op2):

        
        assert( type(g_tau.mesh) == MeshImTime )
        assert( self.beta == g_tau.mesh.beta )
        assert( g_tau.target_shape == (1, 1) )
    
        op1_mat = self.rep.sparse_matrix(op1)
        op2_mat = self.rep.sparse_matrix(op2)        

        tau = np.array([tau for tau in g_tau.mesh])

        g_tau.data[:, 0, 0] = \
            self.ed.get_tau_greens_function_component(
                tau, op1_mat, op2_mat)

        self.set_tail(g_tau, op1_mat, op2_mat)
        
    # ------------------------------------------------------------------
    def set_g2_iwn(self, g_iwn, op1, op2):

        assert( self.beta == g_iwn.mesh.beta )
        assert( g_iwn.target_shape == (1, 1) )
    
        op1_mat = self.rep.sparse_matrix(op1)
        op2_mat = self.rep.sparse_matrix(op2)        

        iwn = np.array([iwn for iwn in g_iwn.mesh])
        
        g_iwn.data[:, 0, 0] = \
            self.ed.get_frequency_greens_function_component(
                iwn, op1_mat, op2_mat, self.xi(g_iwn.mesh))

        self.set_tail(g_iwn, op1_mat, op2_mat)

    # ------------------------------------------------------------------
    def set_tail(self, g, op1_mat, op2_mat):

        tail = g.tail

        tail.data[:tail.order_max, 0, 0] = \
            self.ed.get_high_frequency_tail_coeff_component(
            op1_mat, op2_mat,
            self.xi(g_tau.mesh), Norder=tail.order_max)
            
    # ------------------------------------------------------------------
    def xi(self, mesh):
        if mesh.statistic == 'Fermion': return -1.0
        elif mesh.statistic == 'Boson': return +1.0
        else: raise NotImplementedError

    # ------------------------------------------------------------------
    def set_g40_tau(self, g40_tau, g_tau):

        assert( type(g40_tau.mesh) == MeshProduct )
        assert( type(g_tau.mesh) == MeshImTime )

        mesh_prod = g40_tau.mesh

        # why is a product mesh a list of tuples of meshes?
        
        for mesh_tuple in mesh_prod:
            assert( len(mesh_tuple) == 1 )
            mesh = mesh_tuple[0]
            assert( type(mesh) == MeshImTime )
            assert( mesh.beta == g_tau.mesh.beta )
        
        assert( g_tau.target_shape == g40_tau.target_shape )

        meshes = [ mesh_tuple[0] for mesh_tuple in mesh_prod ]

        for t in g_tau.mesh:
            print t, g_tau(t.real)
            print t, g_tau[[t]]
        
        
        for t1, t2, t3 in itertools.product(*meshes):
            g40_tau[[t1, t2, t3]] = \
                g_tau[[t1-t2]]*g_tau[[t3]] - g_tau[[t1]]*g_tau[[t3-t2]]
        
   # ------------------------------------------------------------------
        #----------------------------------------------------------------------
if __name__ == '__main__':
    
    # ------------------------------------------------------------------
    # -- Hubbard atom with two bath sites, Hamiltonian
    
    beta = 2.0
    V1 = 2.0
    V2 = 5.0
    epsilon1 = 0.00
    epsilon2 = 4.00
    mu = 2.0
    U = 1.0

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
    # -- Green's functions

    from pytriqs.gf import Gf
    from pytriqs.gf import MeshImTime, MeshProduct
    from pytriqs.gf import GfImTime, GfImFreq

    g_tau = GfImTime(name='g_tau', beta=beta,
                     statistic='Fermion', n_points=100,
                     indices=[1])

    g_iwn = GfImFreq(name='g_tau', beta=beta,
                     statistic='Fermion', n_points=10,
                     indices=[1])
    
    fundamental_operators = [
        c(up,0), c(do,0), c(up,1), c(do,1), c(up,2), c(do,2)]
    
    ed = TriqsED(H, fundamental_operators, beta)

    ed.set_g2_tau(g_tau, c(up,0), c_dag(up,0))
    ed.set_g2_iwn(g_iwn, c(up,0), c_dag(up,0))

    ntau = 10
    imtime = MeshImTime(beta, 'Fermion', ntau)
    prodmesh = MeshProduct([imtime, imtime, imtime])
    g40_tau = Gf(name='g40_tau', mesh=prodmesh, indices=[1])

    ed.set_g40_tau(g40_tau, g_tau)

    set
    exit()
    
    from pytriqs.plot.mpl_interface import oplot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 7))
    subp = [2, 1, 1]
    
    plt.subplot(*subp); subp[-1] += 1
    oplot(g_tau)
    plt.subplot(*subp); subp[-1] += 1
    oplot(g_iwn)

    plt.show()
    
    exit()
    
    ntau = 50
    eps = 1e-4 * beta
    tau = np.linspace(0 + eps, beta - eps, num=ntau)

    op_vec = [op.c[0]]
    g_tau = ed.get_tau_greens_function(tau, op_vec)

    nw = 10
    xi = -1.0
    iwn = matsubara_frequencies(nw, beta, xi=xi)
    g_iwn = ed.get_frequency_greens_function(iwn, op_vec, xi)
        
    ntau2 = 5
    tau2 = np.linspace(0+eps, beta-eps, num=ntau2)

    print '--> Dissconnected two-particle Greens function'
    g20_tau = ed.get_g2_dissconnected_tau(tau2, tau, g_tau)

    ops = [op.c[0], op.cdagger[0], op.c[0], op.cdagger[0]]
    g2_tau = ed.get_g2_tau(tau2, ops)

    #exit()
    
    # ------------------------------------------------------------------
    # -- Plotting

    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('pyed')
    plt.plot(tau, g_tau[0, 0], '-g')
    plt.tight_layout()

    plt.figure()
    plt.title('pyed')
    plt.plot(iwn.imag, g_iwn[:, 0, 0].real, '.-g')
    plt.plot(iwn.imag, g_iwn[:, 0, 0].imag, '.-b')
    plt.tight_layout()
    plt.show()
