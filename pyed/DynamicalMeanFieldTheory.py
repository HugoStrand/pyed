import numpy as np
from pytriqs.gf import *
from itertools import product
from scipy.optimize import fmin_l_bfgs_b
from pytriqs.operators import Operator, c, c_dag, n
from TriqsExactDiagonalization import TriqsExactDiagonalization

class DynamicalMeanFieldTheory(object):
    """docstring fs DynamicalMeanFieldTheory."""
    def __init__(self,Hloc,G0,beta,nbath):
        self.norb=G0.data.shape[1]
        self.nbath=nbath
        assert(self.nbath%self.norb==0)
        self.BPO=self.nbath/self.norb # baths per orb
        self.beta=beta
        self.G0=G0
        self.h=np.zeros((self.norb,self.nbath))
        self.ek=np.zeros(self.nbath)
        self.em=np.zeros(self.norb)
        self.nmatsubara=(np.array([iwn for iwn in self.G0.mesh])).size/2
        for i in range(self.norb):
            parameters=self._bath_fit(i)
            self.h[i,self.BPO*i:self.BPO*(i+1)]=parameters[0:self.BPO]
            self.ek[self.BPO*i:self.BPO*(i+1)] =parameters[self.BPO:-1]
            self.em[i]=parameters[-1]

        fundamental_operators = np.array([[c('up',i), c('dn',i)] for i in range(self.norb+self.nbath)]).flatten()
        self.Hkin = sum(self.h[i][j]*c_dag(s,i)*c(s,j+self.norb) for s, i,j in product(['up','dn'], range(self.norb),range(self.nbath)))
        self.Hkin+= sum(self.ek[i]*c_dag(s,i+self.norb)*c(s,i+self.norb) for s,i in product(['up','dn'],range(self.nbath)))
        self.Hkin+= sum(self.em[i]*c_dag(s,i)*c(s,i) for s,i in product(['up','dn'],range(self.norb)))
        self.Hloc=Hloc
        self.H=self.Hkin+self.Hloc

        self.ed = TriqsExactDiagonalization(self.H,fundamental_operators, self.beta,nstates=None)
# ------------------------------------------------------------------
    def _molecular_GF(self,parameters):
        iwn = np.array([iwn for iwn in self.G0.mesh])
        h  = parameters[0:self.BPO]
        ek = parameters[self.BPO:-1]
        em = parameters[-1]
        fitG0 = np.zeros(iwn.size, dtype=np.complex128)
        for i in xrange(iwn.size): fitG0[i] = (iwn[i] - em - np.sum(h ** 2 / (iwn[i] - ek))) ** (-1)
        return fitG0
# ------------------------------------------------------------------
    def _bath_fit(self,i):
        error=lambda parameters:np.sum(np.abs(np.conj(self.G0.data[:,i,i].flatten()-self._molecular_GF(parameters))*(self.G0.data[:,i,i].flatten() - self._molecular_GF(parameters))))
        return fmin_l_bfgs_b(error, x0=2*np.random.random(self.BPO*2+1)-1, approx_grad=True, disp=True)[0]
# ------------------------------------------------------------------
    def get_iwn_GF(self):
        G = GfImFreq(indices = range(self.norb), beta = self.beta, n_points = self.nmatsubara)
        index_combinations=[(i,j) for i,j in product(range(self.norb),range(self.norb))]
        for k in range(len(index_combinations)):
            i,j=index_combinations[k]
            g_iwn=GfImFreq(indices = [0], beta = self.beta, n_points = self.nmatsubara)
            self.ed.set_g2_iwn(g_iwn,c('up',i),c_dag('up',j))
            G.data[:,i,j]=g_iwn.data.flatten()
        return G
# ------------------------------------------------------------------
    def get_w_GF(self,omega):
        G = GfReFreq(indices = range(self.norb), window = (np.min(omega), np.max(omega)), n_points = omega.size)
        index_combinations=[(i,j) for i,j in product(range(self.norb),range(self.norb))]
        for k in range(len(index_combinations)):
            i,j=index_combinations[k]
            g_w=GfReFreq(indices = [0], window = (np.min(omega), np.max(omega)), n_points = omega.size)
            self.ed.set_g2_w(g_w, c('up',i), c_dag('up',j))
            G.data[:,i,j]=g_w.data.flatten()
        return G
# ------------------------------------------------------------------
