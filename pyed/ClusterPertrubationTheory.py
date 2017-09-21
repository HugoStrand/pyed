
"""
General routines for cluster pertrubation theory

Author: Yaroslav Zhumagulov (2017), yaroslav.zhumagulov@gmail.com

New classes will be soon
"""

import numpy as np
from pytriqs.gf import *
from itertools import product
import progressbar
from pytriqs.operators import c, c_dag,n
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
class ClusterPertrubationTheory_2D_Square(object):

    """ Cluster Pertrubation Theory calculator of band structure and Fermi surface of two-dimensional square systems.
        Parameters:
        ed - TriqsExactDiagonalization object
        k_mesh - tuple of kx and ky meshgrid, kx range (-pi,0,pi),ky range (-pi,0,pi)
        V - pertrubation matrix: shape = (N,N,L,L), where L - number of sites of the system, (N,N) - size of kx or ky meshgrid
        omega - frequency meshgrid
        shape - shape of square cluster"""
# ------------------------------------------------------------------
    def __init__(self,ed,k_mesh,V,omega,shape):
        self.ed=ed
        self.kx,self.ky=k_mesh
        self.V=V
        self.N=self.V.shape[0];self.L=self.V.shape[2]
        self.omega=omega
        self.shape=shape
        self._get_green_of_the_system()
        self._coupling_system()
        self._reduce_mix_representation()
# ------------------------------------------------------------------
    def _get_green_of_the_system(self):
        self.G_I=np.zeros((self.L,self.L,self.omega.size),dtype=np.complex)
        print "Calculation green function of full system"
        index_combinations=[(i,j) for i,j in product(range(self.L),range(self.L))]
        bar = progressbar.ProgressBar()
        for k in bar(range(len(index_combinations))):
            i,j=index_combinations[k]
            g_w=GfReFreq(indices = [0], window = (np.min(self.omega), np.max(self.omega)), n_points = self.omega.size)
            self.ed.set_g2_w(g_w, c('up',i), c_dag('up',j))
            self.G_I[i,j]=g_w.data.flatten()
# ------------------------------------------------------------------
    def _coupling_system(self):
        self.G_Q=np.zeros((self.N,self.N,self.L,self.L,self.omega.size),dtype=np.complex)
        index_combinations=[(i,j,k) for i,j,k in product(range(self.N),range(self.N),range(self.omega.size))]
        print "Coupling system"
        bar = progressbar.ProgressBar()
        for l in bar(range(len(index_combinations))):
            i,j,k=index_combinations[l]
            self.G_Q[i,j,:,:,k]=np.dot(self.G_I[:,:,k],np.linalg.inv(np.eye(self.L)-np.dot(self.V[i,j],self.G_I[:,:,k])))
# ------------------------------------------------------------------
    def _reduce_mix_representation(self):
        self.G=np.zeros((self.N,self.N,self.omega.size),dtype=np.complex)
        index_combinations=[(i,j,a,b) for i,j,a,b in product(range(self.N),range(self.N),range(self.L),range(self.L))]
        print "Reduce mixed representation"
        bar = progressbar.ProgressBar()
        for k in bar(range(len(index_combinations))):
            i,j,a,b=index_combinations[k]
            x = a % self.shape[0] - b % self.shape[1]
            y = a //self.shape[0] - b //self.shape[1]
            self.G[i,j]+=np.exp(-1j*self.kx[i,j]*x)*np.exp(-1j*self.ky[i,j]*y)*self.G_Q[i,j,a,b]
# ------------------------------------------------------------------
    def calculation_bandstructure(self):
        bandstructure=[]
        for i in range(self.N/2,self.N,1):bandstructure.append(-self.G[i,i,:].imag/np.pi)
        for i in range(self.N-1,self.N/2,-1):bandstructure.append(-self.G[self.N-1,i,:].imag/np.pi)
        for i in range(self.N-1,self.N/2,-1):bandstructure.append(-self.G[i,self.N/2,:].imag/np.pi)
        self.bandstructure=np.array(bandstructure).T

# ------------------------------------------------------------------
    def calculation_Fermi_surface(self):
        self.FS=-self.G[:,:,np.argmin(abs(self.omega))].imag/np.pi


class ClusterPertrubationTheory_1D(object):
    """ Cluster Pertrubation Theory calculator of band structure of one-dimensional systems.
        Parameters:
        ed - TriqsExactDiagonalization object
        k_mesh -  kx meshgrid, kx range (-pi,0,pi)
        V - pertrubation matrix: shape = (N,L,L), where L - number of sites of the system, N - size of kx meshgrid
        omega - frequency meshgrid
        shape - len of cluster"""
# ------------------------------------------------------------------
    def __init__(self,ed,k_mesh,V,omega,shape):
        self.ed=ed
        self.kx,self.ky=k_mesh
        self.V=V
        self.N=self.V.shape[0];self.L=self.V.shape[2]
        self.omega=omega
        self.shape=shape
        self._get_green_of_the_system()
        self._coupling_system()
        self._reduce_mix_representation()
# ------------------------------------------------------------------
    def _get_green_of_the_system(self):
        self.G_I=np.zeros((self.L,self.L,self.omega.size),dtype=np.complex)
        print "Calculation green function of full system"
        index_combinations=[(i,j) for i,j in product(range(self.L),range(self.L))]
        bar = progressbar.ProgressBar()
        for k in bar(range(len(index_combinations))):
            i,j=index_combinations[k]
            g_w=GfReFreq(indices = [0], window = (np.min(self.omega), np.max(self.omega)), n_points = self.omega.size)
            self.ed.set_g2_w(g_w, c('up',i), c_dag('up',j))
            self.G_I[i,j]=g_w.data.flatten()
# ------------------------------------------------------------------
    def _coupling_system(self):
        self.G_Q=np.zeros((self.N,self.L,self.L,self.omega.size),dtype=np.complex)
        index_combinations=[(i,j) for i,j in product(range(self.N),range(self.omega.size))]
        print "Coupling system"
        bar = progressbar.ProgressBar()
        for k in bar(range(len(index_combinations))):
            i,j=index_combinations[l]
            self.G_Q[i,:,:,j]=np.dot(self.G_I[:,:,j],np.linalg.inv(np.eye(self.L)-np.dot(self.V[i],self.G_I[:,:,j])))
# ------------------------------------------------------------------
    def _reduce_mix_representation(self):
        self.G=np.zeros((self.N,self.omega.size),dtype=np.complex)
        index_combinations=[(i,a,b) for i,j,a,b in product(range(self.N),range(self.L),range(self.L))]
        print "Reduce mixed representation"
        bar = progressbar.ProgressBar()
        for k in bar(range(len(index_combinations))):
            i,a,b=index_combinations[k]
            self.G[i]+=np.exp(-1j*k[i,j]*(a-b))*self.G_Q[i,a,b]
# ------------------------------------------------------------------
    def calculation_bandstructure(self):
        bandstructure=[]
        for i in self.range(L): bandstructure.append(-self.G[i].imag/np.pi)
        self.bandstructure=np.array(bandstructure).T
