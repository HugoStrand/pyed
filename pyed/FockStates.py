"""
Fock states module

Purpose:
    Set up sparse matrix representations for some common operators
    in the occupation number basis.

Author: Hugo U. R. Strand (2017), hugo.strand@gmail.com

Example usage:

>>> from pyGutz.FockStates import *
>>> print 'Creating Fock States'
>>> l = 1
>>> norb = 2*(2*l+1)    
>>> fs = FockStates(norb=norb)

>>> orb = 2
>>> print fs.c[orb]
>>> print fs.cdagger[orb]

>>> print 'Creating Spin Operators'
>>> so = SpinOperators(fs)
>>> print so.Sz

>>> print 'Creating Angular Momentum Operators'
>>> lo = AngularMomentumOperators(fs)
>>> print lo.Lz

>>> print 'Creating Many Body Operators'
>>> mbo = ManyBodyOperators(so)
>>> print mbo.density_density
"""

# ----------------------------------------------------------------------
import numpy as np
from scipy import sparse
from itertools import permutations, combinations

# ----------------------------------------------------------------------
def ManyBodyOperatorFactory(norbs=2):
    """ Convenience Routine """
    fs = FockStates(norbs)
    so = SpinOperators(fs)
    mbo = ManyBodyOperators(so)
    return mbo

# ----------------------------------------------------------------------
class FockStates:
    """Fock states matrix representations

    FockStates.c[orbital] - Annihilation operator of orbital (sparse matrices)
    FockStates.cdagger[orbital] - Creation operator of orbital (sparse matrices)
    FockStates.n[orbital] - Number operator of orbital (sparse matrices)
    FockStates.N - Total number operator (sparse matrices)
    FockStates.norb - Number of orbitals
    FockStates.nstates - Number of many body states
    
    """
    def __init__(self, norb=2*2):
        """ input parameter: norb - number of orbitals including spin
        """
        self.norb = norb

        nstates = self.get_nstates_from_norb(norb)
        self.nstates = nstates
        #print 'nstates =', nstates

        self._construct_c_cdagger_tensor_sparse()
        self._construct_number_operators()

    def get_nstates_from_norb(self, norb):
        return 2**(norb)

    # ------------------------------------------------------------------
    def get_creation_operator(self, orbidx):
        nstates = self.nstates

        # -- Make python based fock states
        numbers = np.arange(nstates, dtype=np.uint32)
        tmp = numbers.flatten().view(np.uint8).reshape((nstates, 4))
        tmp = np.fliplr(tmp)
        states = np.unpackbits(tmp, axis=1)

        # -- Apply creation operator
        orbocc = states[:, -1 - orbidx]
        rightstates = states[:, -1 - orbidx:]

        states_new = np.copy(states)
        states_new[:, -1 - orbidx] = 1

        # -- collect sign
        sign = 1 - 2*np.array(
            np.mod(np.sum(rightstates[:, 1:], axis=1), 2), 
            dtype=np.float64)

        # -- Transform back to uint16
        tmp_new = np.packbits(states_new, axis=1)
        tmp_new = np.fliplr(tmp_new).flatten()
        numbers_new = tmp_new.view(dtype=np.uint32)

        # -- Collect non-zero elements
        idx = orbocc == 0
        I = numbers_new[idx]
        J = numbers[idx]
        D = sign[idx]

        # -- Build sparse matrix repr.
        cdagger = sparse.coo_matrix((D,(I,J)), \
            shape=(nstates, nstates)).tocsr()

        return cdagger

    # ------------------------------------------------------------------
    def _construct_c_cdagger_tensor_sparse(self):
        sparse_cdagger = []
        sparse_c = []

        for orb in np.arange(self.norb):
            cdagger = self.get_creation_operator(orb)
            c = cdagger.getH()

            sparse_cdagger.append(cdagger)
            sparse_c.append(c)

        self.cdagger = sparse_cdagger
        self.c = sparse_c
        
    def _construct_number_operators(self):
        """Use creation and annihilation operator tensors to construct
        the number operators and tota number operator."""
        number_op = []
        N = sparse.csr_matrix((self.nstates, self.nstates))

        for orb in np.arange(self.norb):
            n = self.cdagger[orb] * self.c[orb]
            N = N + n
            number_op.append(n)
        
        self.n = number_op
        self.N = N

    def print_states(self):
        """ Print the full set of the occupation number basis """
        states = np.empty((self.norb, self.nstates))
        for orb  in np.arange(self.norb):
            states[orb] = self.n[orb].diagonal()

        print '--> FockState print_states'
        print states.T

# ----------------------------------------------------------------------
class SpinOperators:
    """
    Spin operator class applying the order,

    spin_orbital index:
        0 = spin_up orb 1,
        1 = spin_down orb 1,
        2 = spin_up orb 2,
        3 = spin_down orb 2,
        etc... 

    Input: An instance of the FockStates class

    Members:
        Sx, Sy, Sz, Sp, Sm, S2, Nup, Ndown, D (tot double occ), I (sparse matrices)
    """
    def __init__(self, fockstates):
        self.__dict__.update(fockstates.__dict__)
        
        # -> Spin operators, general def.
        # S_i = c^T * sigma_i * c
        # S_x =    0.5*(c*_u c_d + c*_d c_u)
        # S_y = -i*0.5*(c*_u c_d - c*_d c_u)
        # S_z =    0.5*(c*_u c_u - c*_d c_d)

        Sx = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')
        Sy = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')
        Sz = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')

        for cup, cdown, cdagup, cdagdown in zip(self.c[0::2], self.c[1::2],
                                                self.cdagger[0::2], self.cdagger[1::2]):
            
            Sx = Sx + np.complex(0.5, 0.0) * (cdagup * cdown + cdagdown * cup)
            Sy = Sy + np.complex(0.0,-0.5) * (cdagup * cdown - cdagdown * cup)
            Sz = Sz + np.complex(0.5, 0.0) * (cdagup * cup   - cdagdown * cdown)

        self.S2 = Sx**2 + Sy**2 + Sz**2
        self.Sx = Sx
        self.Sy = Sy
        self.Sz = Sz

        self.Sp = self.Sx + np.complex(0.0, 1.0) * self.Sy
        self.Sm = self.Sx - np.complex(0.0, 1.0) * self.Sy
        
        self.Nup = 0.5*self.N + self.Sz # Total number of spin up
        self.Ndown = 0.5*self.N - self.Sz # Total number of spin down

        # -- On-site double occupancy
        self.D = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')
        for nup, ndown in zip(self.n[0::2], self.n[1::2]):
            self.D = self.D + nup * ndown 

        self.I = sparse.eye(self.nstates, self.nstates, dtype='complex', format='csr')

# ----------------------------------------------------------------------
class AngularMomentumOperators:
    """
    Angular momentum operator class applying the order,

    spin_orbital index:
        0 = spin_up orb 1, l = -L
        1 = spin_down orb 1, l = -L
        2 = spin_up orb 2, l = -L + 1
        3 = spin_down orb 2, l = -L + 1
        etc... 

    Input: An instance of the FockStates class

    Members:
        Lx, Ly, Lz, Lp, Lm, L2 (sparse matrices)

    Many Body Angular Momentum operators
    Schirmer, J. and Cederbaum, L. S., Phys. Rev. A (16) 1575
        
    """
    def __init__(self, fockstates, use_cubic_harmonics=True):
        self.fs = fockstates
        self.nstates = self.fs.nstates
        self.norb = self.fs.norb

        print 'norb = ', self.norb
        l = (self.norb/2.0 - 1.0)/2.0
        print 'l =', l

        if l not in [0, 1, 2, 3]:
            raise ValueError('ERROR: Creating Angular Momentum Operators ' + \
                             'norb is not a multiple of 2*l + 1')
        
        l = int(l)
        self.l = l

        self._many_body_matrices(l, self.fs, \
            use_cubic_harmonics=use_cubic_harmonics)

        #self._many_body_matrices(self, l)

    # ------------------------------------------------------------------
        
    def _many_body_matrices_deprecated(self, l):
        mvec = np.arange(-l, l+1)
        print 'm = ', mvec
    
        Lx = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')
        Ly = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')
        Lz = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')

        def delta(x,y):
            if x - y == 0:
                return 1
            else:
                return 0

        for m1idx,m1 in enumerate(mvec):
            # print 'm1idx = ', m1idx
            
            cd_up = self.fs.cdagger[2*m1idx]
            cd_down = self.fs.cdagger[2*m1idx+1]
            
            for m2idx,m2 in enumerate(mvec):
                # print 'm2idx = ', m2idx

                c_up = self.fs.c[2*m2idx]
                c_down = self.fs.c[2*m2idx+1]
                
                mat = cd_up*c_up + cd_down*c_down

                coeff_Lx = delta(m1,m2+1)*np.sqrt((l-m2)*(l+m2+1)) + delta(m1+1,m2)*np.sqrt((l+m2)*(l-m2+1))
                coeff_Ly = delta(m1,m2+1)*np.sqrt((l-m2)*(l+m2+1)) - delta(m1+1,m2)*np.sqrt((l+m2)*(l-m2+1))
                coeff_Lz = m1*delta(m1,m2)

                coeff_Lx *= 0.5
                coeff_Ly *= np.complex(0.0,-0.5)

                Lx = Lx + coeff_Lx * mat
                Ly = Ly + coeff_Ly * mat
                Lz = Lz + coeff_Lz * mat

        self.L2 = Lx**2 + Ly**2 + Lz**2
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        self.Lp = self.Lx + np.complex(0.0, 1.0) * self.Ly
        self.Lm = self.Lx - np.complex(0.0, 1.0) * self.Ly

    # ------------------------------------------------------------------

    @classmethod
    def _expand_matrix_to_spin(self, mat):
        """ Take matrix defined only in angular momentum quantum label and
        extend it for half spin fermions."""
        
        dim = tuple(2*np.array(np.shape(mat)))
        print 'dim = ', dim
        ret_mat = np.zeros(dim, dtype=mat.dtype)

        ret_mat[0::2, 0::2] = mat
        ret_mat[1::2, 1::2] = mat

        return ret_mat

    # ------------------------------------------------------------------

    @classmethod
    def _single_particle_matrices(self, l, full_output=False):
        """ Construct the L_i singleparticle matrices for given quantum number l. """

        def L_coeff(l, m):
            return np.sqrt((l-m)*(l+m+1))

        Lp = np.zeros((2*l+1, 2*l+1), dtype='complex128')
        Lm = np.zeros((2*l+1, 2*l+1), dtype='complex128')        

        m_range = np.arange(-l, l+1)
        for idx, m in enumerate(m_range):
            if m+1 in m_range:
                Lp[idx+1, idx] = L_coeff(l,  m)
            if m-1 in m_range:
                Lm[idx-1, idx] = L_coeff(l, -m)

        #print 'Lp = \n', Lp
        #print 'Lm = \n', Lm

        Lx = 0.5 * (Lp + Lm)
        Ly = -np.complex(0.0, 0.5) * (Lp - Lm)
        Lz = np.diag(np.arange(-l, l + 1, dtype='complex128'))

        #print 'Lx = \n', Lx
        #print 'Ly = \n', Ly
        #print 'Lz = \n', Lz

        if full_output:
            return Lx, Ly, Lz, Lp, Lm
        else:
            return Lx, Ly, Lz

    # ------------------------------------------------------------------

    def _many_body_matrices(self, l, fs, use_cubic_harmonics=True):
        """ Take vector of single particle angular momentum matrices and
        calculate the corresponding many body operators.
        """

        Lx_sp, Ly_sp, Lz_sp = self._single_particle_matrices(l)

        Lx_sp = self._expand_matrix_to_spin(Lx_sp)
        Ly_sp = self._expand_matrix_to_spin(Ly_sp)
        Lz_sp = self._expand_matrix_to_spin(Lz_sp)

        if use_cubic_harmonics:
            U = self._get_real_valued_spherical_harmonics(l)
            U = self._expand_matrix_to_spin(U)

            # Debug test
            #U = np.eye(np.shape(Lz_sp)[0], dtype='complex128')
            #print 'U = \n', U

            Lx_sp = np.array( np.matrix(U) * np.matrix(Lx_sp) * np.matrix(U).H )
            Ly_sp = np.array( np.matrix(U) * np.matrix(Ly_sp) * np.matrix(U).H )
            Lz_sp = np.array( np.matrix(U) * np.matrix(Lz_sp) * np.matrix(U).H )

        c = np.array(fs.c)
        cd = np.array(fs.cdagger)

        Lx = np.sum( cd[np.newaxis,:] * (Lx_sp.T * c[:, np.newaxis]) )
        Ly = np.sum( cd[np.newaxis,:] * (Ly_sp.T * c[:, np.newaxis]) )
        Lz = np.sum( cd[np.newaxis,:] * (Lz_sp.T * c[:, np.newaxis]) )

        self.L2 = Lx**2 + Ly**2 + Lz**2
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz

        self.Lp = self.Lx + np.complex(0.0, 1.0) * self.Ly
        self.Lm = self.Lx - np.complex(0.0, 1.0) * self.Ly

    # ------------------------------------------------------------------

    @classmethod
    def _get_real_valued_spherical_harmonics(self, l):
        """ Generator of the unitary transform of the spherical harmonics
        that results in real valued functions in the space angle.

        Not used in current implementation. """

        dim = 2*l + 1
        U = np.zeros((dim, dim), dtype='complex128')
        for m in np.arange(-l,0):
            #print 'm = ', m
            v1 = np.zeros(dim)
            v1[l + m] = (-1.0)**m
            v1[l - m] = 1.0

            U[l+m, :] = v1 / np.sqrt(2)

        v = np.zeros(dim)
        v[l] = 1.0
        U[l,:] = v

        for m in np.arange(1, l+1):
            #print 'm = ', m
            v2 = np.zeros(dim)
            v2[l - m] = -(-1.0)**m
            v2[l + m] = 1.0

            # -- Actual definition with complex prefactor.
            #U[l+m, :] = v2 / (np.complex(0.0, 1.0) * np.sqrt(2))
            U[l+m, :] = np.complex(0.0, 1.0) * v2 / np.sqrt(2)

            # -- TEST: make transform real
            #U[l+m, :] = v2 / np.sqrt(2)

        return U

# ----------------------------------------------------------------------
class ManyBodyOperators():
    """
    Many body operator class applying the order,

    spin_orbital index:
        0 = spin_up orb 1, 
        1 = spin_down orb 1,
        2 = spin_up orb 2, 
        3 = spin_down orb 2, 
        etc... 

    Input: An instance of the SpinOperators class

    Members:
        * spin_flip 
        .. math:: 
           \\sum_{i \\ne j} c^\\dagger_{i \\uparrow} c_{i \\downarrow} 
                            c^\\dagger_{j \\downarrow} c_{j \\uparrow}

        * pair_hopping
        .. math:: 
           \\sum_{i \\ne j} c^\\dagger_{i\\uparrow} c^\\dagger_{i\\downarrow} 
                            c_{j\\uparrow} c_{j\\downarrow}

        * density_density
        .. math::
           \\sum_{i} n_{i\\uparrow} n_{i\\downarrow}

        * density_interband
        .. math:: 
           \\sum_{i < j, s, s'} n_{i, s} n_{j, s'}

        (sparse matrices)    
    """
    def __init__(self, spinop):
        self.__dict__.update(spinop.__dict__)

        spin_flip = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')
        pair_hopping = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')

        for idx1, idx2 in permutations(np.arange(self.norb/2), 2): # all comb. of 1,
            c1_up, c1dag_up, c1_down, c1dag_down = (self.c[0::2][idx1], self.cdagger[0::2][idx1],
                                                    self.c[1::2][idx1], self.cdagger[1::2][idx1])
            c2_up, c2dag_up, c2_down, c2dag_down = (self.c[0::2][idx2], self.cdagger[0::2][idx2],
                                                    self.c[1::2][idx2], self.cdagger[1::2][idx2])

            spin_flip = spin_flip + c1dag_up * c1_down * c2dag_down * c2_up
            pair_hopping = pair_hopping + c1dag_up * c1dag_down * c2_up * c2_down

        self.spin_flip = spin_flip
        self.pair_hopping = pair_hopping

        density_density = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')
        density_interband = sparse.csr_matrix((self.nstates, self.nstates), dtype='complex')

        for idx1, idx2 in combinations(np.arange(self.norb/2), 2):
            n1_up, n1_down = (self.n[0::2][idx1], self.n[1::2][idx1])
            n2_up, n2_down = (self.n[0::2][idx2], self.n[1::2][idx2])
            
            density_density = density_density + (n1_up + n1_down) * (n2_up + n2_down)
            density_interband = density_interband + n1_up * n2_up + n1_down * n2_down

        self.density_density = density_density
        self.density_interband = density_interband

# ----------------------------------------------------------------------              

