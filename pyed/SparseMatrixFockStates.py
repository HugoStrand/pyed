
"""
Sparse matrix representation of fermionic creation
and annihilation operators for a finite Fock space.

Author: Hugo U. R. Strand (2017), hugo.strand@gmail.com
"""

# ----------------------------------------------------------------------

import numpy as np

from scipy import sparse

# ----------------------------------------------------------------------


class SparseMatrixRepresentation(object):

    """ Generator for sparse matrix representations of
    Triqs operator expressions, given a set of fundamental
    creation operators. """

    # ------------------------------------------------------------------
    def __init__(self, fundamental_operators):

        self.fundamental_operators = fundamental_operators

        self.operator_labels = []
        for operator_expression in fundamental_operators:
            for term, coeff in operator_expression:
                assert(coeff == 1)
                assert(len(term) == 1)
                label = term[0]
                dag, idx = label
                # only store annihilation operators
                label = (False, tuple(idx))
                self.operator_labels.append(label)

        # remove operator repetitions
        #self.operator_labels = list(set(self.operator_labels))
        # The set operator changes order!

        # Check for repeated operators
        operator_labels_set = list(set(self.operator_labels))

        assert len(operator_labels_set) == len(self.operator_labels), \
            "ERROR: Repeated operators in fundamental_operators!"

        self.operator_labels = [
            (dag, list(idx)) for dag, idx in self.operator_labels]

        self.nfermions = len(self.operator_labels)
        self.sparse_operators = \
            SparseMatrixCreationOperators(self.nfermions)

    # ------------------------------------------------------------------
    def sparse_matrix(self, triqs_operator_expression):
        """ Convert a general Triqs operator expression to a sparse
        matrix representation. """

        matrix_rep = 0.0 * self.sparse_operators.I

        for term, coef in triqs_operator_expression:

            product = coef * self.sparse_operators.I

            for fact in term:

                dagger, idx = fact
                oidx = self.operator_labels.index((False, idx))

                op = self.sparse_operators.c_dag[oidx]
                if not dagger:
                    op = op.getH()

                product = product * op

            matrix_rep = matrix_rep + product

        return matrix_rep

    # ------------------------------------------------------------------
    def sparse_sympy_matrix(self, triqs_operator_expression):

        Hsp = self.sparse_matrix(triqs_operator_expression)

        from sympy.matrices import SparseMatrix
        from sympy.simplify.simplify import nsimplify

        d = dict([((i, j), nsimplify(val))
                  for (i, j), val in Hsp.todok().items()])

        H = SparseMatrix(Hsp.shape[0], Hsp.shape[1], d)

        return H

# ----------------------------------------------------------------------


class SparseMatrixCreationOperators:

    """ Generator of sparse matrix representation of fermionic
    creation operators, for finite number of fermions. """

    # ------------------------------------------------------------------
    def __init__(self, nfermions):

        self.nfermions = nfermions
        self.nstates = 2**nfermions

        self.c_dag = []
        for fidx in range(nfermions):
            c_dag_fidx = self._build_creation_operator(fidx)
            self.c_dag.append(c_dag_fidx)

        self.I = sparse.eye(
            self.nstates, self.nstates, dtype=np.float64, format='csr')

    # ------------------------------------------------------------------
    def _build_creation_operator(self, orbidx):
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
        sign = 1 - 2 * np.array(
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
        cdagger = sparse.coo_matrix((D, (I, J)),
                                    shape=(nstates, nstates)).tocsr()

        return cdagger

# ----------------------------------------------------------------------

class SparseMatrixBosonicCreationOperators:

    """ Generator for sparse-matrix representations of (soft-core)
    bosonic operators for N : bosonic modes with a maximal boson
    occupation Nmax. """

    # ------------------------------------------------------------------
    def __init__(self, N, Nmax):

        self.N, self.Nmax = N, Nmax
        self.Ngamma = Nmax**N

        # -- Construct bosonic manybody operators
        
        self.b_dag = []
        self.I = sparse.identity(self.Ngamma)

        for fidx in range(self.N):
            b_dag = self._build_destruction_operator(fidx).getH()
            self.b_dag.append(b_dag)

    # ------------------------------------------------------------------
    def _build_destruction_operator(self, fidx):

        J = np.arange(self.Ngamma, dtype=np.uint)
        I = np.zeros(self.Ngamma, dtype=np.uint)
        D = np.zeros(self.Ngamma, dtype=float)

        for idx in J:

            s = self._cubic_idx_to_state(idx, self.Nmax, self.N)
            n = s[fidx]

            if n > 0:
                s[fidx] -= 1
                I[idx] = self._cubic_state_to_idx(s, self.Nmax)
                D[idx] = np.sqrt(n)
            else:
                D[idx] = 0

        b = sparse.coo_matrix(
            (D, (I, J)), 
            shape=(self.Ngamma, self.Ngamma)).tocsr()

        # -- Remove zero occurencies (too lazy to fix them above.. FIXME)
        b.eliminate_zeros()

        return b

    # ------------------------------------------------------------------
    def _cubic_state_to_idx(self, s, Nmax):

        idx = 0
        N = len(s)

        for fidx, n in enumerate(s):
            n = int(n)
            idx += n * Nmax**(N - fidx - 1)

        return idx

    # ------------------------------------------------------------------
    def _cubic_idx_to_state(self, idx, Nmax, N):

        rest = idx
        s = np.zeros(N, dtype=np.uint)

        for fidx in range(N):
            factor = Nmax**(N - fidx - 1)
            s[fidx] = rest / factor # -- Nb! Integer division intended
            rest = np.mod(rest, factor)

        return s


# ----------------------------------------------------------------------

class SparseMatrixFermiBoseCreationOperators:

    """ Generator for sparse-matrix representations of
    fermions and (soft-core) bosonic operators for

    Nf : fermionic modes
    Nb : bosonic modes
    Nb_max : maximal boson occupation """

    # ------------------------------------------------------------------
    def __init__(self, Nf, Nb, Nb_max):

        self.Nf, self.Nb, self.Nb_max = Nf, Nb, Nb_max

        self.fops = SparseMatrixCreationOperators(self.Nf)
        self.bops = SparseMatrixBosonicCreationOperators(self.Nb, self.Nb_max)

        self.If = self.fops.I
        self.Ib = self.bops.I
        self.I = sparse.kron(self.If, self.Ib)

        self.c_dag = []
        for c_dag in self.fops.c_dag:
            c_dag_ext = sparse.kron(c_dag, self.Ib, format='coo')
            self.c_dag.append(c_dag_ext)

        self.b_dag = []
        for b_dag in self.bops.b_dag:
            b_dag_ext = sparse.kron(self.If, b_dag, format='coo')
            self.b_dag.append(b_dag_ext)
