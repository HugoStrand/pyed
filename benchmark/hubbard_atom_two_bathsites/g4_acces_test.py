
""" gf_multivar bracket test: g4_tau[[t1, t2 ,t3]] 

Author: Hugo U.R. Strand (2017) hugo.strand@gmail.com

Gives the error:

Traceback (most recent call last):
  File "g4_acces_test.py", line 38, in <module>
    print g4_tau[[t1, t2, t3]] # <-- How can we access the data?
  File "/Users/hugstr/apps/triqs_devel/lib/python2.7/site-packages/pytriqs/gf/gf.py", line 303, in __getitem__
    assert len(key) == self.rank, "wrong number of arguments in [[ ]]. Expected %s, got %s"%(self.rank, len(key))
AssertionError: wrong number of arguments in [[ ]]. Expected 1, got 3

"""

import itertools
import numpy as np

from pytriqs.gf import Gf
from pytriqs.gf import MeshImTime, MeshProduct

ntau = 10
beta = 1.2345
imtime = MeshImTime(beta, 'Fermion', ntau)
prodmesh = MeshProduct([imtime, imtime, imtime])
g4_tau = Gf(name='g4_tau', mesh=prodmesh, indices=[1])

# -- Try to loop over all times and evaluate

mesh_prod = g4_tau.mesh
meshes = [ mesh_tuple[0] for mesh_tuple in mesh_prod ]

data = g4_tau.data
np.testing.assert_array_almost_equal(data, np.zeros_like(data))

for t1, t2, t3 in itertools.product(*meshes):
    print g4_tau[[t1, t2, t3]] # <-- How can we access the data?
    g4_tau[[t1, t2, t3]] = 1.0

data = g4_tau.data
np.testing.assert_array_almost_equal(data, np.ones_like(data))
