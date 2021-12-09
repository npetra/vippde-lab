# Copyright (c) 2016, The University of Texas at Austin & University of
# California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

from __future__ import absolute_import, division, print_function

import dolfin as dl
from hippylib import *

class _ManifoldR():
    """
    Operator that represent the action of the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, R, P):
        self.R = R
        self.P = P
        self.Px, self.Rx = dl.Vector(), dl.Vector()
        self.P.init_vector(self.Px, 0)
        self.R.init_vector(self.Rx, 0)
        
    def init_vector(self, x, dim):
        self.R.init_vector(x,1)

    def inner(self,x,y):
        self.P.mult(x, self.Px)
        self.P.mult(y, self.Rx)
        return self.R.inner(self.Px, self.Rx)

    def mult(self,x,y):
        self.P.mult(x, self.Px)
        self.R.mult(self.Px, self.Rx)
        self.P.transpmult(self.Rx, y)

    def mpi_comm(self):
        return self.R.mpi_comm()

class _ManifoldRsolver():
    """
    Operator that represent the action of the inverse the regularization/precision matrix
    for the Bilaplacian prior.
    """
    def __init__(self, Rsolver, P):
        self.Rsolver = Rsolver
        self.P = P
        self.Px, self.Pb = dl.Vector(), dl.Vector()
        self.P.init_vector(self.Px, 0)
        self.P.init_vector(self.Pb, 0)

    def init_vector(self, x, dim):
        self.P.init_vector(x, 1)

    def solve(self, x, b):
        self.P.mult(b, self.Pb)
        self.Rsolver.solve(self.Px, self.Pb)
        return self.P.transpmult(self.Px, x)

    def mpi_comm(self):
        return self.P.mpi_comm()

class _ManifoldM():
    def __init__(self, M, P):
        self.M = M
        self.P = P
        self.Px, self.Py = dl.Vector(), dl.Vector()
        self.P.init_vector(self.Px, 0)
        self.P.init_vector(self.Py, 0)

    def mult(self, x, y):
        self.P.mult(x, self.Px)
        self.M.mult(self.Px, self.Py)
        self.P.transpmult(self.Py, y)

class _ManifoldMsolver():
    def __init__(self, Msolver, P):
        self.Msolver = Msolver
        self.P = P
        self.Px, self.Py = dl.Vector(), dl.Vector()
        self.P.init_vector(self.Px, 0)
        self.P.init_vector(self.Py, 0)

    def solve(self, x, b):
        self.P.mult(b, self.Py)
        self.Msolver.solve(self.Px, self.Py)
        self.P.transpmult(self.Py, x)
        

class ManifoldPrior(prior._Prior):
    def __init__(self, Vh, Vsub, bmesh, priorVsub):
        """
        Construct the Prior model.
        Input:
        - Vh:               the finite element space for the parameter
        - Vsub:             the finite element space for the parameter on manifold
        - dummy_mesh:       a dummy subdomain including the manifold in boundary
        - priorVsub:        prior on Vsub
        - bmesh:            BoundaryMesh object of the manifold
        - gamma and delta:  the coefficient in the PDE
        - Theta:            the s.p.d. tensor for anisotropic diffusion of the pde
        - mean:             the prior mean
        """
        import numpy as np
        self.Vh = Vh
        self.Vsub = Vsub
        self.bmesh = bmesh
        self.P = self.ProjectionMatrix()        # projection from Vh to Vsub
        self.priorVsub = priorVsub
        self.Px, self.Ptx = dl.Vector(), dl.Vector()
        self.P.init_vector(self.Px, 0)
        self.P.init_vector(self.Ptx, 1)
        self.mean = None

        prior_type = priorVsub.__class__.__name__
        if str(prior_type) == "BiLaplacianPrior" or str(prior_type)=="SqrtPrecisionPDE_Prior":
            self.M = MatPtAP(self.priorVsub.M, self.P)
            self.Msolver = _ManifoldMsolver(self.priorVsub.Msolver, self.P)
            self.A = MatPtAP(self.priorVsub.A, self.P)
            self.Asolver = _ManifoldMsolver(self.priorVsub.Asolver, self.P)

            self.sqrtM = self.priorVsub.sqrtM 
            self.R = _ManifoldR(prior._BilaplacianR(self.priorVsub.A, self.priorVsub.Msolver), self.P)
            self.Rsolver = _ManifoldRsolver(prior._BilaplacianRsolver(self.priorVsub.Asolver, self.priorVsub.M), self.P)
            if (self.priorVsub.mean is None):# or (self.mean is None):
                self.mean = dl.Vector()
                self.P.init_vector(self.mean, 1)
            else:
                self.P.transpmult(self.priorVsub.mean, self.Ptx)
                self.mean = self.Ptx
        if str(prior_type) == "LaplacianPrior":
            self.M = MatPtAP(self.priorVsub.M, self.P)
            self.Msolver = _ManifoldMsolver(self.priorVsub.Msolver, self.P)
            self.R = _ManifoldR(self.priorVsub.R, self.P)
            self.Rsolver = _ManifoldRsolver(self.priorVsub.Rsolver, self.P)
            if (self.priorVsub.mean is None):
                self.mean = dl.Vector()
                self.P.init_vector(self.mean, 1)
            else:
                self.P.transpmult(self.priorVsub.mean, self.Ptx)
                self.mean = self.Ptx
            self.sqrtR = self.priorVsub.sqrtR 
            
    
    def init_vector(self,x,dim):
        """
        Inizialize a vector x to be compatible with the range/domain of R.
        If dim == "noise" inizialize x to be compatible with the size of
        white noise used for sampling.
        """
        if dim == "noise":
            #self.sqrtM.init_vector(x, 1)
            self.sqrtR.init_vector(x, 1)
        else:
            self.P.init_vector(x,1)
        
    def sample(self, noise, s, add_mean=True):
        """
        Given a noise ~ N(0, I) compute a sample s from the prior.
        If add_mean=True add the prior mean value to s.
        """
        rhs = self.sqrtM*noise
        self.priorVsub.Asolver.solve(self.Px, rhs)
        self.P.transpmult(self.Px, s)
        
        if add_mean:
            s.axpy(1., self.mean)
 
    def ProjectionMatrix(self):
        Vcdofmap = self.Vh.dofmap()
        Vrdofmap = self.Vsub.dofmap()

        from petsc4py import PETSc
        from dolfin import MPI
        mpisize = MPI.size(MPI.comm_world)
        mpirank = MPI.rank(MPI.comm_world)

        rmap = PETSc.LGMap().create(Vrdofmap.dofs(), comm=MPI.comm_world)
        cmap = PETSc.LGMap().create(Vcdofmap.dofs(), comm=MPI.comm_world)
        
        
        P = PETSc.Mat()
        P.create(PETSc.COMM_WORLD)
        try:
            localdimVr = Vrdofmap.local_dimension("owned")
            localdimVc = Vcdofmap.local_dimension("owned")
        except:
            IM = Vrdofmap.index_map()
            localdimVr = IM.size(IM.MapSize(1))
            IM = self.Vh.dofmap().index_map()
            localdimVc = IM.size(IM.MapSize(1))

        P.setSizes([ [localdimVr, self.Vsub.dim()], [localdimVc, self.Vh.dim()] ])
        P.setType('aij')
        P.setUp()
        P.setLGMap(rmap, cmap)

        sub2bmesh = self.Vsub.mesh().data().array('parent_vertex_indices', 0)
        bmesh2mesh = self.bmesh.entity_map(0).array()
        entity_map = bmesh2mesh[sub2bmesh]

        Vr_vert2dof = dl.vertex_to_dof_map(self.Vsub)
        Vc_vert2dof = dl.vertex_to_dof_map(self.Vh)
        for v in dl.vertices(self.Vsub.mesh()):
            i = Vr_vert2dof[v.index()]
            j = Vc_vert2dof[entity_map[v.index()]]
            P[i, j] = 1.
        P.assemblyBegin()
        P.assemblyEnd()
        return dl.Matrix(dl.PETScMatrix(P))

    def Projection(self, x):
        self.P.mult(x, self.Px)
        self.P.transpmult(self.Px, x)
