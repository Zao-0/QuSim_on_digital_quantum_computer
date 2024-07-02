# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:59:47 2024

@author: God_Zao
"""


import numpy as np
import scipy.sparse as sparse
import quantum_sim.hamiltonian_op as hop

class qstate:
    def __init__(self,qlist:list[int]):
        self.N = len(qlist)
        up = sparse.csr_matrix(np.array([[1.],[0.]]))
        dn = sparse.csr_matrix(np.array([[0.],[1.]]))
        self.q = sparse.identity(1,format='csr')
        for i in range(self.N):
            self.q = sparse.kron(self.q, up) if qlist[i]==0 else sparse.kron(self.q, dn)
    
    def local_magnetism(self, i:int):
        assert i<self.N
        assert i>=0
        id1 = sparse.identity(2**i,format='csr')
        id2 = sparse.identity(2**(self.N-i-1),format='csr')
        Z = sparse.csr_matrix(np.array([[1.,0.],[0.,-1.]]))
        result = self.q.conj().T@sparse.kron(sparse.kron(id1,Z),id2)@self.q
        return result.toarray()[0][0]
    
    def time_evolution(self, operator):
        c,r = operator.shape
        assert c==r
        assert r==self.q.shape[0]
        self.q = operator @ self.q
        return self.q
    
    def N_half(self):
        result = 0
        for i in range(int(self.N/2)):
            nhalf = sparse.csr_matrix(np.array([[1.,0.],[0.,0.]]))
            id1 = sparse.identity(2**i,format='csr')
            id2 = sparse.identity(2**(self.N-i-1),format='csr')
            temp = self.q.conj().T@sparse.kron(sparse.kron(id1,nhalf),id2)@self.q
            result+=temp.toarray()[0][0]
        return result
    
    def state_distance(self, other):
        assert isinstance(other, qstate)
        result = self.q-other.q
        return np.linalg.norm(result.data)
    
    def get_fq(self):
        s_list = [1 if i <self.N/2 else -1 for i in range(self.N)]
        fq1 = hop.SumOp([hop.ProductOp([hop.Pz(1, i),hop.Pz(1, j)], s_list[i]*s_list[j]) for i in range(self.N-1) for j in range(i+1,self.N)])
        fq2 = hop.SumOp([hop.Pz(s_list[i], i) for i in range(self.N)])
        fq1 = fq1.as_sparse_matrix(self.N)
        fq2 = fq2.as_sparse_matrix(self.N)
        fq1_result = self.q.conj().T@fq1@self.q
        fq2_result = self.q.conj().T@fq2@self.q
        fq1_result = fq1_result.toarray()[0][0]
        fq2_result = fq2_result.toarray()[0][0]
        return (fq1_result-fq2_result**2)/self.N