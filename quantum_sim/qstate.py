# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 21:59:47 2024

@author: God_Zao
"""

import numpy as np
import scipy.sparse as sparse

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
        result = self.q.T@sparse.kron(sparse.kron(id1,Z),id2)@self.q
        #print(result.shape)
        return result.toarray()[0][0]
    
    def time_evolution(self, operator):
        c,r = operator.shape
        assert c==r
        assert r==self.q.shape[0]
        self.q = operator @ self.q
        return
    