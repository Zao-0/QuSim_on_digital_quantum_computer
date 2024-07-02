# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:56:29 2024

@author: God_Zao
"""

import numpy as np
import scipy.sparse as sparse
import abc

def generate_operator(op,coeff,i:int,N:int):
    if op==0:
        return 0
    if op==1:
        return coeff*sparse.identity(2**N,dtype=np.complex128,format='csr')
    op_dict = {'X':sparse.csr_matrix(np.array([[0.,1.],[1.,0]],dtype = complex)),
               'x':sparse.csr_matrix(np.array([[0.,1.],[1.,0]],dtype=complex)),
               'Y':sparse.csr_matrix(np.array([[0.,0.-1j],[0.+1j,0]])),
               'y':sparse.csr_matrix(np.array([[0.,0.-1j],[0.+1j,0]])),
               'Z':sparse.csr_matrix(np.array([[1.,0.],[0.,-1.]],dtype = complex)),
               'z':sparse.csr_matrix(np.array([[1.,0.],[0.,-1.]],dtype = complex))}
    result = sparse.identity(1,dtype=np.complex128,format='csr')
    eye = sparse.identity(2,dtype=np.complex128,format='csr')
    for j in range(N):
        if j==i:
            result = sparse.kron(result, op_dict[op])
        else:
            result = sparse.kron(result,eye)
    return result*coeff
    

class HamiltonianOp(abc.ABC):
    @abc.abstractmethod
    def __neg__(self):
        """
        Logical negation.
        """

    @abc.abstractmethod
    def __rmul__(self, other):
        """
        Logical scalar product.
        """

    @abc.abstractmethod
    def __add__(self, other):
        """
        Logical sum.
        """

    @abc.abstractmethod
    def __sub__(self, other):
        """
        Logical difference.
        """

    @abc.abstractmethod
    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        """
        Equality test.
        """

    @abc.abstractmethod
    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        
    @abc.abstractmethod
    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
    
    @abc.abstractmethod
    def as_sparse_matrix(self,N):
        """
        The numeric matrix form of the operator
        Parameters
        ----------
        N : INT num of sites

        Returns
        -------
        np.array().

        """
    
    @abc.abstractmethod
    def as_dense_matrix(self,N):
        """
        dense form of the matrix
        """
    @abc.abstractmethod
    def is_zero(self) -> bool:
        """
        

        Returns
        -------
        True if it's zero

        """
    
    @abc.abstractmethod
    def normalize(self) -> tuple:
        """normalize the matrix"""
    
class Id(HamiltonianOp):
    def __init__(self, coeff:float):
        self.coeff=coeff
    
    def __neg__(self):
        """
        Logical negation.
        """
        return Id(-self.coeff)
    
    def __rmul__(self, other):
        return Id(self.coeff*other)
    
    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if not self.proportional(other):
            raise ValueError("can only add another anti-symmetric hopping term acting on same sites")
        return Id(self.coeff+other.coeff)
    
    def __sub__(self, other):
        """
        Logical difference.
        """
        return self+(-other)
    
    def __str__(self) -> str:
        c = "" if self.coeff==1 else f"({self.coeff})"
        return c+"Id"
    
    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, Id):
            return other.coeff==self.coeff
        return False
    
    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, ZeroOp):
            return False
        if isinstance(other, Id):
            return self.coeff<other.coeff
        return True
    
    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        return isinstance(other, Id)
    
    def as_sparse_matrix(self, N):
        return generate_operator(1,self.coeff,0,N)
    
    def as_dense_matrix(self, N):
        assert N<15
        return self.as_sparse_matrix(N).toarray()
    
    def is_zero(self)->bool:
        return self.coeff==0

    def normalize(self)->tuple:
        if self.coeff==1:
            return self,1
        return Id(1),self.coeff
    
class ZeroOp(HamiltonianOp):    
    def __neg__(self):
        """
        Logical negation.
        """
        return ZeroOp()
    
    def __rmul__(self, other):
        return ZeroOp()
    
    def __add__(self, other):
        """
        Logical sum.
        """
        return other
    
    def __sub__(self, other):
        """
        Logical difference.
        """
        return -other
    
    def __str__(self) -> str:
        return "0"
    
    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, ZeroOp()):
            return True
        return False
    
    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, ZeroOp):
            return False
        return True
    
    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        return isinstance(other, ZeroOp())
    
    def as_sparse_matrix(self, N):
        return sparse.csr_matrix((2**N,2**N))
    
    def as_dense_matrix(self, N):
        return 0
    
    def is_zero(self)->bool:
        return True

    
    def normalize(self)->tuple:
        return self,1


class Px(HamiltonianOp):
    def __init__(self, coeff:float, i:int):
        self.coeff=coeff
        self.i=i
    
    def __neg__(self):
        """
        Logical negation.
        """
        
        return Px(-self.coeff,self.i)
    
    def __rmul__(self, other):
        return Px(self.coeff*other,self.i)
    
    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if not self.proportional(other):
            raise ValueError("can only add another anti-symmetric hopping term acting on same sites")
        return Px(self.coeff+other.coeff, self.i)
    
    def __sub__(self, other):
        """
        Logical difference.
        """
        return self+(-other)
    
    def __str__(self) -> str:
        c = "" if self.coeff==1 else f"({self.coeff})"
        return c+f"Px_{self.i}"
    
    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, Px):
            return other.coeff==self.coeff and other.i==self.i
        return False
    
    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, (ZeroOp,Id)):
            return False
        if isinstance(other, Px):
            return (self.i,self.coeff)<(other.i,other.coeff)
        return True
    
    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        if isinstance(other, Px):
            return self.i==other.i
        return False
    
    def as_sparse_matrix(self, N):
        return generate_operator('X',self.coeff,self.i,N)
    
    def as_dense_matrix(self, N):
        assert N<15
        return self.as_sparse_matrix(N).toarray()
    
    def is_zero(self)->bool:
        return self.coeff==0

    def normalize(self)->tuple:
        if self.coeff==1:
            return self,1
        return Px(1,self.i),self.coeff
    
    
class Py(HamiltonianOp):
    def __init__(self, coeff:float, i:int):
        self.coeff=coeff
        self.i=i
    
    def __neg__(self):
        """
        Logical negation.
        """
        
        return Py(-self.coeff,self.i)
    
    def __rmul__(self, other):
        return Py(self.coeff*other,self.i)
    
    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if not self.proportional(other):
            raise ValueError("can only add another anti-symmetric hopping term acting on same sites")
        return Py(self.coeff+other.coeff, self.i)
    
    def __sub__(self, other):
        """
        Logical difference.
        """
        return self+(-other)
    
    def __str__(self) -> str:
        c = "" if self.coeff==1 else f"({self.coeff})"
        return c+f"Py_{self.i}"
    
    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, Py):
            return other.coeff==self.coeff and other.i==self.i
        return False
    
    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, (ZeroOp,Id,Px)):
            return False
        if isinstance(other, Py):
            return (self.i,self.coeff)<(other.i,other.coeff)
        return True
    
    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        if isinstance(other, Py):
            return self.i==other.i
        return False
    
    def as_sparse_matrix(self, N):
        return generate_operator('Y',self.coeff,self.i,N)
    
    def as_dense_matrix(self, N):
        assert N<15
        return self.as_sparse_matrix(N).toarray()
    
    def is_zero(self)->bool:
        return self.coeff==0

    def normalize(self)->tuple:
        if self.coeff==1:
            return self,1
        return Py(1,self.i),self.coeff
    
class Pz(HamiltonianOp):
    def __init__(self, coeff:float, i:int):
        self.coeff=coeff
        self.i=i
    
    def __neg__(self):
        """
        Logical negation.
        """
        
        return Pz(-self.coeff,self.i)
    
    def __rmul__(self, other):
        return Pz(self.coeff*other,self.i)
    
    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if not self.proportional(other):
            raise ValueError("can only add another anti-symmetric hopping term acting on same sites")
        return Pz(self.coeff+other.coeff, self.i)
    
    def __sub__(self, other):
        """
        Logical difference.
        """
        return self+(-other)
    
    def __str__(self) -> str:
        c = "" if self.coeff==1 else f"({self.coeff})"
        return c+f"Pz_{self.i}"
    
    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, Pz):
            return other.coeff==self.coeff and other.i==self.i
        return False
    
    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, (ProductOp, SumOp)):
            return True
        if isinstance(other, Pz):
            return (self.i,self.coeff)<(other.i,other.coeff)
        return False
    
    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        if isinstance(other, Pz):
            return self.i==other.i
        return False
    
    def as_sparse_matrix(self, N):
        return generate_operator('Z',self.coeff,self.i,N)
    
    def as_dense_matrix(self, N):
        assert N<15
        return self.as_sparse_matrix(N).toarray()
    
    def is_zero(self)->bool:
        return self.coeff==0

    def normalize(self)->tuple:
        if self.coeff==1:
            return self,1
        return Pz(1,self.i),self.coeff
    

class ProductOp(HamiltonianOp):
    def __init__(self, op_list:list, coeff:float):
        self.op_list = []
        self.coeff = coeff
        for op in op_list:
            if isinstance(op, ProductOp):
                self.op_list+=op.op_list
                self.coeff*=op.coeff
            elif not isinstance(op, Id):
                if op.is_zero():
                    print(op)
                    self.op_list = [ZeroOp()]
                    self.coeff = 1
                    #assert False
                else:
                    norm_op, op_coeff = op.normalize()
                    self.op_list.append(norm_op)
                    self.coeff*=op_coeff
            else:
                self.coeff*=op.coeff
                
    def __neg__(self):
        return ProductOp(self.op_list, -self.coeff)
    
    def __rmul__(self, other):
        if not isinstance(other, (complex,float)):
            raise ValueError("expecting a scalar argument")
        return ProductOp(self.op_list, other * self.coeff)
    
    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if not self.proportional(other):
            raise ValueError("can only add another product operator with same factors")
        # assuming that each operator in product is normalized
        return ProductOp(self.op_list, self.coeff + other.coeff)
    
    def __sub__(self, other):
        """
        Logical difference.
        """
        return self + (-other)
    
    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        c = "" if self.coeff == 1 else f"({self.coeff}) "
        if not self.op_list:
            # logical identity operator
            return c + "<empty product>"
        s = ""
        for op in self.op_list:
            s += ("" if s == "" else " @ ") + "(" + str(op) + ")"
        return c + s
    
    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, ProductOp):
            if len(self.op_list) == len(other.op_list) and self.coeff == other.coeff:
                if all(op1 == op2 for op1, op2 in zip(self.ops, other.ops)):
                    return True
        return False
    
    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if isinstance(other, SumOp):
            return True
        if not isinstance(other, (ProductOp, SumOp)):
            return False
        
        assert isinstance(other, ProductOp)
        if len(self.op_list)<len(other.op_list):
            return True
            
        for op1,op2 in zip(self.op_list,other.op_list):
            if op1<op2:
                return True
            if op2<op1:
                return False
            assert op1 == op2
        return self.coeff < other.coeff
    
    def proportional(self, other) -> bool:
        if isinstance(other, ProductOp):
            if len(self.op_list) == len(other.op_list):
                if all(op1.proportional(op2) for op1, op2 in zip(self.op_list, other.op_list)):
                    return True
        return False
    
    def as_sparse_matrix(self,N):
        if self.is_zero():
            return sparse.csr_matrix((2**N,2**N))
        if not self.op_list:
            return Id(self.coeff).as_sparse_matrix(N)
        fop = self.op_list[0].as_sparse_matrix(N)
        for op in self.op_list[1:]:
            fop = fop @ op.as_sparse_matrix(N)
        return self.coeff * fop
    
    def as_dense_matrix(self, N):
        assert N>0
        assert N<15
        return self.as_sparse_matrix(N).toarray()
    
    def normalize(self) -> tuple:
        """
        Return a normalized copy of the operator together with its scalar prefactor.
        """
        if self.coeff == 1:
            return self, 1
        return ProductOp(self.op_list, 1), self.coeff
    
    def is_zero(self) -> bool:
        """
        Indicate whether the operator acts as zero operator.
        """
        if self.coeff == 0:
            return True
        for op in self.op_list:
            if op.is_zero():
                return True
        # empty 'ops' logically corresponds to identity operation
        return False

class SumOp(HamiltonianOp):
    def __init__(self,op_list):
        self.op_list = []
        for op in op_list:
            if op.is_zero():
                continue
            if isinstance(op, SumOp):
                self.op_list += op.op_list
            else:
                self.op_list.append(op)
        self.op_list = sorted(self.op_list)
    
    def __neg__(self):
        """
        Logical negation.
        """
        return SumOp([-term for term in self.op_list])
    
    def __rmul__(self, other):
        """
        Logical scalar product.
        """
        if not isinstance(other, (complex,float)):
            raise ValueError("expecting a scalar argument")
        return SumOp([other * term for term in self.op_list])
    
    def __add__(self, other):
        """
        Logical sum.
        """
        if isinstance(other, ZeroOp):
            return self
        if isinstance(other, SumOp):
            return SumOp(self.op_list + other.op_list)
        return SumOp(self.op_list + [other])
    
    def __sub__(self, other):
        """
        Logical difference.
        """
        return self + (-other)
    
    def __str__(self) -> str:
        """
        Represent the operator as a string.
        """
        if not self.op_list:
            # logical zero operator
            return "<empty sum>"
        s = ""
        for op in self.op_list:
            s += ("" if s == "" else " + ") + str(op)
        return s
    
    def __eq__(self, other) -> bool:
        """
        Equality test.
        """
        if isinstance(other, SumOp):
            if len(self.op_list) == len(other.op_list):
                # assuming that terms are sorted
                if all(t1 == t2 for t1, t2 in zip(self.op_list, other.op_list)):
                    return True
        return False
    
    def __lt__(self, other) -> bool:
        """
        "Less than" comparison, used for, e.g., sorting a sum of operators.
        """
        assert isinstance(other, HamiltonianOp)
        if not isinstance(other, SumOp):
            return False
        # assuming that terms are sorted
        for t1, t2 in zip(self.op_list, other.op_list):
            if t1 < t2:
                return True
            if t2 < t1:
                return False
            assert t1 == t2
        if len(self.op_list) < len(other.op_list):
            return True
        if len(other.op_list) < len(self.op_list):
            return False
        # operators are equal
        return False
    
    def proportional(self, other) -> bool:
        """
        Whether current operator is equal to 'other' up to a scalar factor.
        """
        # ignoring a potential global scalar factor, for simplicity
        return self == other
    
    def as_sparse_matrix(self, N):
        if self.is_zero():
            return sparse.csr_matrix((2**N,2**N))
        result = self.op_list[0].as_sparse_matrix(N)
        for op in self.op_list[1:]:
            if not op.is_zero():
                result+=op.as_sparse_matrix(N)
        return result
    
    def as_dense_matrix(self, N):
        assert N<15
        return self.as_sparse_matrix(N).toarray()
    
    def normalize(self):
        return self,1
    
    def is_zero(self):
        return len(self.op_list)==0