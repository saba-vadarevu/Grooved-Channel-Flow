import os
import sys
from flowFieldWavy import *
from warnings import warn

""" Module for time-marching a wavy channel/Couette flow.
Time-stepping involves using the influence matrix technique. 

The first step involves solving the Laplace equation for pressure with BCs
    p_1(-1) = 0, p_1(1) = 1
    p_-1(-1) = 1, p_-1(1) = 0
where p_1 and p_-1 are two solutions for the Laplace equation with the above BCs.
This is achieved in the function 'influence_pSol(flowDict)'
"""

class pFieldArray(np.ndarray):
    """ Defining the class 'pFieldArray' to store solutions for pressure modes
    from the B+ and B- problems"""
    def __new__(cls, flowDict,arr=None):
        """ For now, we always initialize a zero array
        Solutions for different p modes are later assigned
        flowDict takes care of the shape"""
        flowDict['nd'] = 1
        L = flowDict['L']
        M = flowDict['M']; nz = 2*M+1
        N = flowDict['N']
        if arr is None:
            arr = np.zeros((L+1, nz, 2, nz,N), dtype=np.complex)
        obj = np.ndarray.__new__(cls, shape=(L+1,nz,2,nz,N), dtype=np.complex, buffer=arr.copy())

        obj.flowDict = flowDict
        obj.L = L;  obj.nz = nz;   obj.M = M;   obj.N = N
        return obj

    def __array_finalize__(self,obj):
        if obj is None:
            return
        
        self.flowDict = getattr(self,'flowDict',obj.flowDict)
        self.L = getattr(self,'L',obj.L)
        self.M = getattr(self,'M',obj.M)
        self.N = getattr(self,'N',obj.N)
        self.nz = getattr(self,'nz',obj.nz)


    def getFlowField(self,l,m,k):
        """ Cast one of the solutions as a flowFieldRiblet instance
        k = 0 or 1, depending on the needed BC. k=0 corresponds to p=1 at y=-1
        Inputs: 
            self
            l: (int) Streamwise mode number
            m: (int) Spanwise mode number
            k: (0/1) Parity. If k=0, BC is p_{lm}(y=-1) =1, and zero for all others,
                        If k=1, BC is p_{lm}(y=1) =1, and zero for all others
        Outpus:
            pField: flowFieldRiblet instance
        """
        tempDict = self.flowDict.copy()
        if tempDict['nd'] != 1: warn("nd is not 1 in flowDict. Have a look into this")
        assert self.shape[-1] == tempDict['N']
        assert self.shape[-2] == 2*tempDict['M']+1

        pField = flowFieldRiblet(flowDict=tempDict).view4d()
        
        if l>= 0:
            pField[0,self.L+l, :,0] = self[l, m, k] 
        else:
            pField[0,self.L+l, :,0] = np.conj(self[l, m, k] )

        if l != 0:
            pField[0, self.L-l, :, 0] = np.conj(pField[0, self.L+l, ::-1, 0])

        return pField

        




def Lapl_base(flowDict):
    """ A base for the Laplacian matrix. 
        This is essentially the Laplacian matrix for the special case of l=0"""

    # Defining a bunch of variables for easy access later
    a = flowDict['alpha']; b = flowDict['beta']; eps = flowDict['eps']; Re = flowDict['Re']
    g = eps*b; g2 = g**2
    M = flowDict['M'];    N = flowDict['N']
    nz = 2*M + 1           # Total number of Fourier modes, since I do not consider symmetries here
    # I will implement a switch to account for the simpler symmetries

    # Wall-normal nodes and differentiation matrices
    y,DM = chebdif(N,2)
    D = DM[:,:,0]
    D2 = DM[:,:,1]

    # Matrix to be used later
    I = np.identity(N,dtype=np.complex)

    # The Fourier expansion for the Laplacian is:
    # {(d_xx + d_yy + d_zz) p}_{l,m} =      {-l^2*alpha^2  - m^2 * beta^2   +   (2*g2+1)*D2 } p_{l,m}
    #              +  (-g2*D2) p_{l,m-2}  + {g*beta*( 2m-1)*D} p_{l,m-1}    
    #              +  (-g2*D2) p_{l,m+2}  + {g*beta*(-2m-1)*D} p_{l,m+1}     # No pressure terms here
    # where g = eps*beta

    # We set l to zero for Lapl_base
    LaplMat_base = np.zeros((nz*N,nz*N+4*N),dtype=np.complex)
    # We start with 2*N extra columns on each side to make it easier to define the matrix

    # Filling LaplMat by blocks 
    # The +2 diagonal contains entries for p_{l,m} (because of the additional 2*N columns at the start)
    #       +1 diagonal to p_{l,m-1}
    #       principal diagonal to p_{l,m-2}
    #       +3 diagonal to p_{l,m+1}
    #       +4 diagonal to p_{l,m+2}
    for m in range(nz):
        # principal diagonal: p_{l,m-2}
        LaplMat_base[ (m+0)*N : (m+1)*N,  (m+0)*N : (m+1)*N ]  =  -g2 * D2 

        # +1 diagonal: p_{l,m-1}
        LaplMat_base[ (m+0)*N : (m+1)*N,  (m+1)*N : (m+2)*N ]  =  g* b* (2.*m-1.) * D 

        # +2 diagonal: p_{l,m}
        LaplMat_base[ (m+0)*N : (m+1)*N,  (m+2)*N : (m+3)*N ]  =  -(m**2) * (b**2) * I +  (1.+2.*g2) * D2

        # +3 diagonal: p_{l,m+1}
        LaplMat_base[ (m+0)*N : (m+1)*N,  (m+3)*N : (m+4)*N ]  =  g* b* (-2.*m-1.) * D 

        # +4 diagonal: p_{l,m+2}
        LaplMat_base[ (m+0)*N : (m+1)*N,  (m+4)*N : (m+5)*N ]  =  -g2 * D2 

        # Modifying matrix for Dirichlet boundary conditions
        LaplMat_base[ (m+0)*N, :] = 0.
        LaplMat_base[ (m+0)*N, (m+2)*N] = 1.    # Dirichlet BC for top wall
        LaplMat_base[ (m+0)*N+N-1, :] = 0.
        LaplMat_base[ (m+0)*N+N-1, (m+2)*N+N-1] = 1.    # Dirichlet BC for bottom wall
        # These entries make the left hand side of the Dirichlet BC,  1.*p_{wall} = const.
        #   The exact constant is not set within the matrix

    # Getting rid of all the extra columns
    LaplMat_base = LaplMat_base[:, 2*N:-2*N]

    return LaplMat_base




def influence_pSol_naive(flowDict, LaplMat_base=None):
    """ Solve for p_1 and p_-1, the solutions of the Laplace equation with BCs
    p_1(y=1) = 1, p_1(y=-1) =0;
    p_-1(y=1) = 0, p_-1(y=-1) = 1
    This involves solving for a set of modes l,-M to l,M together, and separately for each 'l'.
        This follows from the inter-modal interaction introduced by the waviness
    I call this naive because the LaplMat for different 'l' are the same upto a constant
        term on the diagonal of LaplMat. There are better ways of solving inv(Lapl_base + cI)
        than doing it separately for each l. I will deal with this later.
    Inputs:
        flowDict
    Outputs:
        p_-1, p_1 (flowFieldRiblet class instances)
        """
    if LaplMat_base is None:
        LaplMat_base = Lapl_base(flowDict)

    # The mini-Laplacian for a set of modes (l,-M) through (l,M) is different from
    #   LaplMat_base only on the diagonal blocks. For streamwise mode 'l', 
    #   the mini-Laplacian is obtained by adding -l**2 * a**2 * I to each block,
    #   Or just adding an identity matrix of size LaplMat_base multiplied by -l**2 * a**2

    # So, to get LaplMat_l,
    # LaplMat_l = LaplMat_base.copy(); np.fill_diagonal(LaplMat_l, np.diag(LaplMat_l) - l**2 * a**2)

    # Initializing pFieldArr
    pfArr = pFieldArray(flowDict)
    for l in range(flowDict['L']+1):
        LaplMat = LaplMat_base.copy()

        # Modifying the diagonal
        diagOld = np.diag(LaplMat)
        diagMod = diagOld - (l**2) * (flowDict['alpha']**2) # This changes the BC rows too

        # Reseting the entries on wall-rows to 1
        diagMod = diagMod.reshape((diagMod.size//flowDict['N'], flowDict['N']))
        diagMod[:, [0,-1]] = 1.
        diagNew = diagMod.flatten()
        
        # Replacing diagOld with diagNew
        np.fill_diagonal(LaplMat, diagNew)

        # Inverse of LaplMat_l
        LaplMatInv = np.linalg.pinv(LaplMat)

        for m in range(2*flowDict['M'] + 1):
            for k in range(2):
                # Now we solve for each solution of the B+ and B- problems
                # B- problems are for k=0, B+ for k=1 (refer to documentation)

                # Writing the matrix equation, L * p = f, or p = Linv * f
                #   f is zero for all internal nodes. 
                #   For boundary nodes, f is zero for all but one mode. 
                #   Basically, f is some \bm{e}_q, a unit vector with all but one entry zero.
                # So, the solution p is just the q^th column of Linv 
                #           Because, [a11  a12] [0]  = [a12]
                #                    [a21  a22] [1]    [a22]
                # This means we don't even need matrix vector products. 
                if k == 0: tmpScal = pfArr.N - 1
                else: tmpScal = 0

                columnNumber = m*pfArr.N + tmpScal

                pfArr[l, m, k] =  LaplMatInv[:, columnNumber].reshape((pfArr.nz, pfArr.N))



    return pfArr

