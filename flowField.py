""" 
#####################################################
Author : Sabarish Vadarevu
Affiliation: Aerodynamics and Flight Mechanics group, University of Southampton.


non-class functions:
getDefaultDict(), verify_dict(), read_dictFile()
"""

import numpy as np
import scipy as sp
#from scipy.linalg import norm
from warnings import warn
from pseudo import chebdif, clencurt, chebintegrate, chebint, chebnorm

#from pseudo.py import chebint

defaultDict = {'alpha':1.14, 'beta' : 2.5, 'omega':0.0, 'L': 23, 'M': 23, 'nd':3,'N': 35, 'K':0,
               'Re': 400.0, 'isPois':0.0, 'noise':0.0 , 'lOffset':0.0, 'mOffset':0.0}

divTol = 1.0e-06
pCorrTol = 1.0e-04


    

def getDefaultDict():
    return defaultDict.copy()

def verify_dict(tempDict):
    """Verify that the supplied flowDict has all the parameters required"""
    change_parameters = False
    if tempDict is None:
        tempDict = defaultDict.copy()
        warn('No flowDict was supplied. Assigning the default dictionary')
    else: 
        for key in defaultDict:
            assert key in tempDict, 'Some dictionary keys are missing'
    [tempDict['K'],tempDict['L'],tempDict['M'],tempDict['N'],tempDict['isPois']] = [int(abs(k)) for k in [tempDict['K'],tempDict['L'],tempDict['M'],tempDict['N'],tempDict['isPois']]]
    if tempDict['alpha'] == 0. and tempDict['L'] != 0.: 
        tempDict['L'] == 0 
        warn('alpha is zero in the dictionary, so L has been set to zero too')
    if tempDict['beta'] == 0. and tempDict['M'] != 0.: 
        tempDict['M'] == 0 
        warn('beta is zero in the dictionary, so M has been set to zero too')
    if tempDict['omega'] == 0. and tempDict['K'] != 0.: 
        tempDict['K'] == 0 
        warn('omega is zero in the dictionary, so K has been set to zero too')
    return tempDict

def read_dictFile(dictFile):
    """Read flowDict from file. MUST use "flowConfig.txt" as template. """
    tempDict = {}
    with open("flowConfig.txt",'r') as f:
        for line in f:
            (key,val) = line.split()[:2]
            tempDict[key] = float(val)    
    return tempDict


    

class flowField(np.ndarray):
    """
    This module provides a class to define u,v,w (or scalars such as pressure) in 4D: t, x,z,y. 
    The shape of a class instance is (nt,nx,nz,nd,N): nt,nx,nz are harmonics (in omega, alpha,beta) 
        of Fourier modes in t,x,z respectively.
    nd is the number of components, 3 for [u,v,w]. 
    Scalars and non-3d fields can be created by setting 'nd' appropriately (nd=1 for scalars).
    N refers to Chebyshev collocation nodes

    Class attributes:
        self:   flowField instance of shape (nt,nx,nz,nd,N), inherits np.ndarray
        nt, nx, nz : length of axes 0,1, and 2 respectively
        nd:     Number of components of vector field. =1 for scalars. Length of axis 3
        N:      Number of Chebyshev collocation nodes.
        y:      Chebyshev collocation grid, because there's way too many calls being made to chebdif
        D,D2:   Chebyshev differentiation matrices, same reason as above   
        lOffset:When non-zero, indicates that the streamwise modes are not harmonics of 
                    fundamental frequency (alpha) but are offset from harmonics by a constant `lOffset`
        mOffset:Same as lOffset, for spanwise modes
        flowDict: 
            defaultDict = {'alpha':1.14, 'beta' : 2.5, 'omega':0.0, 'L': 23, 'M': 23, 'nd':3,'N': 35, 'K':0,
                   'Re': 400.0, 'isPois':0.0, 'noise':0.0 }
            'noise' is currently not implemented, but will later be used to initialize "random" flowField instances


    Methods (names only. See doc-strings for methods for template): 
        verify, view1d, view4d 
        slice, getScalar, appendField, copyArray
        real, imag, conjugate, abs
        ddt, ddx, ddx2, ddz, ddz2, ddy, ddy2, intX, intY, intZ
        grad3d, grad2d, grad, div, laplacian, curl3d, curl
        convLinear, convNL, convSemiLinear
        dot, sumAll, norm
        residuals, solvePressure
        ifft, getPhysical, makePhysical, makePhysicalPlanar

    It must always be ensured that the dictionary, self.flowDict, is always consistent with the flowField instance.
    Unless one is absolutely sure that the dictionary attributes don't need to be changed, 
        the arrays should not be accessed directly. Either the methods must be used. 
        For cases when a method isn't appropriate, the dictionary must be appropriately modified.

    self.verify() ensures that at least the shape attributes are self-consistent. 
    alpha, beta, omega, Re are not verified with self.verify()

    Initialization:
        flowField() creates an instance using a default dictionary: a 3 component zero-vector of shape (1,47,24,3,35).
        flowField(flowDict=dictName) creates an instance with shape attributes as defined in the dictionary.
            If the dictionary does not have all the keys needed, an assertion error is printed
        flowField(dictFile='flowConfig.txt') creates an instance using the attributes defined in the file flowConfig.txt
            The file flowConfig.txt has its lines formatted to facilitate being read by a function in this module. DO NOT EDIT IT except for the values
        flowField(arr=initArr, flowDict=dictName)
            Unless an array is passed using the keyword 'arr', the instance is initialized with zeros

    All three arguments can be used to provide a dictionary (arr can be an instance of flowField).
    flowDict argument has highest priority in defining the dictionary, 
        followed by dictFile
        followed by arr.flowDict (when arr is an instance of flowField or its subclass)
    If none of the above arguments provide a flowDict, a default dictionary (defined in the module) is used.
    A warning message is printed when the default dictionary is used.
            
    """
    def __new__(cls, arr=None, flowDict=None, dictFile= None):
        """Creates a new instance of flowField class with arguments (arr=None,flowDict=None,dictFile=None)
        """
        if flowDict is None:
            if dictFile is None:
                if hasattr(arr,'flowDict'):
                    flowDict = verify_dict(arr.flowDict)
                else:
                    flowDict=verify_dict(flowDict)
            else:
                flowDict = verify_dict(read_dictFile(dictFile))
        else:
            flowDict = verify_dict(flowDict)
        
        L = flowDict['L']
        M = flowDict['M']
        N = flowDict['N']
        K = flowDict['K']
        nd = flowDict['nd']
        nt = 2*K+1
        nx = 2*L+1
        nz = 2*M+1 
        
        if arr is None:
            arr=np.zeros(nt*nx*nz*nd*N,dtype=np.complex)
        else:
            if arr.size != (nx*nz*nt*nd*N):
                raise RuntimeError('The parameters in the dictionary are not consistent with the size of the supplied array')
            if arr.dtype == np.float:
                arr = (arr+1.j*np.zeros(arr.shape))
        obj = np.ndarray.__new__(cls,shape=(nt,nx,nz,nd,N),dtype=np.complex,buffer=arr.copy())
        
        obj.flowDict = flowDict.copy()
        obj.nx = nx
        obj.nz = nz
        obj.nt = nt
        obj.N = N
        obj.nd = flowDict['nd']
        yCheb,DM = chebdif(N,2)
        obj.y = yCheb
        obj.D = DM[:,:,0].reshape((N,N)); obj.D2 = DM[:,:,1].reshape((N,N))
        
        return obj
        
    
    def __array_finalize__(self,obj):
        # if self.dtype != np.complex:
            # warn('flowField class is designed to work with complex array entries\n'+
                 # 'To obtain real/imaginary parts of an instance, use class methods "real()" and "imag()"')
        if obj is None: return
         
        self.flowDict = getattr(self,'flowDict',obj.flowDict.copy())
        self.nt = getattr(self,'nt',obj.nt)
        self.nx = getattr(self,'nx',obj.nx)
        self.nz = getattr(self,'nz',obj.nz)
        self.nd = getattr(self,'nd',obj.nd)
        self.N = getattr(self,'N',obj.N)
        self.y = getattr(self,'y',obj.y)
        self.D = getattr(self,'D',obj.D)
        self.D2 = getattr(self,'D2',obj.D2)
        return

    
    def verify(self):
        """Ensures that the size of the class array is consistent with the dictionary entries. 
        Use this when writing new methods or tests"""
        self.flowDict = verify_dict(self.flowDict)  # Check that all keys exist
        # Next, check that the values in the dictionary match the class attributes
        if not ((self.nt == 2*self.flowDict['K']+1) and (self.nx == 2*self.flowDict['L']+1) and (self.nz == 2*self.flowDict['M']+1) and
                (self.N == self.flowDict['N']) and (self.nd == self.flowDict['nd'])   ): 
            raise RuntimeError('The shape attributes of the flowField instance are not consistent with dictionary entries')
        assert self.size == self.nt*self.nx*self.nz*self.nd*self.N, 'The size of the flowField array is not consistent with its shape attributes'
        return
        
        
    def view1d(self):
        """ Returns a 1d view. 
        Don't try to figure out what the ordering is, just use self.view4d() to get an organized view"""
        return self.reshape(self.size)
    
    def view4d(self):
        """ Returns a 4d view (actually, a 5-D array): (omega, alpha, beta, field=u,v,w,p, N)"""
        return self.reshape((self.nt,self.nx,self.nz,self.nd,self.N))

    def slice(self,K=None,L=None,M=None,nd=None,N=None,flowDict=None):
        """
        Returns a class instance with increased/reduced K,L,M,nd,N
        Call as new_inst = myFlowField.slice(K=Knew,L=Lnew,N=Nnew)) to change values of K,L,N without affecting M (and nd)
        When the number of Fourier modes (K,L,M, or nt,nx,nz) are smaller than what is requested, 
            additional zero modes are added. For Chebyshev nodes, interpolation is used"""
        obj = self.copyArray()

        """ THERE MIGHT BE ISSUES WITH ARRAYS NOT BEING CONTIGUOUS.
        IF THAT HAPPENS USE np.ascontiguousarray(arr) WHEREVER THE ERROR SHOWS UP
        """
        nxt = self.nx
        ntt = self.nt
        nzt = self.nz
        ndt = self.nd
        Nt = self.N
        flowDict_temp = self.flowDict.copy()
        if (K is not None) and (K != self.flowDict['K']):
            K = int(abs(K))
            Kt = flowDict_temp['K']               # Temporary name for 'K' of self
            if K <= Kt:
                obj = obj[Kt-K:Kt+K+1]
            else: 
                obj = np.concatenate((  np.zeros((abs(Kt-K),nxt,nzt,ndt,Nt),dtype=np.complex), obj,
                               np.zeros((abs(Kt-K),nxt,nzt,ndt,Nt),dtype=np.complex)  ), axis=0)
            flowDict_temp['K']= K
            ntt = 2*K+1
        
        if (L is not None) and (L != self.flowDict['L']):
            L = int(abs(L))
            Lt = flowDict_temp['L']               # Temporary name for 'L' of self
            if L <= Lt:
                obj = obj[:,Lt-L:Lt+L+1]
            else: 
                obj = np.concatenate((  np.zeros((ntt,abs(Lt-L),nzt,ndt,Nt),dtype=np.complex), obj,
                               np.zeros((ntt,abs(Lt-L),nzt,ndt,Nt),dtype=np.complex)  ), axis=1)
            flowDict_temp['L']= L
            nxt = 2*L+1
        
        if (M is not None) and (M != self.flowDict['M']):
            M = int((abs(M)))
            Mt = flowDict_temp['M']               # Temporary name for 'M' of self
            if M <= Mt:
                obj = obj[:,:,Mt-M:Mt+M+1]
            else: 
                obj = np.concatenate((  np.zeros((ntt,nxt,abs(Mt-M),ndt,Nt),dtype=np.complex), obj,
                               np.zeros((ntt,nxt,abs(Mt-M),ndt,Nt),dtype=np.complex)  ), axis=2)
            flowDict_temp['M']= M
            nzt = 2*M+1 
        
        if (N is not None) and (N != self.flowDict['N']):
            N = abs(int(N))
            Nt = flowDict_temp['N']
            if N != Nt:
                y = chebdif(N,1)[0]
                obj_t = obj.reshape((obj.size//Nt,Nt))
                obj = np.zeros((obj_t.size//Nt,N),dtype=np.complex)
                for n in range(obj_t.size//Nt):
                    obj[n] = chebint(obj_t[n],y)
            obj = obj.reshape(obj.size)
            flowDict_temp['N'] = N

        obj = np.ascontiguousarray(obj)  # Making sure that the array is a continuous block of memory
        
        obj = flowField.__new__(self.__class__,arr=obj, flowDict = flowDict_temp).view4d()
        
        if (nd is not None):
            nd = np.asarray([nd])
            nd = nd.reshape(nd.size)
            obj = obj[:,:,:,nd]
            obj.flowDict['nd'] = nd.size
            obj.nd = nd.size
        
        obj.verify()
        return obj.view4d()
    
    def getScalar(self,nd=0):
        """Returns the field Variable in the flowField instance identified by the argument "nd".
        Default for "nd" is 0, the first scalar in the flowField (u)"""
        if type(nd) != int:
            raise RuntimeError('getScalar(nd=0) only accepts integer arguments')
        obj = self.view4d()[:,:,:,nd:nd+1].copy()
        obj.flowDict['nd'] = 1
        obj.nd = 1
        return obj.view4d()

    def appendField(self,*args):
        """Append one or more fields at the end of "self". To append "p" to "uVec", call as uVec.appendField(p)
        Note: Both uVec and p must be flowField objects, each with their flowDict"""
        tempDict = self.flowDict.copy()
        v1 = self.view4d().copyArray()
        for obj in args:
            if not ( (self.nt==obj.nt) and (self.nx==obj.nx) and (self.nz==obj.nz) and (self.N==obj.N)):
                obj = obj.slice(K = self.flowDict['K'],L = self.flowDict['L'],M = self.flowDict['M'],N = self.flowDict['N'])
                warn('obj did not have the same shape as self, obj has been sliced to match shapes')
            v2 = obj.view4d().copyArray()
            v1=np.append(v1,v2,axis=3)
            tempDict['nd'] += obj.flowDict['nd']
        return flowField.__new__(self.__class__,arr=v1, flowDict=tempDict).view4d()
    
    def copyArray(self):
        """ Returns a copy of the np.ndarray of the instance. 
        This is useful for manipulating the entries of a flowField without bothering with all the checks"""
        return self.view(np.ndarray).copy()
    
    def real(self):
        """ Returns the real part of the flowField (the entries are still complex, with zero imaginary parts)"""
        return flowField.__new__(self.__class__,arr=self.copyArray().real,flowDict=self.flowDict)
    
    def imag(self):
        """ Returns the imaginary part of the flowField (the entries are still complex, with zero imaginary parts)"""
        return flowField.__new__(self.__class__,arr=self.copyArray().imag,flowDict=self.flowDict)
    
    def conjugate(self):
        """ Returns complex conjugate of flowFIeld instance"""
        return self.real()-1.j*self.imag()

    def abs(self):
        """Returns absolute value of entries of flowField instance (still expressed as complex numbers, but with zero imaginary part and positive real part)"""
        return flowField.__new__(self.__class__,arr=np.abs(self.copyArray()),flowDict=self.flowDict.copy())
    
    
    def ddt(self):
        """ Returns a flowField instance that gives the partial derivative along "t" """
        partialT = self.view4d().copy()
        kArr = np.arange(-self.flowDict['K'],self.flowDict['K']+1).reshape(self.nt,1,1,1,1)
        partialT[:] = -1.j*self.flowDict['omega']*kArr*partialT # Fourier modes are e^(i(ax+bz-wt))
        return partialT
    
    def ddx(self):
        """ Returns a flowField instance that gives the partial derivative along "x" """
        lArr = (self.flowDict['lOffset']+np.arange(-self.flowDict['L'],self.flowDict['L']+1)).reshape(1,self.nx,1,1,1)
        partialX = 1.j*self.flowDict['alpha']*lArr*self.view4d().copy()
        return partialX
    
    def ddx2(self):
        """ Returns a flowField instance that gives the second partial derivative along "x" """
        partialX2 = self.view4d().copy()
        l2Arr = ((self.flowDict['lOffset']+np.arange(-self.flowDict['L'],self.flowDict['L']+1))**2).reshape(1,self.nx,1,1,1)
        partialX2[:] = -self.flowDict['alpha']**2*l2Arr*partialX2
        return partialX2
    
    def ddz(self):
        """ Returns a flowField instance that gives the partial derivative along "z" """
        partialZ = self.view4d().copy()
        M = self.flowDict['M']
        mArr = (self.flowDict['mOffset']+np.arange(-self.flowDict['M'],self.flowDict['M']+1 )).reshape((1,1,self.nz,1,1))
        partialZ[:] = 1.j*self.flowDict['beta']*mArr*partialZ
        return partialZ
    
    def ddz2(self):
        """ Returns a flowField instance that gives the second partial derivative along "z" """
        partialZ2 = self.view4d().copy()
        mArr = (self.flowDict['mOffset']+np.arange(-self.flowDict['M'],self.flowDict['M']+1  )).reshape((1,1,self.nz,1,1))
        m2Arr = mArr**2
        partialZ2[:] = -self.flowDict['beta']**2*m2Arr*partialZ2
        return partialZ2
    
    def ddy(self):
        """ Returns a flowField instance that gives the partial derivative along "y" """
        N = self.N
        partialY = self.view1d().copy()
        tempArr = self.reshape(self.size//N,N)
        partialY[:] = np.dot(tempArr,self.D.T).reshape(self.size)
        return partialY.view4d()
    
    def ddy2(self):
        """ Returns a flowField instance that gives the partial derivative along "y" """
        N = self.N
        partialY2 = self.view1d().copy()
        tempArr = self.reshape(self.size//N,N)
        partialY2[:] = np.dot(tempArr,self.D2.T).reshape(self.size)
        return partialY2.view4d()
    
    def intX(self):
        """ Integrate each Fourier mode of each scalar along streamwise 
        Returns a flowField object of the same size of self.
        The constant of integration is decided so that at x=0, the integral is 0 (i.e., starting integration from x=0)"""
        # f(x,y,z) = \sum_l \sum_m  c_lm(y) exp(ilax) exp(imbz)
        # \int f(x,y,z) dx = \sum_l \sum_m  c_lm(y) exp(imbz) [\int exp(ilax) dx ]
        # For l != 0, \int exp(ilax) dx = 1/ila  exp(ilax) - 1/ila
        # For l == 0, \int exp(ilax) dx = x
        #     In the above two lines, the integration was performed from x=0 to some x
        # Fourier( \int f(x,y,z) dx )_lm  =  1/ila* c_lm(y)                           for l != 0
        #                                 =  c_0m(y)*x - \sum_(l!=0) (1/ila)*c_lm(y)  for l == 0
        a = self.flowDict['alpha']
        tol = 1.0e-9
        if a == 0.:
            integralX = self.view4d().copy()
            integralX[:] = 0.+0.j
            if self.norm() >= tol:
                warn("Integral in x cannot be represented by Fourier series for alpha = 0 with non-zero Fourier coeffs, account for c_0m(y)*x separately")
            return integralX

        L = self.flowDict['L']
        lArr = lOffset+np.arange(-L, L+1).reshape((1,self.nx,1,1,1))
        #lArr has a zero, setting that to 1 for now (because I'll divide by lArr in a bit)
        zeroInd = np.squeeze(np.argwhere(lArr==0)) # If an 'l' is zero, set it to 1 in lArr
        lArr[0,zeroInd,0,0,0] = 1.
        
        integralX = self.view4d().copy()/lArr/1.j/a
        integralX[:,zeroInd] =  np.sum(integralX, axis=1) 
        integralX[:,zeroInd] += 1./1.j/a*self[:,zeroInd]
        
        if zeroInd:        
            # All l!=0 modes are now set. Next, to l=0 modes
            # I have added a 1/ia*c_0m to the (0,m) modes that shouldn't actually be added
            # But that's not an issue, because I'll be subtracting that below:

            # Now, the c_0m(y)*x isn't actually a constant- it varies linearly with x
            # The linear function 'x' is not periodic, and hence cannot be represented by a Fourier series (-La,..,0,..,La)
            # This wouldn't be an issue as long as c_0m(y) is 0. Warn if it isn't zero
            zeroMode = integralX[:,zeroInd]
            if chebnorm(zeroMode.reshape(zeroMode.size),self.N) >= tol :
                warn("Integral in x cannot be represented by Fourier series if the zero mode has non-zero Fourier coefficient, account for c_0m(y)*x separately")
        else:
            warn("Integrating in 'x' does not work properly if lOffset is not an integer, because integrating non-zero Fourier modes definitely gives rise to constants of integration that cannot be captured without a zero Fourier basis mode")

        integralX.verify()
        return integralX

    def intZ(self):
        """ Integrate each Fourier mode of each scalar along spanwise
        Returns a flowField object of the same size of self.
        The constant of integration is decided so that at z=0, the integral is 0 (i.e., starting integration from z=0)"""
        # Refer to formulation in method intX()
        b = self.flowDict['beta']
        tol = 1.0e-9
        if b == 0.:
            integralZ = self.view4d().copy()
            integralZ[:] = 0.+0.j
            if self.norm() >= tol:
                warn("Integral in z cannot be represented by Fourier series for alpha = 0 with non-zero Fourier coeffs, account for c_0m(y)*x separately")
            return integralZ

        M = self.flowDict['M']
        mArr = mOffset+np.arange(-M, M+1).reshape((1,1,self.nz,1,1))
        #mArr may have a zero, setting that to 1 for now (because I'll divide by mArr in a bit)
        zeroInd = np.squeeze(np.argwhere(mArr==0)) # If an 'l' is zero, set it to 1 in lArr
        mArr[0,0,zeroInd,0,0] = 1.
        
        integralZ = self.view4d().copy()/mArr/1.j/b
        integralZ[:,:,zeroInd] =  np.sum(integralZ, axis=2) 
        integralZ[:,:,zeroInd] += 1./1.j/b*self[:,:,zeroInd]
        
        if zeroInd:        
            # Refer to comments for method intX()
            zeroMode = integralZ[:,:,zeroInd]
            if chebnorm(zeroMode.reshape(zeroMode.size),self.N) >= tol :
                warn("Integral in z cannot be represented by Fourier series if the zero mode has non-zero Fourier coefficient, account for c_m0(y)*z separately")
        else:
            warn("Integrating in 'z' does not work properly if mOffset is not an integer, because integrating non-zero Fourier modes definitely gives rise to constants of integration that cannot be captured without a zero Fourier basis mode")

        integralZ.verify()
        return integralZ
    
    

    def intY(self):
        """ Integrate each Fourier mode of each scalar along the wall-normal axis
        Returns a flowField object of the same size as self.
        Use this method to compute variables from their wall-normal derivatives"""
        integral = self.copy().reshape((self.size/self.N, self.N))
        arr = integral.copyArray()
        for n in range(np.int(integral.size/integral.N)):
            integral[n] = chebintegrate(arr[n])
        integral.verify()
        return integral.view4d()
    
    def flux(self,nd=0):
        """ Use this method to calculate volume fluxes (supposing 'self' refers to velocity vector)
            Default is the streamwise volume flux (argument: nd=0) at x=0
            Pass nd=1 for wall-normal volume flux at y=0
            Pass nd=2 for spanwise volume flux at z=0
            Return, say for nd=0:  1/lambda_z *\int_(z=0)^(z=lambda_z)   0.5* \int_(y=-1)^(y=1)  scalar* dy * dz
            """
        scalar = self.getScalar(nd=nd)
        if nd == 0:
            # .intY() and .intZ() give definite integrals with integrals being zero at y=-1 and z=0 respectively
            # If homogeneous along z, integrate only along Y and not Z, else integrate along z to z=2*pi/beta
            if self.flowDict['beta']== 0.:
                integratedScalar = scalar.intY()
                flux = 0.5*integratedScalar.ifft()[0,0]
            else:
                L = self.flowDict['L']; M = self.flowDict['M']
                # Zeroth mode cannot be expressed as Fourier mode after integration, so it's accounted for separately
                zeroMode = scalar.copyArray()[0,L,M,0]      # Copying zeroth mode 
                scalar[0,L,M,0] = 0.                        # Getting rid of zeroth mode
                integratedScalar = scalar.intY().intZ()
                lambdaZ = 2.*np.pi/self.flowDict['beta']
                flux = 0.5/lambdaZ*integratedScalar.ifft(zLoc=lambdaZ)[0,0]
                flux += 0.5*np.dot(clencurt(self.N),zeroMode)           # Adding flux due to zeroth mode
        elif nd == 2:
            # If homogeneous along x, return the integral at x=1., else at x=2*pi/alpha
            if self.flowDict['alpha']== 0.:
                integratedScalar = scalar.intY()
                flux = integratedScalar.ifft()[0,0]
            else:
                L = self.flowDict['L']; M = self.flowDict['M']
                # Zeroth mode cannot be expressed as Fourier mode after integration, so it's accounted for separately
                zeroMode = scalar.copyArray()[0,L,(abs(M)-M)//2,0]      # Copying zeroth mode 
                scalar[0,L,(abs(M)-M)//2,0] = 0.                        # Setting rid of zeroth mode
                integratedScalar = scalar.intY().intX()
                lambdaX = 2.*np.pi/self.flowDict['alpha']
                flux = 0.5/lambdaX*integratedScalar.ifft(xLoc=lambdaX)[0,0]
                flux += 0.5*np.dot(clencurt(self.N),zeroMode)           # Adding flux due to zeroth mode
        elif nd == 1: pass
            # I think this follows from divergence-free condition for steady flow. Will derive it later if I need it flux = 0.
        else: raise RuntimeError('nd must be 0,1,or 2')
        return np.real(flux)
    
    def grad(self, nd=0):
        """ Computes gradient (in 3d by default) of either a scalar flowField object, 
            or of one variable (identified by nd) in a vector flowField object (default is first variable in object). 
            """
        scal = self.getScalar(nd=nd)        # Extract the scalar field whose gradient is to be calculated
        gradVec = scal.ddx().appendField(scal.ddy(),scal.ddz())
        return gradVec
        
    def laplacian(self):
        """ Computes Laplacian for a flowField instance """
        return self.ddx2() + self.ddy2() + self.ddz2()
            
    def div(self):
        """ Computes divergence of vector field as u_x+v_y+w_z
        If a flowField with more than 3 components (nd>3) is supplied, takes first three components as u,v,w."""
        assert self.nd >= 3, ('Divergence is defined only for 3C fields')
        return self.getScalar(nd=0).ddx() + self.getScalar(nd=1).ddy() + self.getScalar(nd=2).ddz()
        
    def curl(self):
        assert self.nd >= 3, 'curl3d method is defined only for 3C fields.'
        return makeVector(self.getScalar(nd=2).ddy() - self.getScalar(nd=1).ddz(),\
                         self.getScalar(nd=0).ddz() - self.getScalar(nd=2).ddx(),\
                         self.getScalar(nd=1).ddx() - self.getScalar(nd=0).ddy())
   

    def __sumAll(self):
        """Sums all elements of a flowField object (along all axes)"""
        obj = self.view4d().copyArray()
        return np.sum(np.sum(np.sum(np.sum(np.sum(obj,axis=4),axis=3),axis=2),axis=1),axis=0)
   

    def dot(self, vec2):
        """Computes inner product for two flowField objects, scalar or vector,
            by integrating {self[nd=j]*vec2[nd=j].conj()} along x_j, and adding the integrals for j=1,..,self.nd.
        Currently, only inner products of objects with identical dictionaries are supported"""
        assert (self.flowDict == vec2.flowDict), 'Method for inner products is currently unable to handle instances with different flowDicts'
        
        w = clencurt(self.N).reshape((1,1,1,1,self.N))
        return flowField.__sumAll(self.view4d()*vec2.conjugate().view4d()*w)
   

    def norm(self):
        """Integrates v[nd=j]*v[nd=j].conjugate() along x_j, sums across j=1,..,self.nd , and takes its square-root"""
        return np.sqrt(np.abs(self.dot(self)))
   

    def weighted(self):
        """Weights self by sqrt(W) (where W is the Clenshaw-Curtis quadrature weighting), and returns a 1-D np.ndarray
        When using .dot() or .norm(), what is done is \int W*v1*v2'  
        Another way to do the same is to pre-multiply vectors v1 and v2 with sqrt(W), 
            and then use the regular vector dot product to compute the weighted dot product
        NOTE: RETURNS A NP.NDARRAY OBJECT
        Returning a flowFieldWavy instance makes it ambiguous, because I might use a weighted instance
            as one that isn't weighted, and that would ruin the calculations. 
        """ 
        q = np.sqrt(clencurt(self.N).reshape((1,1,1,1,self.N)))
        return ((q*self.view4d()).view1d()).copyArray()
    
    
    def convNL(self):
        """Computes the non-linear convection term
        Warning: Currently, the code assumes that the flowField supplied is that of a steady flow. Temporal frequencies are not accounted for"""
        
        assert self.flowDict['lOffset'] == 0. and self.flowDict['mOffset']==0. ,\
            'convNL() method is currently not supported for flowFields with offsets in l and m.'
        assert self.flowDict['K'] == 0. and self.flowDict['omega']==0. ,\
            'convNL() method is currently not supported for flowFields with time-dependence'
        y = self.y
        
        # Ensuring a full set -|M|b,...,0b,..,|M|b is available before computing the convection term
        u = self.getScalar(nd=0);  v = self.getScalar(nd=1); w = self.getScalar(nd=2); tempDict = self.flowDict.copy()
        K = tempDict['K']; L = tempDict['L']; M = tempDict['M']; N = tempDict['N']
        nx = self.nx; nz= self.nz

        # Ensuring that u,v and w represent physical quantities, by making sure wavenumber pairs are complex conjugates
        if L != 0:
            u[0,L+1:] = 0.5*( u[0,L+1:] + np.conj( u[0,L-1::-1, ::-1]) )
            u[0,L-1::-1,::-1] = 0.5*( u[0,L-1::-1, ::-1] + np.conj( u[0,L+1:]) )
        
                
        tempDict['nd'] = 3      # Just in case 'self' includes pressure data
        # convTerm = flowField.__new__(self.__class__,flowDict=tempDict).view4d()
        convTerm = self.view4d().copyArray()
        
        ux = u.ddx().copyArray(); uy = u.ddy().copyArray(); uz = u.ddz().copyArray()
        vx = v.ddx().copyArray(); vy = v.ddy().copyArray(); vz = v.ddz().copyArray()
        wx = w.ddx().copyArray(); wy = w.ddy().copyArray(); wz = w.ddz().copyArray()
        u = u.copyArray();  v = v.copyArray();  w = w.copyArray()
        
        # We use this function later when computing all contributing pairs of wavenumber vectors 
        #   to a particular wavenumber vector, such as (3,0),(4,5) contributing to (7,5)
        sumArr = lambda x: np.sum( x.reshape(self.nt,x.size//self.nt//N,N), axis=1)
        for lp in range(self.nx//2+1):
        # for lp in range(self.nx):
            l = lp - L
            l1 = l; l2 = None; l3 = None; l4 = l1-1; 
            if l == 0: l4 = None
            if l < 0:  
                l1 = None; l2 = self.nx+l; l3 = l2-1; l4 = None
                
            for mp in range(self.nz):
                m = mp - M
                m1 = m; m2 = None; m3 = None; m4 = m1-1; 
                if m == 0: m4 = None
                if m < 0: 
                    m1 = None; m2 = self.nz+m; m3 = m2-1; m4 = None
                # Magic happens here:
                convTerm[:,lp,mp,0] = sumArr(u[:,l1:l2,m1:m2]*ux[:,l3:l4:-1,m3:m4:-1]
                                                + v[:,l1:l2,m1:m2]*uy[:,l3:l4:-1,m3:m4:-1]
                                                + w[:,l1:l2,m1:m2]*uz[:,l3:l4:-1,m3:m4:-1])
                
                convTerm[:,lp,mp,1] = sumArr(u[:,l1:l2,m1:m2]*vx[:,l3:l4:-1,m3:m4:-1]
                                               + v[:,l1:l2,m1:m2]*vy[:,l3:l4:-1,m3:m4:-1]
                                               + w[:,l1:l2,m1:m2]*vz[:,l3:l4:-1,m3:m4:-1])
                
                convTerm[:,lp,mp,2] = sumArr(u[:,l1:l2,m1:m2]*wx[:,l3:l4:-1,m3:m4:-1]
                                              + v[:,l1:l2,m1:m2]*wy[:,l3:l4:-1,m3:m4:-1]
                                              + w[:,l1:l2,m1:m2]*wz[:,l3:l4:-1,m3:m4:-1])
                # Just collecting all wavenumber vectors that add up
                #   to give (lp,mp), and doing it for u_j * partial_j(u_i)
                # It might look like should change if we're using wavy walls, but it doesn't,
                #   because the .ddx(), .ddy(), .ddz() methods of flowFieldWavy class already 
                #   account for the effects of the coordinate mapping
        convTerm[0,L+1:] = np.conj(convTerm[0,L-1::-1,::-1])

        convTerm = flowField.__new__(self.__class__,arr=convTerm.reshape(self.size),flowDict=self.flowDict.copy())
        return convTerm
        
    def residuals(self,pField=None, divFree=False):
        """ Computes the residuals of ONLY the momentum equations for a velocity field.
        F(state) =  u_j * partial_j (u_i) + partial_i (p) - 1/Re* partial_jj (u_i) = 0

        Args:
        pField is the pressure field (optional). 
            When not supplied, the pressure is taken to be zero everywhere
        nonLinear (flag) defaults to True
            When set to False, convLinear() is used to evaluate convection term. When true, convNL() is used.
        divFree (flag) defaults to False
            When set to False, nothing is done. This means the field could have a non-zero divergence
            When set to True, wall-normal velocity is changed to ensure divergence is zero.
                But this doesn't change self, instead, the corrected wall-normal velocity is returned as a second argument
                To correct the wall-normal velocity in self, use:
                    >> residual, v = vF.residuals(divFree=True);    vF[:,:,:,1:2] = v 
        When only a velocity field is available, use 
                >> self.solvePressure()[1] 
            to get the residuals instead."""
        assert self.nd == 3, "Method only accepts 3C flowfields, pass pressure using the keyword argument pField"
        tempVec = self.getScalar(nd=1).view4d()
        residual = self.copy(); residual[:] = 0.
        K = self.flowDict['K']; L = self.flowDict['L']; M = self.flowDict['M']; N = self.N
        vCorrection = self.getScalar(nd=0); vCorrection[:] = 0.
        if divFree:
            # u_x + v_y + w_z = div. 
            # To ensure divergence is zero, correct 'v' as v += - \int div * dy
            divergence = self.div()
            divergence[np.abs(divergence.copyArray()) < divTol] = 0.j
            vCorrection = -divergence.intY()
            self.view4d()[:,:,:,1:2] += vCorrection
        
        if pField is None:
            pField = self.getScalar(); pField[:] = 0.  
        else: 
            assert (pField.nd == 1) and (pField.size == self.size//3), 'pField should be a scalar of the same size as each scalar of velocity'
        
        residual[:] = pField.grad() - (1./self.flowDict['Re'])*self.laplacian()
        if self.flowDict['isPois'] ==1:
            residual[K,L,M,0] -= 2./self.flowDict['Re']     # adding dP/dx, the mean pressure gradient

        residual[:] += self.convNL()

        vCorrected = self.getScalar(nd=1)
        self.view4d()[:,:,:,1:2] -= vCorrection 
        
        residual[:,:,:,:,[0,-1]] = self[:,:,:,:,[0,-1]]     
        # Residual at walls is given by the velocities at the walls, this serves as the BC

        # For Couette flow, the BCs on streamwise velocity aren't zero
        if self.flowDict['isPois'] == 0:
            residual[K,L,M,0,0] -= 1. 
            residual[K,L,M,0,-1] -= -1. 

        return residual, vCorrected     
    
    def solvePressure(self, pField=None, residuals=None, divFree=False, nonLinear=True):
        """ Solves for pressure, given a 3C velocity field.
            RETURNS TWO FLOWFIELD INSTANCES: The first one is the corrected pField, the second is the residual when the corrected pressure is used
        If pField is supplied, only corrections about this field need to be calculated. The returned field is pField + corrections computed.
        If residuals is supplied (should be evaluations of the momentum equations using 'self' and 'pField'), 
            the NSE need not be evaluated again, and solving for the pressure field is quite fast
        If residuals is not supplied, the momentum equations are evaluated in the function, and that takes a while
        NOTE: The method does not solve a Poisson equation. 
        Instead, the following approach is used:
            Wall-normal momentum gives wall-normal derivative of each pressure Fourier mode
                Integrating the wall-normal gradient gives the pressure up to a constant 
                In this step, the constant is set such that the pressure at y=1 is zero for each mode
            The residuals of streamwise momentum equation and spanwise momentum equation are averaged to get the actual constant (minimizes the residuals)"""
        
        assert self.nd == 3, 'The flowField instance supplied must be a 3D velocity field'
        tempDict = self.flowDict.copy()
        tempDict['nd'] = 1
        pCorrection = flowField(flowDict=tempDict)
        
        if pField is None:
            pField = flowField(flowDict=tempDict)  # Initializes a zero field
        else:
            assert pField.size == self.size/3, 'pField must be of the same size of each component of the velocity field'
        
        # (u.div(v) - 1/Re* laplacian(v) ) + dp_1/dy + dp_Corr/dy= 0,     where p_1 = pField (input argument),p_Corr is the correction
        # p_Corr = - \int residual dy, where residual is the sum of first three terms on LHS
        
        pCorrection = None
        if residuals is None:
            residuals = self.residuals(pField=pField, divFree=divFree, nonLinear=nonLinear)
        else: 
            assert (residuals.nd == 3) and (residuals.size == self.size), 'residuals must be a 3D flowField object of the same size as self'
            
        pCorrection = -(residuals.getScalar(nd=1)).intY()
        # pCorrection is now determined upto a constant. Next, find the constant that minimizes residual for streamwise, spanwise
        
        residuals[:,:,:,1:2] += pCorrection.ddy()
        residuals[:,:,:,0:1] += pCorrection.ddx()
        residuals[:,:,:,2:3] += pCorrection.ddz()
        #assert residuals.getScalar(nd=1).norm() < 1.0e-6, 'Wall-normal residual has not gone below 1.0e-6 even after correcting dpdy. Weird'
        
        # (....) + ilap = 0
        # (....) + imbp = 0             a is \alpha, b is \beta, l and m identify Fourier mode
        # Constant that minimizes streamwise residual norm is   \int (-(residual_x)/ila) dy
        # Creating arrays to hold the constants
        constX = np.zeros((self.nt,self.nx,self.nz),dtype=np.complex) 
        constZ = constX.copy()
        
        w = clencurt(self.N).reshape((1,1,1,1,self.N))
        L = self.flowDict['L']; M = self.flowDict['M']
        a = self.flowDict['alpha']; b = self.flowDict['beta']
        lArr = np.arange(-L, L+1).reshape((1,self.nx,1,1)) 
        lArr[0,L] = 1.   # Avoiding zeros, will account for this later (about 10 lines below)
        mArr = np.arange( (M-abs(M))/2 , abs(M)+1 ).reshape((1,1,self.nz,1)) 
        mArr[0,0,-(M-abs(M))/2] = 1.   # Avoiding zeros
        
        # Corrections only make sense when la != 0 and mb != 0. So keep entries of constX, constZ as zero when la= 0 or mb=0
        if a != 0.:
            constX[:] = -np.sum( w* (residuals.copyArray()[:,:,:,0]/1.j/lArr/a ), axis=-1)
        if b != 0.:
            constZ[:] = -np.sum( w* (residuals.copyArray()[:,:,:,2]/1.j/mArr/b ), axis=-1)
        
        constX[:,L ]  = 0.
        constZ[:,:,(abs(M)-M)/2] = 0.
        
        const = None
        if a == 0: const = constZ
        elif b==0: const = constX
        else: 
            const = (constX + constZ)/2.
            const[:,L] += constZ[:,L]/2.; const[:,:,(abs(M)-M)/2] += constX[:,:,(abs(M)-M)/2] 
        
        const = const.reshape((self.nt,self.nx,self.nz,1))
        const[const<pCorrTol] = 0.
        
        residuals[:,:,:,0] += 1.j*lArr*a*const
        residuals[:,:,:,2] += 1.j*mArr*b*const
        pCorrection[:,:,:,0] += const
        
        return (pField.view4d()+pCorrection), residuals
    
###########################################################################################################
# 14 March 6 30 PM
    def direcDeriv(self, tLoc=0., xLoc=0., zLoc=0., yLoc=None, nd=0, direc=(1.,0.,0.)):
        """Returns the directional derivative AT A POINT of a single variable
        Arguments: tLoc, xLoc, zLoc, yLoc (pretty obvious what they are), can be floats, arrays, or lists
                        default locations are (0,0,0,-1)
                nd: Variable identifier. 0,1,2 respectively mean u,v,w. Defaults to nd=0 
                direc: a triplet that gives the direction in (x,y,z) along which derivative is needed
                    Defaults to (1,0,0), i.e., the streamwise direction
        """
        # First, calculate the gradient for the variable required
        gradient = self.grad3d(scalDim=nd, nd=3)
        # Obtaining the physical value of the gradient at the required location
        if yLoc is None:
            gradPhysical = gradient.getPhysical(tLoc=tLoc, xLoc=xLoc, zLoc=zLoc)[:,:,:,:,-1]
            # The -1 index refers to y=-1 
        else:
            gradPhysical = gradient.getPhysical(tLoc=tLoc, xLoc=xLoc, zLoc=zLoc, yLoc=yLoc)
        
        xComp = direc[0]; yComp = direc[1]; zComp = direc[2]
        # The direction vector needs to be a unit vector
        vecNorm = np.sqrt(xComp**2 + yComp**2 + zComp**2)
        xComp = xComp/vecNorm; yComp = yComp/vecNorm; zComp = zComp/vecNorm
        
        directionalDerivative = xComp*gradPhysical[:,:,:,0] + yComp*gradPhysical[:,:,:,1] + zComp*gradPhysical[:,:,:,2]
        
        return directionalDerivative
        
    
    def ifft(self,tLoc=0., xLoc=0., zLoc=0.):
        """ Returns a numpy array of shape (self.nd,self.N), the flow field at some location in (t,x,z)
        Arguments: tLoc (default=0.), xLoc (default=0.), zLoc (default=0.))
        Note: The returned array is of dtype np.real (float64), since all flow field variables are real quantities
            If, for some reason, a complex field variable is used, this needs to change
        To obtain field for multiple locations in t,x,z, use .printField(toFile=False)
            Do remember to have a look at the other arguments of .printField()"""
        fields = np.zeros((self.nd,self.N))
        a = self.flowDict['alpha']; b = self.flowDict['beta']; omega = self.flowDict['omega']
        K = self.flowDict['K'];  L = self.flowDict['L']; M = self.flowDict['M']

        if M > 0: 
            obj = self.slice(M=-M); M = -M
        else: obj = self

        if K == 0: kArr = np.ones((1,1,1,1,1))
        else: kArr = np.arange(-K, K+1).reshape((self.nt,1,1,1,1)) 
        if L == 0: lArr = np.ones((1,1,1,1,1))
        else:  lArr = np.arange(-L, L+1).reshape((1,self.nx,1,1,1)) 
        if M == 0: mArr = np.ones((1,1,1,1,1))
        else:  mArr = np.arange( M , abs(M)+1 ).reshape((1,1,obj.nz,1,1)) 
            
        sumArr = lambda arr: np.sum(np.sum(np.sum(arr,axis=0),axis=0),axis=0).real
        field = sumArr(  self.copyArray()*np.exp(1.j*(a*lArr*xLoc + b*mArr*zLoc - omega*kArr*tLoc))  )
        return field
        
    
    def getPhysical(self,tLoc = 0., xLoc=0., zLoc= 0., yLoc = None):
        """Returns the flow field at specified locations in t,x,z,y
        Arguments: tLoc (default=0.), xLoc (default=0.), zLoc (default=0.)
                    yLoc (default:None, corresponds to Chebyshev nodes on [1,-1] of cardinality self.N)
                    if yLoc is specified, it must be either 
        NOTE: yLoc here is referenced from the local wall locations. The walls are ALWAYS at +/-1, including for wavy walls"""
        tLoc = np.asarray(tLoc); xLoc = np.asarray(xLoc); zLoc = np.asarray(zLoc)
        # If, say, tLoc was initially a float, the above line converts it into a 0-d numpy array
        # I can't call tLoc as tLoc[0], since 0-d arrays can't be indexed. So, convert them to 1-d:
        tLoc = tLoc.reshape(tLoc.size); xLoc = xLoc.reshape(xLoc.size); zLoc = zLoc.reshape(zLoc.size)
        
        
        # Ensure that Fourier modes for the full x-z plane are available, not just the half-plane (m>=0)
        M = self.flowDict['M']
        if M > 0: 
            obj = self.slice(M=-M); M = -M
        else: obj = self
            
        if yLoc is None:
            field = np.zeros((tLoc.size, xLoc.size, zLoc.size, self.nd, self.N))
            for tn in range(tLoc.size):
                for xn in range(xLoc.size):
                    for zn in range(zLoc.size):
                        field[tn,xn,zn] = obj.ifft(tLoc=tLoc[tn], xLoc=xLoc[xn], zLoc=zLoc[zn])
            yLoc = chebdif(obj.N,1)[0]
        else:
            yLoc = np.asarray(yLoc).reshape(yLoc.size)
            assert not any(np.abs(yLoc)-1.>1.0e-7), 'yLoc must only have points in [-1,1]'
            field = np.zeros((tLoc.size, xLoc.size, zLoc.size, self.nd, yLoc.size))
            for tn in range(tLoc.size):
                for xn in range(xLoc.size):
                    for zn in range(zLoc.size):
                        fieldTemp = obj.ifft(tLoc=tLoc[tn], xLoc=xLoc[xn], zLoc=zLoc[zn])
                        for scal in range(self.nd):
                            field[tn,xn,zn,scal] = chebint(fieldTemp[scal], yLoc)

        return field
            
    
    def printPhysical(self,xLoc=None, zLoc=None, tLoc=None, yLoc=None,yOff=0.,pField=None, interY=2,fName='ff'):
        """Prints the velocities and pressure in a .dat file with columns ordered as Y,Z,X,U,V,W,P
        Arguments (all keyword):
            xLoc: x locations where field variables need to be computed 
                    (default: [0:2*pi/alpha] 40 points in x when alpha != 0., and just 1 (even when xLoc supplied) when alpha == 0.)
            zLoc: z locations where field variables need to be computed 
                    (default: [0:2*pi/beta] 20 points in z when beta != 0., and just 1 (z=0) when beta == 0.)
            tLoc: temporal locations (default: 7 points when omega != 0, 1 when omega = 0). Fields at different time-locations are printed to different files
            pField: Pressure field (computed with divFree=False, nonLinear=True if pField not supplied)
            interY: Field data is interpolated onto interY*self.N points before printing. Default for interY is 2
            yOff: Use this to define wavy surfaces. For flat walls, yOff = 0. For wavy surfaces, yOff = 2*eps
                    yOff is used to modify y-grid as   y[tn,xn,zn] += yOff*cos(alpha*x + beta*z - omega*t)
            yLoc: Use this to specify a y-grid. When no grid is specified, Chebyshev nodes are used
            fname: Name of .dat file to be printed to. Default: ff.dat
        """
        a = self.flowDict['alpha']; b = self.flowDict['beta']; omega = self.flowDict['omega']
        K = self.flowDict['K']; L = self.flowDict['L']; M=-np.abs(self.flowDict['M'])
        if (a==0.): 
            return self.printPhysicalPlanar(etaLoc=zLoc, tLoc=tLoc, yLoc=yLoc,yOff=yOff,pField=pField, interY=interY,toFile=toFile,fName=fName)
        if (b==0.): 
            return self.printPhysicalPlanar(etaLoc=xLoc, tLoc=tLoc, yLoc=yLoc,yOff=yOff,pField=pField, interY=interY,toFile=toFile,fName=fName)
        if xLoc is None:
            xLoc = np.arange(0., 2.*np.pi/a, 2.*np.pi/a/40.)
        if zLoc is None:
            zLoc = np.arange(0., 2.*np.pi/b, 2.*np.pi/b/20.)
        if tLoc is None:
            if omega != 0.: tLoc = np.arange(0,2.*np.pi/omega, 2.*np.pi/b/7.)
            else: tLoc = np.zeros(1)
        if yLoc is None:
            yLoc = chebdif(interY*self.N,1)[0]
            yLocFlag = False
        else:
            yLocFlag = True
            assert isinstance(yLoc,np.ndarray) and (yLoc.ndim == 1), 'yLoc must be a 1D numpy array'
            assert not any(np.abs(yLoc) > 1), 'yLoc must only have points in [-1,1]' 
            
        assert type(yOff) is float, 'yOff characterizes surface deformation and must be of type float'
        if '.dat' in fName[-4:]: fName = fName[:-4]
        
        assert self.nd == 3, 'makePhysical() is currently written to handle only 3C velocity fields'
        assert isinstance(xLoc,np.ndarray) and isinstance(zLoc,np.ndarray) and isinstance(tLoc,np.ndarray),\
            'xLoc, zLoc, and tLoc must be numpy arrays'
        assert isinstance(fName,str), 'fName must be a string'
        
        if pField is None: pField = self.solvePressure(divFree=False,nonLinear=True)[0]
        else:
            assert pField.size == self.size//3, 'pField must be the same size of each component of self'
        
        obj = self.appendField(pField)        
        
        if interY != 1 and not yLocFlag:
            obj = obj.slice(N=yLoc.size)
        
        
        dataArr = np.zeros((7,tLoc.size,xLoc.size,zLoc.size,yLoc.size))
        
        # Calculating flow field variables at specified t,x,z,y (refer to .ifft() and .getField())
        if not yLocFlag: fields = obj.getPhysical(tLoc=tLoc, xLoc=xLoc, zLoc=zLoc)
        else:  fields = obj.getPhysical(tLoc=tLoc, zLoc=zLoc, xLoc=xLoc, yLoc=yLoc)
        
        # Writing grid point locations:
        #    In output file, columns are ordered as y,z,x, with fields at different time instances in different files
        tLoc = tLoc.reshape(tLoc.size,1,1,1)   # Numpy broadcasting rules repeat entries when size along an axis is 1
        xLoc = xLoc.reshape(1,xLoc.size,1,1)
        zLoc = zLoc.reshape(1,1,zLoc.size,1)
        yLoc = yLoc.reshape(1,1,1,yLoc.size)
        
        dataArr[0] = yLoc + yOff*np.cos(a*xLoc+b*zLoc-omega*tLoc)
        dataArr[1] = zLoc; dataArr[2] = xLoc 
        
        for scal in range(4):
                dataArr[3+scal] = fields[:,:,:,scal]
            
        variables = 'VARIABLES = "Y", "Z", "X", "U", "V", "W", "P"\n'
        zone = 'ZONE T="", I='+str(yLoc.size)+', J='+str(zLoc.size)+', K='+str(xLoc.size)+', DATAPACKING=POINT'
        if tLoc.size == 1:
            #np.savetxt(fName+'.csv', dataArr.reshape((7,dataArr.size//7)).T,delimiter=',')
            #tempArr = dataArr.reshape(dataArr.size)
            title = 'TITLE= "Flow in wavy walled channel with a='+str(a)+', b='+str(b)+\
                ',Re_{\tau}='+str(self.flowDict['Re'])+'"\n'
            hdr = title+variables+zone
            np.savetxt(fName+'.dat',dataArr.reshape(7,dataArr.size//7).T, header=hdr,comments='')
            print('Printed physical field to file %s.dat'%fName)
        else:
            for tn in range(tLoc.size):
                title = 'TITLE= "Flow in wavy walled channel at t='+str(tLoc[tn,0,0,0])+' with a='+str(a)+', b='+str(b)+\
                    ',Re_{\tau}='+str(self.flowDict['Re'])+'"\n'
                hdr = title+variables+zone
                np.savetxt(fName+str(tn)+'.dat', dataArr[:,tn].reshape((7,dataArr[:,tn].size//7)).T,header=hdr,comments='')
            print('Printed %d time-resolved physical fields to files %sX.dat'%(tLoc.size,fName))
        return
    
    def printPhysicalPlanar(self,nLoc=40,etaLoc=None, tLoc=None, yLoc=None,yOff=0.,pField=None, interY=2,fName='ffPlanar'):
        """Prints flowField on the plane beta*x - alpha*z = 0 (this plane has normal (beta,-alpha)),
                in coordinate eta := alpha*x + beta*z
        nLoc: Number of wall-parallel locations (uniform grid is defined along the vector (alpha,beta) 
            starting at a*x+b*z = 0 and ending at a*x+b*z = 2*pi
        Refer to printPhysical() method's doc-string for description of all other input arguments
        """
        assert (type(nLoc) is int), 'nLoc must be int'
        a = self.flowDict['alpha']; b = self.flowDict['beta']; omega = self.flowDict['omega']
        gama = np.sqrt(a*a+b*b)
        K = self.flowDict['K']; L = self.flowDict['L']; M=-np.abs(self.flowDict['M'])
        N = np.int(self.N*interY)
        if etaLoc is None:  etaLoc = np.arange(0., 4.*np.pi/gama, 2.*np.pi/gama/nLoc)
        else: 
            assert isinstance(etaLoc,np.ndarray), 'etaLoc must be a numpy array'
            etaLoc = etaLoc.reshape(etaLoc.size)
        if (a == 0.) and (b == 0.):
            warn('Both alpha and beta are zero for the flowField. Printing a field at (x,z)=(0,0)')
            nLoc=1; etaLoc = np.zeros(1)
        
        if tLoc is None:
            if omega != 0.: tLoc = np.arange(0,2.*np.pi/omega, 2.*np.pi/b/7.)
            else: tLoc = np.zeros(1)
        if yLoc is None:
            yLoc = chebdif(interY*N,1)[0]
            yLocFlag = False
        else:
            yLocFlag = True
            assert isinstance(yLoc,np.ndarray) and (yLoc.ndim == 1), 'yLoc must be a 1D numpy array'
            
        assert type(yOff) is float, 'yOff characterizes surface deformation and must be of type float'
        assert isinstance(fName,str), 'fName must be a string'
        if '.dat' in fName[-4:]: fName = fName[:-4]
        
        assert self.nd == 3, 'printPhysicalPlanar() is currently written to handle only 3C velocity fields'
        assert isinstance(etaLoc,np.ndarray) and isinstance(tLoc,np.ndarray),\
            'xLoc, zLoc, and tLoc must be numpy arrays'
        
        
        if pField is None: pField = self.solvePressure(divFree=False,nonLinear=True)[0]
        else:
            assert pField.size == self.size//3, 'pField must be the same size of each component of self'
        obj = self.appendField(pField)   # Appending pField to velocity field
        
        dataArr = np.zeros((8,tLoc.size,etaLoc.size,yLoc.size))
        
        if interY != 1 and not yLocFlag:
            obj = obj.slice(N=yLoc.size)
        
        
        # .getField() requires x and z locations. So, mapping required etaLoc to xLoc (with zLoc=0), if a != 0
        #       and to zLoc (with xLoc= 0) if a = 0. 
        # gama*eta = a*x + b*z;     If z = 0., x = gama*eta/a.   If x = 0., z = gama*eta/b
        if a != 0.: 
            xLoc = gama*etaLoc/a; zLoc = 0.
        else: 
            xLoc = 0. ; zLoc = gama*etaLoc/b
        
        if yLocFlag: field = obj.getPhysical(tLoc=tLoc, xLoc=xLoc, zLoc=zLoc, yLoc=yLoc)
        else: field = obj.getPhysical(tLoc=tLoc, xLoc=xLoc, zLoc=zLoc, withBase=withBase)
        # The output of .getField(), field, will be of shape (tLoc.size, xLoc.size, zLoc.size, obj.nd, yLoc.size)
        #     But either xLoc.size or zLoc.size is 1, compressing that axis:
        field = field.reshape((tLoc.size, etaLoc.size, obj.nd, yLoc.size))
        
        tLoc = tLoc.reshape(tLoc.size,1,1)
        etaLoc = etaLoc.reshape(1,etaLoc.size,1)
        
        # Assigning grid points:
        dataArr[1] = etaLoc; 
        dataArr[0] = yLoc + yOff*np.cos(gama*etaLoc-omega*tLoc)
        
        # In field, the scalars u,v,w,p are assigned on axis 2. dataArr needs these on axis 0, since the ascii is printed in columns
        for scal in range(4):
            dataArr[2+scal] = field[:,:,scal]
        
        # U_parallel and U_cross:
        dataArr[6] = a/gama*dataArr[2] + b/gama*dataArr[4]
        dataArr[7] = -b/gama*dataArr[2]+ a/gama*dataArr[4]
        
        if 'eps' in self.flowDict: eps = self.flowDict['eps']; g = eps*a
        else: eps = 1.0E-9; g = 0
        if a != 0.: theta = int(np.arctan(b/a)*180./np.pi)
        else: theta = 90
        variables = 'VARIABLES = "Y", "eta", "U", "V", "W", "P", "U_pl", "U_cr" \n'
        zoneName = 'T'+str(theta)+'E'+str(-np.log10(eps))+'G'+str(g)+'Re'+str(self.flowDict['Re'])
        zone = 'ZONE T="'+zoneName+ '", I='+str(yLoc.size)+', J='+str(etaLoc.size)+', DATAPACKING=POINT'
        if tLoc.size == 1:
            #np.savetxt(fName+'.csv', dataArr.reshape((7,dataArr.size//7)).T,delimiter=',')
            #tempArr = dataArr.reshape(dataArr.size)
            title = 'TITLE= "Flow (planar) in wavy walled channel with a='+str(a)+', b='+str(b)+\
                ',Re_{\tau}='+str(self.flowDict['Re'])+'"\n'
            hdr = title+variables+zone
            np.savetxt(fName+'.dat',dataArr.reshape(8,dataArr.size//8).T, header=hdr,comments='')
            print('Printed physical field to file %s.dat'%fName)
        else:
            for tn in range(tLoc.size):
                title = 'TITLE= "Flow (planar) in wavy walled channel at t='+str(tLoc[tn,0,0])+' with a='+str(a)+', b='+str(b)+\
                    ',Re_{\tau}='+str(self.flowDict['Re'])+'"\n'
                hdr = title+variables+zone
                np.savetxt(fName+str(tn)+'.dat', dataArr[:,tn].reshape((8,dataArr[:,tn].size//8)).T,header=hdr,comments='')
            print('Printed %d time-resolved physical fields to files %sX.dat'%(tLoc.size,fName))
        
        return

    def zero(self):
        """Returns an object of the same class and shape as self, but with zeros as entries"""
        obj = self.copy()
        obj[:] = 0.
        return obj


    def identity(self):
        return self



