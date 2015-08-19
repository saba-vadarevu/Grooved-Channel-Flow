''' 
#####################################################
Author : Sabarish Vadarevu
Affiliation: Aerodynamics and Flight Mechanics group, University of Southampton.

This module provides a class to define u,v,w (or scalars such as pressure) in 4D: t, x,z,y. 
The shape of a class instance is (nt,nx,nz,nd,N): nt,nx,nz are harmonics (in omega, alpha,beta) 
    of Fourier modes in t,x,z respectively.
nd is the number of components, 3 for [u,v,w]. 
Scalars and non-3d fields can be created by setting 'nd' appropriately (nd=1 for scalars).
N refers to Chebyshev collocation nodes

Class attributes:
    self:   np.ndarray of shape (nt,nx,nz,nd,N)
    nt, nx, nz : length of axes 0,1, and 2 respectively
    nd:     Number of components of vector field. =1 for scalars. Length of axis 3
    N:      Number of Chebyshev collocation nodes.
    lOffset:When non-zero, indicates that the streamwisemodes are not harmonics of 
                fundamental frequency (alpha) but are offset from harmonics by a constant `lOffset`
    mOffset:Same as lOffset, for spanwise modes
    flowDict: 
        defaultDict = {'alpha':1.14, 'beta' : 2.5, 'omega':0.0, 'L': 23, 'M': 23, 'nd':3,'N': 35, 'K':0,
               'Re': 400.0, 'isPois':0.0, 'noise':0.0 }
        'noise' is currently not implemented, but will later be used to initialize "random" flowField instances
        

Methods (names only. See doc-strings for methods for template): 
    verify. view1d, view4d 
    slice, getScalar, appendField, copyArray
    real, imag, conjugate, abs
    ddt, ddx, ddx2, ddz, ddz2, ddy, ddy2, intY
    grad3d, grad2d, grad, div, laplacian, curl3d, curl
    convLinear, convNL, convSemiLinear
    dot, sumAll, norm
    residuals, solvePressure

It must always be ensured that the dictionary, self.flowDict, is always consistent with the flowField instance.
Unless one is absolutely sure that the dictionary attributes don't need to be changed, 
    the arrays should not be accessed directly. Either the methods must be used, 
    or for cases when a method isn't appropriate, the dictionary must be appropriately modified.
    
self.verify() ensures that at least the shape attributes are self-consistent. 
alpha, beta, omega, Re are not verified with self.verify()

non-class functions:
getDefaultDict(), verify_dict(), read_dictFile(), makeVector()
'''

import numpy as np
import scipy as sp
#from scipy.linalg import norm
from warnings import warn
from pseudo import chebdif, clencurt, chebintegrate, chebint
#from pseudo.py import chebint

defaultDict = {'alpha':1.14, 'beta' : 2.5, 'omega':0.0, 'L': 23, 'M': 23, 'nd':3,'N': 35, 'K':0,
               'Re': 400.0, 'isPois':0.0, 'noise':0.0 , 'lOffset':0.0, 'mOffset':0.0}

defaultBaseDict = {'alpha':0, 'beta' : 0, 'omega':0.0, 'L': 0, 'M': 0, 'nd':1,'N': 35, 'K':0,
               'Re': 400.0, 'isPois':0.0, 'noise':0.0 , 'lOffset':0.0, 'mOffset':0.0}

divTol = 1.0e-06
pCorrTol = 1.0e-04


    

def getDefaultDict(base=False):
    if base:
        return defaultBaseDict.copy()
    else:
        return defaultDict.copy()

def verify_dict(tempDict):
    '''Verify that the supplied flowDict has all the parameters required'''
    change_parameters = False
    if tempDict is None:
        tempDict = defaultDict.copy()
        warn('No flowDict was supplied. Assigning the default dictionary')
    else: 
        for key in defaultDict:
            assert key in tempDict, 'Some dictionary keys are missing'
    [tempDict['K'],tempDict['L'],tempDict['N'],tempDict['isPois']] = [int(abs(k)) for k in [tempDict['K'],tempDict['L'],tempDict['N'],tempDict['isPois']]]
    tempDict['M'] = int(tempDict['M'])
    if tempDict['alpha'] == 0.: assert tempDict['L'] == 0, 'If alpha is zero, L should also be set to zero in the dictionary'
    if tempDict['beta'] == 0.: assert tempDict['M'] == 0, 'If beta is zero, M should also be set to zero in the dictionary'
    if tempDict['omega'] == 0.: assert tempDict['K'] == 0, 'If omega is zero, K should also be set to zero in the dictionary'
    return tempDict

def read_dictFile(dictFile):
    '''Read flowDict from file. MUST use "flowConfig.txt" as template. '''
    tempDict = {}
    with open("flowConfig.txt",'r') as f:
        for line in f:
            (key,val) = line.split()[:2]
            tempDict[key] = float(val)    
    return tempDict

def makeVector(*args):
    '''Concatenate flowField objects. Use this to create a vector flowField from scalar/vector flowFields
    For instance,   uvw = makeVector(u,v,w)
                    uvwp = makeVector(u,v,w,p) = makeVector(uvw,p)'''
    ff = args[0]
    if not isinstance(ff,flowField):
        raise RuntimeError('makeVector takes as arguments only instances of flowField class')
        return
    for v in args[1:]:
        if not isinstance(v,flowField):
            raise RuntimeError('makeVector takes as arguments only instances of flowField class')
        ff = ff.appendField(v)
    return ff
    

class flowField(np.ndarray):
    '''Provides 4d scalars and vectors of shape (nt,nx,nz,nd,N),
    nt, nx, nz: Number of temporal, streamwise, and spanwise Fourier modes contained in the flowField. 
    nd: Number of components of the vector
    N: Number of Chebyshev collocation nodes (on [1,-1])
    nt,nx,nz,nd,N, and a bunch of other parameters: Re, alpha, beta, omega, isPois (flag to say if flow is pressure driven or driven by plate motion)
        are taken from a dictionary that MUST accompany the field array. The class fails to work when the dictionary is not consistent

    Initialization:
        flowField() creates an instance using a default dictionar: a 3 component vector of shape (1,47,24,3,35).
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
            
    '''
    def __new__(cls, arr=None, flowDict=None, dictFile= None):
        '''Creates a new instance of flowField class with arguments (arr=None,flowDict=None,dictFile=None)
        '''
        if flowDict is None:
            if dictFile is None:
                if hasattr(arr,'flowDict'):
                    flowDict = arr.flowDict
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
        nz = int(3.*abs(M)/2. - M/2. + 1)     # = 1 if M=0;    = M+1 if M>0;    = 2*|M|+1 if M<0
        
        if arr is None:
            #obj =  np.zeros((nt,nx,nz,nd,N),dtype=np.complex).view(cls)
            obj = np.ndarray.__new__(flowField,shape=(nt,nx,nz,nd,N),dtype=np.complex,buffer=np.zeros(nt*nx*nz*nd*N,dtype=np.complex))
        else:
            if arr.dtype == np.float:
                arr = (arr+1.j*np.zeros(arr.shape))
            obj = np.ndarray.__new__(flowField,shape=(nt,nx,nz,nd,N),dtype=np.complex,buffer=arr)
        
        #print(norm(obj))
        
        if obj.size != (nx*nz*nt*nd*N):
            raise RuntimeError('The parameters in the dictionary are not consistent with the size of the supplied array')
        
        obj.flowDict = flowDict
        obj.nx = nx
        obj.nz = nz
        obj.nt = nt
        obj.N = N
        obj.nd = flowDict['nd']
        return obj
        
    
    def __array_finalize__(self,obj):
        if self.dtype != np.complex:
            warn('flowField class is designed to work with complex array entries\n'+
                 'To obtain real/imaginary parts of an instance, use class methods "real()" and "imag()"')
        if isinstance(obj, flowField):
            self.flowDict = getattr(self,'flowDict',obj.flowDict.copy())
            self.nt = getattr(self,'nt',obj.nt)
            self.nx = getattr(self,'nx',obj.nx)
            self.nz = getattr(self,'nz',obj.nz)
            self.nd = getattr(self,'nd',obj.nd)
            self.N = getattr(self,'N',obj.N)
            return
        elif obj != None:
            raise RuntimeError('View-casting np.ndarray is not supported since dictionaries cannot be passed. \n'+
                               'To initialize class instance from np.ndarray, use constructor call:flowField(arr=myArray,dictFile=myFile)')
        return

    
    def verify(self):
        '''Ensures that the size of the class array is consistent with the dictionary entries. 
        Use this when writing new methods or tests'''
        self.flowDict = verify_dict(self.flowDict)
        if not ((self.nt == 2*self.flowDict['K']+1) and (self.nx == 2*self.flowDict['L']+1) and 
                (self.nz == int(3.*abs(self.flowDict['M'])/2. - self.flowDict['M']/2. + 1)) and
                (self.N == self.flowDict['N']) and (self.nd == self.flowDict['nd'])): 
            raise RuntimeError('The shape attributes of the flowField instance are not consistent with dictionary entries')
        assert self.size == self.nt*self.nx*self.nz*self.nd*self.N, 'The size of the flowField array is not consistent with its shape attributes'
        
    def view1d(self):
        ''' Returns a 1d view. 
        Don't try to figure out what the ordering is, just use self.view4d() to get an organized view'''
        return self.reshape(self.size)
    
    def view4d(self):
        ''' Returns a 4d view (actually, a 5-D array): (omega, alpha, beta, field=u,v,w,p, N)'''
        return self.reshape((self.nt,self.nx,self.nz,self.nd,self.N))

    def slice(self,K=None,L=None,M=None,nd=None,N=None):
        '''
        Returns a class instance with increased/reduced K,L,M,nd,N
        Call as new_inst = myFlowField.slice(K=Knew,L=Lnew,N=Nnew)) to change values of K,L,N without affecting M (and nd)
        When the number of Fourier modes (K,L,M, or nt,nx,nz) are smaller than what is requested, 
            additional zero modes are added. For Chebyshev nodes, interpolation is used'''
        obj = self.copyArray()
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
                obj = np.concatenate((  np.zeros((Kt-K,nxt,nzt,ndt,Nt),dtype=np.complex), obj,
                               np.zeros((Kt-K,nxt,nzt,ndt,Nt),dtype=np.complex)  ), axis=0)
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
            M = int(M)
            Mt = flowDict_temp['M']               # Temporary name for 'M' of self
            nzt = int(3.*abs(M)/2. - M/2. + 1)     # = 1 if L=0;    = L+1 if L>0;    = 2*|L|+1 if L<0
            
            if M*Mt >=0: 
                if abs(M) <= abs(Mt): # Case 1.A: Truncate
                    nz0 = int((abs(Mt)-Mt)/2)     # = Mt for Mt< 0, = 0 otherwise
                    nzm1 = nz0 - int((abs(M)-M)/2) 
                    nzp1 = nz0 + abs(M) + 1
                    obj = obj[:,:,nzm1:nzp1]
                else:  # Case 1.B: Extend using zero modes
                    nzplus = int(abs(M)-abs(Mt))
                    if M<0: 
                        obj = np.concatenate(( np.zeros((ntt,nxt,abs(Mt-M),ndt,Nt),dtype=np.complex), obj,
                               np.zeros((ntt,nxt,abs(Mt-M),ndt,Nt),dtype=np.complex)  ), axis=2)
                    else:
                        obj = np.concatenate(( obj,
                               np.zeros((ntt,nxt,abs(Mt-M),ndt,Nt),dtype=np.complex) ), axis=2)
            elif M > 0:          # Case 2: Get only modes [0,b,..,|M|b] from [-|Mt|*b,..,0,b,..,|Mt|*b]
                if abs(M) <= abs(Mt): # Case 2.A: |M|< |Mt|, so truncate
                    nz0 = int((abs(Mt)-Mt)/2)
                    nzp1 = nz0 + M + 1
                    obj = obj[:,:,nx0:nzp1]
                else:    # Case 2.B: |M| > |Mt|, so add zero modes 
                    obj = np.concatenate(( obj[:,:,abs(Mt):], 
                               np.zeros((ntt,nxt,abs(Mt-M),ndt,Nt),dtype=np.complex) ), axis=2)
            else: # Case 3: Get modes [-|M|b,...,0,b,..,|M|b], given [0,b,..,|Mt|b]
                if abs(M) <= abs(Mt):        # Case 3.A: Truncate on positive, extend with conjugates on negative
                    obj = np.concatenate(( obj[::-1,::-1,abs(M):0:-1].conjugate(), obj[:,:,:abs(M)+1] ), axis=2)
                else:            # Case 3.B: Extend on positive with zeros, extend on negative with conjugates and zeros
                    # Doing the extension with conjugates on negative first:
                    obj = np.concatenate(( obj[::-1,::-1,:0:-1].conjugate(), obj ), axis=2)
                    # Adding zeros on positive and negative:
                    obj = np.concatenate((  np.zeros((ntt,nxt,abs(Mt-M),ndt,Nt),dtype=np.complex), obj,
                               np.zeros((ntt,nxt,abs(Mt-M),ndt,Nt),dtype=np.complex) ), axis=2)
            flowDict_temp['M']= M
        
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
        
        obj = flowField(arr=obj, flowDict = flowDict_temp).view4d()
        
        if (nd is not None):
            nd = np.asarray([nd])
            nd = nd.reshape(nd.size)
            obj = obj[:,:,:,nd]
            obj.flowDict['nd'] = nd.size
            obj.nd = nd.size
        
        obj.verify()
        return obj
    
    def getScalar(self,nd=0):
        '''Returns the field Variable in the flowField instance identified by the argument "nd".
        Default for "nd" is 0, the first scalar in the flowField (u)'''
        if type(nd) != int:
            raise RuntimeError('getScalar(nd=0) only accepts integer arguments')
        obj = self.view4d()[:,:,:,nd:nd+1].copy()
        obj.flowDict['nd'] = 1
        obj.nd = 1
        return obj.view4d()

    def appendField(self,obj):
        '''Append a field at the end of "self". To append "p" to "uVec", call as uVec.appendField(p)
        Note: Both uVec and p must be flowField objects, each with their flowDict'''
        if not isinstance(obj,flowField):
            raise RuntimeError('Only flowField objects can be appended to a flowField object')
        tempDict = self.flowDict.copy()
        tempDict['nd'] += obj.flowDict['nd']
        v1 = self.view4d().copyArray()
        v2 = obj.view4d().copyArray()
        return flowField(arr=np.append(v1,v2,axis=3), flowDict=tempDict)
    
    def copyArray(self):
        ''' Returns a copy of the np.ndarray of the instance. 
        This is useful for manipulating the entries of a flowField without bothering with all the checks'''
        return self.view(np.ndarray).copy()
    
    def real(self):
        ''' Returns the real part of the flowField (the entries are still complex, with zero imaginary parts)'''
        return flowField(arr=self.copyArray().real,flowDict=self.flowDict)
    
    def imag(self):
        ''' Returns the imaginary part of the flowField (the entries are still complex, with zero imaginary parts)'''
        return flowField(arr=self.copyArray().imag,flowDict=self.flowDict)
    
    def conjugate(self):
        ''' Returns complex conjugate of flowFIeld instance'''
        return self.real()-1.j*self.imag()
    def abs(self):
        '''Returns absolute value of entries of flowField instance (still expressed as complex numbers, but with zero imaginary part and positive real part)'''
        return flowField(arr=np.abs(self.copyArray()),flowDict=self.flowDict.copy())
    
    def ddt(self):
        ''' Returns a flowField instance that gives the partial derivative along "t" '''
        if self.nt == 1:
            return 1.j*self.flowDict['omega']*self.copy()
        partialT = self.view4d().copy()
        kArr = np.arange(-self.flowDict['K'],self.flowDict['K']+1).reshape(self.nt,1,1,1,1)
        return partialT
    
    def ddx(self):
        ''' Returns a flowField instance that gives the partial derivative along "x" '''
        if self.nx == 1:
            return 1.j*(1+self.flowDict['lOffset'])*self.flowDict['alpha']*self.copy()
        partialX = self.view4d().copy()
        lArr = (self.flowDict['lOffset']+np.arange(-self.flowDict['L'],self.flowDict['L']+1)).reshape(1,self.nx,1,1,1)
        partialX[:] = 1.j*self.flowDict['alpha']*lArr*partialX
        return partialX
    
    def ddx2(self):
        ''' Returns a flowField instance that gives the second partial derivative along "x" '''
        if self.nx == 1:
            return -1.*((1+self.flowDict['lOffset'])*self.flowDict['alpha'])**2*self.copy()
        partialX2 = self.view4d().copy()
        l2Arr = ((self.flowDict['lOffset']+np.arange(-self.flowDict['L'],self.flowDict['L']+1))**2).reshape(1,self.nx,1,1,1)
        partialX2[:] = -self.flowDict['alpha']**2*l2Arr*partialX2
        return partialX2
    
    def ddz(self):
        ''' Returns a flowField instance that gives the partial derivative along "z" '''
        if self.nz == 1:
            return 1.j*(1+self.flowDict['mOffset'])*self.flowDict['beta']*self.copy()
        partialZ = self.view4d().copy()
        M = self.flowDict['M']
        mArr = (self.flowDict['mOffset']+np.arange( (M-abs(M))/2,abs(M)+1 ) ).reshape((1,1,self.nz,1,1))
        partialZ[:] = 1.j*self.flowDict['beta']*mArr*partialZ
        return partialZ
    
    def ddz2(self):
        ''' Returns a flowField instance that gives the second partial derivative along "z" '''
        if self.nz == 1:
            return -1.*((1+self.flowDict['mOffset'])*self.flowDict['beta'])**2*self.copy()
        partialZ2 = self.view4d().copy()
        M = self.flowDict['M']
        mArr = (self.flowDict['mOffset']+np.arange( (M-abs(M))/2,abs(M)+1 )).reshape((1,1,self.nz,1,1))
        m2Arr = mArr**2
        partialZ2[:] = -self.flowDict['beta']**2*m2Arr*partialZ2
        return partialZ2
    
    def ddy(self):
        ''' Returns a flowField instance that gives the partial derivative along "y" '''
        partialY = self.view1d().copy()
        N = partialY.flowDict['N']
        D = (chebdif(N,1)[1]).reshape(N,N)
        for n in range(self.nt*self.nx*self.nz*self.nd):
            partialY[n*N:(n+1)*N] = np.dot(D, partialY[n*N:(n+1)*N])
        return partialY.view4d()
    
    def ddy2(self):
        ''' Returns a flowField instance that gives the partial derivative along "y" '''
        partialY2 = self.view1d().copy()
        N = partialY2.flowDict['N']
        D2 = (chebdif(N,2)[1])[:,:,1].reshape(N,N)
        for n in range(self.nt*self.nx*self.nz*self.nd):
            partialY2[n*N:(n+1)*N] = np.dot(D2, partialY2[n*N:(n+1)*N])
        return partialY2.view4d()

    def intY(self):
        ''' Integrate each Fourier mode of each scalar along the wall-normal axis
        Returns a flowField object of the same size as self.
        Use this method to compute variables from their wall-normal derivatives'''
        integral = self.copy().reshape((self.size/self.N, self.N))
        arr = integral.copyArray()
        for n in range(np.int(integral.size/integral.N)):
            integral[n] = chebintegrate(arr[n])
        integral.verify()
        return integral.view4d()
    
    def grad3d(self, scalDim=0, nd=3):
        ''' Computes gradient (in 3d by default) of either a scalar flowField object, 
            or of the first variable in a vector flowField object. 
            Grads of other variables can be calculated by passing scalDim=<index of variable>.
            Gradients in 2D (x and y) can be calculated by passing nd=2'''
        tempDict = self.flowDict.copy()
        tempDict['nd'] = nd
        if self.nd ==1:
            scal = self
        else:
            scal = self.getScalar(nd=scalDim)
        scal.verify()
        if nd == 3:
            gradVec = makeVector(scal.ddx(),scal.ddy(),scal.ddz())
        elif nd ==2:
            gradVec = makeVector(scal.ddx(), scal.ddy())
        return gradVec
    
    def grad(self,**kwargs):
        return self.grad3d(**kwargs)
    
    def grad2d(self, **kwargs):
        ''' Computes gradients in 2D (streamwise & wall-normal) for a scalar flowField object, 
            or for the scalar component of a vector field identified as vecField[:,:,:,scalDim]'''
        kwargs['nd'] = 2
        return self.grad3d(**kwargs)
        
    def laplacian(self):
        ''' Computes Laplacian for a 3C flowField'''
        assert  self.nd in [2,3] , 'Laplacian is defined only for 2C or 3C fields'
        if self.nd == 3: return self.ddx2() + self.ddy2() + self.ddz2()
        else: return self.ddx2() + self.ddy2()
            
    def div(self):
        ''' Computes divergence of vector field as u_x+v_y+w_z
        If a flowField with more than 3 components (nd>3) is supplied, takes first three components as u,v,w.
        Optional: 2-D divergence, u_x+v_y can be requested by passing nd=2'''
        assert self.nd in [2,3], ('Divergence is defined only for 2C or 3C fields')
        if self.nd == 3: return self.getScalar(nd=0).ddx() + self.getScalar(nd=1).ddy() + self.getScalar(nd=2).ddz()
        else: return self.getScalar(nd=0).ddx() + self.getScalar(nd=1).ddy() 
        
    def curl3d(self):
        assert self.nd == 3, 'curl3d method is defined only for 3C fields. To get vorticity of 2D field, use curl() instead'
        return makeVector(self.getScalar(nd=2).ddy() -self.getScalar(nd=1).ddz(),\
                         self.getScalar(nd=0).ddz() -self.getScalar(nd=2).ddx(),\
                         self.getScalar(nd=1).ddx() -self.getScalar(nd=0).ddy())
    
    def curl(self):
        if self.nd == 3: return self.curl3d()
        elif self.nd == 2:
            tempDict = self.flowDict.copy(); tempDict['nd'] = 1;
            zeroField = flowField(flowDict = tempDict)
            return (self.appendField(zeroField).curl()).getScalar(nd=2)
        else:
            raise RuntimeError('Curl is defined only for 2C or 3C fields.')
            return
    
    def sumAll(self):
        '''Sums all elements of a flowField object (along all axes)'''
        obj = self.view4d().copyArray()
        return np.sum(np.sum(np.sum(np.sum(np.sum(obj,axis=4),axis=3),axis=2),axis=1),axis=0)
    
    def dot(self, vec2):
        '''Computes inner product for two flowField objects, scalar or vector,
            by integrating each scalar along x,y,and z, and adding the integrals for each scalar.
        Currently, only inner products of objects with identical dictionaries are supported'''
        assert isinstance(vec2,flowField), 'Inner products are only defined for flowField objects. Ensure passed object is a flowField instance'
        assert (self.flowDict == vec2.flowDict), 'Method for inner products is currently unable to handle instances with different flowDicts'
        
        w = clencurt(self.N).reshape((1,1,1,1,self.N))
        return flowField.sumAll(self*vec2.conjugate()*w)
    
    def norm(self):
        '''Integrates v*v.conjugate() along x,y,z, and takes its square-root'''
        return np.sqrt(np.abs(self.dot(self)))
    
    def convLinear(self,uBase=None):
        ''' Computes linearized convection term as [U u_x + v U',  U v_x,  U w_x ]
        Baseflow, uBase must be a 1D array of size "N" '''
        if isinstance(uBase,flowField) and uBase.size != self.N:
            return self.convSemiLinear(uBase=uBase)
        N = self.N
        y = chebdif(N,1)[0]
        if uBase is not None:
            assert uBase.size == N, 'Base flow must be of size N ([U(y),0,0]). For "richer" base flows, use method .convSemiLinear(uBase=baseFlow)'
        if not isinstance(uBase,flowField): 
            if uBase is None:
                if self.flowDict['isPois'] == 1: baseArr = 1.- y**2
                else: baseArr = y
            baseDict = defaultBaseDict; baseDict['N'] = N
            baseArr = uBase
            uBase = flowField(arr=baseArr, flowDict=baseDict)
            
                
        assert self.nd in [2,3], 'Convection term (linearized) can be calculated only for 2C or 3C vector fields'
        
        convTerm = flowField(flowDict = self.flowDict.copy())
        convTerm[:] = uBase.view4d()*self.ddx()
        convTerm[:,:,:,0:1] += uBase.ddy()*self.view4d()[:,:,:,1:2]
        convTerm.verify()
        return convTerm
    
    def convSemiLinear(self,uBase=None):
        if (uBase is None) or (uBase.size == self.N):
            return self.convLinear(uBase=uBase)
        assert isinstance(uBase,flowField), 'The base flow must be an instance of flowField class'
        assert np.mod(uBase.size,self.nx*self.nz*self.N) == 0., 'The base flow must have the same number of spatial Fourier modes and N as self'
        assert (uBase.flowDict['alpha'] == self.flowDict['alpha']) and (uBase.flowDict['beta'] == self.flowDict['beta']),\
            'For convSemiLinear, uBase and self should have same fundamental wavenumbers and harmonics (except for offsets in l and m)'
        assert (uBase.flowDict['lOffset'] == 0.) and (uBase.flowDict['mOffset'] == 0.), \
            'In uBase, Fourier modes must be harmonics of alpha and beta in flowDict'
            
        if self.flowDict['M'] > 0 :
            obj = self.slice(M=-self.flowDict['M']); 
            u = obj.getScalar(nd=0); v = obj.getScalar(nd=1) ; w = obj.getScalar(nd=2); tempDict = obj.flowDict.copy()
            uBase = uBase.slice(M=-self.flowDict['M'])
        else:
            u = self.getScalar(nd=0);  v = self.getScalar(nd=1); w = self.getScalar(nd=2); tempDict = self.flowDict.copy()
        
        tempDict = self.flowDict.copy()
        tempDict['nd'] = 3
        convTerm = flowField(flowDict=tempDict).view4d()
        
        ux = u.ddx(); uy = u.ddy(); uz = u.ddz()
        vx = v.ddx(); vy = v.ddy(); vz = v.ddz()
        wx = w.ddx(); wy = w.ddy(); wz = w.ddz()
        
        U = uBase.getScalar(nd=0); baseDict = uBase.flowDict.copy(); baseDict['nd'] = 1;
        if uBase.nd >=3 : V = uBase.getScalar(nd=1); W = uBase.getScalar(nd=2);
        elif uBase.nd == 2: V = uBase.getScalar(nd=1); W = flowField(flowDict=baseDict) 
        else: W = flowField(flowDict=baseDict) ;  W = flowField(flowDict=baseDict)
        
        sumArr = lambda v: np.sum( np.sum( np.sum( v, axis=1), axis=1), axis=1)
        
        for lp in range(self.nx):
            l = lp - L
            l1 = l; l2 = None; l3 = None; l4 = l1-1; 
            if l == 0: l4 = None
            if l < 0:  
                l1 = None; l2 = self.nx+l; l3 = l2-1; l4 = None
                
            for mp in range(self.nz):
                m = mp - abs(M)
                m1 = m; m2 = None; m3 = None; m4 = m1-1; 
                if m == 0: m4 = None
                if m < 0: 
                    m1 = None; m2 = self.nz+m; m3 = m2-1; m4 = None
                
                convTerm[:,lp,mp,0] += sumArr(U[:,l1:l2,m1:m2]*ux[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,0] += sumArr(V[:,l1:l2,m1:m2]*uy[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,0] += sumArr(W[:,l1:l2,m1:m2]*uz[:,l3:l4:-1,m3:m4:-1])
                
                convTerm[:,lp,mp,1] += sumArr(U[:,l1:l2,m1:m2]*vx[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,1] += sumArr(V[:,l1:l2,m1:m2]*vy[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,1] += sumArr(W[:,l1:l2,m1:m2]*vz[:,l3:l4:-1,m3:m4:-1])
                
                convTerm[:,lp,mp,2] += sumArr(U[:,l1:l2,m1:m2]*wx[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,2] += sumArr(V[:,l1:l2,m1:m2]*wy[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,2] += sumArr(W[:,l1:l2,m1:m2]*wz[:,l3:l4:-1,m3:m4:-1])
        
        convTerm.verify()
        if self.flowDict['M']>0:
            convTerm = convTerm.slice(M=self.flowDict['M'])
        return convTerm
    
    def convNL(self, uBase=None):
        '''Computes the non-linear convection term, given a fluctuation flow field (base flow is added when calculating)
        Warning: Currently, the code assumes that the flowField supplied is that of a steady flow. Temporal frequencies are not accounted for'''
        
        # If the only modes are, say {(a,-2b),(a,-b),(a,0),(a,b),(a,2b)}, 
        #    then the only non-linear interactions that produce these modes are when the above modes interact with (0,0)
        #    Interactions within the above modes can only produce {(2a,nb)} with doesn't belong to the set
        # Similarly for {(-na,b),(-na+a,b),..,(0,b),(a,b),..,(na,b)}. The convection term is simply the linear part of it (for a given base flow)
        # For a single mode, (a,b), the convection term that produces the mode is, again, its interaction with just (0,0)
        if self.flowDict['L'] == 0 and self.flowDict['M']== 0: return self.convLinear(uBase=uBase)
        if self.flowDict['L'] == 0 and self.flowDict['alpha'] != 0.: return self.convLinear(uBase=uBase)
        if self.flowDict['M'] == 0 and self.flowDict['beta'] != 0.: return self.convLinear(uBase=uBase)
        
        assert self.flowDict['lOffset'] == 0. and self.flowDict['mOffset']==0. ,\
            'convNL() method is currently not supported for flowFields with offsets in l and m. Use convSemiLinear() for linearized term about a base'
        
        # Ensuring a full set -|M|b,...,0b,..,|M|b is available before computing the convection term
        if self.flowDict['M'] > 0 :
            obj = self.slice(M=-self.flowDict['M'])
            u = obj.getScalar(nd=0); v = obj.getScalar(nd=1) ; w = obj.getScalar(nd=2); tempDict = obj.flowDict.copy()
        else:
            u = self.getScalar(nd=0);  v = self.getScalar(nd=1); w = self.getScalar(nd=2); tempDict = self.flowDict.copy()
        L = tempDict['L']; M = tempDict['M']; N = tempDict['N']
        nx = 2*L+1; nz= 2*abs(M)+1
        
        # Computing the base flow and adding it to the flowField before computing the convection term
        y = chebdif(N,1)[0]
        if uBase is None:
            if tempDict['isPois']==1:
                uBase = 1.-y**2
            else:
                uBase = y
        u[0, L, -M ,0] += uBase
                
        tempDict = self.flowDict.copy()
        tempDict['nd'] = 3
        convTerm = flowField(flowDict=tempDict).view4d()
        
        
        ux = u.ddx(); uy = u.ddy(); uz = u.ddz()
        vx = v.ddx(); vy = v.ddy(); vz = v.ddz()
        wx = w.ddx(); wy = w.ddy(); wz = w.ddz()
        
        sumArr = lambda v: np.sum( np.sum( np.sum( v, axis=1), axis=1), axis=1)
        
        for lp in range(self.nx):
            l = lp - L
            l1 = l; l2 = None; l3 = None; l4 = l1-1; 
            if l == 0: l4 = None
            if l < 0:  
                l1 = None; l2 = self.nx+l; l3 = l2-1; l4 = None
                
            for mp in range(self.nz):
                m = mp - abs(M)
                m1 = m; m2 = None; m3 = None; m4 = m1-1; 
                if m == 0: m4 = None
                if m < 0: 
                    m1 = None; m2 = self.nz+m; m3 = m2-1; m4 = None
                
                convTerm[:,lp,mp,0] += sumArr(u[:,l1:l2,m1:m2]*ux[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,0] += sumArr(v[:,l1:l2,m1:m2]*uy[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,0] += sumArr(w[:,l1:l2,m1:m2]*uz[:,l3:l4:-1,m3:m4:-1])
                
                convTerm[:,lp,mp,1] += sumArr(u[:,l1:l2,m1:m2]*vx[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,1] += sumArr(v[:,l1:l2,m1:m2]*vy[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,1] += sumArr(w[:,l1:l2,m1:m2]*vz[:,l3:l4:-1,m3:m4:-1])
                
                convTerm[:,lp,mp,2] += sumArr(u[:,l1:l2,m1:m2]*wx[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,2] += sumArr(v[:,l1:l2,m1:m2]*wy[:,l3:l4:-1,m3:m4:-1])
                convTerm[:,lp,mp,2] += sumArr(w[:,l1:l2,m1:m2]*wz[:,l3:l4:-1,m3:m4:-1])
        
        convTerm.verify()
        if self.flowDict['M']>0:
            convTerm = convTerm.slice(M=self.flowDict['M'])
        return convTerm
        
    def residuals(self,pField=None, nonLinear=True, divFree=False, uBase=None):
        ''' Computes the residuals of the momentum equations for a velocity field.
        Arguments:
        pField is the pressure field (optional). 
            When not supplied, the pressure is taken to be zero everywhere
        nonLinear (flag) defaults to True
            When set to False, convLinear() is used to evaluate convection term. When true, convNL() is used.
        divFree (flag) defaults to False
            When set to False, nothing is done. This means the field could have a non-zero divergence
            When set to True, wall-normal velocity is changed to ensure divergence is zero.
                WARNING: THIS CHANGES THE FLOWFIELD OBJECT THAT IS PASSED. To avoid this, call self.copy().residuals()
        When only a velocity field is available, use solvePressure[1] to get the residuals instead.'''
        assert self.nd == 3, 'Method only accepts 3D flowfields'
        residual = flowField(flowDict=self.flowDict.copy())
        L = self.flowDict['L']; M = self.flowDict['M']
        if divFree:
            # u_x + v_y + w_z = div. 
            # To ensure divergence is zero, correct 'v' as v += - \int div * dy
            divergence = self.div()
            divergence[divergence < divTol] = 0.
            # vCorrection = -(self.div()).intY()
            self.view4d()[:,:,:,1:2] -= divergence.intY()
        
        if pField is None:
            tempDict = self.flowDict.copy(); tempDict['nd'] = 1
            pField = flowField(flowDict=tempDict)
        else: 
            assert (pField.nd == 1) and (pField.size == np.int(self.size/3)), 'pField should be a scalar of the same size as each scalar of velocity'
        
        residual[:] += pField.grad() - (1./self.flowDict['Re'])*self.laplacian()
        if nonLinear:
            residual[:] += self.convNL()
        else:
            residual[:] += self.convLinear()
        
        residual[:,:,:,:,[0,-1]] = self[:,:,:,:,[0,-1]]
        
        return residual
    
    def solvePressure(self, pField=None, residuals=None, divFree=False, nonLinear=True):
        ''' Solves for pressure, given a 3D velocity field.
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
            The residuals of streamwise momentum equation and spanwise momentum equation are averaged to get the actual constant (minimizes the residuals)'''
        
        assert self.nd == 3, 'The flowField instance supplied must be a 3D velocity field'
        assert isinstance(pField,flowField), 'pField must be a flowField instance'
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
            assert isinstance(residuals,flowField) and (residuals.nd == 3) and (residuals.size == self.size), 'residuals must be a 3D flowField object of the same size as self'
            
        pCorrection = - (residuals.getScalar(nd=1)).intY()
        # pCorrection is now determined upto a constant. Next, find the constant that minimizes residual for streamwise, spanwise
        
        residuals[:,:,:,1:2] += pCorrection.ddy()
        residuals[:,:,:,0:1] += pCorrection.ddx()
        residuals[:,:,:,2:3] += pCorrection.ddz()
        assert residuals.getScalar(nd=1).norm() < 1.0e-6, 'Wall-normal residual has not gone below 1.0e-6 even after correcting dpdy. Weird'
        
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
        
    def printCSV(self,xLoc=None, zLoc=None, tLoc=None, yLoc=None,yOff=0.,pField=None, interY=2,toFile=True,fName='ff'):
        '''Prints the velocities and pressure in a CSV file with columns ordered as X,Y,Z,U,V,W,P
        Arguments (all keyword):
            xLoc: streamwise locations where field variables need to be computed (default: [0:4*pi/alpha] 41 points)
            zLoc: spanwise locations where field variables need to be computed (default: [0:2*pi/beta] 13 points)
            tLoc: temporal locations (default: 7 points when omega != 0, 1 when omega = 0). Fields at different time-locations are printed to different files
            pField: Pressure field (computed with divFree=False, nonLinear=True if pField not supplied)
            interY: Field data is interpolated onto interY*self.N points before printing. Default for interY is 2
            yOff: Use this to define wavy surfaces. For flat walls, yOff = 0. For wavy surfaces, yOff = 2*eps
                    yOff is used to modify y-grid as   y[tn,xn,zn] += yOff*cos(alpha*x + beta*z - omega*t)
            yLoc: Use this to specify a y-grid. When no grid is specified, Chebyshev nodes are used
            fname: Name of CSV file to be printed to. Default: ff.csv 
            toFile: If true, prints field to file. If false, returns the physical field as a 5-d array
                    The 5 dimensions are: (variable, time-location, x-loc, z-loc, y-loc)
                    variables are ordered as (x,z,y,u,v,w,p)'''
        print('printCSV called.............')
        a = self.flowDict['alpha']; b = self.flowDict['beta']; omega = self.flowDict['omega']
        K = self.flowDict['K']; L = self.flowDict['L']; M=-np.abs(self.flowDict['M'])
        if xLoc is None:
            if a != 0.: xLoc = np.arange(0,4.*np.pi/a, 2.*np.pi/a/20.)
            else: xLoc = np.zeros(1)
        if zLoc is None:
            if b != 0: zLoc = np.arange(0,2.*np.pi/b, 2.*np.pi/b/12.)
            else: zLoc = np.zeros(1)
        if tLoc is None:
            if omega != 0.: tLoc = np.arange(0,2.*np.pi/omega, 2.*np.pi/b/7.)
            else: tLoc = np.zeros(1)
        if yLoc is None:
            yLoc = chebdif(interY*self.N,1)[0]
            yLocFlag = False
        else:
            yLocFlag = True
            assert isinstance(yLoc,np.ndarray) and (yLoc.ndim == 1), 'yLoc must be a 1D numpy array'
            
        assert type(yOff) is float, 'yOff characterizes surface deformation and must be of type float'
        if '.csv' in fName[-4:]: fName = fName[:-4]
        
        assert self.nd == 3, 'printCSV() is currently written to handle only 3C velocity fields'
        assert isinstance(xLoc,np.ndarray) and isinstance(zLoc,np.ndarray) and isinstance(tLoc,np.ndarray),\
            'xLoc and zLoc must be numpy arrays'
        assert isinstance(fName,str), 'fName must be a string'
        
        if pField is None: pField = self.solvePressure(divFree=False,nonLinear=True)[0]
        else:
            assert isinstance(pField, flowField), 'pField must be an instance of flowField class'
            assert pField.size == self.size//3, 'pField must be the same size of each component of self'
        
        dataArr = np.zeros((7,tLoc.size,xLoc.size,zLoc.size,yLoc.size))
        tLoc = tLoc.reshape(tLoc.size,1,1,1)
        xLoc = xLoc.reshape(1,xLoc.size,1,1)
        zLoc = zLoc.reshape(1,1,zLoc.size,1)
        
        if interY != 1:
            obj = self.slice(M=M, N=interY*self.N).copy()
            p = pField.slice(M=M, N=interY*self.N).copy()
        else: obj = self.slice(M=M).copy(); p = pField.slice(M=M).copy()
        p = p.view4d()
        
        if yLocFlag:
            objTemp = obj.copy(); pTemp = p.copy()
            tempDict = obj.flowDict; tempDict['N'] = yLoc.size
            obj = flowField(flowDict=tempDict.copy())
            tempDict['nd'] = 1; p = flowField(flowDict=tempDict.copy())
            N1 = obj.N ; N2 = objTemp.N
            for ind in range(obj.size//obj.N):
                obj.view1d()[ind*N1:(ind+1)*N1] = chebint(objTemp.view1d()[ind*N2:(ind+1)*N2],yLoc)
            for ind in range(p.size//p.N):
                assert isinstance(pTemp, flowField),'pTemp stops being flowField at ind='+str(ind)
                p.view1d()[ind*N1:(ind+1)*N1] = chebint(pTemp.view1d()[ind*N2:(ind+1)*N2],yLoc)
        
        yLoc = yLoc.reshape(1,1,1,yLoc.size)
        u = obj.getScalar(nd=0).copyArray().reshape(obj.nt,obj.nx,obj.nz, obj.N)
        v = obj.getScalar(nd=1).copyArray().reshape(obj.nt,obj.nx,obj.nz, obj.N)
        w = obj.getScalar(nd=2).copyArray().reshape(obj.nt,obj.nx,obj.nz, obj.N)
        p = p.copyArray().reshape(obj.nt,obj.nx,obj.nz, obj.N)
        
        lArr = (self.flowDict['lOffset']+ np.arange(-L,L+1)).reshape((1,self.nx,1,1))
        mArr = (self.flowDict['mOffset']+ np.arange(M,-M+1)).reshape((1,1,self.nz,1))
        kArr = np.arange(-K,K+1).reshape((self.nt,1,1,1))
        
        sumArr = lambda arr: np.sum(np.sum(np.sum(arr,axis=0),axis=0),axis=0).real
        dataArr[0] = xLoc; dataArr[1] = zLoc; dataArr[2] = yLoc + yOff*np.cos(a*xLoc+b*zLoc-omega*tLoc)
        for tn in range(tLoc.size):
            t = tLoc[tn,0,0,0]
            for xn in range(xLoc.size):
                x = xLoc[0,xn,0,0]
                for zn in range(zLoc.size):
                    z = zLoc[0,0,zn,0]
                    dataArr[3,tn,xn,zn] = sumArr( u * np.exp( 1.j*(lArr*a*x + mArr*b*z - kArr*omega*t) ) )
                    dataArr[4,tn,xn,zn] = sumArr( v * np.exp( 1.j*(lArr*a*x + mArr*b*z - kArr*omega*t) ) )
                    dataArr[5,tn,xn,zn] = sumArr( w * np.exp( 1.j*(lArr*a*x + mArr*b*z - kArr*omega*t) ) )
                    dataArr[6,tn,xn,zn] = sumArr( p * np.exp( 1.j*(lArr*a*x + mArr*b*z - kArr*omega*t) ) )
        
        if toFile:
            if tLoc.size == 1:
                np.savetxt(fName+'.csv', dataArr.reshape((7,dataArr.size//7)).T,delimiter=',')
                print('Printed physical field to file %s.csv'%fName)
            else:
                for tn in range(tLoc.size):
                    np.savetxt(fName+str(tn)+'.csv', dataArr[:,tn].reshape((7,dataArr[:,tn].size//7)).T,delimiter=',')
                print('Printed %d time-resolved physical fields to files %sX.csv'%(tLoc.size,fName))
            return
        return dataArr
            
        