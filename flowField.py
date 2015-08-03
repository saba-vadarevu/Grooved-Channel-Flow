import numpy as np
import scipy as sp
#from scipy.linalg import norm
from warnings import warn
from pseudo import chebdif, clencurt
#from pseudo.py import chebint

defaultDict = {'alpha':1.14, 'beta' : 2.5, 'omega':0.0, 'L': 23, 'M': 23, 'nd':3,'N': 35, 'K':0,
               'ReLam': 400.0, 'isPois':0.0, 'noise':0.0 }

defaultBaseDict = {'alpha':0, 'beta' : 0, 'omega':0.0, 'L': 0, 'M': 0, 'nd':1,'N': 35, 'K':0,
               'ReLam': 400.0, 'isPois':0.0, 'noise':0.0 }

def verify_dict(tempDict):
    '''Verify that the supplied flowDict has all the parameters required'''
    change_parameters = False
    if tempDict is None:
        tempDict = defaultDict
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
    '''Concatenate flowField objects. Use this to create a vector flowField from a scalar flowField as
    uvw = makeVector(u,v,w)'''
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
    ''' Provides a class to define u,v,w,p in 4D: time, x,z,y. 
    Ordered as (omega,alpha,beta,nd,y): omega, alpha, beta are Fourier modes in t,x,z respectively.
    nd is an index going from 0 to 3 for u,v,w,p. 
    y is the array of Chebyshev collocation nodes
    The dictionary is fundamental to the workings of the flowField class. 
        All three arguments can be used to provide a dictionary (arr can be an instance of flowField).
        flowDict argument has highest priority in defining the dictionary, 
            followed by dictFile
            followed by arr.flowDict
        If none of the above arguments provide a flowDict, a default dictionary (defined in the module) is used.
        A warning message is printed when the default dictionary is used.

    Methods: 
        slice(K,L,M,nd,N): Make grid finer or coarser along any direction
        view1d(): Return 1-d array of class flowField
        view4d(): Return 4-d array of class flowField
        etc... 
        Create an object using defaults as ff = flowField() and use tab completion to see all the methods'''
    
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
        if K is not None:
            K = int(abs(K))
            Kt = flowDict_temp['K']               # Temporary name for 'K' of self
            if K <= Kt:
                obj = obj[Kt-K:Kt+K+1]
            else: 
                obj = np.concatenate((  np.zeros((Kt-K,nxt,nzt,ndt,Nt),dtype=np.complex), obj,
                               np.zeros((Kt-K,nxt,nzt,ndt,Nt),dtype=np.complex)  ), axis=0)
            flowDict_temp['K']= K
            ntt = 2*K+1
        
        if L is not None:
            L = int(abs(L))
            Lt = flowDict_temp['L']               # Temporary name for 'L' of self
            if L <= Lt:
                obj = obj[:,Lt-L:Lt+L+1]
            else: 
                obj = np.concatenate((  np.zeros((ntt,abs(Lt-L),nzt,ndt,Nt),dtype=np.complex), obj,
                               np.zeros((ntt,abs(Lt-L),nzt,ndt,Nt),dtype=np.complex)  ), axis=1)
            flowDict_temp['L']= L
            nxt = 2*L+1
        
        if M is not None:
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
        
        if N is not None:
            N = abs(int(N))
            Nt = flowDict_temp['N']
            if N != Nt:
                y = chebdif(Nt,1)[0]
                obj_t = obj.reshape((obj.size/Nt,Nt))
                obj = np.zeros((obj_t.size/Nt,N),dtype=np.complex)
                for n in range(obj_t.size/N):
                    obj[n] = chebint(obj_t[n],y)
            obj = obj.reshape(obj.size)
            flowDict_temp['N'] = N
        
        obj = flowField(arr=obj, flowDict = flowDict_temp).view4d()
        
        if nd is not None:
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
        obj = self.view4d()[:,:,:,nd].copy()
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
            return 1.j*self.flowDict['alpha']*self.copy()
        partialX = self.view4d().copy()
        lArr = np.arange(-self.flowDict['L'],self.flowDict['L']+1).reshape(1,self.nx,1,1,1)
        partialX[:] = 1.j*self.flowDict['alpha']*lArr*partialX
        return partialX
    
    def ddx2(self):
        ''' Returns a flowField instance that gives the second partial derivative along "x" '''
        if self.nx == 1:
            return -1.*(self.flowDict['alpha']**2)*self.copy()
        partialX2 = self.view4d().copy()
        l2Arr = (np.arange(-self.flowDict['L'],self.flowDict['L']+1)**2).reshape(1,self.nx,1,1,1)
        partialX2[:] = -self.flowDict['alpha']**2*l2Arr*partialX2
        return partialX2
    
    def ddz(self):
        ''' Returns a flowField instance that gives the partial derivative along "z" '''
        if self.nz == 1:
            return 1.j*self.flowDict['beta']*self.copy()
        partialZ = self.view4d().copy()
        M = self.flowDict['M']
        mArr = np.arange( (M-abs(M))/2,abs(M)+1 ) .reshape((1,1,self.nz,1,1))
        partialZ[:] = 1.j*self.flowDict['beta']*mArr*partialZ
        return partialZ
    
    def ddz2(self):
        ''' Returns a flowField instance that gives the second partial derivative along "z" '''
        if self.nz == 1:
            return -1.*(self.flowDict['beta']**2)*self.copy()
        partialZ2 = self.view4d().copy()
        M = self.flowDict['M']
        mArr = np.arange( (M-abs(M))/2,abs(M)+1 ).reshape((1,1,self.nz,1,1))
        m2Arr = (mArr**2).reshape(1,1,self.nz,1,1)
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


    def convLinear(self,uBase=None):
        ''' Computes linearized convection term as [U u_x + v U',  U v_x,  U w_x ]
        Baseflow, uBase must be a 1D array of size "N" '''
        N = self.N
        y = chebdif(N,1)[0]
        if uBase == None:
            if self.flowDict['isPois'] == 1: uBase = 1.- y**2
            else: uBase = y
        else: assert uBase.size == N, 'uBase should be 1D array of size "self.N"'
        
        baseDict = defaultBaseDict
        baseDict['N'] = N
        uBaseFF = flowField(arr=uBase.reshape((1,1,1,1,N)),flowDict=baseDict).view4d()
        
        obj=self.view4d()
        if self.nd > 3: 
            warn('Convection term is being requested using a flowField with more than 3 components. \n',
            'Taking only the first 3 components ')
            obj = self.slice(nd=3).view4d()
        elif self.nd < 2: 
            raise RuntimeError('Need at least 2D perturbations for linear stability analysis')
        
        convTerm = flowField(flowDict = obj.flowDict.copy())
        convTerm[:] = uBaseFF*obj.ddx()
        convTerm[:,:,:,0:1] += uBaseFF.ddy()*self.getScalar(nd=1).view4d()
        convTerm.verify()
        return convTerm
    
    def grad3d(self, scalDim=0, nd=3, partialX=None, partialY=None, partialZ=None):
        ''' Computes gradient (in 3d by default) of either a scalar flowField object, 
            or of the first variable in a vector flowField object. 
            Grads of other variables can be calculated by passing scalDim=<index of variable>.
            Gradients in 2D (x and y) can be calculated by passing nd=2'''
        if partialX == None: partialX = flowField.ddx
        if partialY == None: partialY = flowField.ddy
        if partialZ == None: partialZ = flowField.ddz
        tempDict = self.flowDict.copy()
        tempDict['nd'] = nd
        if self.nd ==1:
            scal = self
        else:
            scal = self.getScalar(nd=scalDim)
        scal.verify()
        if nd == 3:
            gradVec = makeVector(partialX(scal), partialY(scal), partialZ(scal))
        elif nd ==2:
            gradVec = makeVector(partialX(scal),partialY(scal))
        
        return gradVec
    
    def grad(self,**kwargs):
        return self.grad3d(**kwargs)
    
    def grad2d(self, **kwargs):
        ''' Computes gradients in 2D (streamwise & wall-normal) for a scalar flowField object, 
            or for the scalar component of a vector field identified as vecField[:,:,:,scalDim]'''
        kwargs['nd'] = 2
        return self.grad3d(**kwargs)
        
    def laplacian(self, partialX2=None, partialY2=None, partialZ2=None):
        if partialX2 == None: partialX2 = flowField.ddx2
        if partialY2 == None: partialY2 = flowField.ddy2
        if partialZ2 == None: partialZ2 = flowField.ddz2
        lapl = self.view4d().copy()
        for scalDim in range(lapl.nd):
            lapl[:,:,:,scalDim] = partialX2(lapl[:,:,:,scalDim])+partialY2(lapl[:,:,:,scalDim])+partialZ2(lapl[:,:,:,scalDim])
        return lapl
        
    def div(self, partialX=None, partialY=None, partialZ=None, nd=3):
        ''' Computes divergence of vector field as u_x+v_y+w_z
        If a flowField with more than 3 scalars (nd>3) is supplied, takes first three components as u,v,w.
        Optional: 2-D divergence, u_x+v_y can be requested by passing nd=2'''
        if partialX == None: partialX = flowField.ddx
        if partialY == None: partialY = flowField.ddy
        if partialZ == None: partialZ = flowField.ddz
        assert nd in [2,3], ('Argument "nd" can only take values 2 or 3')
        assert self.nd >= nd, ('Too few scalar components in the vector')
        divergence = partialX(self.getScalar(nd=0)) + partialY(self.getScalar(nd=1))
        if nd== 3:
            divergence[:] += partialZ(self.getScalar(nd=2))
        
        return divergence
    
    def sumAll(self):
        '''Sums all elements of a flowField object (along all axes)'''
        obj = self.view4d().copyArray()
        return np.sum(np.sum(np.sum(np.sum(np.sum(obj,axis=4),axis=3),axis=2),axis=1),axis=0)
    
    def dot(self, vec2):
        assert isinstance(vec2,flowField), 'Inner products are only defined for flowField objects. Ensure passed object is a flowField instance'
        assert (self.flowDict == vec2.flowDict), 'Method for inner products is currently unable to handle instances with different flowDicts'
        
        w = clencurt(self.N).reshape((1,1,1,1,self.N))
        return flowField.sumAll(self*vec2.conjugate()*w)
    
    def norm(self):
        return np.sqrt(np.abs(self.dot(self)))
    
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
        
        # Ensuring a full set -|M|b,...,0b,..,|M|b is available before computing the convection term
        if self.flowDict['M'] > 0 :
            obj = self.slice(M=-self.flowDict['M'])
            u = obj.getScalar(nd=0); v = obj.getScalar(nd=1) ; w = obj.getScalar(nd=2)
            tempDict = obj.flowDict.copy()
        else:
            u = self.getScalar(nd=0);  v = self.getScalar(nd=1); w = self.getScalar(nd=2)
            tempDict = self.flowDict.copy()
        L = tempDict['L']; M = tempDict['M']; N = tempDict['N']
        nx = 2*L+1; nz= 2*abs(M)+1
        
        # Computing the base flow and adding it to the flowField before computing the convection term
        y = chebdif(N,1)[0]
        if uBase == None:
            if tempDict['isPois']==1:
                uBase = y**2
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
        return convTerm
                
                
        