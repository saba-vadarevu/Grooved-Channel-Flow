from flowField import *
from myUtils import *
import numpy as np
import scipy.io as sio
import os
import matlab.engine


homeFolder = os.environ['HOME']
def mat2ff(arr=None, **kwargs):
    '''Converts state-vectors from my MATLAB solutions to flowField objects.
        Once a solution is converted to a flowField object, it's .printCSV can be called for visualization.
    All parameters are keyword-parameters: arr (state-vec), a,b,eps,Re,N,n'''
    assert isinstance(arr,np.ndarray), 'A numpy array must be passed as the state-vector using keyword "arr"'
    assert set(['a','b','Re','N','eps','n']).issubset(kwargs), 'a,b,Re,eps,N, and n must be supplied as keyword arguments'
    assert all((type(kwargs[k]) is float) or (type(kwargs[k]) is int) for k in kwargs)
    tempDict = {'alpha':kwargs['a'],   'beta' : kwargs['b'], 'omega':0.0,   'K':0, \
                'L': abs(int(kwargs['n'])), 'M': -abs(int(kwargs['n'])),  'nd':3,  'N': abs(int(kwargs['N'])), \
                'Re': kwargs['Re'], 'isPois':1,  'eps': kwargs['eps'], \
                'noise':0.0 , 'lOffset':0.0, 'mOffset':0.0}
    n = tempDict['L']; N = tempDict['N']
    xind = 1; zind = 1
    if kwargs['a'] == 0.: tempDict['L'] = 0; xind = 0
    if kwargs['b'] == 0.: tempDict['M'] = 0; zind = 0
    
    assert arr.size == (2*n+1)*4*N, 'Size of state-vector is not consistent with supplied "n" and "N"'
    arr = arr.reshape((2*n+1, 4, N))
    
    vField = flowFieldWavy(flowDict=tempDict)
    pDict = tempDict.copy(); pDict['nd']= 1; pDict['isPois'] = -1
    pField = flowFieldWavy(flowDict=pDict)
    for k in range(2*n+1):
        vField[0,k*xind, k*zind] = arr[k,:3]
        pField[0,k*xind, k*zind] = arr[k,3:]

    
    return vField, pField

def data2ff(fName=None, ind=None):
    '''Reads a datafile with name fName. 
    Returns vField, pField, dict (containing a,b,g,eps, Re,fnorm)'''
    assert isinstance(fName,str), 'fName must be a string'
    if fName[-4:] != '.mat': fName = fName + '.mat'
    dataFile = sio.loadmat(fName, struct_as_record=False)
    dataStruct = dataFile['solution']
    nCases = dataStruct.shape[0]
    
    if (ind is not None) and (type(ind) is int):
        matArr = dataStruct[ind,0].X
        params = dataStruct[ind,0].Param; params=params.reshape(params.size)
        a=float(params[0]);b=float(params[1]); eps=float(params[2]); Re=float(params[4]); N=int(params[5]); n=int(params[6])
        g = a*eps;  fnorm = float(params[-1])
        if fnorm > 1.0e-5: 
            warn('Residual norm for the case is quite large and the solution cannot be trusted. Initializing a zero flowfield')
            matArr = np.zeros(matArr.shape, dtype=np.complex)
        vF, pF = mat2ff(arr=matArr,a=a,b=b,Re=Re,n=n,eps=eps,N=N)
        
        return vF, pF, {'eps':eps, 'g':g, 'Re':Re, 'a':a, 'b': b, 'fnorm':fnorm}
    else:
        vFieldList = []
        pFieldList = []
        epsArr = np.zeros(nCases); gArr = epsArr.copy(); bArr = epsArr.copy(); ReArr = epsArr.copy(); aArr = epsArr.copy(); fnormArr=epsArr.copy()
        for k in range(nCases):
            matArr = dataStruct[k,0].X
            params = dataStruct[k,0].Param; params=params.reshape(params.size)
            a=float(params[0]);b=float(params[1]); eps=float(params[2]); Re=float(params[4]); N=int(params[5]); n=int(params[6])
            g = a*eps;  fnorm = float(params[-1])
            if fnorm > 1.0e-5: 
                warn('Residual norm for the case is quite large and the solution cannot be trusted. Initializing a zero flowfield')
                matArr = np.zeros(matArr.shape, dtype=np.complex)
            vF, pF = mat2ff(arr=matArr,a=a,b=b,Re=Re,n=n,eps=eps,N=N )
            vFieldList.append(vF)
            pFieldList.append(pF)
            epsArr[k] = eps;  gArr[k] = a*eps; ReArr[k] = Re; bArr[k] = b; aArr[k] = a; fnormArr[k] = fnorm
        return vFieldList, pFieldList, {'eps':epsArr, 'g':gArr, 'Re':ReArr, 'a':aArr, 'b': bArr, 'fnorm':fnormArr}

seprnFolderPath = homeFolder+'/matData/seprn/'
def mapData2ff(eps=0.01, g= 1.0, Re=100, theta=0):
    """Returns the velocity and pressure flowFields along with parameters (as a dictionary) corresponding to the inputs:
    eps, g, Re, theta (streamwise inclination)
    The parameters are returned because the requested case (eps,g,Re,theta) might not exist as a solution"""
    gFileInd = 0
    gInd = int(40*(g-0.2))
    if g> 0.6:
        gFileInd = 1
        gInd = int(20*(g-0.65))
        if gInd > 11: gInd = int((gInd-11)/2+11)
    ReInd = np.int(np.log10(Re)-1)

    eFileInd = 2*int(40*(np.log10(eps)+3)/2)

    fileName = 'dataSeprn'+str(gFileInd)+'b'+str(int(theta))+'_'+str(eFileInd)+'.mat'

    #vfList, pfList,paramDict = data2ff(fName=folderPath+fileName,ind=gInd)
    #vf, pf,paramDict = data2ff(fName=folderPath+fileName,ind=3*gInd+ReInd)
    return data2ff(fName=seprnFolderPath+fileName,ind=3*gInd+ReInd)


MATLABFunctionPath = homeFolder+'/Dropbox/gitwork/matlab/wavy_newt/3D/'
MATLABLibraryPath = homeFolder+'/Dropbox/gitwork/matlab/library/'
def runMATLAB(g=1.0, eps=0.02, theta=0, Re=100., N=60, n=6,multi=False):
    """ Returns vf, pf, fnorm, flag"""
    eng = matlab.engine.start_matlab()
    eng.addpath(MATLABFunctionPath)
    eng.addpath(MATLABLibraryPath)
    if not multi:
        xList,fnorm,a,b = eng.runFromPy(g,eps,float(theta),Re,float(N),float(n),nargout=4)
        flg = 0
    else:
        Nlim = N; nlim = n
        N = 40; n = 4
        xList,fnorm,a,b,N,n,flg = eng.runFromPyMulti(g,eps,float(theta),Re,float(N),float(Nlim),float(n),float(nlim),nargout=7)

    x = np.asarray(xList)
    eng.quit()
    vf,pf = mat2ff(arr=x, a=a,b=b,Re=Re,eps=eps,N=N,n=n)
    return vf,pf,fnorm,flg
    
    
class flowFieldWavy(flowField):
    '''Subclass of flowField
    Corresponds to sinusoidal channel flows (pressure-gradient driven or wall-motion driven)
    Adds an extra attribute "eps" as a key in flowDict
    
    Overloads methods for differentiation: ddx, ddx2,ddy,..., along with .printCSV()'''
    def __new__(cls,arr=None,flowDict=None,dictFile=None):
        #obj = flowField.__new__(flowFieldWavy,arr=arr,flowDict=flowDict,dictFile=dictFile)
        obj = flowField.__new__(cls,arr=arr,flowDict=flowDict,dictFile=dictFile)
        if 'eps' not in obj.flowDict:
            warn('flowFieldWavy object does not have key "eps" in its dictionary. Setting "eps" to zero')
            obj.flowDict['eps'] = 0.
        else:
            assert type(obj.flowDict['eps']) is float, 'eps in flowDict must be of type float'
        obj.verify()
        return obj

#    def __array_finalize__(self,obj):
        # Supporting view-casting only for flowField instances with 'eps' in their dictionary
        #if isinstance(obj,flowField): assert ('eps' in obj.flowDict)
#        return
        

    def verify(self):
        # The only difference, as far as instances are concerned, between flowField and flowFieldWavy
        #   is the flowDict key 'eps'
        assert ('eps' in self.flowDict) and (type(self.flowDict['eps']) is float),\
            "Key 'eps' is missing, or is not float, in flowDict of flowFieldWavy instance"
        flowField.verify(self)
        return

    def slice(self,**kwargs):
        """slice method in flowField only returns a flowField instance. So, overriding it to return flowFieldWavy instances here"""
        return flowField.slice(self,**kwargs)#.view(flowFieldWavy)

    """ Partial derivatives in different coordinate systems:
        In my regular notes, I use \tilde{x}_i to refer to a 'physical system', and x_i to refer to a 'transformed system'
        In this code, I deviate from this convention for convenience. 
            x,y,z refer to the physical system in which the flow happens in a wavy channel
            X,Y,Z refer to the transformed system where the channel walls are flat
        All variables are Fourier decomposed along wall-parallel in the transformed system, i.e., in X-Z. 

        Below, d/dX and d/dZ are first defined as l\alpha u, m\beta u and so on... Similarly for d2/dX2 and d2/dZ2
        d/dx and d/dz are calculated using d/dX, d/dY, and d/dZ as follows:
            d/dx = d/dX + T_x d/dY 
            d/dy = d/dY
            d/dz = d/dZ + T_z d/dZ 
                where T = y - eps. [ exp{i(\alpha x+ \beta z)} + exp{-i(\alpha x + \beta z)}]
            T_x = -i eps. \alpha [ exp{i(\alpha x+ \beta z)} - exp{-i(\alpha x + \beta z)}]
            T_z = -i eps. \beta  [ exp{i(\alpha x+ \beta z)} - exp{-i(\alpha x + \beta z)}]

        When dealing with states, which are collections of Fourier modes, multiplying with, say exp{i l \alpha X},
            is accomplished by shifting all modes in the state vector by 'l' in the appropriate axis of the state-vector. 
        For t-modes, X-modes, and Z-modes, these are the zeroth, first, and second axes respectively.

        The assumption for the current implementation is that the wavy channel dictates the size of the periodic box,
            i.e., the fundamental Fourier mode in the state-vector corresponds to the wavenumber of surface-corrugations

    """ 
    def ddX(self):
        ''' Returns a flowFieldWavy instance that gives the partial derivative along "X", the streamwise coordinate in the transformed system '''
        return flowField.ddx(self) 
    
    def ddX2(self):
        ''' Returns a flowFieldWavy instance that gives the second partial derivative along "X" '''
        return flowField.ddx2(self)
    
    def ddZ(self):
        ''' Returns a flowFieldWavy instance that gives the partial derivative along "Z" '''
        return flowField.ddz(self)
    
    def ddZ2(self):
        ''' Returns a flowFieldWavy instance that gives the second partial derivative along "Z" '''
        return flowField.ddz2(self)

    def ddY(self):
        return self.ddy()

    def ddY2(self):
        return self.ddy2()

    def __ddxzYcomp(self):
        """ This will be used in ddx and ddz methods
            Returns [exp{i(aX+bZ)} - exp{-i(aX+bZ)}] d(field)/dy """

        if self.flowDict['beta'] == 0.: mShift = 0
        else: mShift = 1
        if self.flowDict['alpha'] == 0.: lShift = 0
        else: lShift = 1
        yComponent = self.copy(); yComponent[:] = 0.
        partialY = self.ddY()   # d(field)/dY 
        # Assigning mode '(k,l-1,m-1)' in partialY to mode '(k,l,m)' in yComponent, because that's what multiplying with exp{i(aX+bZ)} does
        #   But, if say a = 0, then (k,l,m-1) mode in partialY must be assigned to (k,l,m). This is what lShift and mShift are for
        yComponent[:, lShift:, mShift:] = partialY[:, :self.nx-lShift, :self.nz-mShift]  

        # Subtracting mode (k,l+1,m+1) in partialY from mode (k,l,m) in yComponent (third term in RHS)
        yComponent[:, :self.nx-lShift, :self.nz-mShift] -= partialY[:, lShift:, mShift:]

        return yComponent


    def ddx(self):
        """ Returns a flowFieldWavy instance that gives the partial derivative along 'x', 
            the streamwise coordinate in the physical system
            d(field)/dx = d(field)/dX - i eps. \alpha [exp{i (aX+bZ)} - exp{-i(aX+bZ)}] d(field)/dY"""
        if self.flowDict['alpha'] != 0.: 
            yComponent = -1.j*self.flowDict['eps']*self.flowDict['alpha']*self.__ddxzYcomp() 
            #.ddxzYcomp() calculates the second and third terms in RHS above, without multiplying  -i.eps.alpha
        else:
            yComponent = flowFieldWavy(flowDict=self.flowDict) # Zero state-vector 

        return self.ddX() + yComponent


    def ddz(self):
        """ Returns a flowFieldWavy instance that gives the partial derivative along 'z', 
            the streamwise coordinate in the physical system
            d(field)/dz = d(field)/dZ - i eps. \beta [exp{i (aX+bZ)} - exp{-i(aX+bZ)}] d(field)/dY"""
        if self.flowDict['beta'] != 0.: 
            yComponent = -1.j*self.flowDict['eps']*self.flowDict['beta']*self.__ddxzYcomp() 
            #.ddxzYcomp() calculates the second and third terms in RHS above, without multiplying  -i.eps.alpha
        else:
            yComponent = flowFieldWavy(flowDict=self.flowDict) # Zero state-vector 

        return self.ddZ() + yComponent

    def ddx2(self):
        """ Returns the second partial derivative along 'x', the streamwise coordinate in the physical system
            Earlier, I just did d/dx twice. But this gave me results dissimilar to my MATLAB code where I did
                a proper ddx2, because with the lazy way, I ignore some contributions that I wouldn't with the proper one
            So, now I do the proper one, given as follows (where C = ax+bz):
                
            d_xx(f) = d_XX(f) - 2.i.g.[e^{iC} - e^{-iC}]. d_XY(f) + g.a.[e^{iC} + e^{-iC}].d_Y(f) 
                        - g.g.[e^{2iC} - 2  + e^{-2iC}]  d_YY(f)
            Here, x and y are coordinates in the physical system, X and Y are coordinates in the transformed system
            For ddx and ddz, I used a single routine to calculate some of the terms, but I'm not doing that here,
                this one seems messier, complicating it would only make it worse
            """
        # If L = 0, it means the only mode is the zero mode, and the derivative would be zero
        if self.nx == 1: return flowFieldWavy(flowDict=self.flowDict)
        # Initiating as above returns a zero-vector
        # In initialization, a copy of the supplied flowDict is assigned as the object's attribute
        eps = self.flowDict['eps']; a=self.flowDict['alpha']; g=eps*a;

        # The first term, d_XX(f) is straight-forward:
        partialx2 = self.ddX2()

        # Later terms involve multiplying with e^{k.iC}, which is the same as shifting modes by 'k' in the state-vector
        #   The shift in x-modes is by 'k', but shift in z-modes could be '0' if the state-vector is only resolved in 'x',
        #       i.e., when M=0. To account for this:
        if self.flowDict['M'] == 0: mShift = 0
        else: mShift =1

        # Group the rest of the terms on the first line as follows:
        #       e^{iC}.[  -2.i.g.d_XY(f) + g.a.d_Y(f) ]  + e^{-iC}.[  2.i.g.d_XY(f)  + g.a.d_Y(f) ]
        # Step 1: Calculate d_XY(f) and d_Y(f):
        tempField1 = self.ddX().ddY()
        tempField2 = self.ddY()
        
        # Step 2.1: Add the above fields, multiply by scalars, shift by +1, and add to partialx2
        sumTemp = -2.j*g*tempField1 + g*a*tempField2
        partialx2[:, 1:  ,   mShift:       ] += sumTemp[:, :-1 , :self.nz-mShift ]
        # Step 2.2: As step 2.1, but shift by -1:
        sumTemp = 2.j*g*tempField1 + g*a*tempField2
        partialx2[:, :-1 , :self.nz-mShift ] += sumTemp[:, 1:  , mShift:         ]

        # Terms on the second line now. First the 1st and 3rd (the ones with shifts)
        # Computing the field that needs to be shifted
        tempField = -g*g*self.ddY2()
        # Adding the field with appropriate shifts:
        partialx2[:, 2:  ,   2*mShift:       ] += tempField[:, :-2 , :self.nz-2*mShift ]
        partialx2[ :, :-2 , :self.nz-2*mShift] += tempField[:, 2:  ,   2*mShift:       ]

        # Finally, adding the term 2*g*g*d_YY(f)
        partialx2 += -2.*tempField
        
        return partialx2

    def ddz2(self):
        """ Returns the second partial derivative along 'z', the spanwise coordinate in the physical system
            Earlier, I just did d/dz twice. But this gave me results dissimilar to my MATLAB code where I did
                a proper ddx2, because with the lazy way, I ignore some contributions that I wouldn't with the proper one
            So, now I do the proper one, given as follows (where C = ax+bz):
                
            d_zz(f) = d_ZZ(f) - 2.i.eps.b.[e^{iC} - e^{-iC}]. d_ZY(f) + eps.b.b.[e^{iC} + e^{-iC}].d_Y(f) 
                        - eps^2.b^2.[e^{2iC} - 2  + e^{-2iC}]  d_YY(f)
            Here, x and y are coordinates in the physical system, X and Y are coordinates in the transformed system
            For ddx and ddz, I used a single routine to calculate some of the terms, but I'm not doing that here,
                this one seems messier, complicating it would only make it worse
            """
        # If M = 0, it means the only mode is the zero mode, and the derivative would be zero
        if self.nz == 1: return flowFieldWavy(flowDict=self.flowDict)
        # Initiating as above returns a zero-vector
        # In initialization, a copy of the supplied flowDict is assigned as the object's attribute
        eps = self.flowDict['eps']; b=self.flowDict['beta'];

        # The first term, d_ZZ(f) is straight-forward:
        partialz2 = self.ddZ2()

        # Later terms involve multiplying with e^{k.iC}, which is the same as shifting modes by 'k' in the state-vector
        #   The shift in z-modes is by 'k', but shift in x-modes could be '0' if the state-vector is only resolved in 'z',
        #       i.e., when L=0. To account for this:
        if self.flowDict['L'] == 0: lShift = 0
        else: lShift =1

        # Group the rest of the terms on the first line as follows:
        #       e^{iC}.[  -2.i.eps.b.d_ZY(f) + eps.b^2.d_Y(f) ]  + e^{-iC}.[  2.i.eps.b.d_ZY(f)  + eps.b^2.d_Y(f) ]
        # Step 1: Calculate d_ZY(f) and d_Y(f):
        tempField1 = self.ddZ().ddY()
        tempField2 = self.ddY()
        
        # Step 2.1: Add the above fields, multiply by scalars, shift by +1, and add to partialz2
        sumTemp = -2.j*eps*b*tempField1 + eps*b*b*tempField2
        partialz2[:, lShift:  ,   1:       ] += sumTemp[:, :self.nx-lShift , :-1 ]
        # Step 2.2: As step 2.1, but shift by -1:
        sumTemp = 2.j*eps*b*tempField1 + eps*b*b*tempField2
        partialz2[:, :self.nx-lShift , :-1 ] += sumTemp[:, lShift:  , 1:         ]

        # Terms on the second line now. First the 1st and 3rd (the ones with shifts)
        # Computing the field that needs to be shifted
        tempField = -(eps**2)*(b**2)*self.ddY2()
        # Adding the field with appropriate shifts:
        partialz2[:, 2*lShift:     , 2:  ] += tempField[:, :self.nx-2*lShift , :-2]
        partialz2[ :,:self.nx-2*lShift, :-2] += tempField[:, 2*lShift:      , 2:  ]

        # Finally, adding the term 2*g*g*d_YY(f)
        partialz2 += -2.*tempField
        
        return partialz2

   
    def printPhysical(self,**kwargs):
        '''Refer to flowField.printCSV()
        This method sets "yOff" to 2*eps'''
        kwargs['yOff'] = 2.*self.flowDict['eps']
        return flowField.printPhysical(self,**kwargs)
    
    def printPhysicalPlanar(self,**kwargs):
        '''Refer to flowField.printCSVPlanar()'''
        kwargs['yOff'] = 2.*self.flowDict['eps']
        return flowField.printPhysicalPlanar(self,**kwargs)
    
    def wallTangent(self, xLoc=0., zLoc=0., direcXZ=(1.,0.)):
        """Returns a triplet that corresponds to the wall-tangent in physical space (x,y,z),
        Arguments: xLoc, zLoc (default to (0,0))
            direcXZ: Refers to the wall-parallel direction (plane) along which the tangent vector is needed"""
        # Wall surfaces are given by
        #    ySurf = +/-1  + 2*eps*cos(ax+bz), or   S(x,y,z) =  y - 2*eps*cos(ax+bz) -/+ 1 = 0
        # The (local) normal to the surface is given by grad(S) = (2*a*eps*sin(ax+bz), 1, 2*b*eps*sin(ax+bz))
        # To obtain the wall tangent along a given direcXZ, 
        #              we need to calculate the projection of direcXZ on the plane normal to grad(S) =: (n1,n2,n3),
        #                     given by n1*x + n2*y + n3*z = 0
        # Given direcXZ = (d1,d3), the y-component (d2) is obtained as  d2 = -(n1*d1 + n3*d3)/n2 = -(n1*d1+n3*d3)
        d1 = direcXZ[0]; d3 = direcXZ[1]
        a = self.flowDict['alpha']; b = self.flowDict['beta']; eps = self.flowDict['eps']
        assert (type(xLoc) is float) and (type(zLoc) is float), "Arguments xLoc and zLoc must be of type float"
        n1 = 2.*a*eps*np.sin(a*xLoc+b*zLoc);    n3 = 2.*b*eps*np.sin(a*xLoc+b*zLoc)
        
        d2 = -(n1*d1 + n3*d3)
        dAmp = np.sqrt(d1**2 + d2**2 + d3**2)
        # Returning a unit vector
        return (d1/dAmp, d2/dAmp, d3/dAmp)
    
    def wallNormal(self,xLoc=0.,zLoc=0.):
        """Returns a triplet that corresponds to the wall-normal in physical space (x,y,z) at bottom surface
        Arguments: xLoc, zLoc (default to (0,0))
            To obtain the wall-normal at top surface, multiply the output of this by -1"""
        # Read comments for .wallTangent()
        a = self.flowDict['alpha'];  b = self.flowDict['beta'];   eps = self.flowDict['eps']
        n1 = 2.*a*eps*np.sin(a*xLoc+b*zLoc);    n3 = 2.*b*eps*np.sin(a*xLoc+b*zLoc);  n2 = 1.
        nAmp = np.sqrt(n1**2+n2**2+n3**2)
        return (n1/nAmp, n2/nAmp, n3/nAmp)
   
def weighted2ff(flowDict=None,arr=None,weights=None):
    """ Converts 1-d np.ndarray into flowFieldWavy object
    Inputs:
        Non-optional: flowDict, arr
        Optional: weights (Clencurt weights that were used to weight the supplied array)
    If weights are not supplied, clencurt weights are calculated using 'N' in flowDict
    But since I wouldn't change 'N' within a GMRES, it's best to calculate the weights once
        and to keep passing it to the method
    """
    assert (flowDict is not None) and (arr is not None), "Both flowDict and arr must be supplied"
    N = flowDict['N']
    if weights is None:
        weights = clencurt(N)

    invRootWeights = (np.sqrt(1./weights)).reshape((1,N))
    deweightedArr =  invRootWeights * (arr.reshape(arr.size//N, N))

    return flowFieldWavy(flowDict=flowDict, arr= deweightedArr.reshape(arr.size) ) 


