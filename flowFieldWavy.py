from flowField import *
from myUtils import *
import numpy as np
import scipy.io as sio
import os
import h5py


homeFolder = os.environ['HOME']
def mat2ff(arr=None, **kwargs):
    '''Converts state-vectors from my MATLAB solutions to flowField objects.
        Once a solution is converted to a flowField object, it's .printCSV can be called for visualization.
    All parameters are keyword-parameters: arr (state-vec), a,b,eps,Re,N,n'''
    assert isinstance(arr,np.ndarray), 'A numpy array must be passed as the state-vector using keyword "arr"'
    assert set(['a','b','Re','N','eps','n']).issubset(kwargs), 'a,b,Re,eps,N, and n must be supplied as keyword arguments'
    # assert all((type(kwargs[k]) is np.float) or (type(kwargs[k]) is int) for k in kwargs)
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

def updateDict(dict1,dict2):
    tempDict = dict1.copy()
    tempDict.update(dict2)
    return tempDict


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
        a=np.float64(params[0]);b=np.float64(params[1]); eps=np.float64(params[2]); Re=np.float64(params[4]); N=int(params[5]); n=int(params[6])
        g = a*eps;  fnorm = np.float64(params[-1])
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
            a=np.float64(params[0]);b=np.float64(params[1]); eps=np.float64(params[2]); Re=np.float64(params[4]); N=int(params[5]); n=int(params[6])
            g = a*eps;  fnorm = np.float64(params[-1])
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

def h52ff(fileName,pres=False):
    """Reads velocity data from h5 file and creates a flowFieldRiblet object out of it
    NOTE 1: Currently, this function only works for Couette flow equilibria, and possibly for
        TWS in Poiseuille flow (but 'isPois' in flowDict has to be manually set to 1, 
            and (-vf.y+1-vf.y**2) must be  added to the zeroth mode)
        Other solutions might have different resolutions, so the defaults I use here may not work.
    NOTE 2: The h5 files currently have only velocity data. ChannelFlow has a Poisson solver for pressure,
        and I have one too. For now, I'll use my own solver. 
    """
    if pres: 
        L = 23; M = 23; nd = 1
    else: 
        L = 15; M = 15; nd = 3

    nx = 2*L+2; nz = 2*M+2

    assert fileName[-3:] == ".h5"
    f = h5py.File(fileName, 'r')
    u = np.array(f['data']['u'])
    x = np.array(f['geom']['x'])
    y = np.array(f['geom']['y'])
    z = np.array(f['geom']['z'])

    # I order my field-objects as (t, x, z, component, y)
    # First, reshaping to (x,z,y)
    uT = np.zeros((nd,nx,nz,35))
    for k in range(nz):
        uT[:,:,k] = u[:,:,:,k].reshape((nd,nx,35))

    # FFT, along with a shift to order modes as -L,..,0,1,..,L instead of numpy's default 0,1,..,L,-L,..,-1
    uSpecArr = np.fft.fftshift(   np.fft.fftn(uT, axes=[1,2]),  axes=[1,2])/nx/nz
    uSpecArr = uSpecArr[:,1:,1:]    # Numpy's fft returns one extra negative mode, removing that for consistency
    nx -= 1; nz -= 1
    if nd == 3:
        # Reshaping so that the component (u,v,w)  axis is between z and y
        u1SpecArr = uSpecArr[0].reshape((nx,nz,1,35))
        u2SpecArr = uSpecArr[1].reshape((nx,nz,1,35))
        u3SpecArr = uSpecArr[2].reshape((nx,nz,1,35))
        uSpecFFArr = np.concatenate((u1SpecArr,u2SpecArr,u3SpecArr), axis=2)
    else: 
        uSpecFFArr = uSpecArr

    flowDict = getDefaultDict()
    # The following parameters describe Couette equilibria from Channel flow
    # For TWS, I must manually change 'isPois' later
    flowDict.update({'L':L,'M':M,'K':0,'N':35, 'nd':nd, 'eps':0.,'Re':400.,'isPois':0})

    # I can only solve cases of Riblets for equilibria and/or TWS. When eps=0. in flowDict, 
    #   flowField, flowFieldWavy, and flowFieldRiblet classes all have equivalent methods
    obj = flowFieldRiblet(flowDict=flowDict, arr=uSpecFFArr.reshape(uSpecFFArr.size))
    if L != 23:
        obj = obj.slice(L=23,M=23)  # Padding with extra modes for computation. Gibson's code uses L=M=23 
    if not pres: obj[0,obj.nx//2, obj.nz//2, 0] += obj.y
    
    return obj

def loadh5(filename):
    """Loads hdf5 files that I write for my flowFieldRiblet solutions
    For loading solutions from channelflow.org, use h52ff()
    Input:
        filename"""
    inFile = h5py.File(filename,"r")
    field = inFile['field']
    tempDict = {}
    for key in field.attrs:
        tempDict[key] = field.attrs[key]
    tempDict['nd']=4    # Solution field must include a pressure field
    ff = flowFieldRiblet(arr=np.array(field), flowDict=tempDict)
    inFile.close()
    return ff


# Standardizing names of flowField files 
def dict2name(flowDict,prefix='solutions/'):
    """ Produce a file name for storing flowfields, given the flowDict
    file name is organized as
        clsName: 'Flat', 'Rib', '' for Wavy
        T+theta; degress, int: 0,15,30, so on... 
        E+epsilon; 4 decimals: 0012 if epsilon =0.0012
        Gx+ slope(x); 4 digits: 1.125 as 1125
        Gz+ slope(z); 4 digits: 1.125 as 1125
        Re+ Re; int
        L + L; int
        M + M; int
        N + N; int
        
    Currently, clsName is left as an empty string. If needed, use the prefix keyword argument to set this
    """
    flowDict=flowDict.copy()
    if 'eps' not in flowDict:
        flowDict['eps'] = 0.
    clsName=''
    if flowDict['beta'] == 0.:
        theta = '00'
    elif flowDict['alpha'] == 0.:
        theta = '90'
    else:
        theta = '%02d' %(int( 180./np.pi* np.arctan(flowDict['beta']/flowDict['alpha'])   ))

    assert flowDict['eps'] == round(flowDict['eps'],4)
    
    epsilon = '%04d' %(int( round(flowDict['eps']*1.0e4)))
    slopeX = '%04d' %(int( round(flowDict['alpha']*flowDict['eps']*1.0e3)))
    slopeZ = '%04d' %(int( round(flowDict['beta']*flowDict['eps']*1.0e3)))
    Re = '%04d' %(int(round(flowDict['Re'])))
    L = '%02d' %(int(flowDict['L']));  M = '%02d' %(int(flowDict['M'])); N = '%02d' %(int(flowDict['N']))
    
    fName = 'T'+theta+'E'+epsilon+'Gx'+slopeX+'Gz'+slopeZ+'Re'+Re+'L'+L+'M'+M+'N'+N
    # The length of fName should be 1+2 + 1+4 + 2+4 + 2+4 + 2+4 + 1+2 + 1+2 + 1+2  = 35
    assert len(fName) == 35
    
    fName = prefix+clsName+fName
    return fName

def name2dict(fName):
    """Populate flowDict from filename. Ignores prefix. 
    Additional defaults: 
        isPois: 1,  nd: 3
        K: 0, omega: 0.
        lOffset:0., mOffset:0., noise:0.
        """
    if fName[-4:] == '.npy':
        fName = fName[:-4]
    # Need only the last 35 characters in fName
    fName = fName[-35:]
    
    flowDict = {'K':0, 'isPois':1, 'lOffset':0., 'mOffset':0., 'nd':4, 'noise':0.0, 'omega':0.0}
    flowDict['N'] = int(fName[-2:])
    flowDict['M'] = int(fName[-5:-3])
    flowDict['L'] = int(fName[-8:-6])
    flowDict['Re'] = np.float64(fName[-13:-9])
    
    gz = np.float64( fName[-19:-15] )/1000.
    gx = np.float64( fName[-25:-21] )/1000.
    flowDict['eps'] = np.float64( fName[-31:-27] )/10000.
    
    try:
        flowDict['alpha'] = gx/flowDict['eps']
    except: 
        flowDict['alpha'] = 1.14
    try: 
        flowDict['beta'] = gz/flowDict['eps']
    except: 
        flowDict['beta'] = 2.5
    
    return flowDict


def saveff(vf,pf,prefix='solutions/'):
    fName = dict2name(vf.flowDict, prefix=prefix)
    x = vf.appendField(pf)
    xArr = x.view1d().copyArray()
    np.save(fName,xArr)
    print('saved to ',fName)
    return

def loadff(fName,prefix='',checkNorm=False):
    flowDict = name2dict(fName)
    if not fName.endswith('.npy'):
        fName = fName + '.npy'
    xArr = np.load(fName)
    x = flowFieldWavy(flowDict=flowDict, arr=xArr)
    vf = x.slice(nd=[0,1,2])
    pf = x.getScalar(nd=3)
    if checkNorm:
        ls = 0; ms = 0
        if vf.nx != 0: ls = 1
        if vf.nz != 0: ms = 1
        res = vf.slice(L=vf.nx//2+2*ls, M=vf.nz//2+2*ms).residuals(pField=pf.slice(L=pf.nx//2+2*ls, M=pf.nz//2+2*ms))
        assert res.slice(L=vf.nx//2,M=vf.nz//2).norm() <= 1.0e-12

    return vf,pf
 

    
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
            assert type(obj.flowDict['eps']) is np.float64 or (type(obj.flowDict['eps']) is np.float), 'eps in flowDict must be of type float'
        obj.verify()
        return obj

#    def __array_finalize__(self,obj):
        # Supporting view-casting only for flowField instances with 'eps' in their dictionary
        #if isinstance(obj,flowField): assert ('eps' in obj.flowDict)
#        return
        

    def verify(self):
        # The only difference, as far as instances are concerned, between flowField and flowFieldWavy
        #   is the flowDict key 'eps'
        assert (type(self.flowDict['eps']) is np.float64) or (type(self.flowDict['eps']) is np.float),\
            "Key 'eps' is missing, or is not float, in flowDict of flowFieldWavy instance"
        flowField.verify(self)
        return


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
            yComponent = self.zero()

        return self.ddX() + yComponent


    def ddz(self):
        """ Returns a flowFieldWavy instance that gives the partial derivative along 'z', 
            the streamwise coordinate in the physical system
            d(field)/dz = d(field)/dZ - i eps. beta [exp{i (aX+bZ)} - exp{-i(aX+bZ)}] d(field)/dY"""
        if self.flowDict['beta'] != 0.: 
            yComponent = -1.j*self.flowDict['eps']*self.flowDict['beta']*self.__ddxzYcomp() 
            #.ddxzYcomp() calculates the second and third terms in RHS above, without multiplying  -i.eps.alpha
        else:
            yComponent = self.zero()

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
        if self.nx == 1: return self.zero()
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
        if self.nz == 1: return self.zero()
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

    def residuals(self,pField=None,**kwargs):
        """
        Overloading the residuals function to slice fields as L+=2 and M+=2
        """
        L = self.flowDict['L']
        M = self.flowDict['M']
        if L != 0: Lnew = L+2
        else: Lnew = L
        if M != 0: Mnew = M+2
        else: Mnew = M
        vf = self.slice(L=Lnew, M=Mnew)
        if pField is None:
            pField = vf.getScalar().zero()
        else:
            pField = pField.slice(L=Lnew,M=Mnew)
        res = flowField.residuals(vf, pField=pField, **kwargs)

        return res.slice(L=L,M=M)
   
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

class flowFieldRiblet(flowFieldWavy):
    """Subclass of flowFieldWavy
    Exclusively for riblet-mounted channels. 
    In flowFieldWavy, flowDict['alpha'] and flowDict['beta'] gave the size of the periodic box
        as well as the orientation of the waviness: (alpha*x + beta*z)
        So, a riblet-mounted geometry also meant that the flow was homogenous in x (since alpha=0)
    In this subclass, we allow for riblet geometries to be resolved along streamwise, defined
        by flowDict['alpha'] and flowDict['L']
    Since there is no streamwise variation in the surface, 
        We overload methods for differentiation along x, ddx() and ddx2() revert to 
        flowField.ddx() and flowField.ddx2()"""
    def __new__(cls,arr=None,flowDict=None,dictFile=None):
        #obj = flowField.__new__(flowFieldWavy,arr=arr,flowDict=flowDict,dictFile=dictFile)
        obj = flowField.__new__(cls,arr=arr,flowDict=flowDict,dictFile=dictFile)
        if 'eps' not in obj.flowDict:
            warn('flowFieldWavy object does not have key "eps" in its dictionary. Setting "eps" to zero')
            obj.flowDict['eps'] = 0.
        else:
            assert type(obj.flowDict['eps']) is np.float64 or (type(obj.flowDict['eps']) is np.float), 'eps in flowDict must be of type float'
        obj.verify()
        return obj

    def ddx(self):
        return self.ddX()

    def ddx2(self):
        return self.ddX2()
    
    def ddz(self):
        if self.nz == 1:
            return self.zero()
        partialZ = self.zero()
        partialZ += self.ddZ()
        partialY = self.ddy()
        gz = self.flowDict['eps'] * self.flowDict['beta']

        partialZ[:,:,1:] += -1.j*gz* partialY[:,:,:-1]
        partialZ[:,:,:-1]+= 1.j* gz* partialY[:,:,1: ]
        return partialZ
    
    
    def ddz2(self):
        # If M = 0, it means the only mode is the zero mode, and the derivative would be zero
        if self.nz == 1: return self.zero()
        # Initiating as above returns a zero-vector
        # In initialization, a copy of the supplied flowDict is assigned as the object's attribute
        eps = self.flowDict['eps']; b=self.flowDict['beta'];

        # The first term, d_ZZ(f) is straight-forward:
        partialz2 = self.ddZ2()

        # Later terms involve multiplying with e^{k.iC}, which is the same as shifting modes by 'k' in the state-vector
        #   The shift in z-modes is by 'k', but shift in x-modes could be '0' if the state-vector is only resolved in 'z',
        #       i.e., when L=0. To account for this:
        # Scratch all that, for riblet geometry, no shift in l
        lShift =0

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
    

    def saveh5fName(self,fNamePrefix='ribEq1',prefix='solutions/ribEq/'):
        fName = 'L'+str(self.flowDict['L'])+'M'+str(self.flowDict['M'])+'N'+str(self.flowDict['N'])
        fName = fName + 'E'+ '%04d' %(int( round(self.flowDict['eps']*1.0e4))) + '.hdf5'

        fName = fNamePrefix + fName
        return fName , prefix


    def saveh5(self,fNamePrefix='ribEq1',prefix='solutions/ribEq/'):
        """ Saves self to a hdf5 file
        Input:
            fNamePrefix (None): prefix to file name, such as LBeq1
            prefix ('solutions/'): Path prefix
        Name of hdf5 file is fNamePrefix+prefix+'LxMxNxExxxx.hdf5'
        
        fName = 'L'+str(self.flowDict['L'])+'M'+str(self.flowDict['M'])+'N'+str(self.flowDict['N'])
        fName = fName + 'E'+ '%04d' %(int( round(self.flowDict['eps']*1.0e4))) + '.hdf5'

        fName = prefix + fNamePrefix + fName
	"""

        fName, pathPrefix = self.saveh5fName(fNamePrefix=fNamePrefix, prefix=prefix)
        fName = pathPrefix + fName

        outFile = h5py.File(fName, "w")
        assert self.nd == 4, "Save flowFields that include a pressure field"

        field = outFile.create_dataset("field",data=self.flatten(),compression='gzip')

        for key in self.flowDict:
            field.attrs[key] = self.flowDict[key]
        print("saved field to ",fName)
        outFile.close()
        return





