
from flowField import *
from myUtils import *
import numpy as np
import scipy.io as sio

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

    # Matlab solutions that I currently have include the base flow in the zeroth mode. So, subtracting for consistency
    vField[0,tempDict['L'], -tempDict['M'], 0] -= vField.uBase
    
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

seprnFolderPath = '/home/sabarish/matData/seprn/'
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
    
MATLABFunctionPath = '/home/sabarish/Dropbox/gitwork/matlab/wavy_newt/3D/'
MATLABLibraryPath = '/home/sabarish/Dropbox/gitwork/matlab/library/'
import matlab.engine
def runMATLAB(g=1.0, eps=0.02, theta=0, Re=100., N=60, n=6):
    eng = matlab.engine.start_matlab()
    eng.addpath(MATLABFunctionPath)
    eng.addpath(MATLABLibraryPath)
    xList,fnorm,a,b = eng.runFromPy(g,eps,float(theta),Re,float(N),float(n),nargout=4)
    print('alpha is:',a)
    x = np.asarray(xList)
    eng.quit()
    vf,pf = mat2ff(arr=x, a=a,b=b,Re=Re,eps=eps,N=N,n=n)
    return vf,pf,fnorm
    
    
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
    
    def verify(self):
        # The only difference, as far as instances are concerned, between flowField and flowFieldWavy
        #   is the flowDict key 'eps'
        assert ('eps' in self.flowDict) and (type(self.flowDict['eps']) is float),\
            "Key 'eps' is missing, or is not float, in flowDict of flowFieldWavy instance"
        flowField.verify(self)
        return
    
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
    