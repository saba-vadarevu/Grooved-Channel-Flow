from flowField import *
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
    pDict = tempDict.copy(); pDict['nd']= 1
    pField = flowFieldWavy(flowDict=pDict)
    for k in range(2*n+1):
        vField[0,k*xind, k*zind] = arr[k,:3]
        pField[0,k*xind, k*zind] = arr[k,3:]
    return vField, pField

def data2ff(fName=None, ind=None):
    '''Reads a datafile with name fName. '''
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
    
    def printCSV(self,**kwargs):
        '''Refer to flowField.printCSV()
        This method sets "yOff" to 2*eps'''
        kwargs['yOff'] = 2.*self.flowDict['eps']
        return flowField.printCSV(self,**kwargs)
    
    def printCSVPlanar(self,**kwargs):
        '''Refer to flowField.printCSVPlanar()'''
        kwargs['yOff'] = 2.*self.flowDict['eps']
        print('yOff in ffwavy is set as',kwargs['yOff'])
        return flowField.printCSVPlanar(self,**kwargs)
    
        