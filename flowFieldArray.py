"""
flowFieldArray: Class that defines arrays of flowField objects
3-D numpy arrays are used, with axes 0,1,2 referring to eps,g,Re

Additionally, a class
flowFieldArray4D may also be defined (later) to allow handling inclinations (beta)"""
from flowField import *
from flowFieldWavy import *
import numpy as np

class flowFieldArray(np.ndarray):
    """Defines a numpy.ndarray object
    Input should be a list of flowField instances, which are then sorted along axes 0,1,2 by eps,g, Re
    Note: Currently, the class needs to know the number of eps, g, Re to work,
            given by arguments neps,ng,nRe"""
    def __new__(cls,fieldList,neps,ng,nRe):
        assert len(fieldList) == neps*ng*nRe, "Length of fieldList is not consistent with neps,ng,nRe"
        objList = np.asarray([flowFieldArrayObject(field) for field in fieldList])
        epsArr = np.asarray([field.eps for field in objList])
        epsInd = np.argsort(epsArr)   # Sort indices by increasing eps
        objList = objList[epsInd].reshape((neps,ng*nRe))
        
        for keps in range(neps):
            gArr = np.asarray([field.g for field in objList[keps]])
            gInd = np.argsort(gArr)
            objList[keps] = objList[keps,gInd]
        
        objList = objList.reshape((neps,ng,nRe))
        for keps in range(neps):
            for kg in range(ng):
                ReArr  = np.asarray([field.Re for field in objList[keps,kg]])
                ReInd = np.argsort(ReArr)
                objList[keps,kg] = objList[keps,kg,ReInd]
        obj = objList.view(cls)
        obj.neps = neps
        obj.ng = ng
        obj.nRe = nRe
        return obj
        
    def __array_finalize__(self,obj):
        if isinstance(obj, flowFieldArray):
            if self.ndim == 3:
                self.neps = self.shape[0]; self.ng = self.shape[1]; self.nRe = self.shape[2]
            else:
                self.neps = getattr(self,'neps',obj.neps)
                self.ng = getattr(self,'ng',obj.ng)
                self.nRe = getattr(self,'nRe',obj.nRe)
            return
        
    def getProperty(self,propStr,**kwargs):
        try:
            propArr = np.asarray([getattr(element,propStr) for element in self.reshape(self.size)])
        except:
            try:
                propArr = np.asarray([getattr(element.field,propStr)(**kwargs) for element in self.reshape(self.size)])
            except TypeError: 
                propArr = np.asarray([getattr(element.field,propStr) for element in self.reshape(self.size)])
            except: "Attribute doesn't exist for either flowFieldArrayObject or flowFieldWavy"
        return propArr.reshape(self.shape)
        
    
class flowFieldArrayObject:
    """Defines an object that contains flowFields, as well as eps,g,alpha,beta,Re as attributes"""
    def __init__(self,field=None):
        if field is None:
            self.field=None; self.eps=None; self.g=None; self.a=None; self.b=None; self.Re=None
        else:
            self.setField(field)
    
    def setField(self,field):
        assert isinstance(field,flowField), "Argument 'field' must be an instance of flowField"
        self.field=field
        self.eps = field.flowDict['eps'];  self.a = field.flowDict['alpha']; self.b = field.flowDict['beta']
        self.Re = field.flowDict['Re']; self.ReTau = np.sqrt(2.*self.Re)
        if self.a != 0.: self.g = self.a*self.eps
        else: self.g = self.b*self.eps    