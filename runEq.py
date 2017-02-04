#!/usr/bin/python3
import os
import argparse
import resource 
import sys
import time
import datetime
from warnings import warn



workingDir = os.getcwd() 
if not workingDir.endswith('/'):
    workingDir = workingDir + '/'
loadPath = workingDir
savePath = workingDir

# Ensure there is a ./tmp/
if not os.path.exists('./tmp'):
    os.mkdirs('./tmp')

##---------------------------------------------------------------------------------------------------------
## Parse commandline arguments 
##---------------------------------------------------------------------------------------------------------

#Creating an argument parser
parser = argparse.ArgumentParser(description=\
        "Refines flowfield (.hdf5 file) in current directory to obtain PCF/Channel flow equilibria.\
        Defines exactRiblet() instance from a hdf5 file and commandline arguments, and runs exactRiblet.iterate().\
        eps1-eps6 and phi1-phi6 default to 0, even if the flowfield in hdf5 file has different values. \
        To run grooved PCF, these arguments must be specified in commandline. eps4-eps6 and phi1-phi6 are currently not supported. ")

# Argparse does not take L,M,N as mandatory arguments anymore. When they are not supplied, 
#   they are set based on the defaults for the symmetries and the method. 
#   When supplied as keyword arguments, they are forced away from the defaults.

parser.add_argument("--eps1",help="Semi-amplitude, Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--eps2",help="Semi-amplitude, Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--eps3",help="Semi-amplitude, Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--eps4",help="Semi-amplitude, Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--eps5",help="Semi-amplitude, Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--eps6",help="Semi-amplitude, Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--phi1",help="Spanwise shift in phase of e1 w.r.t z=0 of original solution, \
        Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--phi2",help="Spanwise shift in phase of e2 w.r.t z=0 of original solution, \
        Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--phi3",help="Spanwise shift in phase of e3 w.r.t z=0 of original solution, \
        Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--phi4",help="Spanwise shift in phase of e4 w.r.t z=0 of original solution, \
        Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--phi5",help="Spanwise shift in phase of e5 w.r.t z=0 of original solution, \
        Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--phi6",help="Spanwise shift in phase of e6 w.r.t z=0 of original solution, \
        Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--L",help="Streamwise modes, Optional, default: None (chosen for method,symms)", type=int)
parser.add_argument("--M",help="Spanwise modes, Optional, default: None (chosen for method,symms)", type=int)
parser.add_argument("--N",help="Wall-normal modes, Optional, default: None (chosen for method,symms)", type=int)
parser.add_argument("--tol",help="Tolerance for N-R method, Optional, default: 1.0e-13", default = 1.0e-13,type=float)
parser.add_argument("--iterMax",help="Total number of iterations, Optional, default: 15", default = 15,type=int)
parser.add_argument("--log",help="Name of log file, Optional, default: outFile.txt (append)", default = 'outFile.txt',type=str)
parser.add_argument("--prefix",help="fNamePrefix for solution files, Optional, default: EQ1", default = 'EQ1',type=str)
parser.add_argument("--method",help="Method to use for solving equations,\
        Options: 'simple' (Newton's+jacobian inversion+line search)\
                'trf', 'dogbox', 'lm' from scipy's least_squares\
                'lm' is currently not supported, since it doesn't allow saving intermediate solutions.\
                Default: 'simple'", default='simple',type=str)
parser.add_argument("--tr_solver",help="Trust region solver to use.\
        Options: 'exact', 'lsmr', 'None' (let scipy choose appropriate solver). Default: None", default=None,type=str)
parser.add_argument("--fName",help="Input file name. If not supplied or invalid, use the first .hdf5 found (warn if multiple files are found)",default='.hdf5',type=str)

symParser1 = parser.add_mutually_exclusive_group(required=False)
symParser2 = parser.add_mutually_exclusive_group(required=False)
symParser3 = parser.add_mutually_exclusive_group(required=False)
symParser4 = parser.add_mutually_exclusive_group(required=False)
symParser5 = parser.add_mutually_exclusive_group(required=False)
symParser6 = parser.add_mutually_exclusive_group(required=False)
symParser1.add_argument("--no-sigma1",help="Do not impose sigma1. Imposed otherwise", dest='sigma1',action="store_false")
symParser1.add_argument("--sigma1",help="Impose sigma1 (default)", dest='sigma1',action="store_true")
symParser2.add_argument("--no-sigma3",help="Do not impose sigma3 (default)", dest='sigma3',action="store_false")
symParser2.add_argument("--sigma3",help="Impose sigma3. Not imposed otherwise", dest='sigma3',action="store_true")
symParser3.add_argument("--no-sigma1T",help="Do not impose sigma1T (default)", dest='sigma1T',action="store_false")
symParser3.add_argument("--sigma1T",help="Impose sigma1T if supplied. Not imposed otherwise. ", dest='sigma1T',action="store_true")
symParser4.add_argument("--no-sigma3T",help="Do not impose sigma3T (default)", dest='sigma3T',action="store_false")
symParser4.add_argument("--sigma3T",help="Impose sigma3T. Not imposed otherwise", dest='sigma3T',action="store_true")
symParser5.add_argument("--jacobian",help="Supply modified Jacobian if using trf,dogbox, or lm methods. Don't otherwise",dest='jacobian',action='store_true')
symParser5.add_argument("--no-jacobian",help="Do not supply modified Jacobian if using trf,dogbox, or lm methods (default)",dest='jacobian',action='store_false')
symParser6.add_argument("--resolutionArr",help="Use resolutionArr to define L,M,N if they are not supplied (default). Don't otherwise",dest='resolutionArr',action='store_true')
symParser6.add_argument("--no-resolutionArr",help="Use L,M,N from the flowfield in the .hdf5 file if they aren ot supplied. Default action is to use resolutionArr when L,M,N are not supplied.",dest='resolutionArr',action='store_false')
parser.set_defaults(sigma1=True)
parser.set_defaults(sigma3=False)
parser.set_defaults(sigma1T=False)
parser.set_defaults(sigma3T=False)
parser.set_defaults(jacobian=False)
parser.set_defaults(resolutionArr=True)


args = parser.parse_args()

L = args.L
M = args.M
N = args.N
iterMax = args.iterMax
logName = args.log
tol = args.tol
fNamePrefix = args.prefix
sigma1 = args.sigma1
sigma1T = args.sigma1T
sigma3 = args.sigma3
sigma3T = args.sigma3T
method = args.method
if method == 'lm':
    method = 'simple'
    print("Method 'lm' is currently not supported. Using 'simple' instead.")
supplyJac = args.jacobian
inFileName = args.fName



##-------------------------------------------------------------------------------
# Use args.method to decide which numpy/scipy version to use.
# This is an issue only when running on IRIDIS. 
##-------------------------------------------------------------------------------

import exactRiblet as rib
import numpy as np
from flowFieldWavy import *



##-------------------------------------------------------------------------------
# Build state-vector based on parsed arguments
##-------------------------------------------------------------------------------

# Default resolutions based on method and symms
# Axis 0: [asym, (sigma1,sigma3), (sigma1T, sigma13), sigma13T]
# Axis 1: ['simple', 'dogbox', 'trf']
# Axis 2: [L,M,N]
resolutionArr = np.zeros((4, 3, 3), dtype=np.int)

resolutionArr[0,0] = [7,12,30]  # asym, simple
resolutionArr[0,1] = [7,10,30]  # asym, dogbox
resolutionArr[0,2] = [7,9 ,25]  # asym, trf

resolutionArr[1,0] = [11,16,35]  # sigma1 or sigma3, simple
resolutionArr[1,1] = [9 ,15,35]  # sigma1 or sigma3, dogbox
resolutionArr[1,2] = [8 ,12,32]  # sigma1 or sigma3, trf

resolutionArr[2,0] = [16,16,35]  # sigma1T or sigma13, simple
resolutionArr[2,1] = [12,16,35]  # sigma1T or sigma13, dogbox
resolutionArr[2,2] = [7 ,15,35]  # sigma1T or sigma13, trf

resolutionArr[3,0] = [16,24,41]  # sigma13T, simple
resolutionArr[3,1] = [12,20,35]  # sigma13T, dogbox
resolutionArr[3,2] = [16,16,35]  # sigma13T, trf

# Obtaining resolution based on method/symm 
# First, parsing symm and method to indices for resolution Arr
methodDict = {'simple':0,'dogbox':1,'trf':2}
def symmNumFun():
    symmNum = 0
    if sigma1: symmNum+= 1
    if sigma3: symmNum+= 1      # sigma13 comes as sigma1 and sigma3
    if sigma1T: symmNum+= 1     # sigma13T comes as sigma1T and sigma3
    return symmNum

resIndMethod = methodDict[method]
resIndSymm = symmNumFun()

print();print();print();print()

# Assign default resolution only if L,M,N are not supplied to argparse
L0,M0,N0 = resolutionArr[resIndSymm, resIndMethod]
# If L is not supplied in commandline (default is None), 
#       if args.resolutionArr is False, use L of the flowfield from the .hdf5 file
#       if args.resolutionArr is True, use L of the flowfield from resolutionArr as above
# If L is supplied in commandline, use this value
# Same for M and N
changeL = True; changeM = True; changeN = True
if (args.L is None) and not args.resolutionArr: changeL = False
elif (args.L is None): L=L0
else: L = args.L
if (args.M is None) and not args.resolutionArr: changeM = False
elif (args.M is None): M=M0
else: M = args.M
if (args.N is None) and not args.resolutionArr: changeN = False
elif (args.N is None): N=N0
else: N = args.N

# Parsing amplitudes into epsArr
epsList = [0.]
if args.eps6 != 0.: epsList.extend((args.eps1, args.eps2, args.eps3, args.eps4, args.eps5, args.eps6))
elif args.eps5 != 0.: epsList.extend((args.eps1, args.eps2, args.eps3, args.eps4, args.eps5))
elif args.eps4 != 0.: epsList.extend((args.eps1, args.eps2, args.eps3, args.eps4))
elif args.eps3 != 0.: epsList.extend((args.eps1, args.eps2, args.eps3))
elif args.eps2 != 0.: epsList.extend((args.eps1, args.eps2))
elif args.eps1 != 0.: epsList.append(args.eps1)
epsArr = np.array(epsList,dtype=np.float)

# Parsing phases into phiArr
phiList = [0.]
if args.phi6 != 0.: phiList.extend((args.phi1, args.phi2, args.phi3, args.phi4, args.phi5, args.phi6))
elif args.phi5 != 0.: phiList.extend((args.phi1, args.phi2, args.phi3, args.phi4, args.phi5))
elif args.phi4 != 0.: phiList.extend((args.phi1, args.phi2, args.phi3, args.phi4))
elif args.phi3 != 0.: phiList.extend((args.phi1, args.phi2, args.phi3))
elif args.phi2 != 0.: phiList.extend((args.phi1, args.phi2))
elif args.phi1 != 0.: phiList.append(args.phi1)
phiArr = np.array(phiList,dtype=np.float)

phiArr = np.array(phiList,dtype=np.float)


print("\n\n\nStarting time:",datetime.datetime.now())
sys.stdout.flush()


# Loading whatever h5 file exists in current directory
files = os.listdir(loadPath)
fileCounter = 0
for fileName in files:
    if fileName.endswith(inFileName):
        h5file = fileName
        fileCounter += 1
    # Find file that matches with supplied --fName, even if it's just '.hdf5'

# If not, try again to find any .hdf5 file
if fileCounter == 0:
    for fileName in files:
        if fileName.endswith('.hdf5'):
            h5file = fileName
            fileCounter += 1

if fileCounter != 1:
    warn("There are multiple files (%d) ending with hdf5 in the folder. Using %s"%(fileCounter,h5file))

tolEps = 1.0e-05
x = loadh5(loadPath+ h5file)



if (x.flowDict['epsArr'].size != epsArr.size) or not (x.flowDict['epsArr'] == epsArr).all():
    print("epsArr in x.flowDict is", x.flowDict['epsArr']," while epsArr from commandline is ",epsArr, ". Changing x.flowDict with new epsArr....")
x.flowDict['epsArr'] = epsArr

if (not ('phiArr' in x.flowDict)) :
    print("phiArr is not in x.flowDict. Assigning phiArr from commandline...")
elif (x.flowDict['phiArr'].size != phiArr.size) or (not (x.flowDict['phiArr'] == phiArr).all()):
    print("phiArr in x.flowDict is", x.flowDict.get('phiArr',None)," while epsArr from commandline is ",epsArr, ". Changing x.flowDict with new phiArr....")
x.flowDict['phiArr'] = phiArr
    

assert x.nd==4, "State-vector in the file %s has to be a 4-component vector."%(loadPath+h5file)
if changeL and (x.nx//2 != L):
    x = x.slice(L=L)
    print("L in x is %d, input/default is %d. Slicing x...."%(x.nx//2,L))
if changeM and (x.nz//2 != M):
    x = x.slice(M=M)
    print("M in x is %d, input/default is %d. Slicing x...."%(x.nz//2,M))
if changeN and (x.N != N):
    x = x.slice(N=N)
    print("N in x is %d, input/default is %d. Slicing x...."%(x.N,N))

x.setWallVel()  # Ensure velocity at the walls isn't affected by N-slicing
print("Revisit flowField.imposeSymms() to include sigma1T and sigma3T")
x.imposeSymms(sigma1=sigma1,sigma3=sigma3)

# Counter used for saving intermediate solutions
if 'counter' not in x.flowDict:
    x.flowDict['counter'] = 0
if x.flowDict['counter']== 0:
    # Saving first flow field in folder tmp/
    x.saveh5(fNamePrefix=fNamePrefix+'_0_',prefix=loadPath+'tmp/')






##----------------------------------------------------------------------------------------------
# Build exactRiblet() class instance out of the state-vector and symmetries
# Iterate using exactRiblet.iterate()
##----------------------------------------------------------------------------------------------


L = x.nx//2; M = x.nz//2; N = x.N
start = time.time()
print("x.flowDict:", x.flowDict)
print("Method: %s, WithJac:%s, iterMax:%d, sigma1:%s, sigma3:%s, sigma1T:%s, sigma3T:%s"%(method,supplyJac,iterMax, sigma1,sigma3,sigma1T,sigma3T))
sys.stdout.flush()
tRun = 0.
start0 = time.time()

ffProb = rib.exactRiblet(x=x, sigma1=sigma1, sigma3=sigma3, method=method, iterMax=iterMax,
        tol=tol, log=logName, prefix=fNamePrefix, supplyJac=supplyJac, saveDir=loadPath+'tmp/')

#sys.exit()

start1 = time.time()
x, fnorm, flg = ffProb.iterate()
if flg == -2:
    print("Residual norm is smaller than the requested tolerance")
    tempFile = open(workingDir+"DELETE_RUNCASE_FILE.txt",'a')
    tempFile.write("\nFlag raised to delete case file at time:",datetime.datetime.now())
    tempFile.close()

tRun = time.time() - start1; start1 = time.time()
x.saveh5(fNamePrefix=fNamePrefix,prefix=loadPath)

print("Total run time (minutes):%d" %((time.time()-start0)/60.))

if fnorm[-1] <= tol:
    print("Iterations have converged")
    sys.stdout.flush()
else:
    print("Iterations have not converged")

sys.stdout.flush()


print("Maximum memory usage (MB):%d"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
print("+++++++++++++++++++++++++++++++++++++")
print("+++++++++++++++++++++++++++++++++++++")
print("\n \n \n \n \n")
sys.stdout.flush()

	
