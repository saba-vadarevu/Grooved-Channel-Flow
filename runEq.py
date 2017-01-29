import os
import argparse
import resource 
import sys
import time
import datetime
from warnings import warn

sys.path.append('../')
import exactRiblet as rib
import numpy as np
from flowFieldWavy import *

workingDir = os.getcwd() 
if not workingDir.endswith('/'):
    workingDir = workingDir + '/'
loadPath = workingDir
savePath = workingDir

#Creating an argument parser
parser = argparse.ArgumentParser(description='Input grid parameters L,M,N\n Riblet amplitude 2*eps is 0.05\n Use keyword --eps to define semi-amplitude')

parser.add_argument("L", help="No. of positive streamwise modes",type=int)
parser.add_argument("M", help="No. of positive spanwise modes",type=int)
parser.add_argument("N", help="No. of wall-normal nodes",type=int)
# I set L,M,N,eps as required variables because I want to ensure that the folder names and geometries always match
# I will compare these arguments later with the hdf5 file that is loaded

parser.add_argument("--eps1",help="Semi-amplitude, Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--eps2",help="Semi-amplitude, Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--eps3",help="Semi-amplitude, Optional, default: 0.0", default = 0.0,type=float)
parser.add_argument("--tol",help="Tolerance for N-R method, Optional, default: 1.0e-13", default = 1.0e-13,type=float)
parser.add_argument("--iterMax",help="Total number of iterations, Optional, default: 6", default = 6,type=int)
parser.add_argument("--log",help="Name of log file, Optional, default: outFile.txt (append)", default = 'outFile.txt',type=str)
parser.add_argument("--prefix",help="fNamePrefix for solution files, Optional, default: ribEq1", default = 'ribEq1',type=str)
parser.add_argument("--method",help="Method to use for solving equations,\
        Options: 'simple' (Newton's+jacobian inversion+line search)\
                'trf', 'dogbox', 'lm' from scipy's least_squares", default='simple',type=str)
parser.add_argument("--fName",help="Input file name. If not supplied or invalid, use whatever .hdf5 is found",default='.hdf5',type=str)

symParser1 = parser.add_mutually_exclusive_group(required=False)
symParser2 = parser.add_mutually_exclusive_group(required=False)
symParser3 = parser.add_mutually_exclusive_group(required=False)
symParser1.add_argument("--no-sigma1",help="Do not impose sigma1. Imposed otherwise", dest='sigma1',action="store_false")
symParser1.add_argument("--sigma1",help="Impose sigma1 (also the default)", dest='sigma1',action="store_true")
symParser2.add_argument("--no-sigma3",help="Do not impose sigma3 (also the default)", dest='sigma3',action="store_false")
symParser2.add_argument("--sigma3",help="Impose sigma3. Not imposed otherwise", dest='sigma3',action="store_true")
symParser3.add_argument("--jacobian",help="Supply modified Jacobian if using trf,dogbox, or lm methods. Don't otherwise",dest='jacobian',action='store_true')
symParser3.add_argument("--no-jacobian",help="Do not supply modified Jacobian if using trf,dogbox, or lm methods (default)",dest='jacobian',action='store_false')
parser.set_defaults(sigma1=True)
parser.set_defaults(sigma2=False)
parser.set_defaults(jacobian=False)


args = parser.parse_args()

L = args.L
M = args.M
N = args.N
iterMax = args.iterMax
logName = args.log
tol = args.tol
fNamePrefix = args.prefix
sigma1 = args.sigma1
sigma3 = args.sigma3
method = args.method
supplyJac = args.jacobian
inFileName = args.fName

sigma2 = sigma1 and sigma3
epsArr = np.array([0., args.eps1])
if args.eps3 != 0.:
    epsArr = np.append(epsArr, [args.eps2, args.eps3]).flatten()
elif args.eps2 != 0.:
    epsArr = np.append(epsArr, args.eps2).flatten()
   
print();print();print();print()
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
for q in range(epsArr.size):
    assert np.abs(epsArr[q] - x.flowDict['epsArr'][q])<= tolEps , "eps_%d in argument does not match the same in x.flowDict, %.3g, %.3g"%(q,epsArr[q],x.flowDict['epsArr'][q])
assert x.nx//2 == L, "L in x is %d, while the argument to script is %d"%(x.nx//2,L)
assert x.nz//2 == M, "M in x is %d, while the argument to script is %d"%(x.nz//2,M)
assert x.N == N, "N in x is %d, while the argument to script is %d"%(x.N,N)


if 'counter' not in x.flowDict:
    x.flowDict['counter'] = 0
if x.flowDict['counter']== 0:
    # Saving first flow field in folder tmp/
    x.saveh5(fNamePrefix=fNamePrefix+'_0_',prefix=loadPath+'tmp/')

start = time.time()
print("Running with N=%d, L=%d, M=%d" %(N,L,M))
print("epsArr is ",epsArr, " and Re is",x.flowDict['Re'])
print("Method: %s, WithJac:%s, iterMax:%d, sigma1:%s, sigma2:%s"%(method,supplyJac,iterMax, sigma1,sigma2))
sys.stdout.flush()
tRun = 0.
start0 = time.time()

x.imposeSymms(sigma1=sigma1, sigma3=sigma3)
x.setWallVel()
ffProb = rib.exactRiblet(x=x, sigma1=sigma1, sigma3=sigma3, method=method, iterMax=iterMax,
        tol=tol, log=logName, prefix=fNamePrefix, supplyJac=supplyJac, saveDir=loadPath+'tmp/')

start1 = time.time()
x, fnorm, flg = ffProb.iterate()
if flg == -2:
    print("Residual norm is smaller than the requested tolerance")
    tempFile = open(workingDir+"DELETE_RUNCASE_FILE.txt",'a')
    tempFile.write("\nFlag raised to delete case file at time:",datetime.datetime.now())
    tempFile.close()

tRun = time.time() - start1; start1 = time.time()
x.saveh5(fNamePrefix=fNamePrefix,prefix=loadPath)

if fnorm <= tol:
    print("Total run time until convergence (minutes):", round((time.time()-start0)/60.,2))
    sys.stdout.flush()
else:
    print("Iterations have not converged")

sys.stdout.flush()


print("Maximum memory usage (MB):%d"%(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
print("+++++++++++++++++++++++++++++++++++++")
print("+++++++++++++++++++++++++++++++++++++")
print("\n \n \n \n \n")
sys.stdout.flush()

	
