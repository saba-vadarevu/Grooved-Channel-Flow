import sys
#sys.path.append('library')
from flowFieldWavy import *
import laminar as lam

Re = 1000.
savePath = 'solutions/Re1000UltraHighRes/'
logFile = 'ultraResRe1000.txt'
orig_stdout = sys.stdout
sys.stdout = open(logFile,'a')

iterMax = 8
epsArr = 0.5*np.arange(0.02,0.21,0.02)
gArr = 0.5*np.arange(0.2,2.1,0.2)

flowDict = {'isPois':1, 'Re':Re, 'M':0,'N':70,'L':20,'nd':4, 'eps':0.01,'epsArr':np.array([0.,0.01]), 'K':0,'alpha':10., 'beta':0., 'omega':0.,'noise':0., 'lOffset':0, 'mOffset':0}
x0 = flowFieldWavy(flowDict=flowDict )
x0[0,x0.nx//2, x0.nz//2,0] = 1. - x0.y**2
vf0  = x0.slice(nd=[0,1,2])
pf0 = x0.getScalar(nd=3)

Narr = 80 - (np.arange(0,19,2)).reshape((10,1)) + (np.arange(0,19,2)).reshape((1,10))
Larr = 10 + (np.arange(0,10)).reshape((10,1)) + (np.arange(0,10)).reshape((1,10))

x = x0.copy()
x0e = x.copy()
for neps in range(len(epsArr)):
    eps = epsArr[neps]
    x = x0e.copy()
    for ng in range(len(gArr)):
        g = gArr[ng]
        a = g/eps
        N = Narr[neps,ng]
        L = Larr[neps,ng]
        print('starting iterations for A=%.3g, S=%.3g'%(2.*eps, 2.*g))
        x = x.slice(L=L, N=N)
        x.flowDict.update({'eps':eps, 'epsArr':np.array([0., eps]), 'alpha':a})
        
        vf = x.slice(nd=[0,1,2])
        pf = x.getScalar(nd=3)

        vf,pf, fnorm, flg = lam.iterate(vf=vf, pf=pf, iterMax=iterMax, tol=1.0e-13)
        sys.stdout.flush()
        x = vf.appendField(pf)
        if x.slice(L=2*L, N=2*N).residuals().norm() > 1.0e-04:
            x = x.slice(L=L+5,N=N+15)
            vf = x.slice(nd=[0,1,2])
            pf = x.getScalar(nd=3)

            vf,pf, fnorm, flg = lam.iterate(vf=vf, pf=pf, iterMax=5, tol=1.0e-13)
            sys.stdout.flush()
            x = vf.appendField(pf)
        if ng == 0:
            x0e = x.copy()
        x.flowDict['class'] = 'flowFieldWavy'
        x.saveh5(prefix=savePath, fNamePrefix='lamRe%03dA%02dS%02d'%(Re,neps, ng) )
        sys.stdout.flush()


sys.stdout = orig_stdout



