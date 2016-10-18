import numpy as np
from flowFieldWavy import *
import laminar as lam
import scipy.integrate as spint

tol = 1.0e-13
linTol = 1.0e-10

def ff2arr(ff):
    return ff.flatten()

def arr2ff(arr=None,flowDict=None):
    nx = 2*flowDict['L']+1; nz = 2*flowDict['M']+1; N = flowDict['N']
    nd = arr.size//(nx*nz*N)
    tempDict = updateDict(flowDict,{'nd':nd})
    ff = flowFieldRiblet(flowDict=tempDict,arr=arr)
    return ff

def setSymms(vfTemp):
    """Set velocities at walls to zero, and ensure that the field is real valued"""
    vf = vfTemp.copy()
    assert vf.nd == 3
    vf.view4d()[:,:,:,:,[0,-1]] = 0.
    if vf.flowDict['isPois']== 0:
        vf.view4d()[0,vf.nx//2, vf.nz//2, 0,[0,-1]] = np.array([1.,-1.])
    
    vf[0,:,:vf.nz//2] = 0.5*(vf[0,:,:vf.nz//2]+ vf[0,:,:vf.nz//2:-1].conj())
    vf[0,:,:vf.nz//2:-1] = vf[0,:,:vf.nz//2].conj()
    return vf

def dict2ff(flowDict):
    """ Returns a velocity flowField with linear/quadratic profile depending on 'isPois'"""
    vf = flowFieldRiblet(flowDict=updateDict(flowDict,{'nd':3})).view4d()
    if flowDict['isPois']==0:
        uProfile = vf.y
    else:
        uProfile = 1. - vf.y**2
    vf[0,vf.nx//2,vf.nz//2,0] = uProfile
    return vf


def linr(flowDict): 
    """Returns matrix representing the linear operator for the equilibria/TWS for riblet case
    Linear operator for exact solutions isn't very different from that for laminar, 
        all inter-modal interaction remains the same for a fixed 'l'. 
    So, we use the function already written in the laminar.py module and modify it to suit
        the current case"""
    assert flowDict['L'] != 0 and flowDict['M'] != 0
    if flowDict['L'] > 4:
        print('L is set to 4 from ', flowDict['L'] )
        flowDict.update({'L':4})
    if flowDict['M'] > 8:
        print('M is set to 8 from ', flowDict['M'] )
        flowDict.update({'M':8})
    Lmat_lam = lam.linr(updateDict(flowDict,{'L':0,'alpha':0.}))

    L = flowDict['L']; M = flowDict['M']
    nx = int(2*L+1)
    nz = int(2*M+1)
    N  = int(flowDict['N']); N4 = 4*N
    s1 = nz*N4
    a = flowDict['alpha']; a2 = a**2
    Re = flowDict['Re']
    I = np.identity(N)
    assert Lmat_lam.shape[0] == nz*N4
    
    # L_lam is built for the case of L= 0. For exact solutions, we have L!= 0
    #   So, we take L_lam, and add i.l.alpha or -l**2.a**2 as appropriate
    Lmat = np.zeros((nx*nz*N4,nx*nz*N4),dtype=np.complex)

    mat1 = np.zeros((nz*N4,nz*N4),dtype=np.complex); mat2 = mat1.copy()
    # Define mat1 and mat2 such that all 'l' terms in the linear matrix can be 
    #   written as l * mat1  + l^2 *mat2
    for mp in range(nz):
        # Row numbers correspond to equation, column numbers to field variable
        #   mp*N4 +     (0:N)   : x-momentum or u
        #               (N:2N)  : y-momentum or v
        #               (2N:3N) : z-momentum or w
        #               (3N:4N) : continuity or p

        # x-momentum
        # (-1/Re)* (d_xx u)_lm = l**2 * a2/Re * I * u_lm
        mat2[mp*N4:mp*N4+N, mp*N4:mp*N4+N]      = a2/Re*I           
        # (d_x p)_lm = l * i*a*I * p_lm
        mat1[mp*N4:mp*N4+N, mp*N4+3*N:mp*N4+N4] = 1.j*a*I

        # y-momentum
        # (-1/Re)* (d_xx v)_lm = l**2 * a2/Re * I * v_lm
        mat2[mp*N4+N:mp*N4+2*N, mp*N4+N:mp*N4+2*N]      = a2/Re*I           

        # z-momentum
        # (-1/Re)* (d_xx u)_lm = l**2 * a2/Re * I * u_lm
        mat2[mp*N4+2*N:mp*N4+3*N, mp*N4+2*N:mp*N4+3*N]  = a2/Re*I           

        # continuity
        mat1[mp*N4+3*N:mp*N4+4*N, mp*N4 : mp*N4+N]      = 1.j*a*I


    for lp in range(nx):
        l = lp-L
        
        # Matrix from laminar case where l=0
        Lmat[lp*s1:(lp+1)*s1, lp*s1:(lp+1)*s1] = Lmat_lam
        # Adding all the l-terms
        Lmat[lp*s1:(lp+1)*s1, lp*s1:(lp+1)*s1] += l* mat1 + l**2 * mat2


    return Lmat

def jcbn(vf):
    eps = vf.flowDict['eps']; a = vf.flowDict['alpha']; b = vf.flowDict['beta']
    N = vf.N; L = vf.nx//2; M = vf.nz//2
    vfArr = vf.view4d().copyArray()
    u = vfArr[0,:,:,0]; v = vfArr[0,:,:,1]; w = vfArr[0,:,:,2]
    vfyArr = vf.ddy().view4d().copyArray()
    uy = vfyArr[0,:,:,0]; vy = vfyArr[0,:,:,1]; wy = vfyArr[0,:,:,2]

    D = vf.D; I = np.identity(vf.N, dtype=np.complex); Z = np.zeros((vf.N,vf.N),dtype=np.complex)

    # Index of the first row/column of the block for any wavenumber vector (l,m)
    iFun = lambda l,m: (l+L)*(vf.nz*4*N) + (m+M)*4*N
    
    G = np.zeros((vf.nx*vf.nz*4*N, vf.nx*vf.nz*4*N),dtype=np.complex)
    # I will be using the functions np.diag() and np.dot() quite often, so,
    diag = np.diag; dot = np.dot
    # And these scalars: 
    ieb = 1.j*eps*b; ia = 1.j*a; ib = 1.j*b

    for l in range(-L,L+1):
        for m in range(-M,M+1):
            # Index of first row in the block for equations for wavenumbers (l,m)
            rInd = iFun(l,m)
            # l1,m2 are the wavenumbers in phi^j_{lm}
            for lp in range(-L,L+1):
                for mp in range(-M,M+1):
                    cInd = iFun(lp,mp)
                    # lp,mp are the wavenumbers for the velocities contributing to phi^j_{lm}, 
                    #   The non-linear Jacobian contains modes with wavenumbers (l-lp,m-mp), and a few (l-lp,m-mp-1) and (l-lp,m-mp+1)
                    if (-L <= (l-lp) <= L):
                        li = l-lp+L # Array index for streamwise wavenumber l-lp
                        if (-M <= (m-mp) <= M):
                            mi = m-mp+M # Array index for spanwise wavenumber m-mp
                            # phi^1_{l,m}: factors of terms with  u_{lp,mp}:
                            G[ rInd+0*N : rInd+1*N , cInd+0*N : cInd+1*N ] += \
                                    l*ia* diag(u[li, mi])  + v[li,mi].reshape((N,1)) *D + diag(w[li,mi])*mp*ib 
                            # phi^1_{l,m}: factors of terms with  v_{lp,mp}:
                            G[ rInd+0*N : rInd+1*N , cInd+1*N : cInd+2*N ] += diag(uy[li, mi])
                            # phi^1_{l,m}: factors of terms with  w_{lp,mp}:
                            G[ rInd+0*N : rInd+1*N , cInd+2*N : cInd+3*N ] += (m-mp)*ib*diag(u[li, mi])

                            # phi^2_{l,m}: factors of terms with  v_{lp,mp}:
                            G[ rInd+1*N : rInd+2*N , cInd+1*N : cInd+2*N ] += \
                                    lp*ia* diag(u[li, mi])  + v[li,mi].reshape((N,1)) *D + diag(w[li,mi])*mp*ib \
                                    + diag(vy[li,mi])
                            # phi^2_{l,m}: factors of terms with  u_{lp,mp}:
                            G[ rInd+1*N : rInd+2*N , cInd+0*N : cInd+1*N ] += (l-lp)*ia*diag(v[li, mi])
                            # phi^2_{l,m}: factors of terms with  w_{lp,mp}:
                            G[ rInd+1*N : rInd+2*N , cInd+2*N : cInd+3*N ] += (m-mp)*ib*diag(v[li, mi])

                            # phi^3_{l,m}: factors of terms with  w_{lp,mp}:
                            G[ rInd+2*N : rInd+3*N , cInd+2*N : cInd+3*N ] += \
                                    lp*ia* diag(u[li, mi])  + v[li,mi].reshape((N,1)) *D + m*ib*diag(w[li,mi]) 
                            # phi^3_{l,m}: factors of terms with  v_{lp,mp}:
                            G[ rInd+2*N : rInd+3*N , cInd+1*N : cInd+2*N ] += diag(wy[li, mi])
                            # phi^3_{l,m}: factors of terms with  u_{lp,mp}:
                            G[ rInd+2*N : rInd+3*N , cInd+0*N : cInd+1*N ] += (l-lp)*ia*diag(w[li, mi])

                        if (-M <= (m-mp-1) <= M):
                            mi = m-mp-1+M # Array index for spanwise wavenumber m-mp
                            # phi^1_{l,m}: factors of terms with u_{lp,mp}
                            G[ rInd+0*N : rInd+1*N , cInd+0*N : cInd+1*N ] += -ieb* w[li,mi].reshape((N,1)) * D 
                            # phi^1_{l,m}: factors of terms with w_{lp,mp}
                            G[ rInd+0*N : rInd+1*N , cInd+2*N : cInd+3*N ] += -ieb* diag(uy[li,mi]) 

                            # phi^2_{l,m}: factors of terms with v_{lp,mp}
                            G[ rInd+1*N : rInd+2*N , cInd+1*N : cInd+2*N ] += -ieb* w[li,mi].reshape((N,1)) * D 
                            # phi^2_{l,m}: factors of terms with w_{lp,mp}
                            G[ rInd+1*N : rInd+2*N , cInd+2*N : cInd+3*N ] += -ieb* diag(vy[li,mi]) 

                            # phi^3_{l,m}: factors of terms with w_{lp,mp}
                            G[ rInd+2*N : rInd+3*N , cInd+2*N : cInd+3*N ] += -ieb* w[li,mi].reshape((N,1)) * D \
                                    -ieb* diag(wy[li,mi]) 
 
                        if (-M <= (m-mp+1) <= M):
                            mi = m-mp+1+M # Array index for spanwise wavenumber m-mp
                            # phi^1_{l,m}: factors of terms with u_{lp,mp}
                            G[ rInd+0*N : rInd+1*N , cInd+0*N : cInd+1*N ] += ieb* w[li,mi].reshape((N,1)) * D 
                            # phi^1_{l,m}: factors of terms with w_{lp,mp}
                            G[ rInd+0*N : rInd+1*N , cInd+2*N : cInd+3*N ] += ieb* diag(uy[li,mi]) 

                            # phi^2_{l,m}: factors of terms with v_{lp,mp}
                            G[ rInd+1*N : rInd+2*N , cInd+1*N : cInd+2*N ] += ieb* w[li,mi].reshape((N,1)) * D 
                            # phi^2_{l,m}: factors of terms with w_{lp,mp}
                            G[ rInd+1*N : rInd+2*N , cInd+2*N : cInd+3*N ] += ieb* diag(vy[li,mi]) 

                            # phi^3_{l,m}: factors of terms with w_{lp,mp}
                            G[ rInd+2*N : rInd+3*N , cInd+2*N : cInd+3*N ] += ieb* w[li,mi].reshape((N,1)) * D \
                                    +ieb* diag(wy[li,mi]) 
                                    


    return G 



def residual(vf=None, pf=None, Lmat=None, G=None):
    """Compute residual: res = (Lmat+0.5*G)x - f"""
    flowDict = vf.flowDict
    N = vf.N 
    if Lmat is None: Lmat = linr(flowDict)
    if G is None: G = jcbn(vf)
    if pf is None: 
        warn('pressure field not supplied, assigning zero pressure for computing residual')                
        x = vf.appendField(vf.getScalar().zero())
    else: x = vf.appendField(pf)
    xArr = x.flatten()

    BCrows = 4*N*np.arange(vf.nx*vf.nz).reshape((vf.nx*vf.nz, 1)) + np.array([0,N-1,N,2*N-1,2*N,3*N-1]).reshape((1,6))
    BCrows = BCrows.flatten()
    G[BCrows,:] = 0.
    Lmat[BCrows,:] =0.
    Lmat[BCrows,BCrows] = 1.
    
    F = np.dot(Lmat+0.5*G, xArr)
    # If flow is not Couette, then mean pressure gradient needs to be added
    F = F.reshape((vf.nx,vf.nz,4,N))
    if flowDict['isPois'] == 1:
        F[vf.nx//2,vf.nz//2,0,1:-1] += -2./flowDict['Re']
    elif flowDict['isPois'] == 0:
        F[vf.nx//2, vf.nz//2,0, [0,-1]] -= np.array([1.,-1.])
        
    F = F.flatten()
    
    return F 


def makeSystem(flowDict=None,vf=None, pf=None, F=None, J=None, Lmat=None, G=None, resNorm=False):
    """
    Create functions for residual and Jacobian matrices, Boundary conditions and symmetries are imposed here. 
    The output functions
    Inputs:
        vf : velocity flowField (pressure field isn't needed)
        flowDict: Needed if vf is not supplied, assigns linear/quadratic velocity field for Couette/Poiseuille flow
        F: (residual) 1-d np.ndarray
        J: (Jacobian) 2-d np.ndarray
        Lmat: 2-d np.ndarray for the linear terms (including divergence) without BCs included
    
    If residual and jacobian are supplied, the BCs and symmetries are added straight away.
    If these are not supplied, they are computed before BCs are added
        If Lmat is supplied, it is used, otherwise, it is computed.
        
    Outputs:
        residualBC: 1-d array
        jacobianBC: 2-d array"""
    # Creating residual array and jacobian matrix if they are not supplied
    
    if vf is None: 
        vf = dict2ff(flowDict)
    else:
        flowDict = vf.flowDict

    N = vf.N; N4 = 4*N
    L = vf.nx//2; M = vf.nz//2
    nx = vf.nx; nz=vf.nz
        
    if (F is None) or (J is None):
        if Lmat is None: Lmat = linr(flowDict)
        if G is None: 
            G = jcbn(vf)
            
        J = Lmat+ G
        if F is None:
            F = residual(vf=vf, pf=pf, Lmat=Lmat, G=G)
    
    # Some simple checks
    assert (F.ndim == 1) and (J.ndim == 2)
    assert (J.shape[0] == vf.nx*vf.nz*N4) and (F.size == vf.nx*vf.nz*N4)
    
    
    # Imposing boundary conditions
    #   Unlike my MATLAB code, I impose all BCs at the end of the jacobian matrix
    #   So, if rect is True, the top rows of jacobian and residual remain unchanged
    #       they just have extra rows at the end
    #   If rect is False, I still add rows for BCs at the end, but I also remove the rows
    #       of the governing equations at the walls
    # Removing wall-equations if rect is False:
    # I don't have to remove wall-equations for divergence
    BCrows = N4*np.arange(vf.nx*vf.nz).reshape((vf.nx*vf.nz,1)) + np.array([0,N-1,N,2*N-1,2*N,3*N-1]).reshape((1,6))
    BCrows = BCrows.flatten()
    
    residualBC = np.delete(F, BCrows)
    jacobianBC = np.delete(J, BCrows, axis=0)

        
    # Now, just append the rows for BCs to jacobian, and zeros to residual
    # Number of rows to add is nx*nz*6
    residualBC = np.append(residualBC, np.zeros(6*nx*nz, dtype=np.complex))
    jacobianBC = np.append(jacobianBC, np.zeros((6*nx*nz, jacobianBC.shape[1]), dtype=np.complex), axis=0)
    # The appendage to jacobianBC doesn't actually give an equation since all coefficients are zero
    #        Replacing some zeros with ones to get Dirichlet BCs:
    #        The column numbers where the zeros must be replaced are the same as the rows we replaced
    #            for rect=False
    jacobianBC[range(-6*nx*nz,0),BCrows] = 1.
    
    # Additionally, I'll also add a 0 BC on the zeroth pressure mode, but only if rect is True
    
    if not resNorm:
        return jacobianBC, residualBC
    else:
        F = F.reshape((nx,nz,4,N))
        F[:,:,:3,[0,-1]] = 0.
        fnorm = chebnorm(F.flatten(),N)

        return jacobianBC, residualBC, fnorm

def iterate(flowDict=None,vf=None, pf=None,iterMax= 6, tol=5.0e-14,rcond=1.0e-14):
    if vf is None: 
        vf = dict2ff(flowDict)
    if pf is None: pf = flowFieldWavy(flowDict=updateDict(vf.flowDict,{'nd':1}))
    Lmat = linr(vf.flowDict)
    fnormArr=[]
    flg = 0
    print('Starting iterations...............')
    for n in range(iterMax):
        print('iter:',n)
        vf0 = vf.copy() # Just in case the iterations fail
        pf0 = pf.copy() 
        J, F, fnorm = makeSystem(vf=vf, pf=pf, Lmat=Lmat, resNorm=True)
        print('fnorm:',fnorm)
        fnormArr.append(fnorm)
        if fnorm <= tol:
            flg = 0
            print('Converged in ',n,' iterations. Returning....................................')
            return vf, pf, np.array(fnormArr), flg
        
        if n>0:
            if fnormArr[n] > fnormArr[n-1]:
                flg = 1
                print('Residual norm is increasing:',fnormArr)
                #print('Returning with initial velocity and pressure fields')
                #return vf0, pf0, np.array(fnormArr),flg
                
        
        dx, linNorm, jRank,sVals = np.linalg.lstsq(J,-F,rcond=rcond)
        linNorm = np.linalg.norm(np.dot(J,dx) + F)
        print('Number of variables, Rank of JacobianBC matrix:', J.shape[1], jRank)
        print('len(sVals:',sVals.size, ' Last 3 singular values:',sVals[-3:])
        print("Inversion success with residual norm ", linNorm)
        if linNorm > linTol:
            print('Least squares problem returned residual norm:',linNorm,' which is greater than tolerance:',linTol)
            
        
        x = vf.appendField(pf)
        x.view1d()[:] += dx
        vf = x.slice(nd=[0,1,2])
        pf = x.getScalar(nd=3)
        
        print('*********************************************************')
    else:
        if fnormArr[-1] > 100.*tol:
            print('Iterations have not converged for iterMax=',iterMax)
            print('fnormArr is ',fnormArr)
            flg = -1
    return vf, pf, np.array(fnormArr), flg

def shearStress(vf):
    """ Returns averaged shear stress at the bottom wall
    NOTE: For Couette flow, stress at the walls is opposite in sign, while
        they have the same sign for Poiseuille flow. So the stress at only one wall
        is considered here
    Inputs: 
        vf: velocity flowFieldRiblet (can include pressure)
    Returns:
        avgShearStress: scalar
    """
    # Building a function that can be handed to scipy.integrate
    u = vf.slice(L=0).getScalar()
    uy = u.ddy(); uz = u.ddz()
    assert u.nx == 1
    eps = vf.flowDict['eps']; a = vf.flowDict['alpha']; b = vf.flowDict['beta']
    
    # I don't need the following code, but I'll keep it just in case I need it later 
    if True:
        avgStrainRate = (1.+2.*eps**2 * b**2)*np.real(uy[0,uy.nx//2, uy.nz//2, 0, -1]) \
                - eps * b**2 * np.real(u[0, u.nx//2, u.nz//2 + 1, 0, -1])\
                - eps**2 * b**2 * np.real(uy[0, uy.nx//2, uy.nz//2+2, 0, -1])
        avgStrainRate = np.real(avgStrainRate)
    else:

        # Arrays for storing the values of Fourier mdoes at the walls
        uzWall = np.zeros(vf.nz,dtype=np.complex)
        uyWall = uzWall.copy()

        uzWall[:] = uz[0,0,:, 0,-1]  
        uyWall[:] = uy[0,0,:, 0,-1]

        mArr = np.arange(-(vf.nz//2), vf.nz//2+1)
        def _ifftWall(zLoc,ffWall):
            """ Returns value of field 'ff' at the wall when Fourier coefficients
                at the wall 'ffWall' are supplied. 
            Inputs:
                zLoc: locations in zLoc, can be array
                ffWall: coefficients at the wall
            Returns:
                ffArr, array if zLoc is input as array
                """
            zLoc = np.array([zLoc]).flatten()
            zLoc = zLoc.reshape((zLoc.size,1))

            ffWall = ffWall.reshape((1,u.nz))
            ffVals = np.real(np.sum(ffWall*np.exp(1.j*mArr*b*zLoc),axis=1))
            return ffVals.flatten()

        def _strainRateFun(zLoc,uyw,uzw):
            """ Local strain rate, du_t/dn"""
            zLoc = np.array([zLoc]).flatten()
            sr = np.zeros(zLoc.shape) 

            # Local strain-rate du/dn, n:normal coordinate
            # du/dn = e_n.e_y * u_y + e_n.e_z * u_z 
            # e_n.e_y = 1/sqrt{1+ (2*eps*b*sin(b*z))^2 }        
            # e_n.e_z = 2*eps*b*sin(bz)/sqrt{1+ (2*eps*b*sin(b*z))^2 }        
            # There is also a dt in the integral (arc-length along the surface)
            #   and dt = sqrt{ 1 + (2*eps*b*sin(b*z))^2 }, which cancels out the 
            #   sqrt in the denominator of the dot products
            
            # e_n.e_y * u_y (without the sqrt denominator)
            sr += _ifftWall(zLoc,uyw)
            # e_n.e_z * u_z (without the sqrt denominator)
            sr += 2.*eps*b*np.sin(b*zLoc) * _ifftWall(zLoc,uzw)

            return sr

        strainRate = lambda zLoc: _strainRateFun(zLoc,uyWall, uzWall)

        Lz = 2.*np.pi/b

        # Integrating du/dn * dt from z=0 to 2*pi/b:
        integratedStrainRate = spint.quad(strainRate, 0., Lz, epsabs = 1.0e-08, epsrel=1.0e-05)[0]
        # e_t.e_x * ds = dx, so don't have to worry about that bit. 
        
        #  Dividing by streamwise wavelength to get the force per unit area (spanwise homogeneous)
        avgStrainRate = integratedStrainRate/Lz
    
    avgShearStress = avgStrainRate/vf.flowDict['Re']    # Follows from non-dimensionalization

    return avgShearStress





def linrInv(flowDict):
    L = flowDict['L']; M = flowDict['M']; N = flowDict['N']
    nx = 2*L+1; nz = 2*M+1; N4 = 4*N
    a = flowDict['alpha']; a2 = a**2; Re = flowDict['Re']
    
    Lmat_lam = lam.linr(updateDict(flowDict,{'L':0,'alpha':0.}))
    
    LmatInv = np.zeros((nx,nz*N4,nz*N4), dtype=np.complex)
    I = np.identity(N)
    assert Lmat_lam.shape[0] == nz*N4
    
    # L_lam is built for the case of L= 0. For exact solutions, we have L!= 0
    #   So, we take L_lam, and add i.l.alpha or -l**2.a**2 as appropriate
    Lmat = np.zeros((nx*nz*N4,nx*nz*N4),dtype=np.complex)

    mat1 = np.zeros((nz*N4,nz*N4),dtype=np.complex); mat2 = mat1.copy()
    # Define mat1 and mat2 such that all 'l' terms in the linear matrix can be 
    #   written as l * mat1  + l^2 *mat2
    for mp in range(nz):
        # Row numbers correspond to equation, column numbers to field variable
        #   mp*N4 +     (0:N)   : x-momentum or u
        #               (N:2N)  : y-momentum or v
        #               (2N:3N) : z-momentum or w
        #               (3N:4N) : continuity or p

        # x-momentum
        # (-1/Re)* (d_xx u)_lm = l**2 * a2/Re * I * u_lm
        mat2[mp*N4:mp*N4+N, mp*N4:mp*N4+N]      = a2/Re*I           
        # (d_x p)_lm = l * i*a*I * p_lm
        mat1[mp*N4:mp*N4+N, mp*N4+3*N:mp*N4+N4] = 1.j*a*I

        # y-momentum
        # (-1/Re)* (d_xx v)_lm = l**2 * a2/Re * I * v_lm
        mat2[mp*N4+N:mp*N4+2*N, mp*N4+N:mp*N4+2*N]      = a2/Re*I           

        # z-momentum
        # (-1/Re)* (d_xx u)_lm = l**2 * a2/Re * I * u_lm
        mat2[mp*N4+2*N:mp*N4+3*N, mp*N4+2*N:mp*N4+3*N]  = a2/Re*I           

        # continuity
        mat1[mp*N4+3*N:mp*N4+4*N, mp*N4 : mp*N4+N]      = 1.j*a*I

    # Boundary conditions on the linear matrix:
    #   Impose them as usual- replace some rows with zeros, and set diagonal 
    #       elements to 1
    BCrows = N4* np.arange(nz).reshape((nz,1)) + np.array([0,N-1,N,2*N-1,2*N,3*N-1]).reshape((1,6))
    BCrows = BCrows.flatten()

    for lp in range(nx):
        l = lp-L 
        # Matrix from laminar case where l=0
        Ltemp = Lmat_lam.copy()
        # Adding all the l-terms
        Ltemp += l* mat1 + l**2 * mat2

        # Imposing BCs
        Ltemp[BCrows,:] = 0.
        Ltemp[BCrows,BCrows] = 1.

        LmatInv[lp] = np.linalg.pinv(Ltemp)

    return LmatInv
    


def testExactRibletModule(L=3,M=5,N=20,eps=0.025):
    vf = h52ff('eq1.h5')
    pf = h52ff('eq1_pressure.h5',pres=True)
    vf = vf.slice(L=L,M=M,N=N); pf = pf.slice(L=L,M=M,N=N)
    Lmat = linr(vf.flowDict)
    G = jcbn(vf)
    x = vf.appendField(pf)
    linTerm = np.dot(Lmat, x.flatten())
    
    linTermClass = -1./vf.flowDict['Re']*vf.laplacian() + pf.grad()
    linTermClass = linTermClass.appendField(vf.div())

    NLterm = 0.5*np.dot(G, x.flatten())

    NLtermClassFine = vf.slice(L=vf.nx//2+3,M=vf.nz//2+3).convNL()
    NLtermClass = NLtermClassFine.slice(L=vf.nx//2, M=vf.nz//2).appendField(pf.zero())

    linTestResult =  np.linalg.norm(linTerm - linTermClass.flatten()) <= tol
    NLtestResult = np.linalg.norm(NLterm - NLtermClass.flatten()) <= tol
    assert linTestResult and NLtestResult
    return linTestResult, NLtestResult






