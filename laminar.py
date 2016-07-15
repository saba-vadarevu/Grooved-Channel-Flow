import numpy as np
from pseudo import *
from warnings import warn
from flowFieldWavy import *
import scipy.integrate as spint
from rootFinder import *

pres0BC = False 
linTol = 1.0e-12
tol0 = 1.0e-14  # Tolerance to use for comparing numbers to zero

def updateDict(dict1,dict2):
    tempDict = dict1.copy()
    tempDict.update(dict2)
    return tempDict

def ff2arr(ff):
    """
    Convert flowField object into a 1d numpy array
    For streamwise or spanwise wavy flowfields, this is just vf.view1d()
    For oblique, I need to collect only modes along k*(l,m)"""
    if ff.flowDict['L']*ff.flowDict['M']==0:
        return ff.view1d().copyArray()
    assert ff.nx == ff.nz
    ffArr = np.zeros((ff.nx,ff.nd,ff.N), dtype=np.complex)
    for k in range(ff.nx):
        ffArr[k] = ff.view4d()[0,k,k]
    
    return ffArr.flatten()


def arr2ff(arr=None,flowDict=None):
    if (flowDict['L'] * flowDict['M']) == 0:
        return flowFieldWavy(flowDict=flowDict, arr=arr)
    ff = flowFieldWavy(flowDict=flowDict)
    assert ff.nx == ff.nz
    arr = arr.reshape((ff.nx,ff.nd,ff.N))
    for k in range(ff.nx):
        ff[0,k,k] = arr[k]
    return ff

def setSymms(vf):
    """Set velocities at walls to zero, and ensure that the field is real valued"""
    vf.view4d()[:,:,:,:,[0,-1]] = 0.
    if vf.flowDict['isPois']== 0:
        vf.view4d()[0,vf.nx//2, vf.nz//2, 0,[0,-1]] = np.array([1.,-1.])
    
    arr = ff2arr(vf)
    m = max(vf.nx,vf.nz)
    arr = arr.reshape((m, vf.nd, vf.N))
    arr[:m//2] = 0.5*(arr[:m//2]+ arr[:m//2:-1].conj())
    arr[:m//2:-1] = arr[:m//2].conj()
    return arr2ff(arr=arr,flowDict=vf.flowDict)

def dict2ff(flowDict):
    """ Returns a velocity flowField with linear/quadratic profile depending on 'isPois'"""
    vf = flowFieldWavy(flowDict=updateDict(flowDict,{'nd':3})).view4d()
    if flowDict['isPois']==0:
        uProfile = vf.y
    else:
        uProfile = 1. - vf.y**2
    vf[0,vf.nx//2,vf.nz//2,0] = uProfile
    return vf

def presDrag(pf):
    """ Returns drag force (as a fraction of force due to mean pressure gradient) due to pressure per unit planform area along streamwise and spanwise
    Use presDrag[0] for just the streamwise"""
    lS = 1; mS = 1
    if pf.flowDict['L'] == 0: lS = 0
    if pf.flowDict['M'] == 0: mS = 0
    pWallIm = np.imag(pf.view4d()[0,pf.nx//2-1*lS,pf.nz//2-1*mS,0,-1])
    eps = pf.flowDict['eps']; a = pf.flowDict['alpha']; b = pf.flowDict['beta']
    
    return np.abs( -4.*eps*a*pWallIm)/4.*pf.flowDict['Re'], np.abs( -4.*eps*b*pWallIm)/4.*pf.flowDict['Re']

def shearStress(vf,seprn=False,localDist=False):
    """ Returns drag due to skin friction along streamwise, only for streamwise waviness (currently)
    """
    assert vf.nz == 1
    # Refer to documentation to meanings of symbols
    # Building a function that can be handed to scipy.integrate

    vfx = vf.ddx().view4d(); vfy = vf.ddy().view4d()

    # Arrays for storing the values of Fourier mdoes at the walls
    uxWall = np.zeros(vf.nx,dtype=np.complex)
    uyWall = uxWall.copy()
    vxWall = uxWall.copy()
    vyWall = uxWall.copy()

    uxWall[:] = vfx[0,:, 0, 0,-1]  
    uyWall[:] = vfy[0,:, 0, 0,-1]
    vxWall[:] = vfx[0,:, 0, 1,-1]
    vyWall[:] = vfy[0,:, 0, 1,-1]

    eps = vf.flowDict['eps']; a = vf.flowDict['alpha']; b = vf.flowDict['beta']
    gma = np.sqrt(a**2 + b**2)
    gx = eps*a; gz = eps*b; g = eps*gma
    K = max(vf.nx, vf.nz)//2
    kArr = np.arange(-K, K+1).reshape((1,2*K+1))

    lArr = np.arange(-(vf.nx//2), vf.nx//2+1)
    def _ifftWall(eta,ffWall):
        """ Returns value of field 'ff' at the wall when Fourier coefficients
            at the wall 'ffWall' are supplied. 
        Inputs:
            eta: locations in eta, can be array
            ffWall: coefficients at the wall, organized as -k(a,b),..,(a,b),2(a,b),..,k(a,b)
        Returns:
            ffEta, array if eta is input as array
            """
        eta = np.array([eta]).flatten()
        eta = eta.reshape((eta.size,1))

        ffWall = ffWall.reshape((1,2*K+1))
        ffVals = np.real(np.sum(ffWall*np.exp(1.j*kArr*gma*eta),axis=1))
        return ffVals.flatten()

    def _strainRateFun(eta,uxw,uyw,vxw,vyw):
        """ Local strain rate, du_t/dn"""
        eta = np.array([eta]).flatten()
        sr = np.zeros(eta.shape) 

        # Local strain-rate du_t/dn, t: tangential, n:normal coordinate
        # Going from (x,y) to (t,n) involves coordinate rotation, 
        # du_t/dn = c21*c11*u_x + c21*c12*v_x + c22*c11*u_y + c22*c12*v_y
        #     where     [ c11  c12] =   [e_t.e_x    e_t.e_y]
        #               [ c21  c22] =   [e_n.e_x    e_n.e_y]
        # Cij := cij/sqrt( 1 + (2*gx*sin(ax))^2 )  # Refer to documentation for details
        
        # C21*C11*u_x
        sr += 2.*g*np.sin(gma*eta)*_ifftWall(eta,uxw)

        # C21*C12*v_x
        sr += -4.*(g*np.sin(gma*eta))**2 * _ifftWall(eta,vxw)

        # C22*C11*u_y
        sr += _ifftWall(eta,uyw)

        # C22*C12*v_y
        sr += -2.*g*np.sin(gma*eta)*_ifftWall(eta,vyw)

        # Dividing by (1+ (2*gx * sin(ax))^2 )
        sr *= 1./(1.+ (2.*g * np.sin(gma*eta))**2 )
        return sr

    strainRate = lambda eta: _strainRateFun(eta,uxWall, uyWall, vxWall, vyWall)

    # Lx = 2.*np.pi/a # Wavelength in x
    # Lz = 2.*np.pi/b
    Leta = 2.*np.pi/gma

    # Integrating du_t/dn * e_t.e_x * ds from x=0 to 2*pi/a:
    integratedStrainRate = spint.quad(strainRate, 0., Leta, epsabs = 1.0e-08, epsrel=1.0e-05)[0]
    # e_t.e_x * ds = dx, so don't have to worry about that bit. 
    
    # Times 2, because the force acts on the top surface too. Dividing by 
    #   streamwise wavelength to get the force per unit area (spanwise homogeneous)
    avgStrainRate = 2.*integratedStrainRate/Leta
    avgShearStress = avgStrainRate/vf.flowDict['Re']    # Follows from non-dimensionalization
    # Refer to documentation
    stressDict = {'avgStress':avgShearStress,'avgStressFraction':avgShearStress/4.*vf.flowDict['Re']}
    if  seprn:
        assert (strainRate(0.) > 0.) and (strainRate(Leta) > 0. )
        etaArr = np.arange(0.,Leta,Leta/100.)
        coarseStrRate = strainRate(etaArr)
        # Calculate strain rate at the wall over a coarse grid of 1000 points
        
        # Find indices (of etaArr) for the first and last zero crossings of local strain rate
        negIndArr = np.arange(etaArr.size)[coarseStrRate<=0.]
        if negIndArr.size > 0:
            sep0 = etaArr[negIndArr[0]-1]; sep1 = etaArr[negIndArr[0]]
            reat0 = etaArr[negIndArr[-1]]; reat1 = etaArr[negIndArr[-1]+1]
            # Separation occurs between etaArr[sepInd0],etaArr[sepInd1],
            # Reattachment occurs between etaArr[reatInd0],etaArr[reatInd1]

            if ( (negIndArr[-1] - negIndArr[0] +1 ) != negIndArr.size):
                stressDict.update({'MultipleCrossings':True})
            else:
                stressDict.update({'MultipleCrossings':False})
            
            # I'm not using a binary search- it's pointless,
            #   It's faster to do ifft and stuff with arrays, 
            #   and even if I use a binary search to do a binary search
            #   I can't say the result is physically valid since I use
            #   only a few Fourier modes
            sepArr = np.arange(sep0,sep1,(sep1-sep0)/100.)
            reatArr = np.arange(reat0,reat1,(reat1-reat0)/100.)

            sepStrRate = strainRate(sepArr)
            reatStrRate = strainRate(reatArr)

            sepNegIndArr = np.arange(sepArr.size)[sepStrRate<=0.]
            reatPosIndArr = np.arange(reatArr.size)[reatStrRate>=0.] 
            xSep = sepArr[sepNegIndArr[0]]/Leta
            xReat = reatArr[reatPosIndArr[0]]/Leta

            yBub = ( 0.5*(np.cos(2.*np.pi*xSep) + np.cos(2.*np.pi*xReat)) +1.)/2.
            # Height of bubble approximated as average of height of separation and reattachment
            #   points measured from the bottom of furrows
        else:
            xSep = None; xReat=None; yBub=None
        stressDict.update({'xSep':xSep, 'xReat':xReat,'yBub':yBub})
    return stressDict


    
def linr(flowDict):
    """ Returns linear operator (without BCs) for the laminar solver
        inputs:
                flowDict
    """    
    N = flowDict['N']
    # Laminar solutions only have energy in modes k(alpha*x + beta*z),
    #    but my flowField objects have modes k1*alpha*x + k2*beta*z
    # In a sense, the flowField objects aren't a good fit for this solver,
    #    I'm using them for consistency later on when I move to non-laminar solutions
    if flowDict['L'] == 0:
        n = flowDict['M']
    else:
        n = flowDict['L']

    # Defining a bunch of variables for easy access later
    a = flowDict['alpha']; b = flowDict['beta']; eps = flowDict['eps']; Re = flowDict['Re']
    N2 = 2*N; N3 = 3*N; N4 = 4*N
    m = 2*n + 1           # Total number of Fourier modes, since I do not consider symmetries here
    # I will implement a switch to account for the simpler symmetries

    gma2 = a*a + b*b      # gma2 is the magnitude of the wavenumber vector
    gma = np.sqrt(gma2)
    g = eps*gma           # g is the semi-slope, since eps is only half of the amplitude
    g2 = g*g              # g2 is the square of the semi-slope
    # In my documentation, I refer to g_x=eps*a and g_z=eps*b for streamwise and spanwise slopes
    # In this code, we just just use g=eps*gma instead

    # Wall-normal nodes and differentiation matrices
    y,DM = chebdif(N,2)
    D = DM[:,:,0]
    D2 = DM[:,:,1]

    # Matrices to be used later
    Z = np.zeros((N,N),dtype=np.complex)
    Z4 = np.zeros((N4,N4),dtype=np.complex)
    I = np.identity(N,dtype=np.complex)

    # Linear terms due to the diffusion terms are:
    # {(d_xx + d_yy + d_zz) u}_k =      {-k^2 * gma^2   +   (2*g2+1)*D2 } u_k
    #              +  (-g2*D2) u_{k-2}  + {g*gma*( 2k-1)*D} u_{k-1}    
    #              +  (-g2*D2) u_{k+2}  + {g*gma*(-2k-1)*D} u_{k+1}     # No pressure terms here
    # Linear terms due to pressure gradient are:
    # {d_x p}_k  = {i*k*a} p_k     +  {-i*eps*a*D} p_{k-1} + {i*eps*a*D} p_{k+1}
    # {d_y p}_k  = {D}p_k
    # {d_z p}_k  = {i*k*b} p_k     +  {-i*eps*b*D} p_{k-1} + {i*eps*b*D} p_{k+1}
    #          The mean pressure gradient is considered a forcing term and not as part of the periodic pressure `p'
    # Linear terms due to the continuity equation:
    # {div.(u,v,w)}_k  =   {i*k*a} u_k  +  {D} v_k   +  {i*k*b} w_k 
    #           + {-i*eps*a*D} u_{k-1}  + {i*eps*a*D} u_{k+1} + {-i*eps*b*D} w_{k-1} + {i*eps*b*D} w_{k-1}           

    # Blocks that are off the principal diagonal by +/- 2 so that they multiply modes (u,v,w,p)_{k+-2}
    #       These only appear in the diffusion term, and don't have any 'k' dependent coefficients
    Lm2 = np.vstack((np.hstack((g2/Re*D2, Z, Z, Z)),
                     np.hstack((Z, g2/Re*D2, Z, Z)),
                     np.hstack((Z, Z, g2/Re*D2, Z)),
                     np.hstack((Z, Z, Z, Z))       ))
    Lp2 = Lm2
    # First 3 row-blocks correspond to streamwise, wall-normal, and spanwise momentum equations respectively
    #   Last row-block corresponds to the divergence of velocity
    # There is no p_{k+-2} mode in the equations above, so all multipliers of p are Z (zero matrices)
    # The divergence condition does not contain any velocity mode of k+-2, so all multipliers are zero

    # Blocks that are off the principal diagonal by +/- 1 so that they multiply modes (u,v,w,p)_{k+-1}
    # These ones have k-dependent factors though, but they're all just constant multipliers. So we'll first
    #    define the constant matrices that they multiply
    d2_m1 = -g*gma/Re*D   # The constant multiplier matrices that appear in the second derivative (diffusion term)
    d2_p1 = d2_m1         #    The laplacian is multiplied by -1/Re (-1 because I move the diffusion term to the other side)
    d1X_m1 = -1.j*eps*a*D
    d1Z_m1 = -1.j*eps*b*D
    d1X_p1 = -d1X_m1
    d1Z_p1 = -d1Z_m1


    # Now everything's in place to define the matrix
    # The matrix should be of size (m*N4,m*N4). However, it's easier to start with an (m*N4, (m+4)*N4) matrix
    #      and then truncate to (m*N4,m*N4). This way, the off-diagonals +- 2 can be assigned without worrying
    #      about if conditions
    L = np.zeros((m*N4, (m+4)*N4), dtype=np.complex)
    for kr in range(m):
        # kr is the row-index for blocks in the array, k is the actual wavenumber
        k = kr-m//2
        # kc is the column-index for blocks. This is needed since we added two block-columns at either end
        kc_Pr = kr+2   # Principal diagonal
        kc_p1 = kr+3;  kc_p2 = kr+4;  kc_m1 = kr+1; kc_m2 = kr

        # Filling the principal diagonal
        L[kr*N4:(kr+1)*N4, kc_Pr*N4:(kc_Pr+1)*N4 ] = \
        np.vstack(( np.hstack(( -(-k**2 * gma2*I + (1.+2.*g2)*D2)/Re , Z, Z, 1.j*k*a*I)),
                    np.hstack(( Z, -(-k**2 * gma2*I + (1.+2.*g2)*D2)/Re , Z,         D)),
                    np.hstack(( Z, Z, -(-k**2 * gma2*I + (1.+2.*g2)*D2)/Re , 1.j*k*b*I)),
                    np.hstack(( 1.j*k*a*I, D, 1.j*k*b*I, Z))      ))
        # Streamwise momentum has -Lapl(u)/Re and p_x, Wall-normal has -Lapl(v)/Re and p_y,
        # Spanwise has -Lapl(w)/Re and p_z,  Divergence has u_x, v_y, w_z

        # Filling the diagonal off by +1
        L[kr*N4:(kr+1)*N4, kc_p1*N4:(kc_p1+1)*N4 ] = \
        np.vstack(( np.hstack(((-2.*k -1.)*d2_p1, Z, Z, d1X_p1 )),
                    np.hstack(( Z, (-2.*k-1.)*d2_p1, Z,      Z )),
                    np.hstack(( Z, Z, (-2.*k-1.)*d2_p1, d1Z_p1 )),
                    np.hstack(( d1X_p1, Z, d1Z_p1, Z))    ))
        # Filling diagonal off by -1
        L[kr*N4:(kr+1)*N4, kc_m1*N4:(kc_m1+1)*N4 ] = \
        np.vstack(( np.hstack(((2.*k -1.)*d2_m1, Z, Z, d1X_m1 )),
                    np.hstack(( Z, (2.*k-1.)*d2_m1, Z,      Z )),
                    np.hstack(( Z, Z, (2.*k-1.)*d2_m1, d1Z_m1 )),
                    np.hstack(( d1X_m1, Z, d1Z_m1, Z))    ))

        # Filling diagonal off by +2
        L[kr*N4:(kr+1)*N4, kc_p2*N4:(kc_p2+1)*N4 ] = Lp2
        # Filling diagonal off by -2
        L[kr*N4:(kr+1)*N4, kc_m2*N4:(kc_m2+1)*N4 ] = Lm2

    # Getting rid of the extra column-blocks on either side
    L = L[ :, 2*N4:(m+2)*N4  ]

    # I'm not imposing boundary conditions in this function/script
    #   Will impose BCs or any other symmetries/simplifications in another function/script
    return L


def jcbn(vf):
    """Defines Jacobian matrix G(chi)
    Inputs:
        flowField object (velocity field), must have nd atleast 3"""
    # I defined functions ff2arr and arr2ff to handle oblique types. The below code isn't needed
    # # For streamwise or spanwise waviness, vf has Fourier modes only along either x or z
    # # For oblique, vf has modes along both x and z, but I only need them along the diagonal
    # #       for this laminar solver
    # # To enable accessing just the modes along k(alpha,beta), I define
    # lS = 0 ; mS = 0
    # if (vf.flowDict['L']*vf.flowDict['alpha']) != 0.: lS = 1
    # if (vf.flowDict['M']*vf.flowDict['beta']) != 0.: mS = 1
    # # When populating the jacobian matrix, we use modes (k*lS, k*mS)
    vf = vf.view4d()   # Just in case I was using a 1d view earlier
    vfy = vf.ddy()   # First derivative along y
    
    m = max(vf.nx,vf.nz)    # Total number of Fourier modes- negative,zero, and positive
    n = m//2        # Number of positive Fourier modes
    N = vf.N
    N2 = 2*N; N3 = 3*N; N4 = 4*N   # Easy typing for later
    a = vf.flowDict['alpha']; b = vf.flowDict['beta']; eps = vf.flowDict['eps']

    vArr = ff2arr(vf)           # Convert flowField to a 1d numpy array, without all the extra mode for oblique case
    DvArr = ff2arr(vfy)
    vArr3d = vArr.reshape((m,3,N))
    DvArr3d = DvArr.reshape((m,3,N))    

    # Ensure that the velocity field is zero at the walls
    WallVel = np.zeros((m,3,2),dtype=np.complex)
    WallVel[:] = vArr3d[:,:,[0,-1]]
    if vf.flowDict['isPois'] == 0:
        WallVel[m//2, 0] -= np.array([1.,-1.])     # Moving walls for Couette flow
    assert np.linalg.norm(WallVel.flatten()) == 0.
    
    # Matrices needed for later:
    I = np.identity(N,dtype=np.complex)
    Z = np.zeros((N,N), dtype=np.complex)
    Z4= np.zeros((N4,N4), dtype=np.complex)
    DM = chebdif(N,2)[1]
    D = np.asmatrix(DM[:,:,0])
    
    G = np.zeros((m*N4, m*N4), dtype=np.complex)  
    # Although the non-linear term appears in only the 3 momentum equations, I add a fourth
    #     equation with all zero multipliers (hence the N4) to keep the same size of the linear operator
    
    """
    The non-linear terms are of the form
    f*g, where f would be in (u,v,w) and g be an (x,y,z) derivative of (u,v,w)
    To get a Fourier component, {f*g}_k, a summation is needed- the wave triads
        {f*g}_k = Sum_r[ f_r * g_{k-r} ]
    The wavenumber triad here is (k, r, k-r)
    However, because of the waviness of the wall, the x and z derivatives introduce additional triads for any k:
        (k, r, k-r), (k, r, k-r-1), (k, r, k-r+1)
        This is the first thing to keep in mind
    The second one concerns the relation between the non-linear term N(vf) and its Jacobian G(vf).
    We show that N(vf) = 0.5* G(vf) * vf:
        First, the entries of the Jacobian go like this:
            d(f_r * g_{k-r})/d(f_r) = g_{k-r};   and  d(f_r * g_{k-r})/d(g_{k-r}) = f_r
        The Jacobian has g_{k-r} and f_{r} in appropriate columns so that they multiply f_r and g_{k-r} respectively
        However, we can also split each term into two halves as below:
            f_r* g_{k-r} = 0.5* f_r* g_{k-r}  +  0.5* g_{k-r}* f_r
        That is, we have
            f_r * g_{k-r} = 0.5*  [d(f_r * g_{k-r})/d(f_r) * f_r  +  d(f_r * g_{k-r})/d(g_{k-r})  * g_{k-r}  ]
        So, multiplying the Jacobian with vf gives the same sum, except that the 0.5 factor is absent.
        This is how we get 
            N(vf) = 0.5 * G(vf) * vf
        It is easy to show that this also holds for the case of k=2r so that r=k-r
    So, populating the Jacobian is enough. 
    The actual math involved in populating the Jacobian involves some tedious bookkeeping, so I ignore that here
    Refer to the documentation
    """
    # We'll also be using these products of variables often, so assigning a short name:
    ie = 1.j*eps;   iea = 1.j*eps*a;  ieb = 1.j*eps*b
    
    # The terms f_r*g_{k-r} involve multiplying values of each of f_r and g_{k-r} at each 
    #    wall-normal collocation node, but not a summation over nodes.
    # So, all blocks of the matrix G(vf) are diagonal matrices
    # Numpy has a diag() function that converts a 1d vector into a diagonal matrix, but
    #     writing diag() everytime is a bit of a pain, so we'll call the function d()
    d = np.diag
        
    
    # Finally, we start populating G
    for rp in range(m):
        r = rp - n   # r is the wavenumber, rp is the index in array for the mode
        U = np.asmatrix(d(vArr3d[rp,0]))
        V = np.asmatrix(d(vArr3d[rp,1]))
        W = np.asmatrix(d(vArr3d[rp,2]))
        DU = np.asmatrix(d(DvArr3d[rp,0]))
        DV = np.asmatrix(d(DvArr3d[rp,1]))
        DW = np.asmatrix(d(DvArr3d[rp,2]))
        
        for kp in range(m):
            k = kp - n
            
            # Diagonal (k, k-r-1) for the wavenumber triad (k, r, k-r-1):
            if ( -n <= (k-r-1) <= n ):
                G[ kp*N4:(kp+1)*N4,  (kp-r-1)*N4: (kp-r)*N4  ] += np.vstack((
                    np.hstack((  -iea*(U*D + DU) - ieb*W*D,   Z,   -ieb*DU,  Z  )), 
                    np.hstack((  -iea*DV,   -iea*U*D - ieb*W*D,   -ieb*DV,   Z  )),
                    np.hstack((  -iea*DW,   Z,   -iea*U*D - ieb*W*D - ieb*DW,  Z  )),
                    np.hstack((  Z,   Z,   Z,   Z  ))
                    ))
            
            # Diagonal (k, k-r+1) for the wavenumber triad (k, r, k-r+1):
            if ( -n <= (k-r+1) <= n ):
                G[ kp*N4:(kp+1)*N4,  (kp-r+1)*N4: (kp-r+2)*N4  ] += -1.*np.vstack((
                    np.hstack((  -iea*(U*D + DU) - ieb*W*D,   Z,   -ieb*DU,  Z  )), 
                    np.hstack((  -iea*DV,   -iea*U*D - ieb*W*D,   -ieb*DV,   Z  )),
                    np.hstack((  -iea*DW,   Z,   -iea*U*D - ieb*W*D - ieb*DW,  Z  )),
                    np.hstack((  Z,   Z,   Z,   Z  ))
                    ))
            
            # Diagonal (k, k-r) for the wavenumber triad (k, r, k-r):
            if ( -n <= (k-r) <= n ):
                G[ kp*N4:(kp+1)*N4,  (kp-r)*N4: (kp-r+1)*N4  ] += np.vstack((
                    np.hstack((  1.j*k*a*U + V*D + 1.j*(k-r)*b*W,   DU,   1.j*r*b*U,   Z )),
                    np.hstack((  1.j*r*a*V,   1.j*(k-r)*(a*U+b*W) + DV + V*D,   1.j*r*b*V,   Z  )), 
                    np.hstack((  1.j*r*a*W,   DW,   1.j*(k-r)*a*U + V*D + 1.j*k*b*W,   Z  )), 
                    np.hstack((  Z,   Z,   Z,   Z  ))
                        ))
    return G
                    
def residual(vf=None, pf=None, L=None, G=None):
    """Compute residual: res = (L+0.5*G)x - f"""
    flowDict = vf.flowDict
    m = max(vf.nx,vf.nz); N = vf.N 
    if L is None: L = linr(flowDict)
    if G is None: G = jcbn(vf)
    if pf is None: 
        warn('pressure field not supplied, assigning zero pressure for computing residual')                
        x = vf.appendField(vf.getScalar().zero())
    else: x = vf.appendField(pf)
    xArr = ff2arr(x)
    F = np.dot(L+0.5*G, xArr)
    # If flow is not Couette, then mean pressure gradient needs to be added
    if flowDict['isPois'] == 1:
        F = F.reshape((m,4,N))
        F[m//2,0] += -2./flowDict['Re']
        F = F.flatten()
    
    return F 



def makeSystem(flowDict=None,vf=None, pf=None, F=None, J=None, L=None, G=None, rect=False, resNorm=False):
    """
    Create functions for residual and Jacobian matrices, Boundary conditions and symmetries are imposed here. 
    The output functions
    Inputs:
        vf : velocity flowField (pressure field isn't needed)
        flowDict: Needed if vf is not supplied, assigns linear/quadratic velocity field for Couette/Poiseuille flow
        rect: True/False. If True, boundary conditions are imposed as additional rows
                If False, boundary conditions replace the governing equations at wall-nodes
        F: (residual) 1-d np.ndarray
        J: (Jacobian) 2-d np.ndarray
        L: 2-d np.ndarray for the linear terms (including divergence) without BCs included
    
    If residual and jacobian are supplied, the BCs and symmetries are added straight away.
    If these are not supplied, they are computed before BCs are added
        If L is supplied, it is used, otherwise, it is computed.
        
    Outputs:
        residualBC: 1-d array
        jacobianBC: 2-d array"""
    # Creating residual array and jacobian matrix if they are not supplied
    
    if vf is None: 
        vf = dict2ff(flowDict)
    else:
        flowDict = vf.flowDict

    N = flowDict['N']; N4 = 4*N
    m = 2*max(flowDict['L'],flowDict['M'])+1
        
    if (F is None) or (J is None):
        if L is None: L = linr(flowDict)
        if G is None: 
            G = jcbn(vf)
            
        J = L+ G
        if F is None:
            F = residual(vf=vf, pf=pf, L=L, G=G)
    
    # Some simple checks
    assert (F.ndim == 1) and (J.ndim == 2)
    assert (J.shape[0] == m*N4) and (F.size == m*N4)
    
    # Imposing boundary conditions
    #   Unlike my MATLAB code, I impose all BCs at the end of the jacobian matrix
    #   So, if rect is True, the top rows of jacobian and residual remain unchanged
    #       they just have extra rows at the end
    #   If rect is False, I still add rows for BCs at the end, but I also remove the rows
    #       of the governing equations at the walls
    # Removing wall-equations if rect is False:
    # I don't have to remove wall-equations for divergence
    BCrows = N4*np.arange(m).reshape((m,1)) + np.array([0,N-1,N,2*N-1,2*N,3*N-1]).reshape((1,6))
    BCrows = BCrows.flatten()

    if not rect: 
        residualBC = np.delete(F, BCrows)
        jacobianBC = np.delete(J, BCrows, axis=0)
    else:
        residualBC = F.copy()
        jacobianBC = J.copy()

        
    
    # Now, just append the rows for BCs to jacobian, and zeros to residual
    # Number of rows to add is m*6
    residualBC = np.append(residualBC, np.zeros(6*m, dtype=np.complex))
    jacobianBC = np.append(jacobianBC, np.zeros((6*m, jacobianBC.shape[1]), dtype=np.complex), axis=0)
    # The appendage to jacobianBC doesn't actually give an equation since all coefficients are zero
    #        Replacing some zeros with ones to get Dirichlet BCs:
    #        The column numbers where the zeros must be replaced are the same as the rows we replaced
    #            for rect=False
    jacobianBC[range(-6*m,0),BCrows] = 1.
    
    # Additionally, I'll also add a 0 BC on the zeroth pressure mode, but only if rect is True
    if rect and pres0BC:
        residualBC = np.append(residualBC, np.array([0.+0.j]))
        jacobianBC = np.append(jacobianBC, np.zeros((1,jacobianBC.shape[1])), axis=0)
        jacobianBC[-1, (m//2)*N4+3*N] = 1.
    
    if not resNorm:
        return jacobianBC, residualBC
    else:
        if rect:
            fnorm = chebnorm(residualBC[:m*N4], N) #+ np.linalg.norm(residualBC[m*N4:])
        else:
            F = F.reshape((m,4,N))
            F[:,:3,[0,-1]] = 0.
            fnorm = chebnorm(F.flatten(),N)

        return jacobianBC, residualBC, fnorm

    
    
def iterate(flowDict=None,vf=None, pf=None,iterMax= 6, tol=5.0e-14,rcond=1.0e-14, rect=False):
    if vf is None: 
        vf = dict2ff(flowDict)
    if pf is None: pf = flowFieldWavy(flowDict=updateDict(vf.flowDict,{'nd':1}))
    L = linr(vf.flowDict)
    fnormArr=[]
    flg = 0
    #print('Starting iterations...............')
    for n in range(iterMax):
        #print('iter:',n)
        vf0 = vf.copy() # Just in case the iterations fail
        pf0 = pf.copy() 
        J, F, fnorm = makeSystem(vf=vf, pf=pf, L=L, resNorm=True,rect=rect)
        #print('fnorm:',fnorm)
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
        #print('Number of variables, Rank of JacobianBC matrix:', J.shape[1], jRank)
        #print('len(sVals:',sVals.size, ' Last 3 singular values:',sVals[-3:])
        if linNorm > linTol:
            print('Least squares problem returned residual norm:',linNorm,' which is greater than tolerance:',linTol)
            
            
        
        dxf = arr2ff(dx,updateDict(vf.flowDict,{'nd':4}))
        dvf = dxf.slice(nd=[0,1,2])
        dpf = dxf.getScalar(nd=3)
        
        #print('Corrections in v and p:',dvf.norm(), dpf.norm())
        vf = vf.view4d() + dvf.view4d()
        vf = setSymms(vf)
        pf = pf.view4d() + dpf.view4d()
        #print('*********************************************************')
    else:
        if fnormArr[-1] > 100.*tol:
            print('Iterations have not converged for iterMax=',iterMax)
            print('fnormArr is ',fnormArr)
            flg = -1
    return vf, pf, np.array(fnormArr), flg
    
    
    
