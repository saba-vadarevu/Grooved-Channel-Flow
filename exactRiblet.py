import numpy as np
from flowFieldWavy import *
import scipy.integrate as spint
import sys

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

def setSymms(xTemp):
    """Set velocities at walls to zero, and ensure that the field is real valued"""
    x = xTemp.copy()
    x.view4d()[:,:,:,:3,[0,-1]] = 0.
    if x.flowDict['isPois']== 0:
        x.view4d()[0,x.nx//2, x.nz//2, 0,[0,-1]] = np.array([1.,-1.])
    
    #vf[0,:,:vf.nz//2] = 0.5*(vf[0,:,:vf.nz//2]+ vf[0,:,:vf.nz//2:-1].conj())
    #vf[0,:,:vf.nz//2:-1] = vf[0,:,:vf.nz//2].conj()
    return x 

def dict2ff(flowDict):
    """ Returns a velocity flowField with linear/quadratic profile depending on 'isPois'"""
    vf = flowFieldRiblet(flowDict=updateDict(flowDict,{'nd':3})).view4d()
    if flowDict['isPois']==0:
        uProfile = vf.y
    else:
        uProfile = 1. - vf.y**2
    vf[0,vf.nx//2,vf.nz//2,0] = uProfile
    return vf


def linr(flowDict,complexType = np.complex, sigma1 = True): 
    """Returns matrix representing the linear operator for the equilibria/TWS for riblet case
    Linear operator for exact solutions isn't very different from that for laminar, 
        all inter-modal interaction remains the same for a fixed 'l'. 
    So, we use the function already written in the laminar.py module and modify it to suit
        the current case
    To use single floats, input argument complexType=np.complex64 
    Inputs:
        flowDict
        complexType (default: np.complex): Use np.complex64 for single precision
        epsArr (default: [0.05,0.,0.]): 1-D numpy array containing amplitudes of surface-modes
                            First element is eps_1, and so on... 
    Outputs:
        Lmat:   Matrix representing linear terms for the complete state-vectors"""
    if 'epsArr' in flowDict:
        epsArr = flowDict['epsArr']
        if epsArr.ndim !=1: warn("epsArr is not a 1-D array. Fix this.")
    else:
        epsArr = np.array([0., flowDict['eps']])

    if complexType is not np.complex:
        warn("complexType is set to np.complex. Other implementations not currently available.")
        complexType = np.complex

    L = flowDict['L']; M = flowDict['M']
    L1 = L+1
    nx = int(2*L+1)
    nz = int(2*M+1)
    a = flowDict['alpha']; a2 = a**2
    b = flowDict['beta']; b2 = b**2
    Re = flowDict['Re']

    N  = int(flowDict['N']); N4 = 4*N
    y,DM = chebdif(N,2)
    D = DM[:,:,0].reshape((N,N)); D2 = DM[:,:,1].reshape((N,N))
    # Change N, D, and D2 if imposing point-wise inversion symmetries

    I = np.identity(N); Z = np.zeros((N,N))

    def _assignL0flat(L0flat,m):
        assert (L0flat.shape[0] == N4) and (L0flat.shape[1] == N4)
        # L_0,flat is built for the case of L= 0 without accounting for wall effects. 
        L0flat[:,:] = np.vstack((
                np.hstack(( -(-m**2 * b2 * I + D2)/Re,  Z,  Z,  Z   )),
                np.hstack(( Z,  -(-m**2 * b2 * I + D2)/Re,  Z,  D   )),
                np.hstack(( Z,  Z,  -(-m**2 * b2 * I + D2)/Re, 1.j*m*b*I )),
                np.hstack(( Z,  D,  1.j*m*b*I,  Z))       ))
        # First row-block is streamwise momentum equation, diffusion term. 
        #       Pressure term isn't set because it's zero for l=0
        # Second is wall-normal momentum
        # Third is spanwise momentum
        # Fourth is continuity
        # Streamwise derivatives in all four equations are set to zero because l=0
        return

    Tz, Tzz, Tz2 = Tderivatives(updateDict(flowDict,{'epsArr':epsArr})) 

    q0 = epsArr.size-1
    # -1 because epsArr includes zeroth mode

    # Number of columns is increased by 4*q0 because wall effects produce
    #   interactions all the way from -M-2*q0 to M+2*q0
    #   I prefer to build the matrix with these included, and then truncate to -M to M
    L0wavy = np.zeros((nz*N4, (nz+4*q0)*N4), dtype=complexType)
    # Wall effects show up in spanwise derivatives only
    # d_z (.) = (-imb)(.) + \sum\limits_q  (T_{z,-q} D) (.)_{l,m+q}
    # d_zz (.) = (-m^2 b^2)(.) + \sum\limits_q \{(T_{zz,-q} + 2i(m+q)bT_{z,-q})D + T^2_{z,-q} D^2\} (.)_{l,m+q}
    for mp in range(nz):
        m = mp-M
        _assignL0flat(L0wavy[mp*N4:(mp+1)*N4, (mp+2*q0)*N4:(mp+2*q0+1)*N4], m)
        # The principal diagonal is shifted by 2*q0= 2*(epsArr.size-1) 

        # Wall-effects enter into the equations as
        #   T_z(-q) * (.)_{l,m+q}
        # Factor of u_{l,m+q} is supposed to be on the q^th diagonal,
        #   however, since I add extra 2*q0 column-blocks on either end,
        #   this factor must be on (q+2*q0)^th diagonal
        for q in range(-q0, q0+1):
            L0wavy[mp*N4:(mp+1)*N4, (mp+q+2*q0)*N4: (mp+q+2*q0+1)*N4] += np.vstack((
                    np.hstack(( -1./Re*( (Tzz[-q+q0] + 2.j*(m+q)*b*Tz[-q+q0])*D) , Z, Z, Z)),
                    np.hstack(( Z, -1./Re*( (Tzz[-q+q0] + 2.j*(m+q)*b*Tz[-q+q0])*D) , Z, Z )), 
                    np.hstack(( Z, Z, -1./Re*( (Tzz[-q+q0] + 2.j*(m+q)*b*Tz[-q+q0])*D), Tz[-q+q0]*D )), 
                    np.hstack((Z, Z, Tz[-q+q0]*D, Z))     ))
        # In the above matrices, I did not include Tz2 terms, that's because 
        #   Tz2 goes from -2*q0 to 2*q0. I add it separately below?
        for q in range(-2*q0, 2*q0+1):
            L0wavy[mp*N4:(mp+1)*N4, (mp+q+2*q0)*N4: (mp+q+2*q0+1)*N4] += np.vstack((
                    np.hstack(( -1./Re*(Tz2[-q+2*q0]*D2), Z, Z, Z)),
                    np.hstack(( Z, -1./Re*( Tz2[-q+2*q0]*D2), Z, Z )), 
                    np.hstack(( Z, Z, -1./Re*(Tz2[-q+2*q0]*D2),Z )), 
                    np.hstack((Z, Z, Z, Z))     ))
    L0wavy = L0wavy[:, N4*2*q0: -N4*2*q0]   # Getting rid of the extra column-blocks
    # And that concludes building L0wavy

    if sigma1:
        nz1 = M+1
    else:
        nz1 = nz
    # Building Lmat for only m<= 0

    nx = 2*L+1
    # For exact solutions, we have L!= 0
    #   So, I take L0wavy, and add i.l.alpha or -l**2.a**2 as appropriate
    Lmat = np.zeros((nx*nz1*N4,nx*nz1*N4),dtype=complexType)
    # If imposing sigma1, build for only m <= 0 


    mat1 = np.zeros((nz1*N4,nz1*N4),dtype=complexType); mat2 = mat1.copy()
    # Define mat1 and mat2 such that all 'l' terms in the linear matrix can be 
    #   written as l * mat1  + l^2 *mat2
    # Their definition remains unchanged with imposition of sigma1
    for mp in range(nz1):
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

        
    s1 = nz1*N4
    s3 = nz*N4
    s2 = s3-s1

    if sigma1:
        # To add linear inter-modal contributions due to m > 0 modes
        #    to equations for m <= 0;
        # u_{l,-m} = (-1)^l u_{l,m}, v_{l,-m} = (-1)^l v_{l,m}, 
        # w_{l,-m} =-(-1)^l w_{l,m}, p_{l,-m} = (-1)^l p_{l,m} 
        # Equations for m <= 0 correspond to L0wavy[:s1]
        # Contributions due to m > 0 correspond to L0wavy[., s1:]
        # They need to be rearranged into N4xN4 blocks (for each Fourier mode),
        #    and then multiplied by (-1)^l. 'w' needs to be multiplied with an extra -1
        L0wavyTemp = L0wavy[:s1, s1:].reshape((s1, s2//N4, N4))

        # Now, reordering so that modes go as M, M-1,...,1
        L0wavyTemp = L0wavyTemp[:, ::-1]

        # Multiplying the spanwise velocity with -1
        L0wavyTemp[ :, :, 2*N:3*N] *= -1.
        # Finally, reshaping
        L0wavyTemp = L0wavyTemp.reshape((s1, s2))
        # Now we're ready to multiply with (-1)^l and add to Lmat
        

    for lp in range(nx):
        l = lp-L

        # Using s1 instead of nz*N4. If sigma1 is False, there is no difference
        # Matrix from laminar case where l=0
        Lmat[lp*s1:(lp+1)*s1, lp*s1:(lp+1)*s1] = L0wavy[:s1, :s1]
        # Adding all the l-terms
        Lmat[lp*s1:(lp+1)*s1, lp*s1:(lp+1)*s1] += l* mat1 + l**2 * mat2
        
        if sigma1:
            # Adding L0wavyTemp:
            Lmat[lp*s1:(lp+1)*s1, lp*s1: lp*s1+s2] += (-1.)**l  * L0wavyTemp

    return Lmat



def jcbn(vf,Lmat=None,sigma1=True):
    if Lmat is None:
        warn('The Jacobian is added in-place to Lmat. Always supply Lmat. Returning.....')
        return

    if sigma1:
        nz1 = vf.nz//2 + 1
    else:
        nz1 = vf.nz

    a = vf.flowDict['alpha']; b = vf.flowDict['beta']
    epsArr = vf.flowDict['epsArr']
    q0 = epsArr.size-1
    Tz = Tderivatives(vf.flowDict)[0]
    N = vf.N; L = vf.nx//2; M = vf.nz//2; N4 = 4*N

    # No reason to keep accessing flowFieldRiblet with all its extra machinery
    #   Copy elements to a regular numpy array instead
    vfArr = vf.view4d().copyArray()
    u = vfArr[0,:,:,0]; v = vfArr[0,:,:,1]; w = vfArr[0,:,:,2]
    vfyArr = vf.ddy().view4d().copyArray()
    uy = vfyArr[0,:,:,0]; vy = vfyArr[0,:,:,1]; wy = vfyArr[0,:,:,2]
    # The state-vectors aren't that big, so memory isn't an issue

    D = vf.D; I = np.identity(vf.N, dtype=Lmat.dtype); Z = np.zeros((vf.N,vf.N),dtype=Lmat.dtype)
    D.astype(Lmat.dtype)

    # Index of the first row/column of the block for any wavenumber vector (l,m)
    iFun = lambda l,m: (l+L)*(nz1*4*N) + (m+M)*4*N

    Gmat = Lmat
    assert (Gmat.shape[0] == vf.nx*nz1*4*N) and (Gmat.shape[1] == vf.nx*nz1*4*N)

    # I will be using the functions np.diag() and np.dot() quite often, so,
    diag = np.diag; dot = np.dot

    # I am deviating from my earlier implementation of defining G to revert to
    #   the implementation used for the laminar case
    # The two ways to do it is this: 
    #   1) Go through each row-column-block of the matrix and figure out which 
    #           u_{l,m} goes there (along with factors for derivatives)
    #   2) Loop over (l,m), get the u_{l,m} and d_y(u_{l,m})
    #       Then, go through row-blocks and assign this u_lm to the appropriate column
    # The first approach is the simpler one. The second is a bit more complex, but
    #   involves one loop fewer. I'm going with the second one, but not for the performance
    # Since I am only building G for non-positive 'l', it's easier if I go with the second

    # This is the notation I shall use:
    # l', m' (written in code as lp, mp) represent the mode numbers of the modes that I
    #   first loop through. That is, I loop over l',m' and populate G with u_{l',m'}
    # l , m represent the mode for which the equation is written
    #   I use phi_lm to represent the convection term in the NSE for mode (l,m)
    #   phi^1 is streamwise convection term: phi^1 = u d_x u + v d_y u + w d_z u
    #   phi^2 is wall-normal, phi^3 is spanwise
    # So, the wave-triads go as { (l', m'), (l,m), (l-l',m-m')} (for terms not involving wall-effects)
    #   u_{l',m'} is populated in row-block corresponding to phi_{l,m} in column-block for u_{l-l',m-m'}

    # Strictly speaking, what I'm building is not the Jacobian G
    # I'm building G such that N(\chi) = 0.5 * G * \chi, where \chi is the state-vector
    # G differs from the Jacobian of N only in terms of
    #                           d/du (u') being written as D instead of u''/ u' (which might produces NaNs)

    # Final piece of notation: 
    ia = 1.j*a; ib = 1.j*b

    if sigma1:
        M1 = 1
    else:
        M1 = M+1

        
    # The convection term has this form:
    # phi^{lm}_{1,2,3} = \sum_lp \sum_mp  u_{l-lp,m-mp} u_{lp,mp}  , 
    #    disregarding the surface influence terms. 
    # Gmat is the actual non-linear jacobian
    # G is a temporary matrix created for each 'lp' in equations for each (l,m) in the Jacobian,
    # If sigma1 is to be imposed,
    #    G is folded and multiplied with (-1)**lp before being assigned to the
    #        corresponding rows and columns in Gmat
    # Otherwise, G is added as is
    G = np.zeros((4*N, vf.nz*4*N), dtype=np.complex)


    for l in range(-L,L+1):
        for m in range(-M,M1):
            # Index of first row in the block for equations for wavenumbers (l,m)
            rInd = iFun(l,m)
            # l1,m2 are the wavenumbers in phi^j_{lm}

            for lp in range(-L,L+1):
                G[:] = 0.    # Getting rid of G from previous 
                rInd = 0     # We're only writing equations for one l,m, lp
                # G contains the part of the Jacobian that multiplies the mode (lp,:)
                #    in the equations for (l,m)
                # I will define this without assuming symmetries
                # When adding to Gmat, I will fold G on itself
                
                for mp in range(-M,M+1):
                    #cInd = iFun(lp,mp)
                    cInd = iFun(-L,mp)  # Because I'm writing G for only one 'l'
                    if (-L <= (l-lp) <= L):
                        li = l-lp+L # Array index for streamwise wavenumber l-lp

                        # First, all the terms not relating to wall-effects
                        if (-M <= (m-mp) <= M):
                            mi = m-mp+M # Array index for spanwise wavenumber m-mp
                            # phi^1_{l,m}: factors of terms with  u_{lp,mp}:
                            G[ rInd+0*N : rInd+1*N , cInd+0*N : cInd+1*N ] += \
                                    ( l*ia* diag(u[li, mi])  + v[li,mi].reshape((N,1)) *D + diag(w[li,mi])*mp*ib )
                            # phi^1_{l,m}: factors of terms with  v_{lp,mp}:
                            G[ rInd+0*N : rInd+1*N , cInd+1*N : cInd+2*N ] += (diag(uy[li, mi]))
                            # phi^1_{l,m}: factors of terms with  w_{lp,mp}
                            G[ rInd+0*N : rInd+1*N , cInd+2*N : cInd+3*N ] += ((m-mp)*ib*diag(u[li, mi]))

                            # phi^2_{l,m}: factors of terms with  v_{lp,mp}:
                            G[ rInd+1*N : rInd+2*N , cInd+1*N : cInd+2*N ] += \
                                    (    lp*ia* diag(u[li, mi])  + v[li,mi].reshape((N,1)) *D + diag(w[li,mi])*mp*ib \
                                    + diag(vy[li,mi])   )
                            # phi^2_{l,m}: factors of terms with  u_{lp,mp}:
                            G[ rInd+1*N : rInd+2*N , cInd+0*N : cInd+1*N ] += ((l-lp)*ia*diag(v[li, mi])  )
                            # phi^2_{l,m}: factors of terms with  w_{lp,mp}:
                            G[ rInd+1*N : rInd+2*N , cInd+2*N : cInd+3*N ] += ((m-mp)*ib*diag(v[li, mi])  )

                            # phi^3_{l,m}: factors of terms with  w_{lp,mp}:
                            G[ rInd+2*N : rInd+3*N , cInd+2*N : cInd+3*N ] += \
                                    (  lp*ia* diag(u[li, mi])  + v[li,mi].reshape((N,1)) *D + m*ib*diag(w[li,mi])  )
                            # phi^3_{l,m}: factors of terms with  v_{lp,mp}:
                            G[ rInd+2*N : rInd+3*N , cInd+1*N : cInd+2*N ] += (   diag(wy[li, mi]) )
                            # phi^3_{l,m}: factors of terms with  u_{lp,mp}:
                            G[ rInd+2*N : rInd+3*N , cInd+0*N : cInd+1*N ] += (   (l-lp)*ia*diag(w[li, mi])  )

                        # Now, the terms arising due to wall effects
                        # The interactions in l are unaffected since Tz only have e^iqb

                        for q in range(-q0,q0+1):

                            if (-M <= (m-mp+q) <= M):
                                mi = m-mp+q+M # Array index for spanwise wavenumber m-mp
                                # phi^1_{l,m}: factors of terms with u_{lp,mp}
                                G[ rInd+0*N : rInd+1*N , cInd+0*N : cInd+1*N ] += Tz[q0-q]* w[li,mi].reshape((N,1)) * D 
                                # phi^1_{l,m}: factors of terms with w_{lp,mp}
                                G[ rInd+0*N : rInd+1*N , cInd+2*N : cInd+3*N ] += Tz[q0-q]* diag(uy[li,mi]) 

                                # phi^2_{l,m}: factors of terms with v_{lp,mp}
                                G[ rInd+1*N : rInd+2*N , cInd+1*N : cInd+2*N ] += Tz[q0-q]* w[li,mi].reshape((N,1)) * D 
                                # phi^2_{l,m}: factors of terms with w_{lp,mp}
                                G[ rInd+1*N : rInd+2*N , cInd+2*N : cInd+3*N ] += Tz[q0-q]* diag(vy[li,mi]) 

                                # phi^3_{l,m}: factors of terms with w_{lp,mp}
                                G[ rInd+2*N : rInd+3*N , cInd+2*N : cInd+3*N ] += Tz[q0-q]* w[li,mi].reshape((N,1)) * D \
                                        +Tz[q0-q]* diag(wy[li,mi]) 
                # Now, G is ready to be folded if sigma1 holds
                if sigma1:
                    if False:
                        Gtemp = np.zeros((4*N, M+1, 4*N), dtype=Gmat.dtype)
                        Gnew = G.reshape((4*N, vf.nz, 4,N))
                        Gnew[:, M:,2] = -1.* Gnew[:, M:, 2]   # w-factors need to be multiplied with an extra -1
                        # Because sigma1 goes u_{l,m} = (-1)^l  u_{l,-m} (same for v and p)
                        #               but   w_{l,m} = (-1)^{l+1} w_{l,-m}
                        Gnew = Gnew.reshape((4*N, vf.nz, 4*N))

                        Gtemp[:] = Gnew[:, :M+1]   # Factors of non-positive modes are assigned as are
                        Gtemp[:,:M] += (-1.)**lp  * Gnew[:, :M:-1]   # Need to assign M to -M, (M-1) to -(M-1) and so on..
                        # Reshaping to a matrix
                        Gtemp = Gtemp.reshape((4*N, nz1*4*N))
                        # And that's it. We're ready to add to Gmat
                    else:
                        Gtemp = G[:, nz1*4*N:].reshape((4*N, M, 4*N))
                        Gtemp = Gtemp[:,::-1]
                        Gtemp[:,:, 2*N:3*N] *= -1.
                        Gtemp = Gtemp.reshape((4*N, M*4*N))
                        
                        
                    
                
                cInd = iFun(lp, -M)
                rInd = iFun(l,m)
                Gmat[rInd: rInd+4*N, cInd: cInd+nz1*4*N] += G[:,:nz1*4*N]
                if sigma1:
                    Gmat[rInd: rInd+4*N, cInd: cInd+M*4*N] += (-1.)**lp * Gtemp


    return  

def _residual(x):
    return (x.slice(nd=[0,1,2]).residuals(pField=x.getScalar(nd=3)).appendField( x.slice(nd=[0,1,2]).div() ) )


def makeSystem(vf=None,pf=None, complexType=np.complex):
    """
    Create functions for residual and Jacobian matrices, Boundary conditions and symmetries are imposed here. 
    The output functions
    Inputs:
        vf : velocity flowField (pressure field isn't needed)
        resNorm: If True, return residual norm
    
        
    Outputs:
        residualBC: 1-d array
        jacobianBC: 2-d array"""
    sigma1=False 
    N = vf.N; N4 = 4*N
    L = vf.nx//2; M = vf.nz//2
    L1 = L+1; nz=vf.nz
    if pf is None:
        pf = vf.getScalar().zero()
        
    J = linr(vf.flowDict, complexType=complexType, sigma1=sigma1)
    jcbn(vf, Lmat=J,sigma1=sigma1)
    F = (vf.residuals(pField=pf).appendField( vf.div() ) ).flatten()
    
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

    jacobianBC = J
    
    jacobianBC[BCrows,:] = 0.
    jacobianBC[BCrows,BCrows] = 1.
    # Equations on boundary nodes now read 1*u_{lm} = .. , 1*v_{lm} = .., 1*w_{lm} = ..
    # The RHS for the equations is set in residualBC below

    return jacobianBC, F 


def lineSearch(normFun,x0,dx,arr=None):
    print("Beginning line search.... Initial residual norm is ",normFun(x0))
    if arr is None:
        arr = np.arange(-0.5,2.1,0.1)
    else:
        arr = np.array(arr).flatten()

    normArr = np.ones(arr.size)
    for k in range(arr.size):
        q = arr[k]
        normArr[k] = normFun(x0+q*dx)

    kMin = np.argmin(normArr)
    normMin = normArr[kMin]
    qMin = arr[kMin]

    if arr[0] < arr[kMin] < arr[-1]:
        arrNew = np.arange( arr[kMin-1], arr[kMin+1], (arr[kMin+1] - arr[kMin-1])/20.)
        normArrNew = np.ones(arrNew.size)
        for k in range(arrNew.size):
            q = arrNew[k]
            normArrNew[k] = normFun(x0+q*dx)

        kMinNew = np.argmin(normArrNew)
        normMinNew = normArrNew[kMinNew]
        qMinNew = arrNew[kMinNew]

        round2 = True
    else:
        round2 = False

    print("Line search.... normArr is",normArr)
    print("Minimal norm is obtained for q in q*dx of %.2g, producing norm of %.3g"%(qMin, normMin))
    if round2:
        print("Finer line search.... normArr is",normArrNew)
        print("Minimal norm is obtained for q in q*dx of %.2g, producing norm of %.3g"%(qMinNew, normMinNew))
        qMin = qMinNew

    oldNorm = normFun(x0); newNorm = normFun(x0+qMin*dx)
    if newNorm > oldNorm:
        print("New norm (%.3g) is greater than the old norm (%.3g) for some weird reason"%(newNorm,oldNorm))

    return x0+qMin*dx



def iterate(vf=None, pf=None,iterMax= 6, tol=5.0e-10,rcond=1.0e-07,complexType=np.complex,doLineSearch=True, sigma1=False):
    sigma1=False
    if pf is None: pf = vf.getScalar().zero()
    resnormFun = lambda x: x.residuals().appendField(x.div()).norm() 

    x = vf.appendField(pf)

    x = setSymms(x)   # Nothing fancy here. Just setting velocities at wall to zero
    # And ensuring field is real-valued

    fnormArr=[]
    flg = 0
    resnorm0 = resnormFun(x)
    if resnorm0 <= tol:
        print("Initial flowfield has zero residual norm (%.3g). Returning..."%(resnorm0))
        return vf,pf,np.array([resnorm0]),flg
    else:
        print("Initial residual norm is %.3g"%(resnorm0))

    print('Starting iterations...............')
    for n in range(iterMax):
        print('iter:',n+1)


        # Ensure BCs on vf, and field is real-valued
        vf = x.slice(nd=[0,1,2]); pf = x.slice(nd=3)


        J, F = makeSystem(vf=vf, pf=pf,complexType=complexType)
                
        sys.stdout.flush()
        
        dx, linNorm, jRank,sVals = np.linalg.lstsq(J,-F,rcond=rcond)
        linNorm = np.linalg.norm(np.dot(J,dx) + F)
        print("Jacobian inversion success with residual norm ", linNorm)
        if linNorm > linTol:
            print('Least squares problem returned residual norm:',linNorm,' which is greater than tolerance:',linTol)
            
       
       
        dxff = x.zero()
        #dxff[0,:x.nx//2+1] = dx.reshape((x.nx//2+1,x.nz,4,x.N))
        #dxff[0,x.nx//2+1:] = np.conj(dxff[0, x.nx//2-1::-1, ::-1])
        dxff[0] = dx.reshape((x.nx, x.nz, 4, x.N))

        #dxff[0] = 0.5*(dxff[0] + np.conj(dxff[0,::-1,::-1]))
        dxff[0,:,:,:3,[0,-1]] = 0.

        if doLineSearch:
            x = lineSearch(resnormFun, x, dxff)
        else:
            x += dxff
        print("residual norm before setSymms:",resnormFun(x))
        x = setSymms(x)

        fnorm = resnormFun(x)
        print('Residual norm after setSymms in %d th iteration is %.3g'%(n+1,fnorm))
        sys.stdout.flush()
        
        fnormArr.append(fnorm)
        if fnorm <= tol:
            flg = 0
            print('Converged in ',n+1,' iterations. Returning....................................')
            return x.slice(nd=[0,1,2]), x.getScalar(nd=3), np.array(fnormArr), flg
        
        if n>0:
            if fnormArr[n] > fnormArr[n-1]:
                flg = 1
                print('Residual norm is increasing:',fnormArr)
                #print('Returning with initial velocity and pressure fields')
                #return vf0, pf0, np.array(fnormArr),flg
        
        print('*********************************************************')
    else:
        if fnormArr[-1] > 100.*tol:
            print('Iterations have not converged for iterMax=',iterMax)
            print('fnormArr is ',fnormArr)
            flg = -1
    return x.slice(nd=[0,1,2]), x.getScalar(nd=3), np.array(fnormArr), flg

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

def averagedU(vf,nd=0, zArr = None, ny = 50):
    """ Velocity averaged in wall-parallel directions in physical domain"""
    b = vf.flowDict['beta']; Lz = 2.*np.pi/b
    epsArr = vf.flowDict['epsArr']
    if zArr is None:
        # Use 500 z-points for averaging
        nz = 500.
        zArr = np.arange(0., Lz, Lz/nz)
    
    
    yWalls = np.zeros(zArr.size)
    # Wall contour
    for k in range(epsArr.size):
        eps = epsArr[k]
        yWalls  += 2.*eps*np.cos(k*b*zArr)
    yMax = np.amax(yWalls); yMin = np.amin(yWalls)
    # At each z location, flowfield class assumes y goes from -1 to 1, 
    #    but it actually goes from -1+yWalls to 1+yWalls
    # I want the velocity profile going from -1+yMax to 1+yMin
    # To keep things simple, I'll use a uniform grid between -1+yMax and 1+yMin
    # At any z, I will need to interpolate the velocity from 
    #      -1 + (yMax -yWalls(z)) to
    #       1 - (yWalls(z) -yMin)
    # Let's define yb = -1 + (yMax - yWalls(z) )  and yt = 1 - (yWalls(z) - yMin)
    yb = -1. + yMax - yWalls
    yt = 1.  + yMin - yWalls
    
    vfArr = vf.getScalar(nd=nd).slice(L=0).copyArray()[0,0,:,0]  
    # Don't need l !=0 modes in averaging
    mArr = np.arange(-(vf.nz//2), vf.nz//2 +1).reshape((vf.nz,1))
    
    def _ifftAvgU(zLoc,yArr):
        # First, getting field on cheb nodes at zLoc:
        vTemp = np.real(np.sum( vfArr* np.exp(1.j*mArr*b*zLoc) , axis=0))
        # Interpolating onto the new grid, yArr and returning
        return chebint(vTemp,yArr)
    
    # Now, start with a zero array,
    vArr = np.zeros(ny+1)
    for k in range(zArr.size):
        yLocs = np.arange(yb[k], 1.0001*yt[k], (yt[k]-yb[k])/ny )
        vArr += _ifftAvgU(zArr[k], yLocs)
    
    vArr = vArr/zArr.size
    
    yArr = np.arange(-1.+yMax, 1.0001*(1.+yMin), (2+yMin-yMax)/ny)
    
    return yArr, vArr
    
    


    


def testExactRibletModule(L=4,M=7,N=35,epsArr=np.array([0.,0.05,0.02,0.03]),sigma1=True,complexType=np.complex):
    vf = h52ff('testFields/eq1.h5')
    pf = h52ff('testFields/pres_eq1.h5',pres=True)
    vf = vf.slice(L=L,M=M,N=N); pf = pf.slice(L=L,M=M,N=N)
    vf.flowDict.update({'epsArr':epsArr}); pf.flowDict.update({'epsArr':epsArr})
    
    Lmat = linr(vf.flowDict,complexType=complexType,sigma1=sigma1)
    x = vf.appendField(pf)
    
    if sigma1:
        xm_ = x.copyArray()[0,:,:x.nz//2+1].flatten()
        linTerm = np.dot(Lmat, xm_)
    else:
        linTerm = np.dot(Lmat, x.flatten())


    nex = 5

    vf1 = vf.slice(L=vf.nx//2+nex, M = vf.nz//2+nex)
    pf1 = pf.slice(L=vf.nx//2+nex, M = vf.nz//2+nex)
    

    linTermClass = (vf1.laplacian()/(-1.*vf1.flowDict['Re']) + pf1.grad()).appendField(vf1.div())
    linTermClass = linTermClass.slice(L=vf.nx//2, M=vf.nz//2)
    if sigma1:
        linTestResult = chebnorm(linTerm - linTermClass.copyArray()[0,:,:x.nz//2+1].flatten(), x.N) <= tol
        print('sigma1 invariance norm of x is', (x - x.reflectZ().shiftPhase(phiX=np.pi) ).norm())
    else:
        linTestResult = chebnorm(linTerm - linTermClass.copyArray().flatten(), x.N) <= tol


    # Lmat = np.zeros(( (x.nx//2 +1) * x.nz *4*N , (x.nx//2 +1) * x.nz *4*N ), dtype=np.complex)
    Lmat0 = Lmat.copy()
    #Lmat[:] = 0.
    jcbn(vf,Lmat=Lmat,sigma1=sigma1)
    Lmat = Lmat-Lmat0

    if sigma1:
        xm_ = x.copyArray()[0,:,:x.nz//2+1].flatten()
        NLterm = 0.5*np.dot(Lmat, xm_)
    else:
        NLterm = 0.5*np.dot(Lmat, x.flatten())

    NLtermClassFine = vf1.convNL()
    NLtermClass = NLtermClassFine.slice(L=vf.nx//2, M=vf.nz//2).appendField(pf.zero())

    if sigma1:
        NLtestResult = chebnorm(NLterm - NLtermClass.copyArray()[0,:,:x.nz//2+1].flatten(), x.N) <= tol
        print('sigma1 invariance norm of x is', (x - x.reflectZ().shiftPhase(phiX=np.pi) ).norm())
    else:
        NLtestResult = np.linalg.norm(NLterm - NLtermClass.flatten()) <= tol
    
    if sigma1: nz1 = vf.nz//2 + 1
    else: nz1 = vf.nz
    if not linTestResult :
        print('Residual norm for linear is:',np.linalg.norm(linTerm - linTermClass[0,:,:nz1].flatten()))
    if not  NLtestResult:
        print('Residual norm for non-linear is:',np.linalg.norm(NLterm - NLtermClass[0,:,:nz1].flatten()))

    if linTestResult and NLtestResult:
        print("Success for both tests!")
    
    return linTestResult, NLtestResult






