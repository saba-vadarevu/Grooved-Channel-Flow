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


def linr(flowDict,complexType = np.complex,sigma1=False): 
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
    if False:
        assert flowDict['L'] != 0 and flowDict['M'] != 0
        if flowDict['L'] > 4: 
            print('L is set to 4 from ', flowDict['L'] )
            flowDict.update({'L':4})
        if flowDict['M'] > 8:
            print('M is set to 8 from ', flowDict['M'] )
            flowDict.update({'M':8})
    # Lmat_lam = lam.linr(updateDict(flowDict,{'L':0,'alpha':0.}))

    L = flowDict['L']; M = flowDict['M']
    L1 = L+1
    nx = int(2*L+1)
    nz = int(2*M+1)
    a = flowDict['alpha']; a2 = a**2
    b = flowDict['beta']; b2 = b**2
    Re = flowDict['Re']

    N  = int(flowDict['N']); N4 = 4*N
    y,DM = chebdif(N,2)
    if complexType is np.complex64:
        DM = np.float32(DM)
    D = DM[:,:,0].reshape((N,N)); D2 = DM[:,:,1].reshape((N,N))
    # Change N, D, and D2 if imposing point-wise inversion symmetries
        
    I = np.identity(N,dtype=complexType); Z = np.zeros((N,N),dtype=complexType)

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

    M1 = M+1
    # For exact solutions, we have L!= 0
    #   So, I take L0wavy, and add i.l.alpha or -l**2.a**2 as appropriate
    # Define mat1 and mat2 such that all 'l' terms in the linear matrix can be 
    #   written as l * mat1  + l^2 *mat2
    mat1 = np.zeros((nz*N4,nz*N4),dtype=complexType); mat2 = mat1.copy()
    # It's okay to define these matrices for nz*N4, I can just trim them to 
    #   shape (M1*N4,M1*N4) later if sigma1 is to be imposed
    #   I say it's okay because the extra memory used isn't significant compared
    #       to the size of Lmat, which is 0.5*L1 times larger.

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
    
    s1 = nz*N4
    s2 = M1*N4
    s3 = s1-s2

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Sigma1=True did not make any difference to the section above this
    # Only below do I start exploiting sigma1
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # If imposing sigma1, then all positive spanwise modes are also disregarded
    if sigma1:
        # mat1 and mat2 only need m<=0 modes if sigma1 is being imposed
        mat1 = mat1[:s2, :s2]
        mat2 = mat2[:s2, :s2]
        Lmat = np.zeros((L1*s2,L1*s2),dtype=complexType)
    else:
        Lmat = np.zeros((L1*s1,L1*s1),dtype=complexType)
    # Building only for non-positive streamwise modes. Wall effects only produce
    #   interactions in the spanwise modes
    
    def _foldMat(L0wavyMat):
        # Doing the folding now. 
        tmpMat = L0wavyMat[:s2, s2:]   # Factors of m>0 fields (u,v,w,p) in equations for m<= 0
        tmpMat = tmpMat.reshape((s2,s3//N4 , 4, N))  
        # Re-arranging columns into chunks of N

        tmpMat = tmpMat[:, ::-1]    # Re-ordering m-modes to go as (M,M-1,..,1) instead of (1,..,M)
        tmpMat[:,:,2] *= -1.
        # w has to be multiplied by an extra -1 than u,v,p
        tmpMat = tmpMat.reshape((s2,s3))
        return tmpMat
    
    for lp in range(L1):
        l = lp-L
      
        if sigma1:
            # Folding the matrix along spanwise modes to impose sigma1:
            #   u(l,-m) = (-1)^l u(l,m)
            #   v(l,-m) = (-1)^l v(l,m)
            #   w(l,-m) = (-1)^(l+1) w(l,m)
            #   p(l,-m) = (-1)^l p(l,m)
            
            # Assigning m<=0 part of L0wavy as is 
            Lmat[lp*s2:(lp+1)*s2, lp*s2:(lp+1)*s2] = L0wavy[:s2 , :s2]
            # First index says only equations for m<=0 are included
            # Second index says only contributions due to m<=0 of u,v,w,p are included (for now)
            
            Lmat[lp*s2:(lp+1)*s2, lp*s2:(lp+1)*s2-N4] += (-1.)**l * _foldMat(L0wavy) 
            # The -N4 in the second index is because the folding only happens for m !=0,
            #   The last N4 entries belong to m=0

            # Adding all the l-terms
            Lmat[lp*s2:(lp+1)*s2, lp*s2:(lp+1)*s2] += l* mat1 + l**2 * mat2
        else:
            # No folding in 'm' if sigma1 isn't imposed
            Lmat[lp*s1:(lp+1)*s1, lp*s1:(lp+1)*s1] = L0wavy
            # Adding all the l-terms
            Lmat[lp*s1:(lp+1)*s1, lp*s1:(lp+1)*s1] += l* mat1 + l**2 * mat2


    return Lmat



def jcbn(vf,Lmat=None,sigma1=False):
    if Lmat is None:
        warn('The Jacobian is added in-place to Lmat. Always supply Lmat. Returning.....')
        return
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
    iFun = lambda l,m: (l+L)*(vf.nz*4*N) + (m+M)*4*N
    
    G = Lmat
    assert (G.shape[0] == (L+1)*vf.nz*4*N) and (G.shape[1] == (L+1)*vf.nz*4*N)

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
    # In earlier implementations, G differed from the Jacobian of N only in terms of
    #                           d/du (u') being written as D instead of u''/ u' (which might produces NaNs)
    # This time, ignoring l > 0 in the state-vector causes greater deviations for G from the true Jacobian
    # For now, I shall ignore all this until I see convergence issues.
    #   Using a modified Jacobian isn't necessarily a bad thing anyway.

    # What the above comments mean is, I will split the below looping into two cases:
    # For a term in phi, u_l'm' * u_{l-l',m-m'}, if both l' and l-l' are > 0, it is accounted for  
    #   by populating the l-l',m-m' column-block with 2*u_{l',m'} (or vice-versa),
    #   so that 0.5* G * \chi returns phi_{l,m}
    # When both l' and l-l' are <=0, the corresponding term is split into 
    #   {0.5*u_{l',m'}} *u_{l-l',m-m'} + {0.5*u_{l-l',m-m'} } * u_{l',m'},
    #   and twice the factors in the curly braces go into the appropriate column-blocks of G
    # This is a really hacky way to do the whole thing, but if it works, that's all that matters.

    # Final piece of notation: 
    ia = 1.j*a; ib = 1.j*b

    for l in range(-L,1):
        for m in range(-M,M+1):
            # Index of first row in the block for equations for wavenumbers (l,m)
            rInd = iFun(l,m)
            # l1,m2 are the wavenumbers in phi^j_{lm}
            for lp in range(-L,l):
                # For these lp,   l-lp must be >0
                # So the u_{l-l', m-m'} must have l-l' >0,
                #   meaning they are populated as 2*u_{l-l',m-m'}
                for mp in range(-M,M+1):
                    cInd = iFun(lp,mp)
                    if (-L <= (l-lp) <= L):
                        li = l-lp+L # Array index for streamwise wavenumber l-lp

                        # First, all the terms not relating to wall-effects
                        if (-M <= (m-mp) <= M):
                            mi = m-mp+M # Array index for spanwise wavenumber m-mp
                            # phi^1_{l,m}: factors of terms with  u_{lp,mp}:
                            G[ rInd+0*N : rInd+1*N , cInd+0*N : cInd+1*N ] += \
                                    2.*( l*ia* diag(u[li, mi])  + v[li,mi].reshape((N,1)) *D + diag(w[li,mi])*mp*ib )
                            # phi^1_{l,m}: factors of terms with  v_{lp,mp}:
                            G[ rInd+0*N : rInd+1*N , cInd+1*N : cInd+2*N ] += 2.*(diag(uy[li, mi]))
                            # phi^1_{l,m}: factors of terms with  w_{lp,mp}:
                            G[ rInd+0*N : rInd+1*N , cInd+2*N : cInd+3*N ] += 2.*((m-mp)*ib*diag(u[li, mi]))

                            # phi^2_{l,m}: factors of terms with  v_{lp,mp}:
                            G[ rInd+1*N : rInd+2*N , cInd+1*N : cInd+2*N ] += \
                                    2.* (    lp*ia* diag(u[li, mi])  + v[li,mi].reshape((N,1)) *D + diag(w[li,mi])*mp*ib \
                                    + diag(vy[li,mi])   )
                            # phi^2_{l,m}: factors of terms with  u_{lp,mp}:
                            G[ rInd+1*N : rInd+2*N , cInd+0*N : cInd+1*N ] += 2.*  ((l-lp)*ia*diag(v[li, mi])  )
                            # phi^2_{l,m}: factors of terms with  w_{lp,mp}:
                            G[ rInd+1*N : rInd+2*N , cInd+2*N : cInd+3*N ] += 2.*  ((m-mp)*ib*diag(v[li, mi])  )

                            # phi^3_{l,m}: factors of terms with  w_{lp,mp}:
                            G[ rInd+2*N : rInd+3*N , cInd+2*N : cInd+3*N ] += \
                                    2.*(  lp*ia* diag(u[li, mi])  + v[li,mi].reshape((N,1)) *D + m*ib*diag(w[li,mi])  )
                            # phi^3_{l,m}: factors of terms with  v_{lp,mp}:
                            G[ rInd+2*N : rInd+3*N , cInd+1*N : cInd+2*N ] += 2.*(   diag(wy[li, mi]) )
                            # phi^3_{l,m}: factors of terms with  u_{lp,mp}:
                            G[ rInd+2*N : rInd+3*N , cInd+0*N : cInd+1*N ] += 2.*(   (l-lp)*ia*diag(w[li, mi])  )

                        # Now, the terms arising due to wall effects
                        # The interactions in l are unaffected since Tz only have e^iqb

                        for q in range(-q0,q0+1):

                            if (-M <= (m-mp+q) <= M):
                                mi = m-mp+q+M # Array index for spanwise wavenumber m-mp
                                # phi^1_{l,m}: factors of terms with u_{lp,mp}
                                G[ rInd+0*N : rInd+1*N , cInd+0*N : cInd+1*N ] += 2.*Tz[q0-q]* w[li,mi].reshape((N,1)) * D 
                                # phi^1_{l,m}: factors of terms with w_{lp,mp}
                                G[ rInd+0*N : rInd+1*N , cInd+2*N : cInd+3*N ] += 2.*Tz[q0-q]* diag(uy[li,mi]) 

                                # phi^2_{l,m}: factors of terms with v_{lp,mp}
                                G[ rInd+1*N : rInd+2*N , cInd+1*N : cInd+2*N ] += 2.*Tz[q0-q]* w[li,mi].reshape((N,1)) * D 
                                # phi^2_{l,m}: factors of terms with w_{lp,mp}
                                G[ rInd+1*N : rInd+2*N , cInd+2*N : cInd+3*N ] += 2.*Tz[q0-q]* diag(vy[li,mi]) 

                                # phi^3_{l,m}: factors of terms with w_{lp,mp}
                                G[ rInd+2*N : rInd+3*N , cInd+2*N : cInd+3*N ] += 2.*Tz[q0-q]* w[li,mi].reshape((N,1)) * D \
                                        +2.*Tz[q0-q]* diag(wy[li,mi]) 
 
            # Repeating the above for lp >= l, so that both lp and l-lp are <= 0                        
            for lp in range(l,1):
                for mp in range(-M,M+1):
                    cInd = iFun(lp,mp)
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
    
    N = vf.N; N4 = 4*N
    L = vf.nx//2; M = vf.nz//2
    L1 = L+1; nz=vf.nz
    if pf is None:
        pf = vf.getScalar().zero()
        
    J = linr(vf.flowDict, complexType=complexType)
    jcbn(vf, Lmat=J)
    F = (vf.residuals(pField=pf).appendField( vf.div() ) )[0,:L1].flatten()
    
    # Some simple checks
    assert (F.ndim == 1) and (J.ndim == 2)
    assert (J.shape[0] == L1*vf.nz*N4) and (F.size == L1*vf.nz*N4)
    
    
    # Imposing boundary conditions
    #   Unlike my MATLAB code, I impose all BCs at the end of the jacobian matrix
    #   So, if rect is True, the top rows of jacobian and residual remain unchanged
    #       they just have extra rows at the end
    #   If rect is False, I still add rows for BCs at the end, but I also remove the rows
    #       of the governing equations at the walls
    # Removing wall-equations if rect is False:
    # I don't have to remove wall-equations for divergence
    BCrows = N4*np.arange(L1*vf.nz).reshape((L1*vf.nz,1)) + np.array([0,N-1,N,2*N-1,2*N,3*N-1]).reshape((1,6))
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



def iterate(vf=None, pf=None,iterMax= 6, tol=5.0e-10,rcond=1.0e-07,complexType=np.complex,doLineSearch=True):
    if pf is None: pf = vf.getScalar().zero()
    resnormFun = lambda x: _residual(x).norm()

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
            
       
        L = x.nx//2; M = x.nz//2 
        dxff = x.zero()
        dx = dx.reshape((L+1,x.nz,4,x.N))
        dxff[0,:L+1] = dx[:]    # Modes l<=0 
        dxff[0,L+1:] = np.conj(dx[ L-1::-1, ::-1])  # Modes l>0

        # Real-valuedness
        dxff[0] = 0.5* (dxff[0] +  np.conj(dx[0, ::-1, ::-1]) )    

        # Velocity BC on the walls 
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
            return vf, pf, np.array(fnormArr), flg
        
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
    


def testExactRibletModule(L=4,M=7,N=35,epsArr=np.array([0.,0.05,0.02,0.03]),complexType=np.complex128):
    vf = h52ff('testFields/eq1.h5')
    pf = h52ff('testFields/pres_eq1.h5',pres=True)
    vf = vf.slice(L=L,M=M,N=N); pf = pf.slice(L=L,M=M,N=N)
    vf.flowDict.update({'epsArr':epsArr}); pf.flowDict.update({'epsArr':epsArr})
   
    # Verifying rib.linr() and rib.jcbn() when sigma1 is not imposed
    print('Verifying rib.linr(sigma1=False)...')
    Lmat = linr(vf.flowDict,complexType=complexType,sigma1=False)
    x = vf.appendField(pf)
   
    xArr = x.copyArray()
    linTerm = np.dot(Lmat, xArr[0,:x.nx//2+1].flatten())

    nex = 5

    vf1 = vf.slice(L=vf.nx//2+nex, M = vf.nz//2+nex)
    pf1 = pf.slice(L=vf.nx//2+nex, M = vf.nz//2+nex)
    
    linTermClass = -1./vf.flowDict['Re']*vf1.laplacian() + pf1.grad()
    linTermClass = linTermClass.appendField(vf1.div()).slice(L=vf.nx//2,M=vf.nz//2)

    Lmat[:] = 0.
    print('Verifying rib.jcbn(sigma1=False)...')
    jcbn(vf,Lmat=Lmat,sigma1=False)


    NLterm = 0.5*np.dot(Lmat, xArr[0,:x.nx//2+1].flatten())

    NLtermClassFine = vf1.convNL()
    NLtermClass = NLtermClassFine.slice(L=vf.nx//2, M=vf.nz//2).appendField(pf.zero())

    linTestResult =  chebnorm(linTerm - linTermClass[0,:x.nx//2+1].flatten(),x.N) <= tol
    NLtestResult = chebnorm(NLterm - NLtermClass[0,:x.nx//2+1].flatten(),x.N) <= tol




    if not linTestResult :
        print('Residual norm for linear without sigma1 is:',chebnorm(linTerm - linTermClass[0,:x.nx//2+1].flatten(),x.N))
    if not  NLtestResult:
        print('Residual norm for non-linear without sigma1 is:',chebnorm(NLterm - NLtermClass[0,:x.nx//2+1].flatten(),x.N))
    
    # Verifying linr() and jcbn() when sigma1 is imposed
    print('Verifying rib.linr(sigma1=True)...')
    Lmat1 = linr(vf.flowDict, complexType=complexType, sigma1=True)
    linTerm1 = np.dot(Lmat1, xArr[0,:x.nx//2+1, :x.nz//2+1].flatten()).flatten()

    s1 = x.nz*4*x.N; M1 = x.nz//2 +1; s2 = M1*4*x.N
    linTerm2 = linTerm.reshape((x.nx//2+1, s1))[:, :s2].flatten()
    # Removing the terms for m > 0 from linTerm calculated without sigma1 

    linTestResult1 =  chebnorm(linTerm1 - linTerm2, x.N) <= tol
    if not linTestResult1 :
        print('Residual norm for linear with sigma1 is:',chebnorm(linTerm1 - linTerm2,x.N))

    if linTestResult and NLtestResult and linTestResult1:
        print("Success for all tests!")
    
    return linTestResult, NLtestResult, linTestResult1






