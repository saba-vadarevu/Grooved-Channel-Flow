import numpy as np
from flowFieldWavy import *
import scipy.integrate as spint
import scipy as sp
from scipy.sparse import bmat, csc_matrix, diags, coo_matrix
import sys
import os
try:
    from scipy.optimize import least_squares
    importLstsq = True
except ImportError:
    importLstsq = False

tol = 1.0e-13
linTol = 1.0e-10

def dict2ff(flowDict):
    """ Returns a flowField with linear/quadratic profile at nd=0 depending on 'isPois'"""
    ff = flowFieldRiblet(flowDict=updateDict(flowDict,{'nd':4}))
    if flowDict['isPois']==0:
        uProfile = ff.y
    else:
        uProfile = 1. - ff.y**2
    ff[0,ff.nx//2,ff.nz//2,0] = uProfile
    return ff


class exactRiblet(object):
    """
    Defines equilibria problem/solver for riblet mounted Couette/channel flows
    Version number is defined as: vY.M.xy, 
        where Y=0 for 2017, M=0 for Jan, x (int) for new versions in the same month and y (char) for same day
    Attributes:
        x: flowFieldRiblet instance
        attributes: dict with keys
            sigma1, sigma3: PCF symmetries (bool)
            method: 'simple' for Newton's method with line search, 
                    and scipy.optimize.least_squares 's 'trf','dogbox', 'lm'
            iterMax: Max no of iterations for method 'simple'
                    used as max_nfev=iterMax+1 for 'trf','dogbox', and 
                            max_nfev=(_ff2symarr(x).size/2)*iterMax for 'lm'
            tol:    tolerance for iterations. Currently implemented only for 'simple'
            log:    file to direct sys.stdout to
            prefix: Prefix of file names to save hdf5 files to
            supplyJac:  If True, supply jacobian matrix to least_squares(),
                        Otherwise, let the algorithms run without a jacobian supplied

    """
    def __init__(self, x=None,fName=None,**kwargs):
        if x is None:
            x = loadh5(fName)
        assert isinstance(x,flowFieldRiblet) and (x.nd==4)
        if 'epsArr' in x.flowDict:
            x.flowDict['epsArr'] = np.array(x.flowDict['epsArr']).flatten()
        else:
            x.flowDict['epsArr'] = np.array([0., x.flowDict['eps']])
        self.x = x
        self.x0 = x.copy()
        self.version_str = u'exactRiblet v0.1.1a'
        self.attributes = {}
        self.attributes['sigma1'] = kwargs.pop('sigma1',True)
        self.attributes['sigma3'] = kwargs.pop('sigma3',True)
        self.attributes['method'] = kwargs.pop('method','simple')   # 'trf', 'dogbox','lm','simple'
        self.attributes['iterMax'] = kwargs.pop('iterMax',6)    # Max number of iterations
        self.attributes['tol'] = kwargs.pop('tol',1.0e-13)      
        self.attributes['log'] = kwargs.pop('log','outFile.txt')    # log file for output
        self.attributes['prefix'] = kwargs.pop('prefix','ribEq1')   # file name prefix for saving hdf5 files
        self.attributes['supplyJac'] = kwargs.pop('supplyJac',False) # Supply jacobian to trf or dogbox
        self.attributes['jacSparsity'] = kwargs.pop('jacSparsity',False) # Supply jacobian to trf or dogbox
        self.attributes['xtol'] = kwargs.pop('xtol',1.0e-13)    # terminate when ||dx|| < xtol*(xtol+||x||) 
        self.attributes['gtol'] = kwargs.pop('gtol',1.0e-13)   # terminate when ||g|| < gtol, g is the gradient (Jacobian) 
        self.attributes['ftol'] = kwargs.pop('ftol',1.0e-10) # terminate when ||dF|| < ftol*||F|| 
        self.attributes['tr_solver'] = kwargs.pop('tr_solver', None)
        self.attributes['version_str'] = self.version_str
        self.attributes['saveDir'] = kwargs.pop('saveDir',None)     
        # Directory to save intermediate solutions when running 'simple'. If None, don't save solutions

        self.attributes.update(kwargs)

    def printLog(self):
        """ Redirect stdout to log file self.attributes['log'], 
            and return a handle to original stdout.
            If self.attributes['log'] is None or 'terminal', don't do anything (return original stdout still)
            """
        bufferSize = 1		# Unbuffered printing to file
        if (self.attributes['log'] is None) or (self.attributes['log'] == 'terminal'):
            print('Not using a log file, printing to terminal')
            return sys.stdout
        workingDir = os.getcwd()
        if not workingDir.endswith('/'):
            workingDir = workingDir + '/'
        logName = self.attributes['log']
        outFile = open(workingDir+logName,'a',bufferSize)
        orig_stdout = sys.stdout
        sys.stdout = outFile
        sys.stderr = outFile
        return orig_stdout

    def _symarr2ff(self,xArr,weighted=True): 
        # Convert 1-d weighted, real, reduced array (reduced if sigma1 or sigma3) to flowFieldRiblet
        N = self.x.N; M = self.x.nz//2; L = self.x.nx//2
        NN = np.int(np.ceil(N/2.)); Nn = np.int(np.floor(N/2.))
        sigma1 = self.attributes['sigma1']; sigma3 = self.attributes['sigma3']
        
        if sigma3:
            if sigma1: nz1 = M+1
            else: nz1 = 2*M+1
            coeffArr = -M + np.arange(nz1).reshape((1,nz1, 1,1,1))
            coeffArr = np.tile(coeffArr, (1,1,4,2,1))
            coeffArr[:,:,:,0] += 1  # For u,v,w, real part multiplies (-1)^(m+1), imaginary multiplies (-1)^m
            coeffArr[:,:,3] += 1    # For p, there's an extra -1 factor compared to u,v,w
            xArr = xArr.reshape((L+1, nz1,4,2,NN))
            xArr = np.concatenate((xArr, np.zeros((L+1,nz1,4,2,Nn),dtype=np.float)),axis=-1)
            xArr[:,:,:,:,:NN-1:-1] = ((-1.)**coeffArr)*xArr[:,:,:,:,:Nn]

        if sigma1:
            xArr = xArr.reshape((L+1, M+1,4,2*N))
            xArrNew = np.zeros((L+1, 2*M+1, 4, 2*N),dtype=np.float)
            xArrNew[:,:M+1] = xArr[:]
            # Assigning coefficients for m > 0:
            # Idea here is to get u_{l,m} = (-1)^l C.u_{l,-m}, with C = (1,1,-1,1) for u,v,w,p 
            compArr = np.array([1., 1., -1., 1.]).reshape((1,1,4,1))
            lArr = np.arange(-L, 1).reshape(( L+1 , 1,1,1)) 
            # l modes go from -L to 0
            # Assigning modes m= M to m=1 using modes m=-M to m=-1:
            xArrNew[:, :M:-1] =  (-1.)**lArr * compArr * xArr[:, :M]
        else:
            xArrNew = xArr
            
        return realField2ff(arr=xArrNew,axis='x', flowDict=self.x.flowDict,weighted=weighted,weights=self.x.w,cls='riblet')

    def _ff2symarr(self,ff,weighted=True):
        # Return weighted, real, reduced (if sigma1 or sigma3) 1d-array from flowfield
        ffArr = ff.realField(axis='x',weighted=weighted)
        L = self.x.nx//2; M = self.x.nz//2; N = self.x.N
        sigma1 = self.attributes['sigma1']; sigma3 = self.attributes['sigma3']
        if sigma1:
            ffArr = ffArr.reshape((L+1, 2*M+1, 4, 2*N))
            ffArr = ffArr[:,:M+1]
        if sigma3:
            NN = np.int( np.ceil(N/2.))
            ffArr = ffArr.reshape((L+1,M+1,4,2,N))
            ffArr = ffArr[:,:,:,:,:NN]            
        return ffArr.flatten()

    def linr(self): 
        """Returns matrix representing the linear operator for the equilibria/TWS for riblet case
        Inputs:
            self
        IMPORTANT: realValued is now always True. DO NOT SUPPLY AS A KEYWORD ARGUMENT
        Outputs:
            Lmat:   Matrix representing linear terms for the complete state-vectors"""
        flowDict = self.x.flowDict
        epsArr = flowDict['epsArr']
        sigma1 = self.attributes['sigma1']
        sigma3 = self.attributes['sigma3']


        L = flowDict['L']; M = flowDict['M']; nx = self.x.nx; nz = self.x.nz
        # nx1 = int(2*(L+1))  # For realvaluedness, lose L>0 modes, but double the number of coefficients
        nx1 = L+1
        a = flowDict['alpha']; a2 = a**2
        b = flowDict['beta']; b2 = b**2
        Re = flowDict['Re']

        N  = int(self.x.N); N4 = 4*N
        D = self.x.D; D2 = self.x.D2

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

        Tz,Tzz, Tz2 = Tderivatives(self.x.flowDict)

        q0 = epsArr.size-1
        # -1 because epsArr includes zeroth mode

        # Number of columns is increased by 4*q0 because wall effects produce
        #   interactions all the way from -M-2*q0 to M+2*q0
        #   I prefer to build the matrix with these included, and then truncate to -M to M
        L0wavy = np.zeros((nz*N4, (nz+4*q0)*N4), dtype=np.complex)
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

        mat1 = np.zeros((nz1*N4,nz1*N4),dtype=np.complex); mat2 = mat1.copy()
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
       

        # Important: s1, s2, and s3 are defined based on N and not NN=ceil(N/2)
        #   This becomes relevant only in the last step where I fold LtempReal and assign to Lmat
        if sigma3:
            NN = np.int(np.ceil(N/2.))
            Nn = np.int(np.floor(N/2.))
            NN4 = 4*NN
            s11 = nz1*NN4   # Count rows with s11 instead of s1 if (sigma3 and realValued)
            # Define a coefficient matrix to use later for folding
            coeffArr = -M + np.arange(M+1).reshape((1,M+1, 1,1,1))
            coeffArr = np.tile(coeffArr, (1,1,4,2,1))
            coeffArr[:,:,:,0] += 1  # For u,v,w, real part multiplies (-1)^(m+1), imaginary multiplies (-1)^m
            coeffArr[:,:,3] += 1    # For p, there's an extra -1 factor compared to u,v,w
        else: 
            s11 = s1
            NN = N; Nn = N; NN4 = 4*N
        
        Lmat = np.zeros(((L+1)*nz1*2*NN4,(L+1)*nz1*2*NN4),dtype=np.float)
        # If imposing sigma1, build for only m <= 0 
        # Because of realValued, build for only l<=0, but with real and imaginary parts separated.
        #   The number of variables remains about the same, and so does the number of equations.
        #   But the size of each element halves, this reduces memory usage.
        # If imposing both sigma3 (=sigma1 and sigma2) also, build for NN=ceil(N/2),
        #   else, build for NN = N.
        #   In this case, we exploit even/odd nature of real and imaginary parts of variables

        Ltemp = np.zeros((s1, s1),dtype=np.complex)
        LtempReal = np.zeros((2*s1, 2*s1), dtype=np.float)
        for l in range(-L,1):
            lp = l+L
            Ltemp[:] = 0.
            # Using s1 instead of nz*N4. If sigma1 is False, there is no difference
            # Matrix from laminar case where l=0
            Ltemp[:] = L0wavy[:s1, :s1]
            # Adding all the l-terms
            Ltemp[:] += l* mat1 + l**2 * mat2
            
            if sigma1:
                # Adding L0wavyTemp:
                Ltemp[:,:s2] += (-1.)**l  * L0wavyTemp

            # Because of realvaluedness, all modes are split as real and imaginary
            # So, each block (for each Fourier mode) is now of size (4x2xN)x(4x2xN),
            #   u_{lm} is of size 2N now, with Real(u_{lm}) first and Imag(u_{lm}) following
            # Az = (A_r z_r - A_i z_i) + i (A_i z_r + A_r z_i)
            LtempReal[:]  = 0.
            LtempReal = LtempReal.reshape((s1//N, 2*N, s1//N, 2*N))
            
            Ltemp = Ltemp.reshape((s1//N,N, s1//N,N))
            LtempReal[:, :N, :, :N] = np.real(Ltemp)        # First term in Az
            LtempReal[:, :N, :, N:] = -np.imag(Ltemp)    # Second term in Az
            LtempReal[:, N:, :, :N] = np.imag(Ltemp)        # Third term in Az
            LtempReal[:, N:, :, N:] = np.real(Ltemp)        # Fourth term in Az

            Ltemp = Ltemp.reshape((s1,s1))
            LtempReal = LtempReal.reshape((2*s1, 2*s1))
            
            if not sigma3:
                LReal = LtempReal
            else:
                # LtempReal is all sorted out for sigma1 and realValued. If sigma3 is True, use it now
                # Let's call the final version LReal
                #   First, reshape columns to separate by 'm', 'nd', real/imag, and N
                LReal = LtempReal.copy().reshape((2*s1, nz1, 4, 2, N))
                # Now, reshape rows to separate by real/imag and N
                LReal = LReal.reshape((s1//N, 2, N, nz1, 4, 2, N)) 
                # Get rid of all the equations for y < 0, i.e., drop rows with N>= NN
                LReal = LReal[:,:, :NN].reshape((2*s11, nz1,4, 2, N)) # Rows are now taken care of
                # Now, separating nodes for + and -
                LReal1 = LReal[:,:,:,:,:NN].copy(); LReal2 = LReal[:,:,:,:,NN:].copy()
                LReal = LReal1
                # Now, I just have to fold LReal2 onto LReal
                # When folding the columns, 
                #   I need to account for even and odd functions. coeffArr does this

                LReal[:, :,:,:, :Nn] += ((-1.)**coeffArr) * LReal2[:,:,:,:, ::-1]
                LReal = LReal.reshape((2*s11, 2*s11))


                
            Lmat[lp*2*s11:(lp+1)*2*s11, lp*2*s11:(lp+1)*2*s11] = LReal

        return Lmat



    def jcbn(self,ff,Lmat=None):
        if Lmat is None:
            raise RuntimeError('jcbn returns None, the matrix has to be supplied so it can be modified in-place')
        sigma1 = self.attributes['sigma1']; sigma3 = self.attributes['sigma3']
        vf = ff.slice(nd=[0,1,2]); pf = ff.getScalar(nd=3)


        if sigma1: nz1 = vf.nz//2 + 1
        else: nz1 = vf.nz

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
        iFun0 = lambda l,m : (l+L)*vf.nz*4*N + (m+M)*4*N 
        # When sigma1 is imposed, I build a G for all l',m' (including l'>0,m'>0). 
        #   I need iFun0 for this case

        if sigma3:
            NN = np.int(np.ceil(N/2.))
            Nn = np.int(np.floor(N/2.))
            NN4 = 4*NN
            # Define a coefficient matrix to use later for folding
            coeffArr = -M + np.arange(M+1).reshape((1,M+1, 1,1,1))
            coeffArr = np.tile(coeffArr, (1,1,4,2,1))
            coeffArr[:,:,:,0] += 1  # For u,v,w, real part multiplies (-1)^(m+1), imaginary multiplies (-1)^m
            coeffArr[:,:,3] += 1    # For p, there's an extra -1 factor compared to u,v,w
            coeffArr = coeffArr.reshape((1,1,M+1,4,2,1))  # The first axis is for equations
        else: 
            NN = N; Nn = N; NN4 = 4*N
        Gmat = Lmat
        #assert (Gmat.shape[0] == nx1*nz1*4*N) and (Gmat.shape[1] == nx1*nz1*4*N)

        # I will be using the functions np.diag() and np.dot() quite often, so,
        diag = np.diag; dot = np.dot

        # To calculate Gmat:
        #   Go through each row-column-block of the matrix and figure out which 
        #           u_{l,m} goes there (along with factors for derivatives)
        # This is the notation I shall use:
        # l , m represent the mode for which the equation is written
        #   I use phi_lm to represent the convection term in the NSE for mode (l,m)
        #   phi^1 is streamwise convection term: phi^1 = u d_x u + v d_y u + w d_z u
        #   phi^2 is wall-normal, phi^3 is spanwise
        # So, the wave-triads go as { (l', m'), (l,m), (l-l',m-m')} (for terms not involving wall-effects)
        #   u_{l-l',m-m'} is populated in row-block corresponding to phi_{l,m} in column-block for u_{l',m'}

        # Strictly speaking, what I'm building is not the Jacobian G
        # I'm building G such that N(\chi) = 0.5 * G * \chi, where \chi is the state-vector
        # G differs from the Jacobian of N only in terms of
        #                           d/du (u') being written as D instead of u''/ u' (which might produces NaNs)

        # Final piece of notation: 
        ia = 1.j*a; ib = 1.j*b

        if sigma1: M1 = 1
        else: M1 = M+1
        # If sigma1, write equations only until m < 1, else, until m<M+1

            
        # The convection term has this form:
        # phi^{lm}_{1,2,3} = \sum_lp \sum_mp  u_{l-lp,m-mp} u_{lp,mp}  , 
        #    disregarding the surface influence terms. 
        # Gmat is the actual non-linear jacobian
        # G is a temporary matrix created for each 'lp' in equations for each (l,m) in the Jacobian,
        # If sigma1 is to be imposed,
        #    G is folded and multiplied with (-1)**lp before being assigned to the
        #        corresponding rows and columns in Gmat
        # Otherwise, G is added as is
        G = np.zeros((4*N, vf.nx*vf.nz*4*N), dtype=np.complex)

        GReal = np.zeros((8*N, L+1,vf.nz,4,2*N), dtype=np.float)

        for l in range(-L,1):
            for m in range(-M,M1):
                # l1,m2 are the wavenumbers in phi^j_{lm}
                G[:] = 0.    # Getting rid of G from previous 
                rInd = 0     # We're only writing equations for one l,m
                # G contains the part of the Jacobian that multiplies all modes  
                #    in the equations for (l,m)
                # I will define this without assuming symmetries
                # Before adding to Gmat, I will fold G on itself according to sigma1, sigma2

                for lp in range(-L,L+1):
                    for mp in range(-M,M+1):
                        cInd = iFun0(lp,mp)
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
                lArr0 = np.arange(-L,L+1).reshape((1,vf.nx, 1,1,1))
                mArr = np.arange(-M,M+1).reshape((1,1, vf.nz,1,1))
                Gnew = G.reshape((4*N,vf.nx, vf.nz, 4,N))
                Gtemp = Gnew.reshape((4,N,vf.nx, vf.nz, 4,N))
                # With realValued=True, all modes are split as real and imaginary
                # So, each block (for each Fourier mode) is now of size (4x2xN)x(4x2xN),
                #   u_{lm} is of size 2N now, with Real(u_{lm}) first and Imag(u_{lm}) following
                # Az+B*conj(z) = ((A_r+B_r) z_r +(B_i- A_i) z_i) + i ((A_i+B_i) z_r + (A_r-B_r) z_i)
                GReal[:]  = 0.
                GReal = GReal.reshape((4, 2*N, L+1, vf.nz, 4, 2*N))

                # Assign terms due to Az = (A_r z_r - A_i z_i) + i (A_i z_r + A_r z_i)
                #   for l <= 0 and all m 
                GReal[:, :N,   :,:,:, :N] =  np.real(Gtemp[:,:   ,:L+1])        # First term in Az
                GReal[:, :N,   :,:,:, N:] = -np.imag(Gtemp[:,:   ,:L+1])    # Second term in Az
                GReal[:, N:,   :,:,:, :N] =  np.imag(Gtemp[:,:   ,:L+1])        # Third term in Az
                GReal[:, N:,   :,:,:, N:] =  np.real(Gtemp[:,:   ,:L+1])        # Fourth term in Az

                # Add terms due to B*conj(z) = (B_r z_r + B_i z_i) + i (B_i z_r - B_r z_i)
                #   for l > 0 and all m.
                # However, m need to be flipped so that (l,m) lines up with (-l,-m)
                GReal[:, :N,   :L,:,:, :N] +=  np.real(Gtemp[:,:   ,:L:-1,::-1])        # First term in Az
                GReal[:, :N,   :L,:,:, N:] +=  np.imag(Gtemp[:,:   ,:L:-1,::-1])    # Second term in Az
                GReal[:, N:,   :L,:,:, :N] +=  np.imag(Gtemp[:,:   ,:L:-1,::-1])        # Third term in Az
                GReal[:, N:,   :L,:,:, N:] += -np.real(Gtemp[:,:   ,:L:-1,::-1])        # Fourth term in Az

                Gnew = GReal.reshape((8*N, L+1, vf.nz, 4,2*N))
                lArr = lArr0[:,:L]

                if sigma1:
                    # The relation between u_{lm} and u_{l,-m} remains unchaged with imposition of realValuedness
                    #   because there's no 1.j in the eqns u_{l,m} = (-1)^l u_{l,-m} etc..
                    compArr = np.array([1., 1., -1., 1.]).reshape((1,1,1,4,1))
                    Gtemp  = Gnew[:,:, :M+1]      # Copying G as is for m <= 0
                    Gtemp[:, :, :M] += ((-1.)**lArr0[:, :Gnew.shape[1]]) * compArr * Gnew[:,:,:M:-1]
                    # For m>0 in G (or Gnew), array is reordered so m lines up with -m,
                    #   u,v,p are multiplied by 1, w by -1,
                    #    and columns for 'l' are multiplied with -1^l
                    Gnew = Gtemp
                    

                if sigma3:
                    # Gnew is all sorted out for sigma1 and realValued. If sigma3 is True, use it now
                    # Let's define a temporary array from Gnew
                    #   First, reshape columns to separate by 'l','m', 'nd', real/imag, and N
                    GnewReal = Gnew.copy()
                    GnewReal = GnewReal.reshape((8*N, L+1, nz1, 4, 2, N))
                    # Now, reshape rows to separate by real/imag and N
                    GnewReal = GnewReal.reshape((4,2, N, L+1, nz1, 4, 2, N)) 
                    # Get rid of all the equations for y < 0, i.e., drop rows with N>= NN
                    GnewReal = GnewReal[:,:, :NN].reshape((8*NN,L+1, nz1,4, 2, N)) # Rows are now taken care of
                    # Now, separating nodes for + and -
                    GnewReal1 = GnewReal[:,:, :,:,:,:NN].copy(); GnewReal2 = GnewReal[:,:,:,:,:,NN:].copy()
                    Gnew = GnewReal1
                    # Now, I just have to fold GnewReal2 onto GnewReal
                    # When folding the columns, 
                    #   I need to account for even and odd functions. coeffArr does this

                    Gnew[:, :,:,:,:, :Nn] += ((-1.)**coeffArr) * GnewReal2[:,:,:,:,:, ::-1]

                Gnew = Gnew.reshape((8*NN, (L+1)*nz1*8*NN))
                rInd = (l+L)*(nz1*8*NN) + (m+M)*8*NN
                Gmat[rInd: rInd+8*NN, : ] += Gnew
        return  

    def _jacSparsity(self):
        """ Returns the sparsity structure of the Jacobian matrix, as type np.int8"""
        L = self.x.nx//2; M = self.x.nz//2; N = self.x.N
        sigma1 = self.attributes['sigma1']; sigma3 = self.attributes['sigma3']
        if sigma1: M1 = 1
        else: M1 = M+1
        if sigma3: NN = np.int(np.ceil(N/2.))
        else: NN = N

        epsArr = self.x.flowDict['epsArr']
        if np.linalg.norm(epsArr)> 0.: q = np.where(epsArr>0.)[-1]
        else: q = 0

        
        oneMat = np.ones((NN,NN),dtype=np.uint8)
        oneArr = np.ones(NN,dtype=np.uint8)
        oneMat0 = oneMat.copy(); oneArr0 = oneArr.copy()

        # I impose BCs at nodes y=+-1, so the sparsity structure must reflect this
        if not sigma3:
            oneMat0[[0,-1],:] = 0
            oneArr0[[0,-1]] = 0
        else:
            oneMat0[0,:] = 0
            oneArr0[0] = 0

        Den = csc_matrix(oneMat0,dtype=np.int8)
        Spa = diags(oneArr0,dtype=np.int8)
        Den_BC = csc_matrix(oneMat,dtype=np.int8)   # For div.u, the rows at y=+-1 aren't set to zero
        Spa_BC = diags(oneArr,dtype=np.int8)
        Z = diags(np.zeros(NN),dtype=np.int8)
        Z2 = diags(np.zeros(2*NN), dtype=np.int8)

        Den0 = bmat([[Den, None],
                      [None, Den]])
        Den1 = bmat([[None, Den],
                    [Den, None]])
        Den2 = bmat([[Den, Den],
                    [Den, Den]])
        Spa0 = bmat([[Spa, None],
                     [None, Spa]])
        Spa1 = bmat([[None, Spa],
                     [Spa, None]])
        Spa2 = bmat([[Spa, Spa],
                     [Spa, Spa]])
       
        # To be used in J0, J1 for divergence equation
        Spa1_BC = bmat([[None, Spa_BC],
                        [Spa_BC, None]])
        Den0_BC = bmat([[Den_BC, None],
                        [None, Den_BC]])
        Den1_BC = bmat([[None, Den_BC],
                        [Den_BC, None]])

        J4 = bmat([[Den2, None, Spa2, Z2  ],
                   [None, Den2, Spa2, None],
                   [None, None, Den2, None],
                   [Z2, None, None, None]])
        J3 = bmat([[Den2, Spa2, Spa2, Z2  ],
                   [Spa2, Den2, Spa2, None],
                   [Spa2, Spa2, Den2, None],
                   [Z2  , None, None, None]])
        J2 = J3
        J1 = bmat([[Den2, Spa2, Spa2, None],
                   [Spa2, Den2, Spa2, None],
                   [Spa2, Spa2, Den2, Den1],
                   [None, None, Den1_BC, None]])
        J0 = bmat([[Den2, Spa2, Spa2, Spa1],
                   [Spa2, Den2, Spa2, Spa0],
                   [Spa2, Spa2, Den2, Den1],
                   [Spa1_BC, Den0_BC, Den1_BC, None]])

        def _returnJac(l,m,lp,mp):
            ld = np.abs(l-lp)
            md = np.abs(m-mp)
            if not ((ld <= L) and (md <= M+q)):
                return None
            elif not (md <= M):
                return J4
            elif not ((ld == 0) and (md < 2*q)):
                return J3
            elif not (md < q):
                return J2
            elif not (md == 0):
                return J1
            else:
                return J0
        def _cIndm(mpp):
            if (sigma1) and (mpp > 0):
                # Add to columns corresponding to (lp,-|mp|)
                return (-mpp+M)*8*NN
            else:
                return (mpp+M)*8*NN
                
        rows = np.array([]); cols = np.array([]); data=np.array([])
        for l in range(-L,1):
            for m in range(-M,M1):
                for lp in range(-L,L+1):
                    for mp in range(-M,M+1):
                        Jmat = _returnJac(l,m,lp,mp)
                        rInd = (l+L)*(M+M1)*8*NN + (m+M)*8*NN
                        # Because of the folding of the Jacobian to impose symmetries/realValuedness
                        #   (see self.linr() and self.jcbn())
                        # I need to make a few changes to cInd before assigning to the Jacobian sparsity structure
                        if lp > 0:
                            # Add to columns corresponding to (-lp,-mp)
                            cInd = (-lp+L)*(M+M1)*8*NN + _cIndm(-mp)
                        else:
                            cInd = (lp+L)*(M+M1)*8*NN + _cIndm(mp) 

                        if Jmat is not None:
                            #print(l,m,lp,mp)
                            rows = np.concatenate((rows, rInd+Jmat.row))
                            cols = np.concatenate((cols, cInd+Jmat.col))
                            data = np.concatenate((data, Jmat.data))

        rows = rows.astype(np.uint32)
        cols = cols.astype(np.uint32)
        data = data.astype(np.int8)
        
        # Finally, imposing BCs  
        BCrows0 = 8*NN*np.arange((L+1)*(M+M1)).reshape(((L+1)*(M+M1),1))
        if sigma3:
            # BCs on real and imag, but only at y=1, because y=-1 isn't part of the vector
            BCrows1 = NN*np.arange(6).reshape((1,6))
        else:
            BCrows1 = N*np.arange(6).reshape((6,1)) + np.array([0,N-1]).reshape((1,2))
            BCrows1 = BCrows1.reshape((1,12))
        BCrows = BCrows0 + BCrows1
        BCrows = BCrows.flatten().astype(np.uint32)

        rows = np.concatenate((rows, BCrows)).astype(np.uint32)
        cols = np.concatenate((cols, BCrows)).astype(np.uint32)
        data = np.concatenate((data, np.ones(BCrows.size,dtype=np.int8))).astype(np.int8)

        Jsparsity = coo_matrix((data, (rows, cols)),dtype=np.int8)
        #Jsparsity = bmat([[Jsparsity,None],
        #                  [None, diags(np.zeros((2*NN)),dtype=np.int8)]] )

        Jsparsity.data[Jsparsity.data>1] = 1
        
        ind0 = (L+1)*(M+M1)*8*NN - Jsparsity.shape[0]
        ind1 = (L+1)*(M+M1)*8*NN - Jsparsity.shape[1]
        if not ((ind0==0) and (ind1==0)):
            print("Something's off. Jsparsity is missing %d rows and %d columns."%(ind0,ind1))
            zeroMat = np.zeros((ind0,ind1),dtype=np.int8)
            Jsparsity = bmat([[Jsparsity,None],
                              [None,  zeroMat]],dtype=np.int8)
            print("Fixed this by adding a dense zero matrix for now.")
        
                    
        return Jsparsity



    def _residual(self):
        return (self.x.slice(nd=[0,1,2]).residuals(pField=x.getScalar(nd=3)).appendField( self.x.slice(nd=[0,1,2]).div() ) )


    def makeSystem(self,ff):
        """
        makeSystem(self)
        Create functions for residual and Jacobian matrices, Boundary conditions and symmetries are imposed here. 
        Outputs:
            residualBC: 1-d array
            jacobianBC: 2-d array"""
        sigma1= self.attributes['sigma1']; sigma3 = self.attributes['sigma3']
        vf = ff.slice(nd=[0,1,2]); pf = ff.getScalar(nd=3)
        N = vf.N; N4 = 4*N
        L = vf.nx//2; M = vf.nz//2
        
        nz1 = vf.nz
        if sigma1: nz1 = M+1
        nx1 = L+1

        J = self.linr()  # Get Lmat
        self.jcbn(ff,Lmat=J)    # Add non-linear jacobian to Lmat
        
        if sigma3: NN = np.int(np.ceil(N/2.))
        else: NN = N
        
        F = self._ff2symarr(vf.residuals(pField=pf).appendField( vf.div() ) )

        # Some simple checks
        assert (F.ndim == 1) and (J.ndim == 2)
        # When realValued, each mode is split into real and imaginary parts, 
        #   so BCs on each Fourier mode block to be applied on 0,N-1,N,...,5*N, 6*N-1
        BCrows0 = 8*NN*np.arange((L+1)*nz1).reshape(((L+1)*nz1,1))
        if sigma3:
            # BCs on real and imag, but only at y=1, because y=-1 isn't part of the vector
            BCrows1 = NN*np.arange(6).reshape((1,6))
        else:
            BCrows1 = N*np.arange(6).reshape((6,1)) + np.array([0,N-1]).reshape((1,2))
            BCrows1 = BCrows1.reshape((1,12))
        BCrows = BCrows0 + BCrows1
        


        BCrows = BCrows.flatten()



        J[BCrows,:] = 0.
        J[BCrows,BCrows] = 1.
        # Equations on boundary nodes now read 1*u_{lm} = .. , 1*v_{lm} = .., 1*w_{lm} = ..
        # The RHS for the equations is set in residualBC below
        F[BCrows] = 0.
        # The residuals are zero because the correction, dx in J*dx = -F, should not change velocity BCs

        return J, F 


    def lineSearch(self,normFun,x0,dx,arr=None):
        print("Beginning line search.... Initial residual norm is ",normFun(x0))
        if arr is None:
            arr = np.arange(-0.5,2.1,0.1)
            arr = np.concatenate((arr, np.arange(2.5,10.1,0.5), np.arange(15.,101.,5.)))
        else:
            arr = np.array(arr).flatten()

        normArr = np.ones(arr.size)
        for k in range(arr.size):
            q = arr[k]
            normArr[k] = normFun(x0+q*dx)

        kMin = np.argmin(normArr)
        normMin = normArr[kMin]
        qMin = arr[kMin]

       
        for kBinary in range(25):
            if arr[0] < arr[kMin] < arr[-1]:
                arr = np.arange( arr[kMin-1], arr[kMin+1], (arr[kMin+1] - arr[kMin-1])/4.)
                normArr = np.ones(arr.size)
                for k in range(arr.size):
                    q = arr[k]
                    normArr[k] = normFun(x0+q*dx)

                kMin = np.argmin(normArr)
                normMin = normArr[kMin]
                qMin = arr[kMin]

            else:
                break

        print("Minimal norm is obtained for q in q*dx of %.2g, producing norm of %.3g"%(qMin, normMin))

        oldNorm = normFun(x0); newNorm = normFun(x0+qMin*dx)
        if newNorm > oldNorm:
            print("New norm (%.3g) is greater than the old norm (%.3g) for some weird reason"%(newNorm,oldNorm))

        return x0+qMin*dx



    def iterate(self):
        """ Iterate using a method specified by 'method' in self.attributes. 
            If method is not one of 'trf', 'dogbox', 'lm', use Newton search with full-rank inversion and linesearch.
            IMPORTANT: keyword argument realValued is now obsolete, since it is always imposed.
        """
        # self.x0 is the first field used to initiate the solver. 
        # self.x is the current flowfield. self.x need not be the same as self.x0
        # For all iterations, use self.x. self.x0 is only for reference
        orig_stdout = self.printLog()
        x = self.x
        rcond = self.attributes.get('rcond',1.0e-06)
        method = self.attributes['method']
        sigma1 = self.attributes['sigma1']; sigma3 = self.attributes['sigma3']
        
        # Ensure initial flowfield has the symmetries that are being imposed
        self.x.imposeSymms(sigma1=sigma1, sigma3=sigma3)
        

        if method in ("trf","dogbox","lm"):
            trustRegion= True
        elif method != 'simple':
            trustReion = False
            warn('Invalid method supplied in kwargs/argparse. Using Newton iterations with full-rank inversion and line search')
        else:
            trustRegion = False

        if trustRegion and importLstsq:
        # If scipy.optimize.least_squares cannot be imported, run simple newton search
            runTrustRegion=True
        else:
            runTrustRegion=False
        
        N = x.N; L = x.nx//2; M=x.nz//2
        w = x.w 
        if sigma3: 
            NN = np.int(np.ceil(N/2.)); Nn = np.int(np.floor(N/2.))
        else: NN = N
        w = w[:NN]  # Because, if foldD, we are only evaluating residuals on y >= 0
        q = np.sqrt(w)
        qinv = 1./q
        Q = np.diag(q)
        Qinv = np.diag(qinv)
        
        def __weightJ(Jacobian):
            for k1 in range(Jacobian.shape[0]//NN):
                for k2 in range(Jacobian.shape[1]//NN):
                    Jacobian[k1*NN:(k1+1)*NN, k2*NN:(k2+1)*NN] = np.dot( Q, np.dot(Jacobian[k1*NN:(k1+1)*NN, k2*NN:(k2+1)*NN], Qinv) )
            return

        def __weightF(residual):
            for k in range(residual.size//NN):
                residual[k*NN: (k+1)*NN] = np.dot(Q, residual[k*NN:(k+1)*NN])
            return

        def __unweightdx(deltaX):
            for k in range(deltaX.size//NN):
                deltaX[k*NN:(k+1)*NN] = np.dot(Qinv, deltaX[k*NN:(k+1)*NN])
            return

        if sigma3:
            coeffArr = -M + np.arange(M+1).reshape((1,M+1, 1,1,1))
            coeffArr = np.tile(coeffArr, (1,1,4,2,1))
            coeffArr[:,:,:,0] += 1  # For u,v,w, real part multiplies (-1)^(m+1), imaginary multiplies (-1)^m
            coeffArr[:,:,3] += 1    # For p, there's an extra -1 factor compared to u,v,w

        
        resnormFun = lambda ff: ff.residuals().appendField(ff.div()).norm() 

        # If eps2 != 0, force sigma3 to be false even if it is supplied as True
        epsArr = x.flowDict['epsArr']
        if (epsArr.size> 2) and abs( epsArr[2])> tol:
            sigma3 = False

        self.x.setWallVel()
        self.x.imposeSymms(sigma1=sigma1,sigma3=sigma3)
        # Impose real-valuedness (by default), and sigma1, sigma3 if needed 

        flowDict = x.flowDict.copy()

        weights = x.w 
        def __resFunReal(xArr):
            ff = self._symarr2ff(xArr)
            res = ff.residuals()
            res = res.appendField(ff.div())

            # dogbox doesn't do well with rank-deficient Jacobians
            # Rank deficiency in the problem is mainly due to the zeroth pressure mode
            # So, to set  p_00 = 0 at both walls, instead of adding extra equations,
            #       I'm adding these to the divergence for the last Fourier modes at the walls
            res[0,0,0,3,0]  += np.abs(ff[0,ff.nx//2, ff.nz//2,3,0])
            res[0,0,0,3,-1] += np.abs(ff[0,ff.nx//2, ff.nz//2,3,-1])
            res[0,-1,-1,3,0]  += np.abs(ff[0,ff.nx//2, ff.nz//2,3,0])
            res[0,-1,-1,3,-1] += np.abs(ff[0,ff.nx//2, ff.nz//2,3,-1])
            # I'll try adding extra equations if this doesn't work out
            resArr = self._ff2symarr(res)

            return resArr.flatten()

        fnormArr=[]
        flg = 0
        resnorm0 = resnormFun(x)
        if resnorm0 <= tol:
            print("Initial flowfield has zero residual norm (%.3g). Returning..."%(resnorm0))
            sys.stdout = orig_stdout
            return x,np.array([resnorm0]),flg
        else:
            print("Initial residual norm is %.3g"%(resnorm0))

        # least_squares did not offer a callback function to save intermediate solutions,
        #   I defined this manually in scipy's libraries. This must be done when running on other systems
        # The following callback functions saves intermediate flowfields.
        if (self.attributes.get('saveDir',None) is not None):
            global saveCounter
            saveCounter = self.x.flowDict.get('counter',0)
            savePath = self.attributes['saveDir']
            def callbackFun(ffArr,_savePath,_fNamePrefix):
                ff = self._symarr2ff(ffArr)
                globals()["saveCounter"] += 1
                fNPrefix = _fNamePrefix+'_'+str(saveCounter)+'_'
                ff.saveh5(prefix=_savePath, fNamePrefix=fNPrefix)
            callback = lambda ffArr: callbackFun(ffArr, savePath, self.attributes['prefix'])
            print('Trying to save to:',savePath,', with file name prefix:', self.attributes['prefix'])
            # callback(self._ff2symarr(self.x))
        else:
            callback = None

        
        print('Starting iterations...............')
        if runTrustRegion:

            def jacFun(ffArr):
                # Return Jacobian matrix for a given state-vector
                ff = self._symarr2ff(ffArr)
                J, F = self.makeSystem(ff)
                return J
                    
            if self.attributes.get('supplyJac',True):
                jac = jacFun
            else:
                jac = '2-point'

            max_nfev = self.attributes['iterMax']
            method = self.attributes['method']
            xtol = self.attributes['xtol']
            ftol = self.attributes['ftol']
            gtol = self.attributes['gtol']
            if method=='lm':
                bounds = (-np.inf,np.inf)
                max_nfev *= (self._ff2symarr(self.x).size//2)
            else:
                bounds = (-1.,1.)
                if method =='trf': max_nfev += 1
                else: max_nfev = np.int(5*max_nfev)
       
            if self.attributes['jacSparsity']:
                jacSparsityMat = self._jacSparsity()
            else:
                jacSparsityMat = None


            x0Arr = self._ff2symarr(self.x)
            print(); print()
            print("IMPORTANT: Change the value of x.flowDict['counter'] manually after exit, since callback() cannot change it")
            print(); print()

            optRes = least_squares(__resFunReal, x0Arr,jac=jac,bounds=bounds,verbose=2,jac_sparsity=jacSparsityMat,
                    method=method,max_nfev=max_nfev,xtol=xtol,ftol=ftol,gtol=gtol,callback=callback)
           
            xArr = optRes.x
            xNew = self._symarr2ff(xArr)
            
            self.x = xNew
            print("Final residual norm from class method is %.4g"%(self.x.residuals().norm()))
            print("Final residual norm from symarr is %.4g"%(np.linalg.norm(__resFunReal(xArr))))
            sys.stdout = orig_stdout
            return xNew, optRes.cost, optRes.status

        iterMax = self.attributes['iterMax']
        for n in range(iterMax):
            saveDir = self.attributes['saveDir']
            workingDir = os.getcwd()
            if saveDir is not None:
                saveDir = str(saveDir)
                if os.path.exists(saveDir): saveSolns = True
                else:
                    try: 
                        os.makedirs(saveDir)
                        saveSolns = True
                    except:
                        saveSolns = False
                    

            print('iter:',n+1)


            # Ensure BCs on vf, and field is real-valued
            J, F = self.makeSystem(self.x)
                    
            sys.stdout.flush()
            chebWeight = True
            # Weight Jacobian and residual matrices for clencurt weighing
            if chebWeight:
                __weightJ(J)
                __weightF(F)
            
            dx, linNorm, jRank,sVals = np.linalg.lstsq(J,-F,rcond=rcond)
            print('Jacobian inversion returned with residual norm:',linNorm)
       
            dxff = self._symarr2ff(dx)
            # Ensuring correction fields are real-valued and obey the required symmetries
            # imposeSymms has realValued=True by default
            dxff.imposeSymms(sigma1=self.attributes['sigma1'], sigma3=self.attributes['sigma3'])
            dxff[0,:,:,:3,[0,-1]] = 0.   # Correction field should not change velocity BCs, for Couette or channel flow

            self.x = self.lineSearch(resnormFun, self.x, dxff)
           
            # I don't have to keep using imposeSymms(), but it doesn't reduce performance, so might as well
            
            self.x.imposeSymms(sigma1=sigma1, sigma3=sigma3)
            if callback is not None:
                callback(self._ff2symarr(self.x))
                self.x.flowDict['counter'] = saveCounter

            if False:
                # Old way to save intermediate solutions. 
                # Now using the same function that saves for trf and dogbox
                if (self.attributes.get('saveDir',None) is not None):
                    if 'counter' not in self.x.flowDict:
                        self.x.flowDict['counter'] = 0
                    savePath = saveDir
                    fNamePrefix = self.attributes['prefix']+'_'+str(self.x.flowDict['counter'])+'_'
                    print(savePath, fNamePrefix)
                    self.x.saveh5(prefix=savePath, fNamePrefix=fNamePrefix)


            fnorm = resnormFun(self.x)
            print('Residual norm after %d th iteration is %.3g'%(n+1,fnorm))
            sys.stdout.flush()
            
            fnormArr.append(fnorm)
            if fnorm <= tol:
                flg = 0
                print('Converged in ',n+1,' iterations. Returning....................................')
                sys.stdout = orig_stdout
                return x, np.array(fnormArr), flg
            
            if n>0:
                if fnormArr[n] > fnormArr[n-1]:
                    flg = 1
                    print('Residual norm is increasing:',fnormArr)
            
            print('*********************************************************')
        else:
            if fnormArr[-1] > 100.*tol:
                print('Iterations have not converged for iterMax=',iterMax)
                print('fnormArr is ',fnormArr)
                flg = -1
        sys.stdout = orig_stdout
        return self.x, np.array(fnormArr), flg

    def shearStress(self):
        """ Returns averaged shear stress at the bottom wall
        NOTE: For Couette flow, stress at the walls is opposite in sign, while
            they have the same sign for Poiseuille flow. So the stress at only one wall
            is considered here
        Inputs: 
            vf: velocity flowFieldRiblet (can include pressure)
        Returns:
            avgShearStress: scalar
        """
        vf = self.x.slice(nd=[0,1,2])
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



    def averagedU(self,nd=0, zArr = None, ny = 50):
        """ Velocity averaged in wall-parallel directions in physical domain"""
        vf = self.x.slice(nd=[0,1,2])
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
    
    


    


def testExactRibletModule(ffProb,nex=5):
    sigma1 = ffProb.attributes['sigma1']; sigma3 = ffProb.attributes['sigma3']; 
    x = ffProb.x 
    tol = ffProb.attributes['tol']
    print('Testing for symmetries sigma1=%r and sigma3=%r to tolerance %.3g'%(sigma1,sigma3,tol))
    print('epsArr is ', x.flowDict['epsArr'])
   
    x1 = x.slice(L=x.nx//2+nex, M=x.nz//2+nex)    # Up-slicing to avoid aliasing effects
    vf1 = x1.slice(nd=[0,1,2]); pf1 = x1.getScalar(nd=3)
    N = x.N
    if sigma3: NN = np.int(np.ceil(x.N/2.))
    else: NN = x.N
    w = x.w 
    w = w[:NN]  # Because, if foldD, we are only evaluating residuals on y >= 0
    q = np.sqrt(w); qinv = 1./q;  Q = np.diag(q);  Qinv = np.diag(qinv)
    def __weightJ(Jacobian):
        for k1 in range(Jacobian.shape[0]//NN):
            for k2 in range(Jacobian.shape[1]//NN):
                Jacobian[k1*NN:(k1+1)*NN, k2*NN:(k2+1)*NN] = np.dot( Q, np.dot(Jacobian[k1*NN:(k1+1)*NN, k2*NN:(k2+1)*NN], Qinv) )
        return

    # Reduced state-vector (weighted numpy array):
    xm_ = ffProb._ff2symarr(ffProb.x)
    xm2 = ffProb._ff2symarr(ffProb.x,weighted=False)

    
    # Calculating linear matrix, and the product with state-vector
    Lmat = ffProb.linr()
    LmatUnweighted = Lmat.copy()
    __weightJ(Lmat)
    linTermArr = np.dot(Lmat, xm_)
    Lmat = LmatUnweighted

    # Calculating linear term from class methods
    linTermClass = (vf1.laplacian()/(-1.*vf1.flowDict['Re']) + pf1.grad()).appendField(vf1.div())
    linTermClass = linTermClass.slice(L=x.nx//2, M=x.nz//2)


    # Calculating non-linear matrix, and its product with the state-vector
    Lmat0 = Lmat.copy()
    ffProb.jcbn(ffProb.x,Lmat=Lmat)
    Lmat -= Lmat0
    __weightJ(Lmat)
    NLtermArr = 0.5*np.dot(Lmat, xm_)
    #NLtermArr = 0.5*np.dot(Lmat, xm2)

    # Calculating non-linear term from class methods
    NLtermClassFine = vf1.convNL(fft=True)
    NLtermClass = NLtermClassFine.slice(L=x.nx//2, M=x.nz//2).appendField(x.getScalar().zero())


    # Reducing terms from class methods so positive Fourier modes are discarded according to sigma1,sigma2
    # linTermMat = ffProb._symarr2ff(linTermArr)
    linTermMat = ffProb._symarr2ff(linTermArr)
    NLtermMat  = ffProb._symarr2ff(NLtermArr)
    #NLtermMat  = ffProb._symarr2ff(NLtermArr,weighted=False)
    NLtermMat[0,:,:,:,[0,-1]] = 0.

    linTermClassArr = ffProb._ff2symarr(linTermClass)
    NLtermClassArr  = ffProb._ff2symarr(NLtermClass)
    linResNorm = (linTermMat - linTermClass).norm()
    NLresNorm  = (NLtermMat  - NLtermClass ).norm()


    #linTestResult = linResNorm <= tol
    linTestResult = linResNorm <= 1.0e-14
    NLtestResult  = NLresNorm  <= tol
    if sigma1:
        print('sigma1 invariance norm of x is', (x - x.reflectZ().shiftPhase(phiX=np.pi) ).norm())
    if sigma3:
        print('sigma3 invariance norm of x is', (x - x.pointwiseInvert().shiftPhase(phiZ=np.pi) ).norm())
    
    if not linTestResult :
        print('Residual norm for linear is:',linResNorm)
    if not  NLtestResult:
        print('Residual norm for non-linear is:',NLresNorm)
        print('Residual norm for arrays is:', chebnorm(NLtermClassArr - NLtermArr, x.N))

    if linTestResult and NLtestResult:
        print("Success for both tests!")

    print("*******************************")
    sys.stdout.flush()
    
    return linTestResult, NLtestResult










