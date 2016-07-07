# pseudo.py
#
# Contains functions:
#		y,DM = chebdif(N,M): Takes number of nodes 'N', and required number of derivatives'M'
#					and returns Chebyshev nodes 'y', and differentiation matrices in 'DM'
#		DM = poldif(x,M): Takes a general grid 'x', and required number of derivatives'M'
#					and returns differentiation matrices in 'DM' based on Lagrange interpolation
#		w = clencurt(N): Takes number of nodes 'N'
#					and returns Clenshaw-Curtis weights 'w', for 'N' Chebyshev nodes between -1 and 1
##--------------------------------------------------------------------------------------------------------
# Update:
# Earlier version used 2-D arrays, as used by Matlab, as the default. Vectors
#  were stored as (N,1) arrays. 
# This version uses 1-D arrays as the default for vectors


## ----------------------------------------------------
## ACKNOWLEDGEMENT

# Implementation for chebdif and poldif is based on Weideman and Reddy's differentiation suite for Matlab.
# Most of the code is A line-by-line translation of their functions.
# This translation is done, in spite of available pseudo-spectral differentiation codes since the Matlab
# suite incorporates features that improve accuracy, which becomes important for calculating higher derivatives.
# Notably, the chebdif function uses trigonometric identities in [0,pi/2] to calculate x(k)-x(j), and higher 
# derivatives are explicitly calculated, instead of using D*D*..

# The webpage for the differentiation suite:
# http://dip.sun.ac.za/~weideman/research/differ.html


# 'clencurt' is from a blog by Dr. Greg von Winckel:
#	http://www.scientificpython.net/pyblog/clenshaw-curtis-quadrature	
# However, a minor change has been made- only the weight matrix is returned, as opposed to returning both the weights
#		and the nodes, as in the origion version on the blog



###--------------------------------------------------------

import scipy as sp 
from operator import mul
from scipy.linalg import toeplitz
from scipy.fftpack import ifft

def chebdif(N,M):

	I = sp.identity(N)		# Identity matrix
	#L = bool(I)

	n1 = int(sp.floor(N/2.))		# Indices for flipping trick
	n2 = int(sp.ceil(N/2.))

	k=sp.vstack(sp.linspace(0.,N-1.,N))	# Theta vector
	th=k*sp.pi/(N-1.)

	x=sp.sin(sp.pi*sp.vstack(sp.linspace(N-1.,1.-N,N))/2./(N-1.))	# Chebyshev nodes

	T = sp.tile(th/2.,(1,N))
	DX= 2.*sp.sin(T.T+T)*sp.sin(T.T-T)	# Compute dx using trigonometric identity, improves accuracy

	DX= sp.vstack((DX[0:n1,:], sp.flipud(sp.fliplr([-1.*el for el in DX[0:n2,:]]))))
	DX = DX+I			# Replace 0s in diagonal by 1s (required for calculating 1/dx)

	C= toeplitz(pow(-1,k))			# Matrix with entries c(k)/c(j)
	C[0,:] = [2.*el for el in C[0,:]]	
	C[N-1,:] = [2.*el for el in C[N-1,:]]
	C[:,0] = [el/2. for el in C[:,0]]
	C[:,N-1] = [el/2. for el in C[:,N-1]]


	Z = [1./el for el in DX]		# Z contains 1/dx, with zeros on diagonal
	Z = Z - sp.diag(sp.diag(Z))


	D = sp.identity(N)
	DM = sp.zeros((N,N,M))			# Output matrix, contains 'M' derivatives

	for ell in range(0,M):
		D[:,:] = Z*(C*(sp.tile(sp.diag(D),(N,1)).T)-D)	
		D[:,:] = [(ell+1)*l for l in D]			# Off-diagonal elements
		trc = sp.array([-1.*l for l in sp.sum(D.T,0)])
		D = D - sp.diag(sp.diag(D)) + sp.diag(trc)	# Correcting the main diagonal
		DM[:,:,ell] = D

   # Return collocation nodes as a 1-D array
	return  x[:,0] , DM


def poldif(x,M):
	N= sp.size(x)
	x = sp.vstack(sp.array(x))
	a,b = x.shape
	if b != 1:
		print ('x is not a vector')

	alpha = sp.ones((N,1))
	B = sp.zeros((M,N))

	I = sp.identity(N)

	XX = sp.tile(x,(1,N))
	DX=XX-XX.T
	DX = DX + I

	c = sp.ones((N,1))
	sp.prod(DX,axis=1,out=c)

	C = sp.tile(c, (1,N))
	C = C/C.T

	Z = sp.array([1./l for l in DX])
	Z = Z - sp.diag(sp.diag(Z))

	X = Z.T

	X = sp.delete(X.T,sp.linspace(0,N*N-1,N))
	X = (X.reshape((N,N-1))).T

	Y = sp.ones((N-1,N))
	D = sp.identity(N)
	DM = sp.ones((N,N,M))

	for ell in range(0,M):
		Y = sp.cumsum(sp.vstack((B[ell,:], (ell+1.)*Y[0:N-1,:]*X)) , axis=0,dtype=float)
		D = (ell+1.)*Z*(C*(sp.tile(sp.diag(D),(N,1)).T)-D)
		D = D - sp.diag(sp.diag(D)) + sp.diag(Y[N-1,:])
		DM[:,:,ell] = D
	
	return DM
	
	
	
def clencurt(n1):
  """ Computes the Clenshaw Curtis weights """
  if n1 == 1:
    x = 0
    w = 2
  else:
    n = n1 - 1
    C = sp.zeros((n1,2))
    k = 2*(1+sp.arange(int(sp.floor(n/2))))
    C[::2,0] = 2/sp.hstack((1, 1-k*k))
    C[1,1] = -n
    V = sp.vstack((C,sp.flipud(C[1:n,:])))
    F = sp.real(ifft(V, n=None, axis=0))
    x = F[0:n1,1]
    w = sp.hstack((F[0,0],2*F[1:n,0],F[n,0]))
  return w


def chebnorm(vec,N):
	vect = sp.asarray(vec)
	vect = vect.reshape((vect.size,1))
	wvec = clencurt(N)
	wvec = wvec.reshape((1,N))
	
	repN = vect.size//N
	
	return sp.sqrt(  abs(sp.dot( sp.tile(wvec,(1,repN)), vect.conjugate()*vect ))[0,0]   )

# Clencurt-weighted norm on internal Chebyshev collocation nodes- i.e., excluding 1,-1
def chebnorm_int(vec,N):
   m = vec.size/(N-2)
   vect = sp.asarray(vec).reshape((m,N-2))
   wvec = clencurt(N)
   wvec = wvec[1:N-1]

   return sp.sqrt( abs(  sp.sum(sp.dot( vect*vect.conjugate(),wvec )) ))

def chebdotvec(vec1,vec2,N):
    # This function computes inner products of vectors ONLY.
    if (vec1.ndim != 1) or (vec2.ndim != 1) or (not isinstance(N,int)):
        raise RuntimeError("chebdotvec only accepts vector arguments, and integer 'N'")

    if (vec1.size % N != 0) or (vec2.size % N != 0) or (vec1.size != vec2.size):
        raise RuntimeError("Vector sizes not consistent with each other or with 'N'")

    wvec = clencurt(N)
    res = 0.
    wvec = sp.tile(wvec,(vec1.size/N,))

    return sp.dot(wvec, vec1.conjugate()*vec2)

def chebdotvec_int(vec1,vec2,N):
    m = vec1.size/(N-2)
    wvec = clencurt(N)
    wvec = wvec[1:N-1]
    vec1t = vec1.reshape((m,N-2))
    vec2t = vec2.reshape((m,N-2))

    return sp.sum( sp.dot( vec1t*vec2t.conjugate(), wvec ))

def chebdot(arr1,arr2,N):
    # The arguments arr1 and arr2 can either be vectors or matrices
    if (arr1.ndim ==1) and (arr2.ndim == 1):
        return chebdotvec(arr1,arr2,N)


    if (arr1.ndim == 2) and (arr2.ndim == 1):
        l = arr1.shape[0]
        dot_prod = sp.zeros(l)
        for k in range(0,l):
            dot_prod[k] = chebdotvec(arr1[k],arr2,N)

    elif (arr1.ndim == 1) and (arr2.ndim == 2):
        l = arr2.shape[0]
        dot_prod = sp.zeros(l)
        for k in range(0,l):
            dot_prod[k] = chebdotvec(arr1,arr2[k],N)

    elif (arr1.ndim == 2) and (arr2.ndim ==2):
        l1 = arr1.shape[0]
        l2 = arr2.shape[0]
        dot_prod = sp.zeros((l1,l2))
        for k1 in range(0,l1):
            for k2 in range(0,l2):
                dot_prod[k1,k2] = chebdotvec(arr1[k1],arr2[k2],N)

    else:
        raise RuntimeError("Input arguments are neither vectors nor 2D matrices")

    return dot_prod

def chebnorm1(vec,N):
	vect = sp.asarray(vec)
	vect = vect.reshape((vect.size,1))
	wvec = clencurt(N)
	wvec = wvec.reshape((1,N))
	
	repN = vect.size/N
	
	return abs(sp.dot( sp.tile(wvec,(1,repN)), abs(vect) ))[0,0]   

def chebnorm1_int(vec,N):
   m = vec.size/(N-2)
   vect = sp.asarray(vec).reshape((m,N-2))
   wvec = clencurt(N)
   wvec = wvec[1:N-1]

   return abs(  sp.sum(sp.dot( abs(vect),wvec )) )   

def chebnorm2(vec,N):
	return chebnorm(vec,N)

def chebnorm2_int(vec,N):
    return chebnorm_int(vec,N)
# Given function values at Cheb collocation nodes, returns the
# coefficients of Chebyshev polynomials of the 1st kind
def chebcoeffs(f):
	if f.ndim != 1:
		raise RuntimeError("Input is not a 1-D array")
		
	N = f.size
	a = sp.fft(sp.append(f,f[N-2:0:-1]))
	 
	a = a[:N]/(N-1.)*sp.concatenate(([0.5],sp.ones(N-2),[0.5]))
	
	return a



def chebcoll_vec(a):
	if a.ndim !=1:
		raise RuntimeError("Input is not a 1-D array")
	
	N = a.size
	a = a*(N-1.)/sp.concatenate(([0.5],sp.ones(N-2),[0.5]))

	f = sp.ifft(sp.append(a,a[N-2:0:-1]))
	
	return f[:N]
	
	


def presdif(N):
	x = chebdif(N,1)[0]
	D1 = poldif(x[1:N-1],1)
	D1=D1[:,:,0]
	
	return D1



def chebdifBL(N,Y):
	eta,DM = chebdif(2*N,2)
	x = -Y*sp.log(eta[:N])
	
	D1 = (-1./Y)*eta[:N].reshape((N,1))*DM[0:N,0:N,0]
	D2 = (1./Y/Y)*(eta[:N].reshape((N,1))*DM[0:N,0:N,0] + (eta[0:N].reshape((N,1))**2)*DM[0:N,0:N,1])

	return x,D1,D2


def presdifBL(N,Y):
	eta,DM = chebdif(2*N,2)
	Dpeta = poldif(eta[1:2*N-1],1)
	
	Dp = (-1./Y)*eta[1:N]*Dpeta[0:N-1,0:N-1,0]
	
	return Dp


def chebint (fk, x):
    speps = sp.finfo(float).eps # this is the machine epsilon
    N = sp.size(fk)
    M = sp.size(x)
    xk = sp.sin(sp.pi*sp.arange(N-1,1-N-1,-2)/(2*(N-1)))
    w = sp.ones (N)*(-1)**(sp.arange(0,N))
    w[0] = w[0]/2
    w[N-1] = w[N-1]/2
    D = sp.transpose(sp.tile(x,(N,1)))-sp.tile(xk,(M,1))
    D = 1/(D+speps*(D==0))
    p = sp.dot(D,(w*fk))/(sp.dot(D,w))
    return p

def chebintegrate(v):
    ''' Integrates 'v' over Chebyshev nodes, assuming v(y=-1) (or, v[-1]) = 0'''

    coeffs = chebcoeffs(v)
    int_coeffs = sp.zeros(v.size, dtype=coeffs.dtype)
    N = v.size

    # T_0 = 1,  T_1 = x, T_2 = 2x^2 -1
    # \int T_0 dx = T_1
    int_coeffs[1] = coeffs[0]

    # \int T_1 dx = 0.25*T_2 + 0.25*T_0
    int_coeffs[2] += 0.25*coeffs[1]
    int_coeffs[0] += 0.25*coeffs[1]

    # \int T_n dx = 0.5*[T_{n+1}/(n+1) - T_{n-1}/(n-1)]
    nvec = sp.arange(0,N)
    int_coeffs[3:] += 0.5/nvec[3:]*coeffs[2:N-1]
    int_coeffs[1:N-1] -= 0.5/nvec[1:N-1]*coeffs[2:]

    int_coll_vec = chebcoll_vec(int_coeffs)
    return int_coll_vec - int_coll_vec[-1]
