import scipy as sp
from scipy.integrate import quad
import numpy as np
import scipy.io
from pseudo import *

# This script looks at a state-vector and tells if separation occurs. And if
#separation does occur, it gives the locations of separation and
#reattachment- hence the extent of the recirculation bubble, and the height
#of the reattachment bubble.

# Separation is identified by looking for the appearance of negative
# streamwise velocities. The x-location of the separation is given by the
# lowest 'x' at which negative streamwise velocity appears, and the
# location of the reattachment given by the highest 'x' at which negative
# streamwise velocity appears. 

# The height of the bubble is a bit more tricky. I take particles at one
# crest of the surface waviness and integrate them towards the next crest.
# All particles that encounter a negative velocity are dropped. Among those
# that remain, the one that goes to the lowest 'y' is used to determine the
# height of the bubble. 

# mat_contents = scipy.io.loadmat('C:\\Users\\adrie_000\\variables_mat2python\\x')
# x = mat_contents['x']
# N=40
# n=5
# eps=0.0251
# a=39.8107170553497
# b=0

# x is the state vector with states [x_(-n),...x_0,x_1,...x_n]
# x_k = [u_k, v_k, w_k, p_k]^T
# And they're all column vectors, not row vectors
# N is the number of wall-normal nodes
# eps is a quarter of the amplitude of the surface waviness, the surface is
#       given as y = +/-1 + 2.eps.cos(ax + bz)
# Even for b != 0, I can just integrate along 'x' because the flow doesn't
# change in the direction normal to the waviness, and spanwise changes are
# quite small. 


# ALL STREAMWISE LENGTHS ARE NORMALIZED AS xl = a*x, so that consecutive
# crests are at 0 and 2*pi
    
def seprn (x,N,n,eps,a,b,Re):
    ySolution = None
    U_ysolution = None
    utol = max(-1.0e-4,-abs(eps*1.0e-3))
    # Size of smallest separation bubbles that can be captured
    xsres = 1.0*(10**(-2))
    # Resolution in identifying zero-crossings, needed for determining the height
    # of separation bubbles
    ysres = 0.01*eps
    # Number of steps needed to achieve that accuracy (described later):
    sep_res=1.0*10**(-4)
    #Number of steps needed to achieve that accuracy (described later)
    nsteps = sp.ceil(sp.log(sep_res)/sp.log(0.5))+1
    nsteps = int(nsteps)
    m = 2*n+1
    N4 = 4*N
    N3 = 3*N
    N2 = 2*N
    y,Dm = chebdif(N,2)
    D = Dm[:,:,0]
    D2 = Dm[:,:,1]
    
    ##Unpacking the state-vector into Fourier modes
    modes = sp.reshape(x,(m,N4))
    # u = modes[:,0:N];
    # v = modes[:,N:N2];
    # w = modes[:,N2:N3];
    # Pressure isn't needed here

    # I don't need the full wall-normal extent. A half of it should do- that
    # reduces the computation (which isn't a big deal anyways). The more
    # important reason to be doing this is to make sense of using 'crests' and
    # 'troughs'. I may not accurately identify separation and reattachment if I
    # consider both the top and bottom walls- since the recirculation bubbles
    # may not be symmetric about a trough. 
    # So I'm only using the flow in the bottom, which corersponds to the
    # indices
    # ceil(N/2): N
    NN = sp.ceil(N/2);
    u0 = modes[:,0:N]
    u = modes[:,NN:N]
    U = sp.zeros(N)
    w0 = modes[:,N2:N3]
    W = sp.zeros(N)
    if b!= 0 and a != 0:
        #vfluxu = Vflux for the spanwise direction
        k = 0
        kp = k+n
        U = U + u0[kp]
        w = clencurt (N)
        Vfluxu = sp.absolute(sp.dot(w,U))
        
        #Vfluxw = Vflux for the streamwise direction
        k = 0
        kp = k+n
        W = W + w0[kp]
        w = clencurt (N)
        Vfluxw = sp.absolute(sp.dot(w,W))
        
    else:    
        #Vfluxu:    
        for k in range (-n,n+1):
            kp = k+n
            U = U + u0[kp]
        w = clencurt (N)
        Vfluxu = sp.absolute(sp.dot(w,U))
        
        #Vfluxw    
        for k in range (-n,n+1):
            kp = k+n
            W = W + w0[kp]
        w = clencurt (N)
        Vfluxw = sp.absolute(sp.dot(w,W))
    #NN = sp.ceil(N/2);
    #u0 = modes[:,0:N]
    # u = modes[:,NN:N]
    #U = sp.zeros(N)   
    #for k in range (-n,n+1):
        #kp = k+n
        #U = U + u0[kp]*sp.exp(1j*k*sp.pi)
    #w = clencurt (N)
    #Vflux = sp.absolute(sp.dot(w,U))
    
    #Working out drag due to pressure
    po = modes[:,N3:N4]
    P = sp.zeros(N)
    pplus1 = po[1+n,-1]# fourier mode of p at wall position and index +1
    pminus1 = po[-1+n,-1]# Fourier mode of p at wall position and index -1
    Pfx = abs(eps*1j*a*(pplus1-pminus1)) #drag/(2pi*2pi/alph*beta) due to pressure along x
    Pfz = abs(eps*1j*b*(pplus1-pminus1)) #drag/(2pi*2pi/alph*beta) due to pressure along z
    
    #Working out drag due to skin friction
    Pskinx = (2/Re-Pfx) #drag/(2pi*2pi/alph*beta) due to friction along x
    Pskinz = -Pfz #drag/(2pi*2pi/alph*beta) due to friction along z
    
    
    # Interpolating velocity fields to get a better approximation for
    # point of separation and reattachment
    # I'm NOT DOING THIS because interpolation doesn't improve the quality of
    # data unless I solve for the flow again. 
    # y1 = chebdif(3*N,1);
    # NN1 = sp.ceil(1.5*N);
    # y1 = y1[NN1+1:3*N];
    # 
    # for k = 1:m
    #     u1[:,k] = chebint(u[:,k],y1);
    #     v1[:,k] = chebint(v[:,k],y1);
    # end
    # u = u1;
    # v = v1;
    # y = y1;
    # NN = NN1;
    # N = 3*N;

    ## Finding if there is separation, and location of separation
    # I use z = 0, WLG
    # At a trough, alpha.x = pi. At alpha.x = 0, a crest occurs.

    # I can make some general observations about where separation is likely to
    # occur.
    # Any separation (on the bottom wall) must occur before the trough. 
    # Separation is more likely to occur between 0 and pi/2 (the slope starts
    # to increase after alpha.x = pi/2)

    # I will be looking for separation in [0,pi], at a resolution given by
    # 'xsres', which gives the size of the largest recirculation zone that can 
    # go uncaptured. 

    # Once separation is encountered, I use a binary-search-like algorithm to
    # find the location of circulation- I can do this with much better
    # resolution- An accuracy ~ 10^-5 needs only about 15 steps with a binary
    # search algorithm. 

    # The binary search algorithm: 
    # Proceeds by evaluating some function f[xl], where xl is the current location
    # xl* such that f[xl*] = 0 is the solution sought. 
    # sep_flag is a flag that is 1 if there is separation (f[xl]<0) and 0 if
    # there is no separation. 

    # I use two variables 'xl1' and 'xl2'. 
    # 'xl1' always stores the location explored immediately before the current.
    # 'xl2' gives the direction in which the exploration must proceed. 
    # As long as sep_flag doesn't change from the previous 'xl', 'xl' is
    # the next 'xl' is (xl + xl2)/2
    # Anytime sep_flag changes, then I start moving in the opposite direction,
    # i.e., I set xl2 = xl1, and then go for xl = (xl+xl2)/2
    # Initially, xl2  is set to 0, since the location of separation must lie
    # between 0 and 'xl', given that the flow is separated at some 'xl'



    # I do the same as above for reattachment
    # I look for reattachment between 2*pi and the location of separation,
    # starting at 2*pi, and going in steps of xsres. Once separation is
    # encountered, a binary search is used between the location of separation
    # and the location of non-separation. 
    xsmat  = sp.arange (0,sp.pi,xsres)
    sep_flag = 0
    
    #Sweeping through xsmat
    for l in range(0,xsmat.size):
        xl = xsmat[l]
        U = sp.zeros(N-NN)
        for k in range(-n,n+1):
            kp = k+n
            U = U+u[kp]*sp.exp(1j*k*xl)
        ind = np.arange(0,U.size)[U<utol] #Finds the index satisfying the condition U<utol
        if sp.size(ind)!=0:
            sep_flag = 1;
            break
    if sep_flag == 0:
        #print 'Could not find a sepparated region in the flow for the given resolution'
        sflag = 0
        xsep = sp.pi
        xreat = sp.pi
        ybub = 0
        bubArea = 0
        bAreaRatio = 0
        Upmax = 0   
        Uphb = 0
        #return  sflag, xsep, xreat, ybub, bubArea, bAreaRatio, Vfluxu,Upmax,Uphb,ySolution, U_ysolution,Vfluxw,Pfx,Pfz,Pskinx,Pskinz
    else:
        sflag =1;
        
        ## Finding the point of separation
        # If the function got to this point, it means I found a location where
        # separation happens. Now, I need to improve on this location using a binary search. 
        
        # I exit the previous for loop after finding a single location of
        # separation, so lsep only has one element. 
        xl = xsmat[l]
        xl1 = xsmat[l-1]
        xl2 = xsmat[l-1]
        
        for l in range(0,nsteps):
            sep_flag0 = sep_flag
            xl = (xl+xl2)/2
            U = sp.zeros(N-NN)
            for k in range(-n,n+1):
                kp = k+n
                U = U + u[kp]*sp.exp(1j*k*xl)
            ind = np.arange(0,U.size)[U<utol]# If 'ind' has elements, it means there is at least one location with negative u
            if ind.size!=0:
                sep_flag = 1
            else:
                sep_flag = 0
    
            if sep_flag != sep_flag0:
                xl2 = xl1 # If the flag changes, switch direction towards previous location
            xl1 = xl
    
        xsep = xl/2/sp.pi
        # It doesn't really matter if sep_flag is 0 or 1 in the end. 
    
        ## Finding location of reattachment
        # I can start directly with the binary search. But if there are multiple
        # separations and reattachments, a direct binary search would give
        # inaccurate results. 
    
        # Working on a coarse grid first:
    
        xrmat = sp.arange(2*sp.pi,xl,-xsres)
        reat_flag = 1
        for l in range(0,xrmat.size):
            xl = xrmat[l]
            U = sp.zeros(N-NN)
            for k in range(-n,n+1):
                kp = k+n
                U = U +u[kp]*sp.exp(1j*k*xl)
            ind = np.arange(0,U.size)[U<utol] 
            if ind.size!=0:
                reat_flag = 0;
                break
            
        xl2 = xrmat[l-1]
        xl1 = xl
        for l in range(0,nsteps):
            reat_flag0 = reat_flag
            xl = (xl + xl2)/2
            U = sp.zeros(N-NN)
            for k in range(-n,n+1):
                kp = k + n
                U = U +u[kp]*sp.exp(1j*k*xl)
            ind = np.arange(0,U.size)[U<utol]  #If 'ind' has elements, it means there is at least one location with negative u
                            # size(ind) !=0 means that flow is separated, hence not reattached
            if sp.size(ind) != 0:
                reat_flag = 0
            else:
                reat_flag = 1
    
            if (reat_flag!=reat_flag0):
                xl2 = xl1;  # If the flag changes, switch direction towards previous location
            xl1 = xl
        xreat = xl/2/sp.pi
        ## Finding the height of the separation bubble
        # The height of the bubble is defined as the maximum wall-normal distance
        # for the last zero-crossing of \int (Udy). 
        # As before, I'm only considering the bottom-half of the flow      
        W = sp.tile(w,(N,1))    
        for k in range(1,N):
            W[k, 0:k] = 0
    
        xsmat = sp.arange(2* sp.pi*xsep,2*sp.pi*xreat,2*sp.pi*(xreat-xsep)/10)
        
        if (xsep< 0.5)&(xreat>0.5):
            xsmat = sp.concatenate((sp.array([sp.pi]),xsmat))   
        xsmat = sp.sort(xsmat)
        hbub = sp.zeros(xsmat.size)
        y = chebdif(N,1)[0]
        bubArea = 0;
        for k in range(0,xsmat.size):
            xl = xsmat[k]
            U = sp.zeros(N)
            for l in range(-n,n+1):
                lp = l+n
                U = U + u0[lp]*sp.exp(1j*l*xl)
            Uint = sp.dot(W,U)
            Uint1 = Uint[NN:]
            ind = np.arange(0,Uint1.size)[Uint1<0]
            if sp.size(ind) !=0:
                ycr1 = y[NN+min(ind)]# Location farthest from wall with negative \int(Udy)
                ycr2 = y[NN-1+min(ind)]# Location just above that on Cheb grid
                ystep = max((ycr2-ycr1)/25,ysres)
            else :
                ycr1 = y[NN]
                ycr2 = y[NN-1]
                ystep = ysres
            ygrid = sp.arange (ycr1,ycr2+ystep,ystep)
            Uint_inter = chebint (Uint ,ygrid)
            ind = np.arange(0,Uint_inter.size)[Uint_inter<0]
            if ind.size!=0:
                hbub[k] = ygrid[max(ind)]+1
        #When calculating area, 'x' was normalized by the wavelength, and 'hbub'
        #by 4*eps, the crest-to-trough height. In that case, the area of the
        # valley is given by [1- \int_0^1  0.5*{1+cos(2*pi*x)} dx] = 0.5
        # The ratio of the area of the bubble to that of the valley is:
        ybub = max(hbub)/4/eps
        ysep = 2.*eps*np.cos(2.*np.pi*xsep)
        yreat = 2.*eps*np.cos(2.*np.pi*xreat)
        ybubMax = np.max([ysep+2.*eps, yreat+2.*eps])/4./eps
        if ybub > ybubMax: ybub=ybubMax

    return  sflag, xsep, xreat, ybub
