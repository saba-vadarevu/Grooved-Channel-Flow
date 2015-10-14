
import os
import numpy as np
def changeLine(fOld= None, fNew=None, strOld = None, strNew = None, deleteOld=False):
    """
    Function for file manipulation (very inefficient for editing multiple lines)
    fOld, fNew are strings of filenames
    strOld is the string that needs to be matched in fOld.
        It only needs to have enough characters of a line to uniquely identify the line. 
        Can be a list of strings if multiple lines have to be modified.
    strNew is the line that replaces the line identified by strOld.
        If strOld is a list, strNew must be a list of the same size
    deleteOld is a boolean flag, it applies only for cases when fOld==fNew
        If set to True, the new file replaces the old file
        If set to False (default), the old file is renamed to fOld+'~'  """
    if isinstance(strOld, list):
        changeMultipleLines(fOld=fOld, fNew=fNew, strOld=strOld, strNew=strNew, deleteOld=deleteOld)
    assert (fOld is not None) and (fNew is not None) and (strOld is not None) and (strNew is not None)
    changeName=False
    if fOld == fNew:
        os.rename(fOld, fOld+'~')
        fOld = fOld+'~'
        changeName=True
        
    with open(fOld, 'r') as fRead:
        with open(fNew,'w') as fWrite:
            for line in fRead:
                if line[:len(strOld)] == strOld:
                    fWrite.write(strNew+'\n')
                else:
                    fWrite.write(line)
    if changeName and bool(deleteOld):
        os.remove(fOld)
    return

def changeMultipleLines(**kwargs):
    if isinstance(kwargs['strOld'], list):
        oldList = kwargs['strOld']; newList = kwargs['strNew']
        assert isinstance(newList, list) and (len(oldList) ==  len(newList))
        fOld = kwargs['fOld']; fNew = kwargs['fNew']
        for k in range(len(oldList)):
            if k == 0:
                changeLine(fOld=fOld, fNew=fNew, strOld=oldList[k], strNew=newList[k])
            else:
                changeLine(fOld=fNew, fNew=fNew, strOld=oldList[k], strNew=newList[k], deleteOld=True)
    elif isinstance(kwargs['strOld'],str):
        changeLine(**kwargs)
    else:
        raise RuntimeError("Argument 'strOld' must be either a string or a list of strings")
    return

def findRoot(func, a, b, nSteps=20, tol= 1.0e-9):
    """ Find the root of a function 'func(x)' between 'x=a' and 'x=b' using binary search for 'nSteps' steps.
    func(a)*func(b) must be negative (i.e., there must be at least one zero crossing between a and b)
    func must take exactly one argument
    nSteps (defaults to 20), determines the number of steps in binary search
    tol (defaults to 1.0e-9), determines the tolerance on func(root). 
         Note: If the function is too 'flat' near the root, the root found could be quite inaccurate"""
    if abs(func(a)) < tol: return a 
    if abs(func(b)) < tol: return b
    assert func(a)*func(b) <= 0., "There must be at least one zero crossing at or betweeen a and b"
    assert (type(a) is float) and (type(b) is float), "This function only finds roots for functions defined over real numbers"
    
    # Start at 'a', head towards 'b'
    x0 = a;
    xf = b;
    fx0 = np.real(func(a))
    fxf = np.real(func(b))
    
    # Given that the function changes sign between x0 and xf, it's guaranteed that the root is between x0 and xf
    for n in range(nSteps):
        x = 0.5*(x0+xf)
        fx = np.real(func(x))
        
        if abs(fx) < tol: return x
        
        if fx*fx0 < 0.:
            # func changes sign between x and x0, meaning the root must be between x and x0
            #     Starting at x, move towards x0
            xf = x0    # Go towards previous x0
        # If func hand't changed signs between x and x0, it means the root is still between x and xf
        #    So, the next step should be (x+xf)/2, i.e., still moving towards xf
        x0 = x
        fx0 = fx
    return x0
        
    
    
    
    


def writeMeshPoints(a=50, eps=0.02, scale=0.1, nx = 50, ny= 500, nz=1, fName='temp'):
    """Using blockMesh for meshing wavy walled flow for OpenFOAM produces lots of cells with skewness ~ 0.5
    This is because the mesh lines follow the wall surface even into the core. 
    To work around this, I'm printing my own points (with same neighbour, face relations as blockMesh)
    Arguments: 
        a = 2*pi/lambda_x (defaults to 50)
        eps (wall is 2*eps*cos(a*x)), defaults to 0.02
        scale, scaling factor for channel geometry (convertToMeters in blockMesh), defaults to 0.1
        nx, number of cells along x, defaults to 50
        ny, number of cells along y, defaults to 500
        nz, number of cells along z, defaults to 1
    """
    # Points need to be written in a column as ordered pairs (x,y,z)
    # The columns must sweep points first along increasing x, then along increasing y, then z.
    xArr = np.zeros((nz+1, ny+1, nx+1))
    yArr = np.zeros(xArr.shape)
    zArr = np.zeros(xArr.shape)
    
    x1D = np.arange(0.,  2.*np.pi/a+1.0e-9  , 2.*np.pi/a/nx)
    xArr[:] = x1D.reshape((1,1,nx+1))
    zArr[:] = np.asarray([0,2.*np.pi/a/nx]).reshape((nz+1,1,1))
    
    # Defining the wall:
    yWall = 2.*eps*np.cos(a*x1D)
    
    # In the region going from y=0 to y=8*eps (at a*x=pi/2), we use (16*eps)*ny/2 cells with uniform grading
    # In the region going from y=8*eps to y=1 (at a*x=pi/2), we use (1-16*eps)*ny/2.
    #       with appropriate grading so that the cell sizes match at y=8*eps
    # Similarly from y=1 to y=2-8*eps, and y=2-8*eps to y=2
    # The plan is to make the wall-parallel mesh line 'flat' by the time it gets to y=8*eps (at a*x=pi/2)
    assert abs(eps)< 0.11, "This code is only written for sufficiently small eps"
    ny1 = int(8*eps*ny);  ny2 = int(ny//2-ny1)
    ny4 = ny1
    ny3 = ny-ny1-ny2-ny4
    
    ny1 += 1
    for k in range(nx+1):
        # First, cells between y=0 and y=10*eps
        yLen = 8.*eps
        yArr[0,:ny1,k] = np.arange( yWall[k], yLen+1.0e-9, (yLen-yWall[k])/(ny1-1) )
        
    # Next, figuring out the cell expansion ratio for the next sub-block:
    dy0 = yArr[0,ny1-1,nx//4] - yArr[0,ny1-2,nx//4]   # Length of last cell in previous sub-block
    yLen = 1.-8.*eps        # Total length of the sub-block
    func = lambda r: yLen - ( dy0*  (r**ny2 - 1.)/ (r - 1.)  )
    # The above function 'func' would be positive  or negative depending on if 'r' is greater or less than the required value
    r2 = findRoot(func, 0.5, 1.5)
    yArr[0,ny1:ny1+ny2]  =  (yArr[0,ny1-1,0]  + dy0**( r2*np.arange(1.,ny2+1.0e-9) )).reshape((ny2,1))
    
    # Next sub-block: y=1 to y=2.-8*eps
    r3 = 1./r2
    dy0 = yArr[0,ny1+ny2-1, 0] - yArr[0,ny1+ny2-2,0]
    yArr[0,ny1+ny2:ny1+ny2+ny3]  =  (yArr[0,ny1+ny2-1,0]  + dy0**( r3*np.arange(0.,ny3) )).reshape((ny3,1))
    
    # Final sub-block: y=2.-8*eps to y=2
    for k in range(nx+1):
        yLen = 8.*eps
        yArr[0,ny1+ny2+ny3:,k] = np.arange( 2.-yLen, 2.+yWall[k]+1.0e-9, (yLen+yWall[k])/ny4 )[1:]
    
    # Same mesh at all z as at z0
    yArr[1:] = yArr[0]
    
    xArr = xArr.reshape((xArr.size,1))
    yArr = yArr.reshape((yArr.size,1))
    zArr = zArr.reshape((zArr.size,1))
    points = scale*np.concatenate((xArr,yArr,zArr), axis=1)
    
    if '.dat' in fName[-4:]: fName = fName[:-4]
    
    np.savetxt(fName+'.dat', points,fmt='%.10f', newline=')\n(',)
    return
    

def readPoints(fName='points', nx= 41, ny=301, nz=2, vertices=False):
    """Reads the 'points' file written by blockMesh for OpenFOAM simulations, returns numpy arrays x,y,z
        The returned arrays are of size (nx,nz,ny)"""
    
    with open(fName,'r') as fOpen:
        fRead = fOpen.readlines()
        with open(fName+'~', 'w') as fWrite:
            for line in fRead[20:-4]:
                fWrite.write(line[1:-2]+'\n')
                
    xT,yT,zT = np.genfromtxt(fName+'~', unpack=True)
    print(xT.size)
    assert xT.size == nx*ny*nz, "The shape attributes given by nx, ny, nz are not compatible with the size of data in "+str(fName)
    xT = xT.reshape((nz, ny, nx)); yT=yT.reshape(xT.shape); zT = zT.reshape(xT.shape)
    
    x = np.zeros((nx,nz,ny))
    y = np.zeros(x.shape); z = np.zeros(x.shape)
    
    for k in range(nx):
        x[k] = xT[:,:,k]
        y[k] = yT[:,:,k]
        z[k] = zT[:,:,k]
    
    # OpenFOAM solution files print velocities at cell centers instead of at vertices. 
    # Since I'm writing this function to compare my N-R solutions to OpenFOAM solutions, 
    #        I should interpolate my solutions on the same points
    if not vertices:
        # Return x,y,z of cell centers instead of vertices
        xCC = np.zeros((nx-1,nz-1,ny-1)); yCC = np.zeros(xCC.shape); zCC = np.zeros(xCC.shape)
        # Averaging x,y,z coordinates of cell vertices to obtain cell centre
        xCC[:] = 0.5*(x[:-1,:1,:1] + x[1:,:1,:1])
        yCC[:] = 0.25*(y[:-1,:1,:-1] + y[:-1,:1,1:] + y[1:,:1,:-1] + y[1:,:1,1:]) # \sum 0.25*y for 'y' of each vertex
        zCC[:] = 0.5*(z[0,0,0]+z[0,1,0])
        return xCC, yCC, zCC
    
    return x,y,z
    
def readFOAMData(fName='U', nx=40, ny=300, nz=1):
    """Reads velocity data file from OpenFOAM solutions and returns the velocity field components u,v,w as numpy arrays of size (nx,nz,ny)"""
    with open(fName,'r') as fOpen:
        fRead = fOpen.readlines()
        with open(fName+'~', 'w') as fWrite:
            for line in fRead[22:-35]:
                fWrite.write(line[1:-2]+'\n')
                
    uT,vT,wT = np.genfromtxt(fName+'~', unpack=True)
    print('max U in readFOAMData():',np.max(uT))
    assert uT.size == nx*ny*nz, "The shape attributes given by nx, ny, nz are not compatible with the size of data in "+str(fName)
    uT = uT.reshape((nz, ny, nx)); vT=vT.reshape(uT.shape); wT = wT.reshape(uT.shape)
    
    u = np.zeros((nx,nz,ny))
    v = np.zeros(u.shape); w = np.zeros(u.shape)
    
    for k in range(nx):
        u[k] = uT[:,:,k]
        v[k] = vT[:,:,k]
        w[k] = wT[:,:,k]
    
    
    return u,v,w

def writeFOAMData(vF, pF, nx=60, ny=500,nz=1,filePath='./'):
    """Writes uField and pField on points defined in 'fName' as output file 'outFile' as required by OpenFOAM"""
    uSample = 'Usample'    # Sample output file that contains formatting except for the field values
    pSample = 'psample'
    writePath=filePath+'0/'
    fName = filePath+'constant/polyMesh/'+'points'
    try:
        os.stat(writePath)
    except:
        os.mkdir(writePath)
    uOutFile = writePath+'U'; pOutFile = writePath+'p'
    
    print(nx,ny,nz)
    x,y,z = readPoints(nx=nx+1,ny=ny+1,fName=fName)    # By default, this gives coordinates for cell centers, not vertices
    nx = x.shape[0]
    ny = x.shape[2]
    nz = x.shape[1]
    h = 0.1
    x = x/h; y=(y-h)/h; z=z/h

    u = vF.getScalar(nd=0)
    v = vF.getScalar(nd=1)
    w = vF.getScalar(nd=2)
    uData = np.zeros(x.shape); vData = np.zeros(x.shape); wData = uData.copy(); pData = uData.copy()
    for kx in range(x.shape[0]):
        for kz in range(x.shape[1]):
            yBottom = np.min(y[kx,kz])
            yT = y[kx,kz]-yBottom-1.
            uData[kx,kz] = u.getPhysical(xLoc=x[kx,0,0], zLoc=z[0,kz,0], yLoc = yT)+ (1.-yT**2)
            vData[kx,kz] = v.getPhysical(xLoc=x[kx,0,0], zLoc=z[0,kz,0], yLoc = yT)
            wData[kx,kz] = w.getPhysical(xLoc=x[kx,0,0], zLoc=z[0,kz,0], yLoc = yT)
            pData[kx,kz] = pF.getPhysical(xLoc=x[kx,0,0], zLoc=z[0,kz,0], yLoc = yT)
    
    uT = np.zeros((nz,ny,nx)); vT = uT.copy(); wT = uT.copy(); pT = uT.copy()
    
    for k in range(nx):
        uT[:,:,k] = uData[k]; vT[:,:,k] = vData[k]; wT[:,:,k] = wData[k]; pT[:,:,k] = pData[k]
    uT=uT.reshape(uT.size,1); vT=vT.reshape(vT.size,1); wT=wT.reshape(wT.size,1); pT=pT.reshape(pT.size,1)
    
    with open(uSample,'r') as uOpen:
        uRead = uOpen.readlines()
        uRead1 = uRead[:19]
        uRead2 = uRead[-34:]
    with open(pSample,'r') as pOpen:
        pRead = pOpen.readlines()
        pRead1 = pRead[:19]
        pRead2 = pRead[-32:]
    
    uvwT = 0.015*np.concatenate((uT,vT,wT),axis=1)
    print(uvwT.shape)
    pT = 0.015**2*pT
    np.savetxt(uOutFile+'~',uvwT)
    np.savetxt(pOutFile+'~',pT)
    
    with open(uOutFile,'w') as uOpen:
        with open(pOutFile,'w') as pOpen:
            nCells = nx*ny*nz
            uRead1.append(str(nCells)+'\n(\n')
            pRead1.append(str(nCells)+'\n(\n')
            for line in uRead1:
                uOpen.write(line)
            for line in pRead1:
                pOpen.write(line)
            with open(uOutFile+'~','r') as uTemp:
                uDataLines = uTemp.readlines()
                for line in uDataLines:
                    uOpen.write('('+line[:-1]+')\n')
            with open(pOutFile+'~','r') as pTemp:
                pDataLines = pTemp.readlines()
                for line in pDataLines:
                    pOpen.write(line)
            uOpen.write(')\n')
            pOpen.write(')\n')
            for line in uRead2:
                uOpen.write(line)
            for line in pRead2:
                pOpen.write(line)
    os.remove(uOutFile+'~')
    os.remove(pOutFile+'~')
    