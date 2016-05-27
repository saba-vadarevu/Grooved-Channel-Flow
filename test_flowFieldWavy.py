import unittest
import numpy as np
from scipy.linalg import norm

from flowFieldWavy import *

# K=2, L=7, M=3, N=21
ind1 = np.index_exp[2,3,1,2,11]     # k= 0, l=-4, m=-2, y=yCheb[11]
ind2 = np.index_exp[1,9,2,0,5]      # k=-1, l= 2, m=-1, y=yCheb[5]
ind3 = np.index_exp[3,4,0,2,10]     # k= 1, l=-3, m=-3, y=0.
ind4 = np.index_exp[0,8,4,1,17]     # k=-2, l= 1, m= 1, y=yCheb[17]

ind5 = np.index_exp[2,9,4,0,:]      # k= 0, l= 2, m= 1, y=yCheb
ind6 = np.index_exp[0,3,6,1,:]      # k=-2, l=-4, m= 3, y=yCheb
ind7 = np.index_exp[1,10,3,2,:]     # k=-1, l= 3, m= 0, y=yCheb

testDict = getDefaultDict()
testDict.update({'L':7,'M':3,'N':21,'K':2,'eps':3.0e-2,'alpha':25., 'beta':10.,'omega':5.,'isPois':1})
K = testDict['K']; L = testDict['L']; M = testDict['M']; N = testDict['N']
vf = flowFieldWavy(flowDict=testDict)
lArr = np.arange(-L,L+1).reshape((1,vf.nx,1,1,1))
mArr = np.arange(-M,M+1).reshape((1,1,vf.nz,1,1))
yArr = (vf.y).reshape((1,1,1,1,vf.N))
vf[:] = (lArr**2)*(mArr**2)*(1-yArr**2)



vNew = flowFieldWavy(flowDict=testDict.copy())
vNew[:,:,:,0:1] = 1./( (1.+lArr**2)*(1.+mArr**2) ) * (1.-yArr**2)
vNew[:,:,:,1:2] = 1./( (1.+lArr**2)*(1.+mArr**4) ) * (1.-yArr**4)*testDict['eps'] 
vNew[:,:,:,2:3] = 1./( (2.+lArr**2)*(2.+mArr**2) ) * (1.-yArr**6)*testDict['eps'] 


class WavyTestCase(unittest.TestCase):
    """ Defines a simple flowFieldWavy instance, and verifies that operations on it,
            such as ddx(), ddy(), etc.. return accurate flowFieldWavy instances 
            by checking a few of their elements
    IMPORTANT: The tests start with first derivatives, and the derivatives are validated
                only for the flowFieldWavy defined below. Do not modify it.
    To ensure that I don't modify it by mistake, I'm including a copy of it here:
        testDict = getDefaultDict()
        testDict.update({'L':7,'M':3,'N':21,'K':2,'eps':3.0e-2,'alpha':25., 'beta':10.,'omega':5.})
        K = testDict['K']; L = testDict['L']; M = testDict['M']; N = testDict['N']
        vf = flowFieldWavy(flowDict=testDict)
        lArr = np.arange(-L,L+1).reshape((1,vf.nx,1,1,1))
        mArr = np.arange(-M,M+1).reshape((1,1,vf.nz,1,1))
        yArr = (vf.y).reshape((1,1,1,1,vf.N))
        vf[:] = (lArr**2)*(mArr**2)*(1-yArr**2)
    On second thought, the field above isn't appropriate, because it has much more energy in 
        higher modes than in lower modes, which isn't what happens in my flow fields

    So, a second field with the same dictionary, but with fields defined as
        vNew[:,:,:,0] = 1/(l^2 +1)  1/(m^2+1)  (1-y^2)
        vNew[:,:,:,1] = eps* 1/(l^2 +1)  1/(m^4+1)  (1-y^4)
        vNew[:,:,:,2] = eps* 1/(l^2 +2)  1/(m^2+2)  (1-y^6)


    """
    print("Note: when testing for derivatives, if looking at the last modes in x or z,"+\
            "remember that the general formulae I use do not apply, because one/some of their "+\
            "neighbouring modes are missing")

    def test_ddy(self):
        partialY = vNew.ddy()
        yCheb = vNew.y
        eps = vNew.flowDict['eps']

        # u.ddy() should be 1/(l*l+1)/(m*m+1)* (-2y)
        # v.ddy() should be eps/(l*l+1)/(m**4+1)* (-4y**3)
        # w.ddy() should be eps/(l*l+2)/(m**2+2)* (-6y**5)

        # ind1 = np.index_exp[2,3,1,2,11]     # k= 0, l=-4, m=-2, y=yCheb[11]
        # At ind1 (refers to w), w.ddy() = eps/18/6*(-6.*yCheb[11]**5)
        self.assertAlmostEqual(partialY[ind1] , -eps/18.*(yCheb[11]**5) )
       
        # ind2 = np.index_exp[1,9,2,0,5]      # k=-1, l= 2, m=-1, y=yCheb[5]
        # At ind2 (refers to u), u.ddy() = 1/5/2*(-2.*yCheb[5]) = -yCheb[5]/5.
        self.assertAlmostEqual(partialY[ind2] , -yCheb[5]/5. )
        
        # ind4 = np.index_exp[0,8,4,1,17]     # k=-2, l= 1, m= 1, y=yCheb[17]
        # At ind4 (refers to v), v.ddy() = eps/2/2*(-4.*yCheb[17]**3) = -eps*yCheb[17]**3
        self.assertAlmostEqual(partialY[ind4] , -eps*yCheb[17]**3 )

        # ind5 = np.index_exp[2,9,4,0,:]      # k= 0, l= 2, m= 1, y=yCheb
        # At ind5 (refers to u), u.ddy() = 1/5/2*(-2.*yCheb) = -yCheb/5
        self.assertAlmostEqual(norm(partialY[ind5] +yCheb/5.) , 0. )
        
        # ind6 = np.index_exp[0,3,6,1,:]      # k=-2, l=-4, m= 3, y=yCheb
        # At ind6 (refers to v), v.ddy() = eps/17/82*(-4.*yCheb**3) = -4*eps/17/82*yCheb**3
        self.assertAlmostEqual(norm(partialY[ind6] +4.*eps/17./82.*yCheb**3 ), 0.)
        return

    def test_ddx(self):
        """ Refer to testCases.pdf in ./doc/"""
        partialX = vNew.ddx()
        y = vNew.y;  a = vNew.flowDict['alpha'];  eps = vNew.flowDict['eps']; g = eps*a;
        # ind5 = np.index_exp[2,9,4,0,:]      # k= 0, l= 2, m= 1, y=yCheb
        # At ind5 (refers to u), 
        #     u.ddx() = i.2.a.(1-y^2)/5/2  + 2.i.g.y.[ 1/2/1 - 1/10/5 ]
        tempVec = 2.j*a*(1.-y**2)/5./2.  + 2.j*g*y*(1./2. -1./50.)
        self.assertAlmostEqual(norm(partialX[ind5]-tempVec) , 0. )
        
        # ind6 = np.index_exp[0,3,6,1,:]      # k=-2, l=-4, m= 3, y=yCheb
        # At ind6 (refers to v), 
        #   v.ddx() =  i.-4.a.eps.(1-y**4)/17/82   + 4.i.g.eps.y**3.[1/26 - 1/10]/82
        tempVec = -4.j*a*eps*(1.-y**4)/17./82.   + 4.j*g*eps*(y**3)*(1./26./17.)
        self.assertAlmostEqual(norm(partialX[ind6] -tempVec ), 0.)
        
        ind7 = np.index_exp[0,8,3,1,:]      # k=-2, l=1, m= 0, y=yCheb
        # At ind7 (refers to v), 
        #   v.ddx() =  i.a.eps.(1-y**4)/2/1  + 4.i.g.y**3.[1 - 1/5]/1
        tempVec = 1.j*a*eps*(1.-y**4)/2.  + 4.j*g*eps*y**3*(1./1./2. - 1./5./2.)
        self.assertAlmostEqual(norm(partialX[ind7] - tempVec ), 0.)
        return

    def test_ddz(self):
        partialZ = vf.ddz()
        yCheb = vf.y;  b = vf.flowDict['beta'];  eps = vf.flowDict['eps']
        # Testing with vf_{k,l,m} (y) = l^2 m^2 (1-y^2)
        #   tilde{z} derivative of vf for mode (k,l,m) should be   
        #       i.b.m^3.l^2.(1-y^2)+ 2i.eps.b.y.[(l-1)^2.(m-1)^2 - (l+1)^2.(m+1)^2]
        self.assertAlmostEqual( partialZ[4,3,2,2,0], 2.j*eps*b*100)  # l=-4, m=-1, y=1.
        self.assertAlmostEqual( partialZ[2,8,3,1,17], 2.j*eps*b*yCheb[17]*(-4.))    # l=1,m=0
        self.assertAlmostEqual( partialZ[3,9,1,0,8], 
                                1.j* b*(-8.)*4.*(1.-yCheb[8]**2) )    
                                # l=2, m=-2, y = yCheb[8]
        return

    def test_secondDerivatives(self):
        ddx2      = vNew.ddx2()
        ddy2      = vNew.ddy2()
        ddz2      = vNew.ddz2()

        y = vNew.y
        eps = vNew.flowDict['eps']; a = vNew.flowDict['alpha']; b = vNew.flowDict['beta']
        g = eps*a;  gz = eps*b

        # ind5 = np.index_exp[2,9,4,0,:]      # k= 0, l= 2, m= 1, y=yCheb
        # Field is u_xx
        # Refer to eq. 0.5 in /doc/testCases.pdf
        # First, collecting all terms with 'y' in them so that tempVec is defined as an array
        tempVec = -2.*g*a*y*3./2./1.  - 2.*g*a*y*(-5.)/10./5. - 4.*a*a*(1.-y**2)/5./2.
        # Now the terms without y
        tempVec += 2.*g*g/1./2. + 2.*g*g/17./10. - 4.*g*g/5./2.
        self.assertAlmostEqual( norm(ddx2[ind5] - tempVec),0.)


        # ind6 = np.index_exp[0,3,6,1,:]      # k=-2, l=-4, m= 3, y=yCheb
        # Testing ddx2(), for v:
        tempVec = 12.*g*g*(y**2)/37./2. - 4.*g*a*(y**3)*(-9.)/26./17. -(16.*a*a*(1.-y**4) + 24.*g*g*y**2)/17./82.
        tempVec = tempVec*eps
        self.assertAlmostEqual( norm(ddx2[ind6] - tempVec),0.)

        # ind5 = np.index_exp[2,9,4,0,:]      # k= 0, l= 2, m= 1, y=yCheb
        # Testing ddz2() for x:
        tempVec = -2.*gz*b*y*1./2./1.  - 2.*gz*b*y*(-3.)/10./5. - 1.*b*b*(1.-y**2)/5./2.
        tempVec += 2.*gz*gz/1./2. + 2.*gz*gz/17./10. - 4.*gz*gz/5./2.
        self.assertAlmostEqual( norm(ddz2[ind5] - tempVec),0.)
        

        # ind7 = np.index_exp[1,10,3,2,:]     # k=-1, l= 3, m= 0, y=yCheb
        # Testing ddz2(), for w:
        tempVec = 30.*(gz**2)*(y**4)* (1./3./6. + 1./27./6.) \
                - 6.*gz*b*(y**5)* (-1./6./3. - 1./18./3.) \
                - 60.*gz*gz*(y**4)/11./2.
        tempVec = eps* tempVec
        self.assertAlmostEqual( norm(ddz2[ind7] - tempVec),0.)


        # Also testing ddy2() for u (ind5) and v(ind6) and w(ind7):
        tempVec = -2./5./2.*np.ones(y.shape)
        self.assertAlmostEqual( norm(ddy2[ind5] - tempVec),0.)
        tempVec = -12.*(y**2)/17./82.*eps
        self.assertAlmostEqual( norm(ddy2[ind6] - tempVec),0.)
        tempVec = -30.*(y**4)/11./2.*eps
        self.assertAlmostEqual( norm(ddy2[ind7] - tempVec),0.)

        return

    def test_norm(self):
        """ The norm is defined as integral over X,Y,Z of v*v.conj()
            in the transformed space. If I redefine the norm later, 
            expect this test to fail."""
        # vf = \sum_k \sum_l \sum_m l^2 m^2 (1-y^2) e^{i(kwt + lax + mbz)}
        # ||vf||^2 := 1/(T.L_x.L_z)  \int_t \int_x \int_z \int_y vf * vf  dy dz dx dt
        #    since \int_{x=0}^{2pi/a} e^{ilax} = 0   for any non-zero integer 'l', 
        #    it can be shown that 
        # ||v||^2 = \sum_k \sum_l \sum_m   (\int_y v_klm * v_klm.conj() dy)
        # For vf,
        #   ||vf||^2 = 16/15 * \sum_{k=-2}^2 \sum_{l=-7}^7 \sum_{m=-3}^3 l^4 m^4 
        #           = 16/15 * 5 \sum_{l=-7}^7 \sum_{m=-3}^3 l^4 m^4 
        # NOTE: The above norm is only for one scalar, not all of them. For 3 scalars, it should be thrice
        l4Sum = 0.; m4Sum = 0.
        for l in range(-L,L+1): l4Sum += l**4
        for m in range(-M,M+1): m4Sum += m**4

        Lx = 2.*np.pi/vf.flowDict['alpha']
        Lz = 2.*np.pi/vf.flowDict['beta']
        scal = 1./Lx/Lz/2.
   
        # 
        self.assertAlmostEqual(vf.getScalar().norm()**2,scal*16./3.*l4Sum*m4Sum)
        self.assertAlmostEqual(vf.norm()**2, scal*16.*l4Sum*m4Sum)
        return



    def test_gradient(self):
        v0 = vf.getScalar()
        v0Grad = v0.grad()

        xError = v0Grad.getScalar() - v0.ddx()
        yError = v0Grad.getScalar(nd=1) - v0.ddy()
        zError = v0Grad.getScalar(nd=2) - v0.ddz()

        self.assertAlmostEqual(xError.norm(),0.)
        self.assertAlmostEqual(yError.norm(),0.)
        self.assertAlmostEqual(zError.norm(),0.)
        return




    def test_laplacian(self):
        lapl = vf.laplacian()
        laplSum0 = vf.getScalar().ddx2() + vf.getScalar().ddy2() + vf.getScalar().ddz2()
        lapl0 = lapl.getScalar()

        ind1 = np.index_exp[2,3,1,0,11]     # k= 0, l=-4, m=-2, y=yCheb[11]
        ind2 = np.index_exp[1,9,2,0,5]      # k=-1, l= 2, m=-1, y=yCheb[5]
        ind3 = np.index_exp[3,4,0,0,10]     # k= 1, l=-3, m=-3, y=0.
        ind4 = np.index_exp[0,8,4,0,17]     # k=-2, l= 1, m= 1, y=yCheb[17]

        self.assertAlmostEqual( laplSum0[ind1] , lapl0[ind1])
        self.assertAlmostEqual( laplSum0[ind2] , lapl0[ind2])
        self.assertAlmostEqual( laplSum0[ind3] , lapl0[ind3])
        self.assertAlmostEqual( laplSum0[ind4] , lapl0[ind4])
        return

    def test_convection(self):
        vTest = vNew.slice(K=0,L=7,M=0)     # Not testing time-modes here
        vNewSlice = vTest.copy()
        vTest.flowDict['omega'] = 0.
        vTest.flowDict['beta'] = 0.

        vTest[:,:,:,1:] = 0.        # Setting v and w to be zero
        convTerm = np.zeros(N,dtype=np.complex)
        a = vTest.flowDict['alpha']; b = vTest.flowDict['beta']; eps = vTest.flowDict['eps']
        y = vTest.y
        _L = vTest.flowDict['L']; _M = vTest.flowDict['M']
        for l in range(-_L,_L+1):
            for m in range(-_M,_M+1):
                convTerm += -1.j*l*a*(1.-y**2)**2/ (l*l +1.)**2 / (m*m+1.)**2 \
                            + 2.j*a*eps*y*(1.-y**2) / (m*m+1.)**2 / (l*l+1.) * (
                                    1./( (l+1.)**2 + 1.)  -  1./( (l-1.)**2 + 1.)  )
        convFromClass = vTest.convNL().getScalar()[0,_L,_M,0]
        self.assertAlmostEqual(norm(convTerm-convFromClass), 0.)

        # Next, we add v u_y 
        for l in range(-_L,_L+1):
            for m in range(-_M, _M+1):
                convTerm += -2.*eps*y*(1.-y**4)/(  (l**2+1.)**2 * (m*m+1.) * (m**4+1.)  )

        vTest[0,:,:,1] = vNewSlice[0,:,:,1]
        convFromClass = vTest.convNL().getScalar()[0,_L,_M,0]

        self.assertAlmostEqual(norm(convTerm-convFromClass), 0.)

        # Finally, adding w u_x
        for l in range(-_L,_L+1):
            for m in range(-_M, _M+1):
                convTerm += -1.j*m*b*(1.-y**2) * (1.-y**6)/ (l*l +1.) / (l*l+2.) / (m*m+1.) / (m*m+2.) \
                            + 2.j*b*eps*y*(1.-y**6) / (m*m+2.) / (l*l+2.) / (l*l+1.) * (
                                    1./( (m+1.)**2 + 1.)  -  1./( (m-1.)**2 + 1.)  )

        vTest[0,:,:,2] = vNewSlice[0,:,:,2]
        convFromClass = vTest.convNL().getScalar()[0,_L,_M,0]
        self.assertAlmostEqual(norm(convTerm-convFromClass), 0.)
        return

    def test_weighting(self):
        self.assertAlmostEqual( (vNew - weighted2ff(flowDict=vNew.flowDict, arr=vNew.weighted()) ).norm(), 0.)
        return


    # @unittest.skip("Need to reset epsilon after debugging")
    def test_dict(self):
        """ Verifies that flowDict entries have not changed during previous tests"""
        self.assertEqual( vf.flowDict['eps'],3.0e-2)
        self.assertEqual( vf.flowDict['alpha'],25.)
        self.assertEqual( vf.flowDict['beta'],10.)
        self.assertEqual( vf.flowDict['omega'],5.)
        self.assertEqual( vf.size, 5*15*7*3*21)
        self.assertEqual( vf.N, 21)
        self.assertEqual( vf.nd,3)
        self.assertIsNone(vf.verify())
        return



if __name__ == '__main__':
    unittest.main()
