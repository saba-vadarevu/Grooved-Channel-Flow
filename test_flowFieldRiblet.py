import unittest
import numpy as np
from scipy.linalg import norm

from flowFieldWavy import *

# K=2, L=7, M=3, N=21
ind1 = np.index_exp[0,3,1,2,11]     # k= 0, l=-4, m=-2, y=yCheb[11]
ind2 = np.index_exp[0,9,2,0,5]      # k= 0, l= 2, m=-1, y=yCheb[5]
ind3 = np.index_exp[0,4,0,2,10]     # k= 0, l=-3, m=-3, y=0.
ind4 = np.index_exp[0,8,4,1,17]     # k= 0, l= 1, m= 1, y=yCheb[17]
ind5 = np.index_exp[0,9,4,0,:]      # k= 0, l= 2, m= 1, y=yCheb
ind6 = np.index_exp[0,3,6,1,:]      # k= 0, l=-4, m= 3, y=yCheb
ind7 = np.index_exp[0,10,3,2,:]     # k= 0, l= 3, m= 0, y=yCheb

# eps_1 = 0.05, eps_2 = 0.025, eps_3 = 0.03
# A_1 = 10%, A_2 = 5%, A_3 = 6%
testDict = getDefaultDict()
testDict.update({'L':3,'M':7,'N':21,'K':0,'eps':5.0e-2,'epsArr':np.array([0.,0.05,0.025,0.03]),'alpha':10., 'beta':2.5,'omega':0.,'isPois':0})
K = testDict['K']; L = testDict['L']; M = testDict['M']; N = testDict['N']
epsArr = testDict['epsArr']

vf = flowFieldRiblet(flowDict=testDict)
L = vf.nx//2; M = vf.nz//2

# yArr is added to u because it must satisfy u(y=\pm1) = \pm 1
#   For all other Fourier modes of u,v,w 
vf[0,L,M,0] = vf.y 
vf[0,L+1,M+1,1] =  (1.-vf.y**4)*epsArr[1] + (1.-vf.y**8)*epsArr[2]  + 1.j* ( epsArr[3]* (1.-vf.y**2)**3 )
vf[0,L-1,M-1,1] =  (1.-vf.y**4)*epsArr[1] + (1.-vf.y**8)*epsArr[2]  - 1.j* ( epsArr[3]* (1.-vf.y**2)**3 )
vf[0,L,M+3,2] = (1.-vf.y**6)*( epsArr[2]+epsArr[3])
vf[0,L,M-3,2] = (1.-vf.y**6)*( epsArr[2]+epsArr[3])

# EQ1 solution from channel flow is only accurate to about 3e-06
#   and 32-bit floats are also not very accurate
tol32 = 5.0e-07
tol1 = 1.0e-05
tol2 = 2.0e-09


class WavyTestCase(unittest.TestCase):
    """ 
    This testing is more to ensure changes I make don't break the code,
        to validate new changes rather than to assist in debugging.
    I shall do three tests
        1) Verify that residual norm for EQ1-flat is below tolerance
        2) Verify that .ddz() and .ddz2() are accurate for a simple flow field
            involving eps_1, eps_2, and eps_3
        3) Verify laminar solutions (periodic in one groove-wavelength)
            produce residual norm below tolerance when introduced in 
            periodic boxes containing integral multiples of these wavelengths with L != 0

    The first test ensures that for eps_q = 0, all the derivatives work fine
    The second test ensures that ddz and ddz2 are accurate for a simple case
        The laminar solutions to be used in the third test come are only valid
        when the second test works
    The third test ensures that ddz and ddz2 for a more complex field than the
        one used in the second test are valid

    I will have a directory called testFields where the solutions for the first 
        and second tests will be stored

    IMPORTANT: In the second test, the derivatives are validated
                only for the flowFieldRiblet defined above the class definition. Do not modify it.
    To ensure that I don't modify it by mistake, I'm including a copy of it here:

    testDict.update({'L':3,'M':7,'N':21,'K':0,'eps':5.0e-2,'epsArr':np.array([0.05,0.025,0.03]),'alpha':25., 'beta':10.,'omega':0.,'isPois':0})
    vf[0,L,M,0] = vf.y 
    vf[0,L+1,M+1,1] =  (1.-vf.y**4)*epsArr[0] + (1.-vf.y**8)*epsArr[1]  + 1.j* ( epsArr[2]* (1.-vf.y**2)**3 )
    vf[0,L-1,M-1,1] =  (1.-vf.y**4)*epsArr[0] + (1.-vf.y**8)*epsArr[1]  - 1.j* ( epsArr[2]* (1.-vf.y**2)**3 )
    vf[0,L,M+3,2] = (1.-yArr**6)*( epsArr[2]+epsArr[1])
    vf[0,L,M-3,2] = (1.-yArr**6)*( epsArr[2]+epsArr[1])

    """
    print("Note: when testing for derivatives, if looking at the last modes in x or z,"+\
            "remember that the general formulae I use do not apply, because one/some of their "+\
            "neighbouring modes are missing")

    #@unittest.skip("Need to reset epsilon after debugging")
    def test_ddx_ddy(self):

        vf0 = h52ff('testFields/eq1.h5')
        pf0 = h52ff('testFields/pres_eq1.h5',pres=True)
        resnorm = vf0.residuals(pField=pf0).appendField(vf0.div()).norm()

        x0 = loadh5('testFields/ribEq1L7M10N30E0000.hdf5')
        vf0 = x0.slice(nd=[0,1,2]); pf0 = x0.getScalar(nd=3)
        resnorm0 = vf0.residuals(pField=pf0).appendField(vf0.div()).norm()

        x5 = loadh5('testFields/ribEq1L7M10N30E0500.hdf5')
        vf0 = x5.slice(nd=[0,1,2]); pf0 = x5.getScalar(nd=3)
        resnorm5 = vf0.residuals(pField=pf0).norm()
        
        x2 = loadh5('testFields/laminarChannel2L0M12N50E0000.hdf5')
        resnorm2 = x2.slice(nd=[0,1,2]).residuals(x2.slice(nd=3)).appendField(x2.slice(nd=[0,1,2]).div()).norm()
        x3 = loadh5('testFields/laminarChannelL0M18N50E0000.hdf5')
        resnorm3 = x3.slice(nd=[0,1,2]).residuals(x3.slice(nd=3)).appendField(x3.slice(nd=[0,1,2]).div()).norm()

        self.assertTrue(0. < resnorm <= tol1)
        self.assertTrue(0. < resnorm0 <= tol2)
        self.assertTrue(0. < resnorm5 <= tol2)
        self.assertTrue(0. < resnorm2 <= tol32)
        self.assertTrue(0. < resnorm3 <= tol32)
        return

    def test_Tderivatives(self):
        Tz, Tzz, Tz2 = Tderivatives(vf.flowDict)
        TzPreCal = 1.j* np.array([0.225, 0.125, 0.125, 0., -0.125, -0.125, -0.225])
        TzzPreCal = np.array([1.6875, 0.625, 0.3125, 0., 0.3125, 0.625, 1.6875])
        Tz2PreCal = np.array([-0.050625, -0.05625, -0.071875, -0.03125, 0.040625, 0.0875, 0.16375,\
                0.0875, 0.040625, -0.03125, -0.071875, -0.05625, -0.050625])


        resnormTz  = np.linalg.norm(Tz-TzPreCal)
        resnormTzz = np.linalg.norm(Tzz - TzzPreCal)
        resnormTz2 = np.linalg.norm(Tz2 - Tz2PreCal)

        self.assertLess( resnormTz , tol32)
        self.assertLess( resnormTzz, tol32)
        self.assertLess( resnormTz2, tol32)
        return
        

    def test_ddz(self):
        partialz = vf.ddz()
        uz = partialz.getScalar(); vz = partialz.getScalar(nd=1); wz = partialz.getScalar(nd=2)
        epsArr = vf.flowDict['epsArr']
        e1 = epsArr[1]; e2 = epsArr[2]; e3 = epsArr[3]
        yCheb = vf.y;  b = vf.flowDict['beta']
        
        # Testing with vf for modes (1,-2), (-1,0), (0,2), (0,0), and (0,-3) 
        #   Refer to documentation for derivation
        ndef = np.linalg.norm
        L = vf.nx//2; M = vf.nz//2
        
        # Validating u_z
        self.assertLess( ndef(uz[0,L+1,M-2,0]),tol32)
        self.assertLess( ndef(uz[0,L-1,M  ,0]),tol32)
        self.assertLess( ndef(uz[0,L+0,M+2,0] + 2.j*b*e2),tol32)
        self.assertLess( ndef(uz[0,L+0,M+0,0]),tol32)
        self.assertLess( ndef(uz[0,L+0,M-3,0] - 3.j*b*e3),tol32)

        # Validating v_z
        vzp1m2 = 3.j*b*e3*(-4.*vf.y**3*e1 - 8.*vf.y**7*e2) + 18.*b*e3*e3*vf.y*(1.-vf.y**2)**2
        vzm1p0 =-1.j*b*e1*(-4.*vf.y**3*e1 - 8.*vf.y**7*e2) +  6.*b*e1*e3*vf.y*(1.-vf.y**2)**2
        self.assertLess( ndef(vz[0,L+1,M-2,0] - vzp1m2  ),tol32)
        self.assertLess( ndef(vz[0,L-1,M  ,0] - vzm1p0 ),tol32)
        self.assertLess( ndef(vz[0,L+0,M+2,0]),tol32)
        self.assertLess( ndef(vz[0,L+0,M+0,0]),tol32)
        self.assertLess( ndef(vz[0,L+0,M-3,0]),tol32)

        # Validating w_z
        self.assertLess( ndef(wz[0,L+1,M-2,0]),tol32)
        self.assertLess( ndef(wz[0,L-1,M  ,0]),tol32)
        self.assertLess( ndef(wz[0,L+0,M+2,0] + 6.j*e1*b* vf.y**5 * (e3+e2) ),tol32)
        self.assertLess( ndef(wz[0,L+0,M+0,0]),tol32)
        self.assertLess( ndef(wz[0,L+0,M-3,0] + 3.j*b* (e3+e2)*(1.-vf.y**6) ),tol32)

        return
    
    #@unittest.skip("Need to reset epsilon after debugging")
    def test_ddz2(self):
        partialzz = vf.ddz2()
        uzz = partialzz.getScalar(); wzz = partialzz.getScalar(nd=2)
        epsArr = vf.flowDict['epsArr']
        e1 = epsArr[1]; e2 = epsArr[2]; e3 = epsArr[3]
        yCheb = vf.y;  b = vf.flowDict['beta']
        Tz, Tzz, Tz2 = Tderivatives(vf.flowDict)
        
        # Testing with u,w for modes (1,-2), (-1,0), (0,2), (0,0), and (0,-3) 
        #   Refer to documentation for derivation
        ndef = np.linalg.norm

        # Validating u_zz
        self.assertLess( ndef(uzz[0,L+1,M-2,0]),tol32)
        self.assertLess( ndef(uzz[0,L-1,M  ,0]),tol32)
        self.assertLess( ndef(uzz[0,L+0,M+2,0] - 4.*b**2 * e2),tol32)
        self.assertLess( ndef(uzz[0,L+0,M+0,0]),tol32)
        self.assertLess( ndef(uzz[0,L+0,M-3,0] - 9.*b**2 * e3),tol32)

        # Validating w_zz
        wzz0p2 = -6.* b**2 *(e2+e3) * (-5.*e1) * vf.y**5  - 30.*(e2+e3)* vf.y**4 * ( Tz2[6-1] + Tz2[6+5] )
        wzz00  = -36.*b**2 *(e2+e3) * (-3.*e3) * vf.y**5  - 30.*(e2+e3)* vf.y**4 * ( Tz2[6-3] + Tz2[6+3] )
        wzz0m3 = -9  *b**2 *(e2+e3) * (1.-vf.y**6)        - 30.*(e2+e3)* vf.y**4 * ( Tz2[6-6] + Tz2[6+0] )
        self.assertLess( ndef(wzz[0,L+1,M-2,0]),tol32)
        self.assertLess( ndef(wzz[0,L-1,M  ,0]),tol32)
        self.assertLess( ndef(wzz[0,L+0,M+2,0] - wzz0p2),tol32)
        self.assertLess( ndef(wzz[0,L+0,M+0,0] - wzz00 ),tol32)
        self.assertLess( ndef(wzz[0,L+0,M-3,0] - wzz0m3),tol32)

        return

    @unittest.skip("Need to reset epsilon after debugging")
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

    @unittest.skip("Need to reset epsilon after debugging")
    def test_weighting(self):
        self.assertAlmostEqual( (vf - weighted2ff(flowDict=vf.flowDict, arr=vf.weighted()) ).norm(), 0.)
        return


    # @unittest.skip("Need to reset epsilon after debugging")
    def test_dict(self):
        """ Verifies that flowDict entries have not changed during previous tests"""
        self.assertEqual( vf.flowDict['eps'],5.0e-2)
        self.assertEqual( vf.flowDict['alpha'],10.)
        self.assertEqual( vf.flowDict['beta'],2.5)
        self.assertEqual( vf.flowDict['omega'],0.)
        self.assertEqual( vf.size, 1*15*7*3*21)
        self.assertEqual( vf.N, 21)
        self.assertEqual( vf.nd,3)
        self.assertIsNone(vf.verify())
        return



if __name__ == '__main__':
    unittest.main()
