import numpy as np
import torch
import scipy as sp
import scipy.linalg
import unittest

from lie_conv.lieGroups import SO3,SE3,SE2,SO2
test_groups = [SO2(),SO3(),SE3()]
class TestGroups(unittest.TestCase):
    def test_exp_correct(self,num_trials=3,tol=1e-4):
        for group in test_groups:
            for i in np.linspace(-5,2,10):
                for _ in range(num_trials):
                    w = torch.rand(group.embed_dim)*(10**i)
                    R = group.exp(w).data.numpy()
                    A = group.components2matrix(w).data.numpy()
                    R2 = sp.linalg.expm(A)
                    err = np.abs(R2-R).mean()
                    if err>tol: print(f'{group} exp check failed with {err:.2E} at |w|={w.abs().mean():.2E}')
                    self.assertTrue(err<tol)
    def test_log_correct(self,num_trials=3,tol=1e-4):
        for group in test_groups:
            for i in np.linspace(-2,2,10):
                for _ in range(num_trials):
                    w = (torch.rand(group.embed_dim)*(10**i))
                    A = group.components2matrix(w).data.numpy()
                    R = sp.linalg.expm(A)
                    lR = sp.linalg.logm(R,disp=False)
                    logR = group.matrix2components(torch.from_numpy(lR[0].real.astype(np.float32)))
                    logR2 = group.log(torch.from_numpy(R.astype(np.float32)))
                    err = (((logR2-logR).abs()).mean()/logR.abs().mean()).data
                    if err>tol: 
                        print(f'{group} log check failed with {err:.2E} at |w|={w.abs().mean():.2E}')
                        print(logR,logR2,w)
                    self.assertTrue(err<max(tol,lR[-1]))


    def test_exp_log_cycle_consistent(self,num_trials=3,tol=3e-4):
        for group in test_groups:
            for i in np.linspace(-2,2,10):
                for _ in range(num_trials):
                    w = torch.rand(3,5,group.embed_dim)*(10**i)
                    R = group.exp(w)
                    logR = group.log(R)
                    R2 = group.exp(logR)
                    err = ((R2-R).abs()).mean()/logR.abs().mean()
                    if err>tol: print(f'{group} cycle failed with {err:.2E} at |w|={w.norm():.2E}')
                    self.assertTrue(err<tol)
        
if __name__=="__main__":
    unittest.main()