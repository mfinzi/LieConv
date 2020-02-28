import torch
from lie_conv.datasets import QM9datasets
from moleculeTrainer import MoleculeTrainer,MolecLieResNet
from moleculeTrainer import MolecResNet
from oil.utils.utils import LoaderTo
from corm_data.collate import collate_fn
from torch.utils.data import DataLoader
from lie_conv.lieGroups import norm, SO3
import unittest
import numpy as np

class TestPad(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        device = torch.device('cuda')
        datasets, num_species, charge_scale = QM9datasets()
        dataloaders = {key:LoaderTo(DataLoader(dataset,batch_size=5,num_workers=0,shuffle=(key=='train'),
                    pin_memory=True,collate_fn=collate_fn),device) for key,dataset in datasets.items()}
        for mb in dataloaders['train']:
            self.mb = mb
            break
        #meanstd = datasets['train'].stats['homo']
        self.model = MolecLieResNet(num_species,charge_scale,nbhd=10,mean=True,radius=1.5,liftsamples=6).to(device)
    def test_padding_noeffects(self,tol=1e-4):
        self.model.eval()
        mb = self.mb
        outs = self.model(mb).cpu().data.numpy()
        #print('first done')
        outs2 = self.model(mb).cpu().data.numpy()
        #print('second done')
        mb['positions'] = torch.cat([mb['positions'],torch.randn_like(mb['positions'])],dim=1)
        mb['one_hot'] = torch.cat([mb['one_hot'],torch.zeros_like(mb['one_hot'])],dim=1)
        mb['charges'] = torch.cat([mb['charges'],torch.zeros_like(mb['charges'])],dim=1)
        mb['atom_mask'] = torch.cat([mb['atom_mask'],torch.zeros_like(mb['atom_mask'])>1],dim=1)
        outs3 = self.model(mb).cpu().data.numpy()
        diff = np.abs(outs2-outs).mean()/np.abs(outs).mean()
        print('run through twice rel err:',diff)
        diff = np.abs(outs2-outs3).mean()/np.abs(outs2).mean()
        print('increase padding rel err:',diff)
        self.assertTrue(diff<tol)
    def test_equivariance(self,tol=1e-4):
        self.model.eval()
        mb = self.mb
        outs = self.model(mb).cpu().data.numpy()
        #print('first done')
        outs2 = self.model(mb).cpu().data.numpy()
        #print('second done')
        bs = mb['positions'].shape[0]
        q = torch.randn(bs,1,4,device=mb['positions'].device,dtype=mb['positions'].dtype)
        q /= norm(q,dim=-1).unsqueeze(-1)
        theta_2 = torch.atan2(norm(q[...,1:],dim=-1),q[...,0]).unsqueeze(-1)
        so3_elem = theta_2*q[...,1:]
        Rs = SO3.exp(so3_elem)
        #print(Rs.shape)
        #print(mb['positions'].shape)
        mb['positions'] = (Rs@mb['positions'].unsqueeze(-1)).squeeze(-1)
        outs3 = self.model(mb).cpu().data.numpy()
        diff = np.abs(outs2-outs).mean()/np.abs(outs).mean()
        print('run through twice rel err:',diff)
        diff = np.abs(outs2-outs3).mean()/np.abs(outs2).mean()
        print('rotation equivariance rel err:',diff)
        self.assertTrue(diff<tol)

    def test_random_sample_ball(self):
        dists = torch.arange(8).reshape(1,2,4)
        k=2
        bs,m,n = dists.shape
        within_ball = (dists < 3) | (dists>6) # (bs,m,n)
        print(within_ball)
        # random_perm = torch.rand(bs,m,n).argsort(dim=-1) # (bs,m,n)
        # print(random_perm)
        # inverse_perm_indices = random_perm.argsort(dim=-1) # (bs,m,n)
        # print(inverse_perm_indices)

        # B = torch.arange(bs)[:,None,None].expand(*random_perm.shape)
        # M = torch.arange(m)[None,:,None].expand(*random_perm.shape)
        # print("inv[perm]",inverse_perm_indices[B,M,random_perm])
        #print(within_ball[B,M,random_perm].shape)
        # valid,permed_idx = torch.topk((within_ball[B,M,random_perm]).float(),k,dim=-1,largest=True,sorted=False)
        #print(permed_idx) # (bs,m,k)
        # B = torch.arange(bs)[:,None,None].expand(*permed_idx.shape)
        # M = torch.arange(m)[None,:,None].expand(*permed_idx.shape)
        # nbhd_idx = random_perm[B,M,permed_idx]
        valid, permed_nbhd_idx =torch.topk(within_ball+torch.rand(bs,m,n,device=within_ball.device),
                                                                k,dim=-1,largest=True,sorted=False)
        valid = (valid>1)
        #nbhd_idx = rand_perm[B.expand(*permed_nbhd_idx.shape),M.expand(*permed_nbhd_idx.shape),permed_nbhd_idx]
        nbhd_idx = permed_nbhd_idx
        print('nbhd_idx',nbhd_idx)
        print('valid',valid)
        self.assertTrue(True)

class TestEquivariance(unittest.TestCase):
    pass

        
if __name__=="__main__":
    unittest.main()