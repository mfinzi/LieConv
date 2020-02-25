import copy
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, FixedNumpySeed, cosLr
from lie_conv.datasets import SpringDynamics
from lie_conv import datasets
from lie_conv.dynamicsTrainer import IntegratedDynamicsTrainer, FCHamNet, RawDynamicsNet, LieResNetT2
from lie_conv.graphnets import OGN,HOGN, VOGN
import lie_conv.liegroups as liegroups
from lie_conv import dynamicsTrainer
from lie_conv.dynamics_trial import DynamicsTrial


def makeTrainer(*,network,net_cfg,lr=1e-2,n_train=3000,regen=False,dataset=SpringDynamics,
                dtype=torch.float32,device=torch.device('cuda'),bs=200,num_epochs=2,
                trainer_config={}):
    # Create Training set and model
    splits = {'train':n_train,'val':200,'test':2000}
    dataset = dataset(n_systems=10000,regen=regen)
    with FixedNumpySeed(0):
        datasets = split_dataset(dataset,splits)
    model = network(sys_dim=dataset.sys_dim,d=dataset.space_dim,**net_cfg).to(device=device,dtype=dtype)
    # Create train and Dev(Test) dataloaders and move elems to gpu
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,n_train),num_workers=0,shuffle=(k=='train')),
                                device=device,dtype=dtype) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],len(dataloaders['val']))
    # Initialize optimizer and learning rate schedule
    opt_constr = lambda params: Adam(params, lr=lr)
    lr_sched = cosLr(num_epochs)
    return IntegratedDynamicsTrainer(model,dataloaders,opt_constr,lr_sched,
                                    log_args={'timeFrac':1/4,'minPeriod':0.0},**trainer_config)

best_hypers = {
    LieResNetT2: {'net_cfg':{'k':384, 'num_layers':4},'lr':1e-3},
    VOGN: {'net_cfg':{'k':512},'lr':3e-3},
    HOGN: {'net_cfg':{'k':256},'lr':1e-2},
    OGN: {'net_cfg':{'k':256},'lr':1e-2},
    FCHamNet: {'net_cfg':{'k':256,'num_layers':4},'lr':1e-2},
    RawDynamicsNet: {'net_cfg':{'k':256},'lr':3e-3},
}

Trial = DynamicsTrial(makeTrainer)
if __name__=='__main__':
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults.update({'network':LieResNetT2,'net_cfg':{'k':384, 'num_layers':4},'lr':1e-3})
    #defaults['early_stop_metric']='val_MSE'
    print(Trial(argupdated_config(defaults,namespace=(dynamicsTrainer,liegroups,datasets))))