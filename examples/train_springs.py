import copy, warnings
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
from oil.tuning.study import train_trial
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, FixedNumpySeed, cosLr
from lie_conv.datasets import SpringDynamics
from lie_conv import datasets
from lie_conv.dynamicsTrainer import IntegratedDynamicsTrainer, FC, HLieResNet

import lie_conv.lieGroups as lieGroups
from lie_conv.lieGroups import Tx
from lie_conv import dynamicsTrainer
#from lie_conv.dynamics_trial import DynamicsTrial
try:
    import lie_conv.graphnets as graphnets
except ImportError:
    import lie_conv.lieConv as graphnets
    warnings.warn('Failed to import graphnets. Please install using \
                `pip install .[GN]` for this functionality', ImportWarning)

def makeTrainer(*,network=FC,net_cfg={},lr=1e-2,n_train=3000,regen=False,dataset=SpringDynamics,
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

Trial = train_trial(makeTrainer)
if __name__=='__main__':
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults['save']=False
    defaults['trainer_config']['early_stop_metric']='val_MSE'
    print(Trial(argupdated_config(defaults,namespace=(dynamicsTrainer,lieGroups,datasets,graphnets))))