import copy
from oil.tuning.study import Study, train_trial
from oil.datasetup.datasets import split_dataset
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, FixedNumpySeed
from lie_conv.datasets import SpringDynamics
from lie_conv.dynamicsTrainer import IntegratedDynamicsTrainer,FCHamNet, RawDynamicsNet, LieResNetT2
from graphnets import OGN,HOGN, VOGN


def makeTrainer(*,network,net_cfg,lr=1e-2,n_train=3000,regen=False,
                dtype=torch.float32,device=torch.device('cuda'),bs=200,num_epochs=10,
                trainer_config={}):
    # Create Training set and model
    splits = {'train':n_train,'val':min(n_train,2000),'test':2000}
    dataset = SpringDynamics(N=100000,regen=regen)
    with FixedNumpySeed(0):
        datasets = split_dataset(dataset,splits)
    model = network(**net_cfg).to(device=device,dtype=dtype)
    # Create train and Dev(Test) dataloaders and move elems to gpu
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,n_train),num_workers=0,shuffle=(k=='train')),
                                device=device,dtype=dtype) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],len(dataloaders['val']))
    # Initialize optimizer and learning rate schedule
    opt_constr = lambda params: Adam(params, lr=lr)
    lr_sched = lambda e: 1#cosLr(cfg['num_epochs'])
    return IntegratedDynamicsTrainer(model,dataloaders,opt_constr,lr_sched,
                                    log_args={'timeFrac':1/4,'minPeriod':0.0},**trainer_config)

Trial = train_trial(makeTrainer)
#r = lambda *options: np.random.choice(options)
hyper_choices = {
    LieResNetT2: {'net_cfg':{'k':[384,512,768],'num_layers':[4,6,8]},'lr':[1e-3,3e-3,1e-2]},
    VOGN: {'net_cfg':{'k':[384,512,768]},'lr':[1e-3,3e-3,1e-2]},
    HOGN: {'net_cfg':{'k':[256,384,512]},'lr':[1e-3,3e-3,1e-2]},
    OGN: {'net_cfg':{'k':[128,256,512]},'lr':[1e-3,3e-3,1e-2]},
    FCHamNet: {'net_cfg':{'k':[128,256,512],'num_layers':[2,4,6]},'lr':[1e-3,3e-3,1e-2]},
    RawDynamicsNet: {'net_cfg':{'k':[128,256,512],'num_layers':[2,4,6]},'lr':[1e-3,3e-3,1e-2]},
}

if __name__ == '__main__':
    thestudy = Study(Trial,{},study_name='tune_dynamics_hypers2')
    for network, net_config_spec in hyper_choices.items():
        config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
        config_spec.update({
            'network':network,
            'num_epochs':100, 'n_train':3000,
            'early_stop_metric':'val_MSE',
        })
        config_spec.update(net_config_spec)
        thestudy.run(num_trials=-1,new_config_spec=config_spec,ordered=True)
    print(thestudy.results_df())