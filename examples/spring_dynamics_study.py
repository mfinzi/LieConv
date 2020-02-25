import copy
from oil.tuning.study import Study, train_trial
from oil.tuning.args import argupdated_config
from oil.datasetup.datasets import split_dataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, FixedNumpySeed, cosLr
from lie_conv.datasets import SpringDynamics
from lie_conv.dynamicsTrainer import IntegratedDynamicsTrainer,FCHamNet, RawDynamicsNet, LieResNetT2, HLieResNet
from lie_conv.liegroups import T, SO2, Trivial
from graphnets import OGN,HOGN, VOGN
from lie_conv.dynamics_trial import DynamicsTrial
from oil.tuning.configGenerator import sample_config, flatten_dict,grid_iter
import os

def makeTrainer(*,network,net_cfg,lr=1e-2,n_train=5000,regen=False,
                dtype=torch.float32,device=torch.device('cuda'),bs=200,num_epochs=2,
                trainer_config={'log_dir':'data_scaling_study_final'}):
    # Create Training set and model
    splits = {'train':n_train,'val':min(n_train,2000),'test':2000}
    dataset = SpringDynamics(n_systems=100000, regen=regen)
    with FixedNumpySeed(0):
        datasets = split_dataset(dataset,splits)
    model = network(**net_cfg).to(device=device,dtype=dtype)
    # Create train and Dev(Test) dataloaders and move elems to gpu
    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=min(bs,n_train),num_workers=0,shuffle=(k=='train')),
                                device=device,dtype=dtype) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],len(dataloaders['val']))
    # Initialize optimizer and learning rate schedule
    opt_constr = lambda params: Adam(params, lr=lr)
    lr_sched = cosLr(num_epochs)
    return IntegratedDynamicsTrainer(model,dataloaders,opt_constr,lr_sched,
                                    log_args={'timeFrac':1/4,'minPeriod':0.0},**trainer_config)

class MiniTrial(object):

    def __init__(self, make_trainer):
        self.make_trainer = make_trainer

    def __call__(self, cfg, i=None):
        cfg.pop('local_rank', None)  # TODO: properly handle distributed
        if i is not None:
            orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
            cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
        trainer = self.make_trainer(**cfg)
        trainer.logger.add_scalars('config', flatten_dict(cfg))
        trainer.train(cfg['num_epochs'])
        outcome = trainer.logger.scalar_frame.iloc[-1:]
        trainer.logger.save_object(trainer.model.state_dict(),suffix=f'checkpoints/final.state')
        trainer.logger.save_object(trainer.logger.scalar_frame,suffix=f'scalars.df')

        return cfg, outcome

Trial = MiniTrial(makeTrainer)

best_hypers = [
    {'network':RawDynamicsNet,'net_cfg':{'k':256},'lr':3e-3},
    {'network':FCHamNet,'net_cfg':{'k':256,'num_layers':4},'lr':1e-2},
    {'network':HLieResNet, 'net_cfg':{'k':384, 'num_layers':4, 'group':T(2)}, 'lr':1e-3},
    {'network':HLieResNet, 'net_cfg':{'k':384, 'num_layers':4, 'group':SO2()}, 'lr':3e-4},
    {'network':HLieResNet, 'net_cfg':{'k':384, 'num_layers':4, 'group':Trivial(2)}, 'lr':1e-3},
    {'network':VOGN,'net_cfg':{'k':512},'lr':3e-3},
    {'network':HOGN,'net_cfg':{'k':256},'lr':1e-2},
    {'network':OGN,'net_cfg':{'k':256},'lr':1e-2},
]

if __name__ == '__main__':
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    config_spec.update({
        'num_epochs':(lambda cfg: int(np.sqrt(1e7/cfg['n_train']))),
        'n_train':[10,25,50,100,400,1000,3000,10000,30000,100000-4000],
    })
    config_spec = argupdated_config(config_spec)
    name = 'data_scaling_dynamics_final'#config_spec.pop('study_name')
    num_repeats = 3#config_spec.pop('num_repeats')
    thestudy = Study(Trial,{},study_name=name,base_log_dir=config_spec['trainer_config'].get('log_dir',None))
    for cfg in best_hypers:
        the_config = copy.deepcopy(config_spec)
        the_config.update(cfg)
        thestudy.run(num_trials=-1*num_repeats,new_config_spec=the_config,ordered=True)
    print(thestudy.results_df())