import torch
from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, islice, FixedNumpySeed
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.utils.parallel import try_multigpu_parallelize
from oil.model_trainers.classifier import Classifier
from functools import partial
from torch.optim import Adam
from oil.tuning.args import argupdated_config
import copy
import lie_conv.lieGroups as lieGroups
import lie_conv.lieConv as lieConv
from lie_conv.lieConv import ImgLieResnet
from lie_conv.datasets import MnistRotDataset
from torchcontrib.optim import SWA
import os
from oil.tuning.configGenerator import flatten_dict

def makeTrainer(*, dataset=MnistRotDataset, network=ImgLieResnet, num_epochs=100,
                bs=50, lr=3e-3, swa_lr=1e-4,aug=True, optim=Adam, device='cuda', trainer=Classifier,
                split={'train':12000}, small_test=False, net_config={}, opt_config={},
                trainer_config={'log_dir':None}):

    # Prep the datasets splits, model, and dataloaders
    with FixedNumpySeed(0):
        datasets = split_dataset(dataset(f'~/datasets/{dataset}/'),splits=split)
    datasets['test'] = dataset(f'~/datasets/{dataset}/', train=False)
    device = torch.device(device)
    model = network(num_targets=datasets['train'].num_targets,**net_config).to(device)
    if aug: model = torch.nn.Sequential(datasets['train'].default_aug_layers(),model)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    if small_test: dataloaders['test'] = islice(dataloaders['test'],1+len(dataloaders['train'])//10)
    # Add some extra defaults if SGD is chosen
    opt_constr = lambda params: optim(params,lr=lr,**opt_config)
    lr_sched = lambda e:1#cosLr(num_epochs)
    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

class swa_finetune(object):
    """ Assumes trainer is an object of type Trainer, trains for num_epochs which may be an
        integer or an iterable containing intermediate points at which to save.
        Pulls out special (resume, save, early_stop_metric, local_rank) args from the cfg """
    def __init__(self,make_trainer):
        self.make_trainer = make_trainer
    def __call__(self,cfg,i=None):
        cfg.pop('local_rank',None) #TODO: properly handle distributed
        resume = cfg.pop('resume',False)
        save = cfg.pop('save',False)
        swa_lr = cfg.pop('swa_lr')
        if i is not None:
            orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
            cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
        trainer = self.make_trainer(**cfg)
        try: cfg['params(M)'] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)/10**6
        except AttributeError: pass
        trainer.logger.add_scalars('config',flatten_dict(cfg))
        epochs = cfg['num_epochs']
        if resume: trainer.load_checkpoint(None if resume==True else resume)
        trainer.optimizer = SWA(trainer.optimizer,swa_start=0,swa_freq=100,swa_lr=swa_lr)
        trainer.train_to(epochs)
        trainer.optimizer.swap_swa_sgd()
        trainer.optimizer.bn_update(trainer.dataloaders['train'],trainer.model)
        trainer.logStuff(trainer.epoch*len(trainer.dataloaders['train']))
        if save: cfg['saved_at']=trainer.save_checkpoint()
        outcome = trainer.ckpt['outcome']
        del trainer
        return cfg, outcome

if __name__=="__main__":
    Trial = swa_finetune(makeTrainer)
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults['save'] = False
    defaults['resume'] = True
    Trial(argupdated_config(defaults,namespace=(lieConv,lieGroups)))
