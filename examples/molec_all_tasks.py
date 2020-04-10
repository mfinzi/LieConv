import torch
from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, islice
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.utils.parallel import try_multigpu_parallelize
from oil.model_trainers.classifier import Classifier
from functools import partial
from torch.optim import Adam
from oil.tuning.args import argupdated_config
import copy
import lie_conv.moleculeTrainer as moleculeTrainer
import lie_conv.lieGroups as lieGroups
from lie_conv.lieGroups import T,Trivial,SE3,SO3
import lie_conv.lieConv as lieConv
from lie_conv.lieConv import ImgLieResnet
from lie_conv.datasets import MnistRotDataset
from examples.train_molec import makeTrainer,Trial
from oil.tuning.study import Study

def trial_name(cfg):
    ncfg = cfg['net_config']
    return f"molec_f{ncfg['fill']}_n{ncfg['nbhd']}_{ncfg['group']}_{cfg['lr']}"

def nsamples(cfg):
    return 4 if isinstance(cfg['net_config']['group'],(SE3,SO3)) else 1

if __name__ == '__main__':
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    config_spec.update({
        'num_epochs':500,
        'net_config':{'fill':1.,'nbhd':25,'liftsamples':lambda cfg: nsamples(cfg)},
        'lr':3e-3,'bs':75,'task':['alpha','gap','homo','lumo','mu','Cv','G','H','r2','U','U0','zpve']
        'recenter':True,'trainer_config':{'log_dir':'se3_molec_all','log_suffix':lambda cfg:trial_name(cfg)},
    })
    config_spec = argupdated_config(config_spec,namespace=(moleculeTrainer,lieGroups)))
    thestudy = Study(Trial,config_spec,study_name='tune_se3_molec_hypers')
    thestudy.run(num_trials=-1,ordered=True)
    print(thestudy.results_df())