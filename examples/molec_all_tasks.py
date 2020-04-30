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


def bigG(cfg):
    return isinstance(cfg['net_config']['group'],(SE3,SO3))

if __name__ == '__main__':
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    config_spec.update({
        'num_epochs':500,
        'net_config':{'fill':lambda cfg: (1.,1/2)[bigG(cfg)],'nbhd':lambda cfg: (100,25)[bigG(cfg)],
        'group':T(3),'liftsamples':lambda cfg: (1,4)[bigG(cfg)]},'recenter':lambda cfg: bigG(cfg),
        'lr':3e-3,'bs':lambda cfg: (100,75)[bigG(cfg)],'task':['alpha','gap','homo','lumo','mu','Cv','G','H','r2','U','U0','zpve'],
        'trainer_config':{'log_dir':'molec_all_tasks4','log_suffix':lambda cfg:trial_name(cfg)},
    })
    config_spec = argupdated_config(config_spec,namespace=(moleculeTrainer,lieGroups))
    thestudy = Study(Trial,config_spec,study_name='molec_all_tasks4')
    thestudy.run(num_trials=-1,ordered=True)
    print(thestudy.results_df())
