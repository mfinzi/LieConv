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
import lie_conv.lieGroups as lieGroups
import lie_conv.lieConv as lieConv
from lie_conv.lieConv import ImgLieResnet
from lie_conv.datasets import MnistRotDataset
from lie_conv.examples.train_molec import makeTrainer,Trial
from oil.tuning.study import Study

def trial_name(cfg):
    ncfg = cfg['net_config']
    return f"molec_f{ncfg['fill']}_n{ncfg['nbhd']}_{ncfg['group']}_{cfg['lr']}"

if __name__ == '__main__':
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    config_spec.update({
        'num_epochs':500,
        'net_config':{'fill':[1/2,1.,3/4],'nbhd':[25,20,15],'liftsamples':4,
                     'group':[lieGroups.SE3(a) for a in [.15,.2,.25]]},
        'lr':[1e-3],'bs':75,'aug':True,'task':'homo',
        'subsample':20000,
        'save':True,
        'trainer_config':{'log_dir':'se3_molec_study','log_suffix':lambda cfg:trial_name(cfg)},
        
    })
    thestudy = Study(Trial,config_spec,study_name='tune_se3_molec_hypers')
    thestudy.run(num_trials=-1,ordered=False)
    print(thestudy.results_df())