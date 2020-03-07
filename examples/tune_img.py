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
from lie_conv.examples.train_img import makeTrainer,Trial
from oil.tuning.study import Study

if __name__ == '__main__':
    thestudy = Study(Trial,{},study_name='tune_se2_img_hypers')
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    config_spec.update({
        'num_epochs':200,
        'net_config':{'k':128,'total_ds':.1,'fill':[1/15,1/12,1/8,1/22,1/30],'nbhd':25,
                    'liftsamples':4, 'group':[lieGroups.SE2(a) for a in [.2,.25,.3]]},
        'split':{'train':10000,'val':2000},
        'lr':[3e-3,1e-3],'bs':25,'aug':True,
        'save':True,
    })
    thestudy.run(num_trials=-1,new_config_spec=config_spec,ordered=False)
    print(thestudy.results_df())