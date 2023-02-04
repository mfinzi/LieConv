import copy
import os

import numpy as np
import torch
from oil.datasetup.datasets import split_dataset
from oil.tuning.args import argupdated_config
from oil.tuning.configGenerator import flatten_dict
from oil.tuning.study import Study
from oil.utils.utils import LoaderTo, islice, FixedNumpySeed, cosLr
from torch.optim import Adam
from torch.utils.data import DataLoader

from lie_conv.datasets import SpringDynamics
from lie_conv.dynamicsTrainer import IntegratedDynamicsTrainer, HLieResNet
from lie_conv.lieGroups import SE2


def make_trainer(*, network, net_cfg, lr=1e-2, n_train: int = 5000, regen=False,
                 dtype=torch.float32, device=torch.device('cuda'), bs=200, num_epochs=2,
                 trainer_config={'log_dir': 'data_scaling_study_final'}):
    # Create Training set and model
    splits = {'train': n_train, 'val': min(n_train, 2000), 'test': 2000}
    dataset = SpringDynamics(n_systems=100000, regen=regen)
    with FixedNumpySeed(0):
        datasets = split_dataset(dataset, splits)
    model = network(**net_cfg).to(device=device, dtype=dtype)
    # Create train and Dev(Test) dataloaders and move elems to gpu
    dataloaders = {k: LoaderTo(DataLoader(v, batch_size=min(bs, n_train), num_workers=0, shuffle=(k == 'train')),
                               device=device, dtype=dtype) for k, v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'], len(dataloaders['val']))
    # Initialize optimizer and learning rate schedule
    opt_constr = lambda params: Adam(params, lr=lr)
    lr_sched = cosLr(num_epochs)
    return IntegratedDynamicsTrainer(model, dataloaders, opt_constr, lr_sched,
                                     log_args={'timeFrac': 1 / 4, 'minPeriod': 0.0}, **trainer_config)


class MiniTrial(object):

    def __init__(self, make_trainer):
        self.make_trainer = make_trainer

    def __call__(self, cfg, i=None):
        cfg.pop('local_rank', None)  # TODO: properly handle distributed
        if i is not None:
            orig_suffix = cfg.setdefault('trainer_config', {}).get('log_suffix', '')
            cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix, f'trial{i}/')
        trainer = self.make_trainer(**cfg)
        trainer.logger.add_scalars('config', flatten_dict(cfg))
        trainer.train(cfg['num_epochs'])
        outcome = trainer.logger.scalar_frame.iloc[-1:]
        trainer.logger.save_object(trainer.model.state_dict(), suffix=f'checkpoints/final.state')
        trainer.logger.save_object(trainer.logger.scalar_frame, suffix=f'scalars.df')

        return cfg, outcome


Trial = MiniTrial(make_trainer)

best_hypers = [
    {'network': HLieResNet, 'net_cfg': {'k': 384, 'num_layers': 1, 'group': SE2()}, 'lr': 3e-4},
]

if __name__ == '__main__':
    config_spec = copy.deepcopy(make_trainer.__kwdefaults__)
    config_spec.update({
        'num_epochs': (lambda cfg: int(np.sqrt(1e7 / cfg['n_train']))),
        'n_train': [10, 25, 50, 100, 400, 1000, 3000, 10000, 30000, 100000 - 4000],
    })
    config_spec = argupdated_config(config_spec)
    name = 'simple_spring_dynamics_playground'
    num_repeats = 1
    thestudy = Study(Trial, {}, study_name=name, base_log_dir=config_spec['trainer_config'].get('log_dir', None))
    for cfg in best_hypers:
        the_config = copy.deepcopy(config_spec)
        the_config.update(cfg)
        thestudy.run(num_trials=-1 * num_repeats, new_config_spec=the_config, ordered=True)
    print(thestudy.results_df())
