from oil.tuning.configGenerator import flatten_dict
import os

class DynamicsTrial(object):
    """ Assumes trainer is an object of type Trainer, trains for num_epochs which may be an
        integer or an iterable containing intermediate points at which to save.
        Pulls out special (resume, save, early_stop_metric, local_rank) args from the cfg """

    def __init__(self, make_trainer):
        self.make_trainer = make_trainer
        self.trainer = None

    def __call__(self, cfg, i=None):
        cfg.pop('local_rank', None)  # TODO: properly handle distributed
        if i is not None:
            orig_suffix = cfg.setdefault('trainer_cfg',{}).get('log_suffix','')
            cfg['trainer_cfg']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
        trainer = self.make_trainer(**cfg)
        trainer.logger.add_scalars('config', flatten_dict(cfg))

        # trainer.train(round(cfg['n_epochs'] / 2))
        # trainer.model.load_state_dict(trainer.ckpt[1])
        # # trainer.model.train()
        # # trainer.model.eval()

        # trainer.train(round(cfg['n_epochs'] / 2))
        trainer.train(cfg['num_epochs'])
        save = cfg.pop('save',True)
        # trainer.model.load_state_dict(trainer.ckpt[1])
        # trainer.model.train()
        # trainer.model.eval()

        # # cast to double
        # trainer.model.double()
        # optimizer_sd = trainer.optimizer.state_dict()
        # for val in optimizer_sd['state'].values():
        #     if torch.is_tensor(val):
        #         val.double()
        # trainer.optimizer.load_state_dict(optimizer_sd)
        # trainer.dataloaders = {k:LoaderTo(v, dtype=torch.float64) for k, v in trainer.dataloaders.items()}
        # trainer.train(int(round(0.1*cfg['n_epochs'])))
        # if trainer.traj_data is not None:
        #     trainer.logger.add_scalars('metrics', {'rollout_mse': trainer._get_rollout_mse()})
        outcome = trainer.logger.scalar_frame.iloc[trainer.ckpt[0]:trainer.ckpt[0]+1]
        if save: trainer.logger.save_object(trainer.ckpt[1],suffix=f'checkpoints/{trainer.ckpt[0]}.state')
        self.trainer = trainer
        return cfg, outcome
