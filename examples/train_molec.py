import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, cosLr, FixedNumpySeed
from oil.tuning.args import argupdated_config
from oil.tuning.study import train_trial
from oil.utils.parallel import try_multigpu_parallelize
from lie_conv.datasets import QM9datasets
from corm_data.collate import collate_fn
from lie_conv.moleculeTrainer import MolecLieResNet, MoleculeTrainer
from oil.datasetup.datasets import split_dataset
import lie_conv.moleculeTrainer as moleculeTrainer
import lie_conv.lieGroups as lieGroups
import functools
import copy


def makeTrainer(*, task='homo', device='cuda', lr=3e-3, bs=75, num_epochs=500,network=MolecLieResNet, 
                net_config={'k':1536,'nbhd':100,'act':'swish','group':lieGroups.T(3),
                'bn':True,'aug':True,'mean':True,'num_layers':6}, recenter=False,
                subsample=False, trainer_config={'log_dir':None,'log_suffix':''}):#,'log_args':{'timeFrac':1/4,'minPeriod':0}}):
    # Create Training set and model
    device = torch.device(device)
    with FixedNumpySeed(0):
        datasets, num_species, charge_scale = QM9datasets()
        if subsample: datasets.update(split_dataset(datasets['train'],{'train':subsample}))
    ds_stats = datasets['train'].stats[task]
    if recenter:
        m = datasets['train'].data['charges']>0
        pos = datasets['train'].data['positions'][m]
        mean,std = pos.mean(dim=0),pos.std()
        for ds in datasets.values():
            ds.data['positions'] = (ds.data['positions']-mean[None,None,:])/std
    model = network(num_species,charge_scale,**net_config).to(device)
    model,bs = try_multigpu_parallelize(model,bs)
    # Create train and Val(Test) dataloaders and move elems to gpu
    dataloaders = {key:LoaderTo(DataLoader(dataset,batch_size=bs,num_workers=0,
                    shuffle=(key=='train'),pin_memory=False,collate_fn=collate_fn,drop_last=True),
                    device) for key,dataset in datasets.items()}
    # subsampled training dataloader for faster logging of training performance
    dataloaders['Train'] = islice(dataloaders['train'],len(dataloaders['train'])//10)
    
    # Initialize optimizer and learning rate schedule
    opt_constr = functools.partial(Adam, lr=lr)
    cos = cosLr(num_epochs)
    lr_sched = lambda e: min(e / (.01 * num_epochs), 1) * cos(e)
    return MoleculeTrainer(model,dataloaders,opt_constr,lr_sched,
                            task=task,ds_stats=ds_stats,**trainer_config)

Trial = train_trial(makeTrainer)
if __name__=='__main__':
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults['trainer_config']['early_stop_metric']='valid_MAE'
    defaults['save']=False
    print(Trial(argupdated_config(defaults,namespace=(moleculeTrainer,lieGroups))))

    # thestudy = Study(simpleTrial,argupdated_config(config_spec,namespace=__init__),
    #                 study_name="point2d",base_log_dir=log_dir)
    # thestudy.run(ordered=False)
