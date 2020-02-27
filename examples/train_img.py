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


def makeTrainer(*, dataset=MnistRotDataset, network=ImgLieResnet, num_epochs=100,
                bs=50, lr=3e-3, aug=True, optim=Adam, device='cuda', trainer=Classifier,
                split={'train':12000}, small_test=False, net_config={}, opt_config={},
                trainer_config={'log_dir':None}):

    # Prep the datasets splits, model, and dataloaders
    datasets = split_dataset(dataset(f'~/datasets/{dataset}/'),splits=split)
    datasets['test'] = dataset(f'~/datasets/{dataset}/', train=False)
    device = torch.device(device)
    model = network(num_targets=datasets['train'].num_targets,**net_config).to(device)
    if aug: model = torch.nn.Sequential(datasets['train'].default_aug_layers(),model)
    model,bs = try_multigpu_parallelize(model,bs)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    if small_test: dataloaders['test'] = islice(dataloaders['test'],1+len(dataloaders['train'])//10)
    # Add some extra defaults if SGD is chosen
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)
    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)


if __name__=="__main__":
    Trial = train_trial(makeTrainer)
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults['save'] = False
    Trial(argupdated_config(defaults,namespace=(lieConv,lieGroups)))
