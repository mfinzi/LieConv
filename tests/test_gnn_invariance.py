import torch
from oil.datasetup.datasets import split_dataset
from oil.utils.utils import LoaderTo
from torch.utils.data import DataLoader

from lie_conv.datasets import MnistRotDataset
from lie_conv.lieConv import ImgGCNLieResnet, ImgLieResnet
from lie_conv.lieGroups import SO2


def test_gnn_model_invariance(device='cpu', dataset=MnistRotDataset, bs: int = 20,
                              net_config: dict = {'k': 128, 'total_ds': 1., 'fill': 1., 'nbhd': 50,
                                                  'group': SO2(0.)}, MAX_ERR: float = 0.3):
    datasets = split_dataset(dataset(f'~/datasets/{dataset}/'), splits={'train': bs})

    device = torch.device(device)

    model = ImgGCNLieResnet(num_targets=datasets['train'].num_targets, **net_config).to(device)
    model = torch.nn.Sequential(datasets['train'].default_aug_layers(), model)

    dataloaders = {k: LoaderTo(DataLoader(v, batch_size=bs, shuffle=(k == 'train'),
                                          num_workers=0, pin_memory=False), device) for k, v in datasets.items()}
    data = next(iter(dataloaders['train']))[0]

    with torch.inference_mode():
        norm_res = model(data)
        rot_data = data
        for _ in range(3):
            rot_data = rot_data.transpose(-2, -1).flip(-2)
            rot_res = model(rot_data)

            print(abs(norm_res - rot_res).max())
            assert abs(norm_res - rot_res).max() < MAX_ERR, \
                'Error - too high error, model is not equivariant!'

if __name__ == "__main__":
    test_gnn_model_invariance()
