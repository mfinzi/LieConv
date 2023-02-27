import torch
from oil.datasetup.datasets import split_dataset
from oil.utils.utils import LoaderTo
from torch.utils.data import DataLoader

from lie_conv.datasets import MnistRotDataset
from lie_conv.lieConv import ImgGCNLieResnet, ImgLieResnet, LieConvGCN
from lie_conv.lieGroups import SO2, T, Trivial


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


def test_gnn_conv_equivariance(bs: int = 2, device='cpu', h: int = 10, w: int = 10, liftsamples: int = 1,
                               group=T(2), n_mc_samples: int = 25, ds_frac: float = 1., fill: float = 1.,
                               conv_layer=LieConvGCN):
    device = torch.device(device)

    # Construct coordinate grid
    i = torch.linspace(-h / 2., h / 2., h)
    j = torch.linspace(-w / 2., w / 2., w)
    coords = torch.stack(torch.meshgrid([i, j]), dim=-1).float().view(-1, 2).unsqueeze(0).repeat(bs, 1, 1)
    coords[1, :, 0] += 1  # shift x by 1

    values = torch.randn((bs, coords.shape[1], 1))

    mask = torch.ones(bs, values.shape[1], device=device) > 0

    abq, lifted_vals, lifted_mask = group.lift((coords, values, mask), liftsamples)

    gnn_conv = conv_layer(1, 1, mc_samples=n_mc_samples, ds_frac=ds_frac, bn=True, act='swish',
                          mean=True, group=group, fill=fill, cache=True, knn=False)

    conv_vals = gnn_conv.point_convolve(abq, lifted_vals, lifted_mask)

    assert (conv_vals[0] == conv_vals[1]).all(), 'Error - layer is not equivariant!'


if __name__ == "__main__":
    test_gnn_conv_equivariance()
