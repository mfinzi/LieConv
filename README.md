# LieConv

# Installation
To install as a package, run `pip install git+https://github.com/mfinzi/LieConv#egg=LieConv`. Dependencies will be checked and installed from the setup.py file.

To run the scripts you will need to clone the repo and install it locally. You can use the commands below.
```
git clone https://github.com/mfinzi/LieConv.git
cd LieConv
pip install -e .
```

For the optional graphnets and tensorboardX functionality you can replace the last line with
`pip install -e .[GN,TBX]`

# Dependencies
* Python 3.7+
* [PyTorch](http://pytorch.org/) 1.3.0+
* [torchvision](https://github.com/pytorch/vision/)
* [snake-oil-ml](https://github.com/mfinzi/snake-oil-ml)
* [torchdiffeq](https://github.com/rtqichen/torchdiffeq)
* (optional) [torch-scatter,torch-sparse,torch-cluster,torch-geometric]
* (optional) [tensorboardX](https://github.com/lanpa/tensorboardX)

# QM9 Molecular Exps

```bash
python examples/train_molec.py --task 'homo' --lr 3e-3 --aug True --num_epochs 1000 --num_layers 6 \
  --log_suffix 'run_name_here' --network MolecLieResNet \
  --net_config "{'group':T(3),'fill':1.}"

```
# RotMNIST Exps
We provide comands to run LieConv on the RotMNIST data for different groups. The comands share hyper-parameters except for the name of the group and the `alpha` parameter that defines the metric on the group.

```bash
# Trivial
python examples/train_img.py --num_epochs=500 --aug=True --trainer_config "{'log_suffix':'mnistTrivial'}" \
  --net_config "{'k':128,'total_ds':.1,'fill':1/15,'nbhd':25,'group':Trivial(2)}" \
  --bs 25 --lr .003 --split "{'train':12000}"

# T2
python examples/train_img.py --num_epochs=500 --aug=True --trainer_config "{'log_suffix':'mnistT2'}" \
   --net_config "{'k':128,'total_ds':.1,'fill':1/15,'nbhd':25,'group':T(2)}" \
   --bs 25 --lr .003 --split "{'train':12000}"

# SO2
python examples/train_img.py --num_epochs=500 --aug=True --trainer_config "{'log_suffix':'mnistSO2'}" \
  --net_config "{'k':128,'total_ds':.1,'fill':1/15,'nbhd':25,'group':SO2(.2)}" \
  --bs 25 --lr .003 --split "{'train':12000}"

#RxSO2
python examples/train_img.py --num_epochs=500 --aug=True --trainer_config "{'log_suffix':'mnistRxSO2'}" \
  --net_config "{'k':128,'total_ds':.1,'fill':1/15,'nbhd':25,'group':RxSO2(.3)}" \
  --bs 25 --lr 3e-3 --split "{'train':12000}" 

#SE2
python examples/train_img.py --num_epochs=500 --aug=True --trainer_config "{'log_suffix':'mnistSE2'}" \
  --net_config "{'k':128,'total_ds':.1,'fill':1/15,'nbhd':25,'group':SE:2(.2)}" \
  --bs 25 --lr 3e-3 --split "{'train':12000}" 
```

## RotMNIST Results

Using the comands above we obtain the following test errors for different groups:

| Trivial | T2   | SO2  | RxSO2 | SE2  |
|---------|------|------|-------|------|
| 1.57    | 1.50 | 1.40 | 1.33  | 1.39 |

# Spring Dynamics Exps
Group substitutions: `Trivial(2)`, `T(2)`, `SO2()`
```bash
python examples/train_springs.py --num_epochs 100 --n_train 3000 \
  --network HLieResNet --net_cfg "{'group':T(2)}" --lr 1e-3
```
