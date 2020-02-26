# LieConv

LieConv is an equivariant convolutional layer that can be applied on generic coordinate-value data and instantiated with the symmetries of a given Lie Group. LieConv was developed with rapid prototyping in mind, we believe that researchers and engineers should be able to experiment with multiple symmetry groups rather than being locked into using a given symmetry by the method.

To accomplish this, we provide an interface for implementing new equivariances by defining the lifting procedure and the matrix exponential and logarithm maps. With these three pieces, a new convolutional layer can be instantiated that reflects the given symmetry. Currently implemented are the Trivial, T(d), SO(2), Rx, RxSO(2), SE(2), and SE(3) groups.

The framework is especially effective for Abelian groups, as for these groups the LieConv layer without subsampling is exactly and deterministically equivariant to the transformations. The approach also extends to non-commutative groups, but relies on a sampling procedure that means that the layer is only equivariant in distribution.

We develop a ResNet model using LieConv as an internal layer, and this network can act on inputs that are any collection of coordinates and values `{x_i,v_i}_{i=1}^N`. We apply this same network architecture to RotMNIST dataset, the QM9 molecular property prediction dataset, and to the modeling of Hamiltonian dynamical systems.

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
<p align="center">
  <img src="https://user-images.githubusercontent.com/12687085/75298868-9e2a3800-5801-11ea-9b87-b02886d78e95.png" width=50>
</p>

|Task|Alpha|Delta|HOMO|LUMO|Mu|Cv|G|H|R2|U|U0|ZPVE|
|-----|-----|---|---|---|-----|-----|---|---|-----|---|---|---|
|Units|bohr^3|meV|meV|meV|Debye|cal/mol K|meV|meV|bohr^2|meV|meV|meV|
|MAE|.084|49|30|25|.032|.038|22|24|.800|19|19|2.280|

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
