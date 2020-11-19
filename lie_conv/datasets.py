import math
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import h5py
import os
from torch.utils.data import Dataset
from .utils import Named, export, Expression, FixedNumpySeed, RandomZrotation, GaussianNoise
from oil.datasetup.datasets import EasyIMGDataset
from lie_conv.hamiltonian import HamiltonianDynamics, KeplerH, SpringH
from lie_conv.lieGroups import SO3
from torchdiffeq import odeint_adjoint as odeint
from corm_data.utils import initialize_datasets
import torchvision


#ModelNet40 code adapted from 
#https://github.com/DylanWusee/pointconv_pytorch/blob/master/data_utils/ModelNetDataLoader.py

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label

def load_data(dir,classification = False):
    data_train0, label_train0,Seglabel_train0  = load_h5(dir + 'ply_data_train0.h5')
    data_train1, label_train1,Seglabel_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, label_train2,Seglabel_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, label_train3,Seglabel_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, label_train4,Seglabel_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_test0, label_test0,Seglabel_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1, label_test1,Seglabel_test1 = load_h5(dir + 'ply_data_test1.h5')
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel


@export
class ModelNet40(Dataset,metaclass=Named):
    ignored_index = -100
    class_weights = None
    stratify=True
    num_targets=40
    classes=['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
        'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
        'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
        'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
        'wardrobe', 'xbox']
    default_root_dir = '~/datasets/ModelNet40/'
    def __init__(self,root_dir=default_root_dir,train=True,transform=None,size=1024):
        super().__init__()
        #self.transform = torchvision.transforms.ToTensor() if transform is None else transform
        train_x,train_y,test_x,test_y = load_data(os.path.expanduser(root_dir),classification=True)
        self.coords = train_x if train else test_x
        # SWAP y and z so that z (gravity direction) is in component 3
        self.coords[...,2] += self.coords[...,1]
        self.coords[...,1] = self.coords[...,2]-self.coords[...,1]
        self.coords[...,2] -= self.coords[...,1]
        # N x m x 3
        self.labels = train_y if train else test_y
        self.coords_std = np.std(train_x,axis=(0,1))
        self.coords /= self.coords_std
        self.coords = self.coords.transpose((0,2,1)) # B x n x c -> B x c x n
        self.size=size
        #pt_coords = torch.from_numpy(self.coords)
        #self.coords = FarthestSubsample(ds_frac=size/2048)((pt_coords,pt_coords))[0].numpy()

    def __getitem__(self,index):
        return torch.from_numpy(self.coords[index]).float(), int(self.labels[index])
    def __len__(self):
        return len(self.labels)
    def default_aug_layers(self):
        subsample = Expression(lambda x: x[:,:,np.random.permutation(x.shape[-1])[:self.size]])
        return nn.Sequential(subsample,RandomZrotation(),GaussianNoise(.01))#,augLayers.PointcloudScale())#


try: 
    import torch_geometric
    warnings.filterwarnings('ignore')
    @export
    class MNISTSuperpixels(torch_geometric.datasets.MNISTSuperpixels,metaclass=Named):
        ignored_index = -100
        class_weights = None
        stratify=True
        num_targets = 10
        # def __init__(self,*args,**kwargs):
        #     super().__init__(*args,**kwargs)
        # coord scale is 0-25, std of unif [0-25] is 
        def __getitem__(self,index):
            datapoint = super().__getitem__(int(index))
            coords = (datapoint.pos.T-13.5)/5 # 2 x M array of coordinates
            bchannel = (datapoint.x.T-.1307)/0.3081 # 1 x M array of blackwhite info
            label = int(datapoint.y.item())
            return ((coords,bchannel),label)
        def default_aug_layers(self):
            return nn.Sequential()
except ImportError:
    warnings.warn('torch_geometric failed to import MNISTSuperpixel cannot be used.', ImportWarning)

class RandomRotateTranslate(nn.Module):
    def __init__(self,max_trans=2):
        super().__init__()
        self.max_trans = max_trans
    def forward(self,img):
        if not self.training: return img
        bs,c,h,w = img.shape
        angles = torch.rand(bs)*2*np.pi
        affineMatrices = torch.zeros(bs,2,3)
        affineMatrices[:,0,0] = angles.cos()
        affineMatrices[:,1,1] = angles.cos()
        affineMatrices[:,0,1] = angles.sin()
        affineMatrices[:,1,0] = -angles.sin()
        affineMatrices[:,0,2] = (2*torch.rand(bs)-1)*self.max_trans/w
        affineMatrices[:,1,2] = (2*torch.rand(bs)-1)*self.max_trans/h
        flowgrid = F.affine_grid(affineMatrices.to(img.device), size = img.shape)
        transformed_img = F.grid_sample(img,flowgrid)
        return transformed_img

@export
class RotMNIST(EasyIMGDataset,torchvision.datasets.MNIST):
    """ Unofficial RotMNIST dataset created on the fly by rotating MNIST"""
    means = (0.5,)
    stds = (0.25,)
    num_targets = 10
    def __init__(self,*args,dataseed=0,**kwargs):
        super().__init__(*args,download=True,**kwargs)
        # xy = (np.mgrid[:28,:28]-13.5)/5
        # disk_cutout = xy[0]**2 +xy[1]**2 < 7
        # self.img_coords = torch.from_numpy(xy[:,disk_cutout]).float()
        # self.cutout_data = self.data[:,disk_cutout].unsqueeze(1)
        N = len(self)
        with FixedNumpySeed(dataseed):
            angles = torch.rand(N)*2*np.pi
        with torch.no_grad():
            # R = torch.zeros(N,2,2)
            # R[:,0,0] = R[:,1,1] = angles.cos()
            # R[:,0,1] = R[:,1,0] = angles.sin()
            # R[:,1,0] *=-1
            # Build affine matrices for random translation of each image
            affineMatrices = torch.zeros(N,2,3)
            affineMatrices[:,0,0] = angles.cos()
            affineMatrices[:,1,1] = angles.cos()
            affineMatrices[:,0,1] = angles.sin()
            affineMatrices[:,1,0] = -angles.sin()
            # affineMatrices[:,0,2] = -2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/w
            # affineMatrices[:,1,2] = 2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/h
            self.data = self.data.unsqueeze(1).float()
            flowgrid = F.affine_grid(affineMatrices, size = self.data.size())
            self.data = F.grid_sample(self.data, flowgrid)
    def __getitem__(self,idx):
        return (self.data[idx]-.5)/.25, int(self.targets[idx])
    def default_aug_layers(self):
        return RandomRotateTranslate(0)# no translation

from PIL import Image
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, \
    verify_str_arg
from torchvision.datasets.vision import VisionDataset
# !wget -nc http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip
# # uncompress the zip file
# !unzip -n mnist_rotation_new.zip -d mnist_rotation_new
class MnistRotDataset(VisionDataset,metaclass=Named):
    """ Official RotMNIST dataset."""
    ignored_index = -100
    class_weights = None
    balanced = True
    stratify = True
    means = (0.130,)
    stds = (0.297,)
    num_targets=10
    resources = ["http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"]
    training_file = 'mnist_all_rotation_normalized_float_train_valid.amat'
    test_file = 'mnist_all_rotation_normalized_float_test.amat'
    def __init__(self,root, train=True, transform=None,download=True):
        if transform is None:
            normalize = transforms.Normalize(self.means, self.stds)
            transform = transforms.Compose([transforms.ToTensor(),normalize])
        super().__init__(root,transform=transform)
        self.train = train
        if download:
            self.download()
        if train:
            file=os.path.join(self.raw_folder, self.training_file)
        else:
            file=os.path.join(self.raw_folder, self.test_file)
        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
            
        self.images = data[:, :-1].reshape(-1, 28, 28).astype(np.float32)
        self.labels = data[:, -1].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def _check_exists(self):
        return (os.path.exists(os.path.join(self.raw_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.raw_folder,
                                            self.test_file)))
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')
    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder,exist_ok=True)
        os.makedirs(self.processed_folder,exist_ok=True)

        # download files
        for url in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=None)
        print('Downloaded!')

    def __len__(self):
        return len(self.labels)
    def default_aug_layers(self):
        return RandomRotateTranslate(0)# no translation


class DynamicsDataset(Dataset, metaclass=Named):
    num_targets = 1

    def __len__(self):
        return self.Zs.shape[0]

    def __getitem__(self, i):
        inputs = (self.Zs[i, 0], self.SysP[i], self.Ts[i])
        targets = self.Zs[i]
        return inputs, targets

    def generate_trajectory_data(self, n_systems, sim_kwargs, batch_size=5000):
        """
        Parameters
        ----------
        n_systems: int
        batch_size: int

        Returns
        -------
        ts: torch.Tensor, [n_systems, traj_len]
        zs: torch.Tensor, [n_systems, traj_len, z_dim]
        sys_params: torch.Tensor, [n_systems, param_dim]
        """
        batch_size = min(batch_size, n_systems)
        n_gen = 0
        t_batches, z_batches, sysp_batches = [], [], []
        while n_gen < n_systems:
            z0s, sys_params = self.sample_system(n_systems=batch_size, space_dim=self.space_dim)
            dynamics = self._get_dynamics(sys_params)
            new_ts, new_zs = self.sim_trajectories(z0s, dynamics, **sim_kwargs)
            t_batches.append(new_ts)
            z_batches.append(new_zs)
            sysp_batches.append(torch.stack(sys_params, dim=-1))
            n_gen += new_ts.shape[0]
        print(n_gen)
        ts = torch.cat(t_batches, dim=0)[:n_systems]
        zs = torch.cat(z_batches, dim=0)[:n_systems]
        sys_params = torch.cat(sysp_batches, dim=0)[:n_systems]
        return ts, zs, sys_params

    def sim_trajectories(self, z0, dynamics, traj_len, delta_t):
        """
        This method should be implemented in a subclass with the following interface:
        Parameters
        ----------
        z0: torch.Tensor, [batch_size, z_dim]
        traj_len: int
        delta_t: float or torch.Tensor, [batch_size] (must be greater than 0)
        dynamics: function that computes dz/dt

        Returns
        -------
        ts: torch.Tensor, [batch_size, traj_len]
        zs: torch.Tensor, [batch_size, traj_len, z_dim]
        """
        batch_size, _ = z0.shape
        with torch.no_grad():
            ts = torch.linspace(0, traj_len * delta_t, traj_len).double()
            zs = odeint(dynamics, z0, ts, rtol=1e-8, method='rk4').detach()
            ts = ts.expand(batch_size, -1)
        zs = zs.transpose(1, 0)
        return ts, zs

    def format_training_data(self, ts, zs, chunk_len):
        """
        Randomly samples chunks of trajectory data, returns tensors shaped for training.
        Parameters
        ----------
        ts: torch.Tensor, [batch_size, traj_len]
        zs: torch.Tensor, [batch_size, traj_len, z_dim]
        chunk_len: int
        Returns
        -------
        chosen_ts: torch.Tensor, [batch_size, chunk_len]
        chosen_zs: torch.Tensor, [batch_size, chunk_len, z_dim]
        """
        batch_size, traj_len, z_dim = zs.shape
        n_chunks = traj_len // chunk_len
        chunk_idx = torch.randint(0, n_chunks, (batch_size,), device=zs.device).long()
        chunked_ts = torch.stack(ts.chunk(n_chunks, dim=1))
        chunked_zs = torch.stack(zs.chunk(n_chunks, dim=1))
        chosen_ts = chunked_ts[chunk_idx, range(batch_size)]
        chosen_zs = chunked_zs[chunk_idx, torch.arange(batch_size).long()]
        return chosen_ts, chosen_zs

    def sample_system(self, n_systems, space_dim, **kwargs):
        """
        This method should be implemented in a subclass with the following interface:
        Parameters
        ----------
        n_systems: int
        space_dim: int

        Returns
        -------
        z0: torch.Tensor, [n_systems, z_dim]
        sys_params: tuple (torch.Tensor, torch.Tensor, ...
        """
        raise NotImplementedError

    def _get_dynamics(self, sys_params):
        """
        Parameters
        ----------
        sys_params: tuple(torch.Tensor, torch.Tensor, ...)
        """
        raise NotImplementedError
    
@export
class SpringDynamics(DynamicsDataset):
    default_root_dir = os.path.expanduser('~/datasets/ODEDynamics/SpringDynamics/')
    sys_dim = 2
    
    def __init__(self, root_dir=default_root_dir, train=True, download=True, n_systems=100, space_dim=2, regen=False,
                 chunk_len=5):
        super().__init__()
        filename = os.path.join(root_dir, f"spring_{space_dim}D_{n_systems}_{('train' if train else 'test')}.pz")
        self.space_dim = space_dim
        if os.path.exists(filename) and not regen:
            ts, zs,self.SysP = torch.load(filename)
        elif download:
            sim_kwargs = dict(
                traj_len=500,
                delta_t=0.01,
            )
            ts, zs, self.SysP = self.generate_trajectory_data(n_systems=n_systems, sim_kwargs=sim_kwargs)
            os.makedirs(root_dir, exist_ok=True)
            print(filename)
            torch.save((ts, zs, self.SysP),filename)
        else:
            raise Exception("Download=False and data not there")
        self.sys_dim = self.SysP.shape[-1]
        self.Ts, self.Zs = self.format_training_data(ts, zs, chunk_len)
    
    def sample_system(self, n_systems, space_dim, ood=False):
        """
        See DynamicsDataset.sample_system docstring
        """
        n = np.random.choice([6]) #TODO: handle padding/batching with different n
        if ood: n = np.random.choice([4,8])
        masses = (3 * torch.rand(n_systems, n).double() + .1)
        k = 5*torch.rand(n_systems, n).double()
        q0 = .4*torch.randn(n_systems, n, space_dim).double()
        p0 = .6*torch.randn(n_systems, n, space_dim).double()
        p0 -= p0.mean(0,keepdim=True)
        z0 = torch.cat([q0.reshape(n_systems, n * space_dim), p0.reshape(n_systems, n * space_dim)], dim=1)
        return z0, (masses, k)

    def _get_dynamics(self, sys_params):
        H = lambda t, z: SpringH(z, *sys_params)
        return HamiltonianDynamics(H, wgrad=False)

@export
class NBodyDynamics(DynamicsDataset):
    default_root_dir = os.path.expanduser('~/datasets/ODEDynamics/NBodyDynamics/')

    def __init__(self, root_dir=default_root_dir, train=True, download=True, n_systems=100, regen=False,
                 chunk_len=5, space_dim=3, delta_t=0.01):
        super().__init__()
        filename = os.path.join(root_dir, f"n_body_{space_dim}D_{n_systems}_{('train' if train else 'test')}.pz")
        self.space_dim = space_dim

        if os.path.exists(filename) and not regen:
            ts, zs, self.SysP = torch.load(filename)
        elif download:
            sim_kwargs = dict(
                traj_len=200,
                delta_t=delta_t,
            )
            ts, zs, self.SysP = self.generate_trajectory_data(n_systems, sim_kwargs)
            os.makedirs(root_dir, exist_ok=True)
            print(filename)
            torch.save((ts, zs, self.SysP), filename)
        else:
            raise Exception("Download=False and data not there")
        self.sys_dim = self.SysP.shape[-1]
        self.Ts, self.Zs = self.format_training_data(ts, zs, chunk_len)

    def sample_system(self, n_systems, n_bodies=6, space_dim=3):
        """
        See DynamicsDataset.sample_system docstring
        """
        grav_const = 1.  # hamiltonian.py assumes G = 1
        star_mass = torch.tensor([[32.]]).expand(n_systems, -1, -1)
        star_pos = torch.tensor([[0.] * space_dim]).expand(n_systems, -1, -1)
        star_vel = torch.tensor([[0.] * space_dim]).expand(n_systems, -1, -1)

        planet_mass_min, planet_mass_max = 2e-2, 2e-1
        planet_mass_range = planet_mass_max - planet_mass_min

        planet_dist_min, planet_dist_max = 0.5, 4.
        planet_dist_range = planet_dist_max - planet_dist_min

        # sample planet masses, radius vectors
        planet_masses = planet_mass_range * torch.rand(n_systems, n_bodies - 1, 1) + planet_mass_min
        rho = torch.linspace(planet_dist_min, planet_dist_max, n_bodies - 1)
        rho = rho.expand(n_systems, -1).unsqueeze(-1)
        rho = rho + 0.3 * (torch.rand(n_systems, n_bodies - 1, 1) - 0.5) * planet_dist_range / (n_bodies - 1)
        planet_vel_magnitude = (grav_const * star_mass / rho).sqrt()

        if space_dim == 2:
            planet_pos, planet_vel = self._init_2d(rho, planet_vel_magnitude)
        elif space_dim == 3:
            planet_pos, planet_vel = self._init_3d(rho, planet_vel_magnitude)
        else:
            raise RuntimeError("only 2-d and 3-d systems are supported")

        # import pdb; pdb.set_trace()
        perm = torch.stack([torch.randperm(n_bodies) for _ in range(n_systems)])

        pos = torch.cat([star_pos, planet_pos], dim=1)
        pos = torch.stack([pos[i, perm[i]] for i in range(n_systems)]).reshape(n_systems, -1)
        momentum = torch.cat([star_mass * star_vel, planet_masses * planet_vel], dim=1)
        momentum = torch.stack([momentum[i, perm[i]] for i in range(n_systems)]).reshape(n_systems, -1)
        z0 = torch.cat([pos.double(), momentum.double()], dim=-1)

        masses = torch.cat([star_mass, planet_masses], dim=1).squeeze(-1).double()
        masses = torch.stack([masses[i, perm[i]] for i in range(n_systems)])

        return z0, (masses,)

    def _init_2d(self, rho, planet_vel_magnitude):
        n_systems, n_planets, _ = rho.shape
        # sample radial vectors
        theta = 2 * math.pi * torch.rand(n_systems, n_planets, 1)
        planet_pos = torch.cat([
            rho * torch.cos(theta),
            rho * torch.sin(theta)
        ], dim=-1)
        # get radial tangent vector, randomly flip orientation
        e_1 = torch.stack([-planet_pos[..., 1], planet_pos[..., 0]], dim=-1)
        flip_dir = 2 * (torch.bernoulli(torch.empty(n_systems, n_planets, 1).fill_(0.5)) - 0.5)
        e_1 = e_1 * flip_dir / e_1.norm(dim=-1, keepdim=True)
        planet_vel = planet_vel_magnitude * e_1
        return planet_pos, planet_vel

    def _init_3d(self, rho, planet_vel_magnitude):
        n_systems, n_planets, _ = rho.shape
        # sample radial vectors
        theta = 2 * math.pi * torch.rand(n_systems, n_planets, 1)
        phi = torch.acos(2 * torch.rand(n_systems, n_planets, 1) - 1)  # incorrect to sample \phi \in [0, \pi]
        planet_pos = torch.cat([
            rho * torch.sin(phi) * torch.cos(theta),
            rho * torch.sin(phi) * torch.sin(theta),
            rho * torch.cos(phi)
        ], dim=-1)

        # get radial tangent plane orthonormal basis
        e_1 = torch.stack([torch.zeros(n_systems, n_planets), -planet_pos[..., 2], planet_pos[..., 1]], dim=-1)
        e_2 = torch.cross(planet_pos, e_1, dim=-1)
        e_1 = e_1 / e_1.norm(dim=-1, keepdim=True)
        e_2 = e_2 / e_2.norm(dim=-1, keepdim=True)

        # sample initial velocity in tangent plane
        omega = 2 * math.pi * torch.rand(n_systems, n_planets, 1)
        planet_vel = torch.cos(omega) * e_1 + torch.sin(omega) * e_2
        planet_vel = planet_vel_magnitude * planet_vel
        return planet_pos, planet_vel

    def _get_dynamics(self, sys_params):
        H = lambda t, z: KeplerH(z, *sys_params)
        return HamiltonianDynamics(H, wgrad=False)

@export
class T3aug(nn.Module):
    def __init__(self,scale=.5,train_only=True):
        super().__init__()
        self.train_only = train_only
        self.scale=scale
    def forward(self,x):
        if not self.training and self.train_only: return x
        coords,vals,mask = x
        bs = coords.shape[0]
        unifs = torch.randn(bs,1,3,device=coords.device,dtype=coords.dtype)
        translations = self.scale*unifs
        return (coords+translations,vals,mask)
@export
class SO3aug(nn.Module):
    def __init__(self,train_only=True):
        super().__init__()
        self.train_only = train_only
    def forward(self,x):
        if not self.training and self.train_only: return x
        coords,vals,mask = x
        # coords (bs,n,c)
        Rs = SO3().sample(coords.shape[0],1,device=coords.device,dtype=coords.dtype)
        return ((Rs@coords.unsqueeze(-1)).squeeze(-1),vals,mask)
@export
def SE3aug(scale=.5,train_only=True):
    return nn.Sequential(T3aug(scale,train_only),SO3aug(train_only))

default_qm9_dir = '~/datasets/molecular/qm9/'
def QM9datasets(root_dir=default_qm9_dir):
    root_dir = os.path.expanduser(root_dir)
    filename= f"{root_dir}data.pz"
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        datasets, num_species, charge_scale = initialize_datasets((-1,-1,-1),
         "data", 'qm9', subtract_thermo=True,force_download=True)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)
            dataset.num_species = 5
            dataset.charge_scale = 9
        os.makedirs(root_dir, exist_ok=True)
        torch.save((datasets, num_species, charge_scale),filename)
        return (datasets, num_species, charge_scale)


# class SchPackQM9(Dataset,metaclass=Named):
#     default_qm9_dir = '~/datasets/molecular/qm9/'
#     max_atoms = 29
#     num_species = 5
#     charge_scale = 9
#     def __init__(self,root_dir=default_qm9_dir):
#         super().__init__()
#         filename = f"{root_dir}sch_data.pz"
#         if os.path.exists(filename):
#             self.data = torch.load(filename)
#         else:
#             schqm9 = schnetpack.datasets.QM9(os.path.join(root_dir,'qm9.db'),download=True)
#             self.data = self.collect_and_pad_data(schqm9)
#             os.makedirs(root_dir, exist_ok=True)
#             torch.save(self.data,filename)
#         self.calc_stats()
    
#     def __getitem__(self, idx):
#         return {key: val[idx] for key, val in self.data.items()}

#     def calc_stats(self):
#         self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() 
#                       if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}
#         self.median_stats = {key: (val.median(), torch.median(torch.abs(val - val.median()))) for key, val in self.data.items() 
#                              if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

#     def collect_and_pad_data(self,sch_dataset):
#         datapoints = [sch_dataset[i] for i in range(len(sch_dataset))]
#         properties = datapoints[0].keys()-{'_cell_offset','_cell','_neighbors'}
#         batched = {prop: batch_stack([mol[prop] for mol in datapoints]) for prop in properties}
#         return batched


md17_subsets = {'benzene','uracil','naphthalene','aspirin','salicylic_acid',
               'malonaldehyde','ethanol','toluene','paracetamol','azobenzene'}
default_md17_dir = '~/datasets/molecular/md17'
def MD17datasets(root_dir=default_md17_dir,task='benzene'):
    root_dir = os.path.expanduser(root_dir)
    filename= f"{root_dir}data.pz"
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        datasets, num_species, charge_scale = initialize_datasets((-1,-1,-1), 
        "data", 'md17',subset=task,force_download=True)
        mean_energy = datasets['train'].data['energies'].mean()
        for dataset in datasets.values():
            dataset.data['energies'] -= mean_energy
        os.makedirs(root_dir, exist_ok=True)
        torch.save((datasets,num_species,charge_scale),filename)
        return (datasets,num_species,charge_scale)

if __name__=='__main__':
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import cv2
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    i = 0
    # a = load_data(os.path.expanduser('~/datasets/ModelNet40/'))[0]
    # a[...,2] += a[...,1]
    # a[...,1] = a[...,2]-a[...,1]
    # a[...,2] -= a[...,1]
    D = ModelNet40()
    def update_plot(e):
        global i
        if e.key == "right": i+=1
        elif e.key == "left": i-=1
        else:return
        ax.cla()
        xyz,label = D[i]#.T
        x,y,z = xyz.numpy()*D.coords_std[:,None]
        # d[2] += d[1]
        # d[1] = d[2]-d[1]
        # d[2] -= d[1]
        ax.scatter(x,y,z,c=z)
        ax.text2D(0.05, 0.95, D.classes[label], transform=ax.transAxes)
        #ax.contour3D(d[0],d[2],d[1],cmap='viridis',edgecolor='none')
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event',update_plot)
    plt.show()
