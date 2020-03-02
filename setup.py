from setuptools import setup,find_packages
import sys, os

setup(name="lie-conv",
      description="Equivariant Convolutions on Lie Groups",
      version='0.1',
      author='Marc Finzi, Samuel Stanton, Pavel Izmailov',
      author_email='maf820@nyu.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['h5py',
      'olive-oil-ml @ git+https://github.com/mfinzi/olive-oil-ml',
      'torchdiffeq @ git+https://github.com/rtqichen/torchdiffeq'],#
      extras_require = {
          'GN':['torch-scatter','torch-sparse','torch-cluster','torch-geometric'],
          'TBX':['tensorboardX']
      },
      packages=find_packages(),#["oil",],#find_packages()
      long_description=open('README.md').read(),
)
