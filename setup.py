from setuptools import setup,find_packages
import sys, os

setup(name="lie-conv",
      description="Equivariant Convolutions on Lie Groups",
      version='0.1',
      author='Marc Finzi, Pavel Izmailov, Samuel Stanton',
      author_email='maf388@cornell.edu',
      license='MIT',
      python_requires='>=3.6',
      install_requires=['h5py',
      'snake-oil-ml @ git+https://github.com/mfinzi/snake-oil-ml',
      'torchdiffeq @ git+https://github.com/rtqichen/torchdiffeq'],#
      extras_require = {
          'GN':['torch-scatter','torch-sparse','torch-cluster','torch-geometric'],
          'TBX':['tensorboardX']
      },
      packages=find_packages(),#["oil",],#find_packages()
      long_description=open('README.md').read(),
)
