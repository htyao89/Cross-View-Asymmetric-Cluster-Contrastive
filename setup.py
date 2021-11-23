from setuptools import setup, find_packages


setup(name='DualClusterContrastive',
      version='1.0.0',
      description='Dual Cluster Contrastive learning for Person Re-Identification',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages())
