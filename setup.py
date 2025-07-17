import setuptools

requirements = [
    'numpy',
    'torch',
    'tqdm',
    'scikit-learn',
    'h5py',
    'anndata',
    'scanpy',
    'matplotlib',
    'scipy',
    'pandas',
    'toml'
]

setuptools.setup(
    name='scGPD',
    version='0.0.1',
    author='Tina Guo',
    description='scGPD: single-cell informed Gene Panel Desig',
    url='https://github.com/TinaGuo/scGPD',
    packages=['scGPD'],
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering'
    ],
    python_requires='>=3.6',
)

