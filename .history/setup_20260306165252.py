from setuptools import setup, find_packages
setup(
    name='jclsquant',
    version='2',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'cython',
        'icecream',
        'matplotlib',
        'scikit-optimize',
        'joblib'
    ],
    description='Jorge Martinez Romeral version of LSQUANT including some BLAS functions and hamiltonian modifiers in C++ plus its veresions in cuda',
    author='Jorge Martinez Romeral',
)
