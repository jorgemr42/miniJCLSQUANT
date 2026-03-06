from setuptools import setup, find_packages
setup(
    name='minijclsquant',
    version='1',
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
    description='Jorge Martinez Romeral pruned version of JCLSQUANT',
    author='Jorge Martinez Romeral',
)
