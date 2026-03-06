from setuptools import setup, find_packages
setup(
    name='JLSQUANT',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib.pyplot',
        'math',
        'numba',
        'tqdm',
        'time',
        'copy'
    ],
    description='Jorge Martínez Romeral version of LSQUANT',
    author='Jorge Martinez Romeral',
)