import os
from os.path import join as pjoin
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from icecream import ic
import numpy


def find_in_path(name, path):
    """Find a file in a search path"""
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system"""
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin')
    else:
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be located in your $PATH.')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        'home': home,
        'nvcc': nvcc,
        'include': pjoin(home, 'include'),
        'lib64': pjoin(home, 'lib64'),
    }

    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError(f'The CUDA {k} path could not be located in {v}')

    return cudaconfig


CUDA = locate_cuda()
ic(CUDA)

ext = Extension(
    'obv_gpu',
    sources=['obv.pyx'],
    libraries=['obv', 'cudart'],  # Make sure 'kernel' is correct (not libkernel)
    language='c++',
    include_dirs=[".",CUDA['include'],numpy.get_include()],
    library_dirs=[ ".",CUDA['lib64']],
    extra_compile_args=['-fopenmp']
)

setup(
    name='obv_gpu',
    include_dirs=[CUDA['include']],
    ext_modules=[ext],
    cmdclass={'build_ext': build_ext},
)
