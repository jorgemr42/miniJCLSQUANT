from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy
# In order to compile this we should write python3 matrix_vector_setup.py build_ext --inplace
# The first name of the extension is the name the module will have 
# 
ext_modules = [
    Extension(
        "blas_funs", 
        ["blas_funs.pyx"],
        include_dirs=[numpy.get_include()], 
        extra_compile_args=["-O3",'-fopenmp',"-ftree-vectorize","-mfma","-mavx2","-m3dnow","-fprefetch-loop-arrays","-march=native"],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='blas_funs',
    ext_modules=cythonize(ext_modules, annotate=True),
)