from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='MultiClassTsetlinMachine',
    ext_modules=cythonize("MultiClassTsetlinMachine.pyx"),
    include_dirs=[numpy.get_include()],
)