from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


setup(
    name='deep_point',
    version='v1.0',
    description='deep layers used to convert between point and voxels',
    author='gang.zhang',
    author_email='zhanggang11021136@gmail.com',
    ext_modules=[
        CppExtension(name = 'point_deep.cpu_kernel',
                    sources = ['src/point_deep.cpp']),
        CUDAExtension(name = 'point_deep.cuda_kernel',
                    sources = ['src/point_deep_cuda.cpp', 'src/point_deep_cuda_kernel.cu'],
                    include_dirs = ['src']),
    ],
    cmdclass={'build_ext': BuildExtension},
    packages=find_packages()
)