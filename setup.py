import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='EJRGF',  # 与 pyproject.toml 中的项目名匹配
    ext_modules=[
        CUDAExtension(
            name='EJRGF._C',  # 扩展模块名，可在 Python 中导入为 EJRGF._C
            sources=[
                'src/cpp_cuda/ext.cpp',
                'src/cpp_cuda/EJRGFgpu.cpp',
                'src/cpp_cuda/EJRGF_GPUL2G.cpp',
                'src/cpp_cuda/EJRGF.cu',
                'src/cpp_cuda/permutohedral_lattice_kernel.cu',
            ],
            include_dirs=[os.path.join(os.path.dirname(__file__), 'src/cpp_cuda')],
            extra_compile_args={
                'cxx': ['-O3'],  # C++ 编译优化
                'nvcc': ['-O3']  # CUDA 编译优化
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)