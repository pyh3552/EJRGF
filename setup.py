# 导入setuptools的setup函数，用于配置和构建Python包
from setuptools import setup, find_packages
# 导入PyTorch提供的CUDA扩展构建工具：CUDAExtension用于定义CUDA扩展模块，BuildExtension用于构建过程中的配置
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
# 导入os模块，用于处理文件路径和目录操作
import os
# 获取项目根目录绝对路径
project_root = os.path.dirname(os.path.abspath(__file__))
# 调用setup函数配置并构建包
setup(
    name="EJRGF",
    version='0.0.0',
    author='Yihan Pan',
    author_email='Y20210060@mail.ecust.edu.cn',
    license='GPL-3.0',
    packages=find_packages(where="src/python"),# 指定包含的Python包列表，此处'diff_gaussian_rasterization_taming'目录将被视为Python包
    package_dir={"":"src/python"},
    # install_requires=['torch'],
    # 定义要编译的扩展模块列表
    ext_modules=[
        # 使用CUDAExtension类定义CUDA扩展模块
        CUDAExtension(
            name="EJRGF._C",  # 确保扩展模块在包内
            sources=[
            "src/cpp_cuda/EJRGF.cu",
            "src/cpp_cuda/EJRGF_GPUL2G.cpp",
            "src/cpp_cuda/EJRGFgpu.cpp",
            "src/cpp_cuda/permutohedral_lattice_kernel.cu",
            "src/cpp_cuda/ext.cpp"],
            # 指定额外的编译参数，针对不同编译器传递选项
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--expt-extended-lambda"]
            }
            )
        ],
    # 自定义构建命令类，覆盖标准的build_ext命令以使用PyTorch的构建扩展
    cmdclass={
        'build_ext': BuildExtension# 使用PyTorch的BuildExtension处理CUDA编译和兼容性
    }
)