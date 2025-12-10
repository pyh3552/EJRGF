# 导入setuptools的setup函数，用于配置和构建Python包
from setuptools import setup, find_packages
# 导入PyTorch提供的CUDA扩展构建工具：CUDAExtension用于定义CUDA扩展模块，BuildExtension用于构建过程中的配置
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
# 导入os模块，用于处理文件路径和目录操作
import os
# 获取当前脚本文件的绝对路径所在的目录（此处未保存结果，实际未使用，可能为冗余代码或调试遗留）
os.path.dirname(os.path.abspath(__file__))
print("test")
print(os.path.dirname(os.path.abspath(__file__)))
# 获取项目根目录绝对路径
project_root = os.path.dirname(os.path.abspath(__file__))
# 调用setup函数配置并构建包
setup(
    name="pyhEJRGF",
    packages=find_packages(where="src/python"),# 指定包含的Python包列表，此处'diff_gaussian_rasterization_taming'目录将被视为Python包
    package_dir={"":"src/python"},
    # 定义要编译的扩展模块列表
    ext_modules=[
        # 使用CUDAExtension类定义CUDA扩展模块
        CUDAExtension(
            name="pyhEJRGF._C",  # 确保扩展模块在包内
            # 列出所有需要编译的源文件（CUDA和C++文件）

            # include_dirs=[
            #     # "src/cpp_cuda/",
            #     # "src/third_party/permutohedralGPU/"
            #     # 关键：使用绝对路径
            #     os.path.join(project_root, "src/cpp_cuda"),
            #     os.path.join(project_root, "src/third_party/permutohedralGPU"),
            #     # 其他第三方头文件路径（如果有）
            # ],
            sources=[
            "src/cpp_cuda/EJRGF.cu",
            "src/cpp_cuda/EJRGF_GPUL2G.cpp",
            "src/cpp_cuda/EJRGF.cpp",
            "src/cpp_cuda/permutohedral_lattice_kernel.cu",
            "src/cpp_cuda/ext.cpp"],
            # 指定额外的编译参数，针对不同编译器传递选项
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--expt-extended-lambda"]
            }
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]}
            )
        ],
    # 自定义构建命令类，覆盖标准的build_ext命令以使用PyTorch的构建扩展
    cmdclass={
        'build_ext': BuildExtension# 使用PyTorch的BuildExtension处理CUDA编译和兼容性
    }
)
# from setuptools import setup
# from torch.utils.cpp_extension import CUDAExtension, BuildExtension
# import os
# import torch  # 新增：获取 PyTorch 的库路径
#
# project_root = os.path.dirname(os.path.abspath(__file__))
#
# # 获取 PyTorch 的库目录和头文件目录
# torch_lib_dir = os.path.dirname(torch.__file__) + "/lib"
# torch_include_dir = os.path.dirname(torch.__file__) + "/include"
# print(torch_lib_dir)
# print(torch_include_dir)
# setup(
#     name="pyhEJRGF",
#     packages=["pyhEJRGF"],
#     package_dir={"pyhEJRGF": "src/python/pyhEJRGF"},
#     ext_modules=[
#         CUDAExtension(
#             name="pyhEJRGF._C",
#             include_dirs=[
#                 os.path.join(project_root, "src/cpp_cuda"),
#                 os.path.join(project_root, "src/third_party/permutohedralGPU"),
#                 torch_include_dir,  # 添加 PyTorch 头文件路径
#             ],
#             libraries=["c10", "torch", "torch_cpu", "torch_python"],  # 需要链接的库
#             library_dirs=[torch_lib_dir],  # 添加 PyTorch 库路径
#             sources=[
#                 "src/cpp_cuda/EJRGF.cu",
#                 "src/cpp_cuda/FastJRMPC_GPUL2G.cpp",
#                 "src/cpp_cuda/FastJRMPCgpu.cpp",
#                 "src/third_party/permutohedralGPU/permutohedral_lattice_kernel.cu",
#                 "src/cpp_cuda/ext.cpp"
#             ],
#             extra_compile_args={
#                 "cxx": ["-O3"],
#                 "nvcc": ["-O3", "--expt-extended-lambda"]
#             },
#             extra_link_args=[f"-L{torch_lib_dir}", f"-L{torch_lib_dir}/libc10.so"]  # 强制链接路径
#         )
#     ],
#     cmdclass={'build_ext': BuildExtension},
#     zip_safe=False,
# )