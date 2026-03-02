import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.path.join(os.path.dirname(os.path.abspath(__file__)), "../glm/")

setup(
    name='diff_gaussian_rasterization_omni',
    packages=['diff_gaussian_rasterization_omni'],
    ext_modules=[
        CUDAExtension(
            name='diff_gaussian_rasterization_omni._C', 
            sources=[
            'cuda_rasterizer/rasterizer_impl.cu',
            'cuda_rasterizer/forward.cu',
            'cuda_rasterizer/backward.cu',
            'rasterize_points.cu',
            'ext.cpp'],
            extra_compile_args={'nvcc': ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "../glm/")]})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
