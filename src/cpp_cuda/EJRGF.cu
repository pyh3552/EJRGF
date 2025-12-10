#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <fstream>
#include <string>
#include <functional>
#include "EJRGF.h"
#include "EJRGFgpu.h"

std::vector<torch::Tensor>
EJRGF_L2G_CUDA(std::vector<torch::Tensor> input,
                int sub_group_size,
                int gmm_mean_local_num,
                int gmm_mean_global_num,
                float epsilon,
                float local_sigma,
                int local_iteration_num,
                float global_sigma,
                int global_iteration_num, 
                bool global_refinement)
{
    return EJRGF_GPUL2G(input,
                sub_group_size,
                gmm_mean_local_num,
                gmm_mean_global_num,
                epsilon,
                local_sigma,
                local_iteration_num,
                global_sigma,
                global_iteration_num, 
                global_refinement);
}