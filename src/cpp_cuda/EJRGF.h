#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::vector<torch::Tensor>
EJRGF_L2G_CUDA(std::vector<torch::Tensor> input,
                int sub_group_size,
                int gmm_mean_local_num,
                int gmm_mean_global_num,
                float epsilon,
                float local_sigma,
                float local_threshold,
                int local_iteration_num,
                float global_sigma,
                float global_threshold,
                int global_iteration_num, 
                bool global_refinement);
