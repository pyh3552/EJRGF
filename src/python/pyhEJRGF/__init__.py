import torch
from . import _C
def EJRGF_register(input_list, sub_group_size, gmm_mean_local_num, gmm_mean_global_num, epsilon, local_sigma, local_threshold, local_iteration_num, global_sigma, global_threshold, global_iteration_num=0, global_refinement=False):
    return  _C.EJRGF(input_list, sub_group_size, gmm_mean_local_num, gmm_mean_global_num, epsilon, local_sigma, local_threshold, local_iteration_num, global_sigma, global_threshold, global_iteration_num, global_refinement)
