import torch
from . import _C

def EJRGF_register( input_list, \
                    sub_group_size, \
                    gmm_mean_local_num, \
                    epsilon, \
                    local_sigma, \
                    local_iteration_num, \
                    global_refinement=False, \
                    gmm_mean_global_num=0, \
                    global_sigma=0, \
                    global_iteration_num=0 \
                    ):
    return  _C.EJRGF(input_list, sub_group_size, gmm_mean_local_num, epsilon, local_sigma, local_iteration_num, global_refinement, gmm_mean_global_num, global_sigma, global_iteration_num)
