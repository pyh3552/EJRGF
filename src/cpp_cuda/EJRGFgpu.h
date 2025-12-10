//
// Created by pyh on 24-8-13.
//

#ifndef EJRGF_EJRGFGPU_H
#define EJRGF_EJRGFGPU_H
#include "permutohedral_lattice_kernel.cuh"
#include <torch/torch.h>
std::vector<torch::Tensor> EJRGF_GPUL2G(std::vector<torch::Tensor> input_list, int N_PiInSub, int gmm_mean_l_num, int gmm_mean_g_num, float epsilon, float sigma_l, float threshold_l, int num_iters_l, float sigma_g, float threshold_g, int num_iters_g, bool global_refinement);
class EJRGFgpu
{
public:
    EJRGFgpu(std::vector<torch::Tensor> input_list, torch::Tensor gmm_mean, float gamma, float sig2=0, float epsilon=1e-6, float threshold=1e-4, int num_iters=500);
    std::vector<torch::Tensor> Register();
    torch::Tensor get_gmm_mean();
    torch::Tensor get_Mk0(){return _Mk0;}
private:
    std::vector<torch::Tensor> _src_pcd_list;
    torch::Tensor _gmm_mean;
    // 源点云帧数
    const int _M;
    // 每个源点云的点数
    std::vector<int> _Nj;
    // 源点云总点数
    long _all_pts_num;
    // 目标点云的点数
    int _K;
    // gamma
    float _gamma;
    // 用于迭代的R list
    std::vector<torch::Tensor> R_list;
    // 用于迭代的t list
    std::vector<torch::Tensor> t_list;
    // 结果的g list
    std::vector<torch::Tensor> g_list;
    // 用于迭代的T_pcd_list
    std::vector<torch::Tensor> T_pcd_list;

    float _sigma2;
    float _epsilon;
    float _beta;
    float _threshold;
    int _num_iters;

    float initial_sigma();
    float good_initial_sigma();

    float get_bbox_volume(torch::Tensor pcd);
    float get_bbox_diag(torch::Tensor pcd);

    float _c_const_part;

    torch::Tensor _zeros_src1;
    torch::Tensor _zeros_srcd;
    torch::Tensor _zeros_trg1;
    torch::Tensor _zeros_trgd;

//    torch::Tensor _ones_src1_bool;
    torch::Tensor _ones_trg1;

    torch::Tensor _vin0_k;
    torch::Tensor _vin1_k;
    torch::Tensor _vin2_k;
    torch::Tensor _vin0_ji;
    torch::Tensor _vin1_ji;
    torch::Tensor _vin2_ji;

    torch::Tensor _Mk0;



    std::vector<long> starter;

    void pose_solver(torch::Tensor * R, torch::Tensor * t, torch::Tensor src, torch::Tensor trg, torch::Tensor weight);

    torch::Tensor get_SE3(torch::Tensor R, torch::Tensor t);
};
#endif //EJRGF_EJRGFGPU_H
