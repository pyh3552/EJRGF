//
// Created by pyh on 24-8-27.
//
#include "EJRGFgpu.h"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <omp.h>
//#include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <ATen/cuda/CUDAEvent.h>

/**
 * @brief Sample GMM means from source point clouds.
 *        从源点云中采样GMM均值。
 * 
 * Concatenate all source point clouds and uniformly sample points based on the specified number.
 * 拼接所有源点云，并根据指定数量均匀采样点。
 * 
 * @param input_list Vector of input point cloud tensors. 输入点云张量的向量。
 * @param gmm_mean_num The target number of GMM means. 目标GMM均值数量。
 * @return torch::Tensor The sampled GMM means tensor. 采样得到的GMM均值张量。
 */
torch::Tensor RandomSampleGMM_from_src(std::vector<torch::Tensor> input_list, int gmm_mean_num)
{
    torch::Tensor all_source = torch::concatenate(input_list, 0);
    int step = ceilf(float(all_source.size(0))/float(gmm_mean_num));
    std::vector<int64_t> indices(ceilf(all_source.size(0)/step));
    for (int i = 0; i < ceilf(all_source.size(0)/step); ++i) {
        indices[i] = i * step;  // 每step行取一行
    }
    // 将索引转换为张量
    torch::Tensor indices_tensor = torch::tensor(indices, torch::dtype(torch::kInt64)).to(torch::kCUDA);

    // 使用index_select来选取特定的行
    torch::Tensor gmm = all_source.index_select(0, indices_tensor);
    return gmm;
}

/**
 * @brief Transform a point cloud using a given transformation matrix.
 *        使用给定的变换矩阵变换点云。
 * 
 * @param input The input point cloud tensor. 输入点云张量。
 * @param g The transformation matrix tensor. 变换矩阵张量。
 * @return torch::Tensor The transformed point cloud tensor. 变换后的点云张量。
 */
torch::Tensor trans_cloud(torch::Tensor input, torch::Tensor g)
{
    return  torch::matmul(input, g.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}).t()) + g.index({torch::indexing::Slice(0, 3), 3});
}

/**
 * @brief Transform multiple point clouds using a list of transformation matrices.
 *        使用变换矩阵列表变换多个点云。
 * 
 * @param input_list Vector of input point cloud tensors. 输入点云张量的向量。
 * @param g_list Vector of transformation matrix tensors. 变换矩阵张量的向量。
 * @return std::vector<torch::Tensor> Vector of transformed point cloud tensors. 变换后的点云张量的向量。
 */
std::vector<torch::Tensor> trans_clouds(std::vector<torch::Tensor> input_list, std::vector<torch::Tensor> g_list)
{
    std::vector<torch::Tensor> output_list;
    for (int i = 0; i < input_list.size(); ++i) {
        auto trans = torch::matmul(input_list[i], g_list[i].index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}).t()) + g_list[i].index({torch::indexing::Slice(0, 3), 3});
        output_list.push_back(trans);
    }
    return output_list;
}

/**
 * @brief Transform multiple point clouds using a list of transformation matrices.
 *        使用变换矩阵列表变换多个点云。
 * 
 * @param input_list Vector of input point cloud tensors. 输入点云张量的向量。
 * @param g_list Pointer to the transformation matrix tensor. 变换矩阵张量的指针。
 * @return std::vector<torch::Tensor> Vector of transformed point cloud tensors. 变换后的点云张量的向量。
 */
std::vector<torch::Tensor> trans_clouds(std::vector<torch::Tensor> input_list, torch::Tensor * g_list)
{
    std::vector<torch::Tensor> output_list;
    for (int i = 0; i < input_list.size(); ++i) {
        auto trans = torch::matmul(input_list[i], g_list[i].index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}).t()) + g_list[i].index({torch::indexing::Slice(0, 3), 3});
        output_list.push_back(trans);
    }
    return output_list;
}

/**
 * @brief Align multiple transformation matrices to the first one.
 *        将多个变换矩阵对齐到第一个变换矩阵。
 * 
 * @param g_list Vector of transformation matrix tensors. 变换矩阵张量的向量。
 * @return std::vector<torch::Tensor> Vector of aligned transformation matrix tensors. 对齐后的变换矩阵张量的向量。
 */
std::vector<torch::Tensor> align2first(std::vector<torch::Tensor> g_list)
{
    std::vector<torch::Tensor> ali_g_list;
    torch::Tensor first_inv = torch::linalg_inv(g_list[0]);
    for (const auto & i : g_list) {
        ali_g_list.push_back(torch::matmul(first_inv, i));
    }
    return ali_g_list;
}

/**
 * @brief Register multiple point clouds using EJRGF. 
 *        The point cloud range is [s_start, s_end]. 
 *        GMM means are sampled from the source point cloud. 
 *        The result is aligned to the first point cloud.
 *        使用EJRGF配准多个点云。
 *        点云范围为[s_start, s_end]。
 *        GMM均值从源点云随机采样。
 *        返回结果对齐于第一帧点云，即第一个变换矩阵为单位矩阵。
 * @param input_list Vector of input point cloud tensors. 输入点云张量的向量。
 * @param s_start Start index of the source point cloud range. 源点云范围的起始索引。
 * @param s_end End index of the source point cloud range. 源点云范围的结束索引。
 * @param gmm_mean_l_num Number of GMM means for local registration. 局部配准时的GMM均值数量。
 * @param epsilon Epsilon value for EJRGF. EJRGF的epsilon值。
 * @param sigma_l Sigma value for local registration. 局部配准时的sigma值。
 * @param threshold_l Threshold value for local registration. 局部配准时的阈值。
 * @param num_iters_l Number of iterations for local registration. 局部配准时的迭代次数。
 * @return std::vector<torch::Tensor> Vector of registered transformation matrix tensors. 配准后的变换矩阵张量的向量。
 */

std::vector<torch::Tensor> REG(std::vector<torch::Tensor> input_list, int s_start, int s_end, int gmm_mean_l_num, float epsilon, float sigma_l, float threshold_l, int num_iters_l)
{
    std::vector<torch::Tensor> Pss(input_list.begin()+s_start, input_list.begin()+s_end+1);
    torch::Tensor gmms = RandomSampleGMM_from_src(Pss, gmm_mean_l_num);
    EJRGFgpu ejrgf(Pss, gmms, 1.0/gmms.size(0), sigma_l, epsilon, threshold_l, num_iters_l);
    std::vector<torch::Tensor> g_list_s = ejrgf.Register();
    g_list_s = align2first(g_list_s);
    return g_list_s;
}


/**
 * @brief Register multiple point clouds using EJRGF. 
 *        GMM means are sampled from the source point cloud. 
 *        The result is aligned to the first point cloud.
 *        使用EJRGF配准多个点云。
 *        GMM均值从源点云随机采样。
 *        返回结果对齐于第一帧点云，即第一个变换矩阵为单位矩阵。
 * @param input_list Vector of input point cloud tensors. 输入点云张量的向量。
 * @param N_PiInSub Number of point clouds in each subgroup. 每个子组中的点云数量。
 * @param gmm_mean_l_num Number of GMM means for local registration. 局部配准时的GMM均值数量。
 * @param gmm_mean_g_num Number of GMM means for global registration. 全局细化时的GMM均值数量。
 * @param epsilon Epsilon value for EJRGF. EJRGF的epsilon值。
 * @param sigma_l Sigma value for local registration. 局部配准时的sigma值。
 * @param threshold_l Threshold value for local registration. 局部配准时的阈值。
 * @param num_iters_l Number of iterations for local registration. 局部配准时的迭代次数。
 * @param sigma_g Sigma value for global registration. 全局细化时的sigma值。
 * @param threshold_g Threshold value for global registration. 全局细化时的阈值。
 * @param num_iters_g Number of iterations for global registration. 全局细化时的迭代次数。
 * @param global_refinement Whether to perform global refinement. 是否进行全局细化。
 * @return std::vector<torch::Tensor> Vector of registered transformation matrix tensors. 配准后的变换矩阵张量的向量。
 */
std::vector<torch::Tensor> EJRGF_GPUL2G(std::vector<torch::Tensor> input_list, int N_PiInSub, int gmm_mean_l_num, int gmm_mean_g_num, float epsilon, float sigma_l, float threshold_l, int num_iters_l, float sigma_g, float threshold_g, int num_iters_g, bool global_refinement)
{
    // 输入点云的总数量
    int N_P = input_list.size();
    std::cout << "Frame Number = " << N_P << std::endl;
    std::vector<int> idxs_P;
    for (int i = 0; i < N_P; i=i+N_PiInSub-1) {
        idxs_P.push_back(i);
        std::cout << i << "\t";
    }
    if (idxs_P.back() != N_P-1) {
        idxs_P.push_back(N_P-1);
        std::cout << N_P-1 << "\t";
    }
    std::cout << "\n";

    int N_subP = idxs_P.size() - 1;
    std::cout << "Number of subgroups = " << N_subP << std::endl;

    if (N_subP == 1)
    {
        std::cout << "One Subgroup." << std::endl;

        torch::Tensor gmm = RandomSampleGMM_from_src(input_list, gmm_mean_l_num);

        EJRGFgpu ejrgf(input_list, gmm, 1.0/gmm.size(0), sigma_l, epsilon, threshold_l, num_iters_l);
        std::vector<torch::Tensor> g_list = ejrgf.Register();
        std::vector<torch::Tensor> g_list_ali = align2first(g_list);
        return g_list_ali;
    }
    else{
        std::cout << "Multiple Subgroups." << std::endl;
        std::vector<torch::Tensor> g_list_g;
        std::vector<torch::Tensor> gmm_list(N_subP);
        for (int i = 0; i < N_P; ++i) {
            g_list_g.push_back(torch::eye(4));
        }
        std::vector<std::vector<torch::Tensor>> g_local_group_list(N_subP);

        for (int i = 0; i < N_subP; ++i) {
            int s_start = idxs_P[i];
            int s_end = idxs_P[i+1];
            g_local_group_list[i] = REG(input_list, s_start, s_end,gmm_mean_l_num, epsilon, sigma_l, threshold_l, num_iters_l);
        }


        for (int i = 0; i < g_local_group_list[0].size(); ++i) {
            g_list_g[i] = g_local_group_list[0][i];
        }

        int idx = g_local_group_list[0].size();
        for (int i = 1; i < g_local_group_list.size(); ++i) {
            for (int j = 1; j < g_local_group_list[i].size(); ++j) {
                g_local_group_list[i][j] = torch::matmul(g_local_group_list[i-1].back(), g_local_group_list[i][j]);
                {
                    g_list_g[idx+j-1] = g_local_group_list[i][j];
                }

            }
            idx += (g_local_group_list[i].size() - 1);
        }

        if (global_refinement)
        {
            // global refinement
            std::vector<torch::Tensor> TP_l_end = trans_clouds(input_list, g_list_g);
            torch::Tensor gmm = RandomSampleGMM_from_src(TP_l_end, gmm_mean_g_num);
            EJRGFgpu ejrgf_g(TP_l_end, gmm, 1.0/gmm.size(0), sigma_g, epsilon, threshold_g, num_iters_g);
            std::vector<torch::Tensor> g_list_refine = ejrgf_g.Register();
            g_list_refine = align2first(g_list_refine);
            std::vector<torch::Tensor> g_list_result;
            for (int k = 0; k < g_list_refine.size(); ++k) {
                g_list_result.push_back(torch::matmul(g_list_refine[k], g_list_g[k]));
            }
            return g_list_result;
        }
        else
        {
            return g_list_g;
        }
    }
}