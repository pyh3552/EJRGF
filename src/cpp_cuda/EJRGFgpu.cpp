//
// Created by pyh on 24-8-13.
//
#include "EJRGFgpu.h"
#include <omp.h>
#include <fstream>

EJRGFgpu::EJRGFgpu(std::vector<torch::Tensor> input_list, torch::Tensor gmm_mean, float gamma, float sig2, float epsilon, float threshold, int num_iters)
        :_M(input_list.size()), _K(gmm_mean.size(0)), _gamma(gamma), _epsilon(epsilon), _threshold(threshold), _num_iters(num_iters)
{
    _all_pts_num = 0;
    for (int i = 0; i < _M; ++i) {
        R_list.push_back(torch::eye(3, torch::dtype(torch::kFloat32)).to(torch::kCUDA));
        t_list.push_back(torch::zeros({3}, torch::dtype(torch::kFloat32)).to(torch::kCUDA));
        g_list.push_back(torch::eye(4, torch::dtype(torch::kFloat32)).to(torch::kCUDA));
        _Nj.push_back(input_list[i].size(0));
        starter.push_back(_all_pts_num);
        _all_pts_num += _Nj[i];
    }
    _src_pcd_list = input_list;
    T_pcd_list = input_list;
    _gmm_mean = gmm_mean;


    if(sig2 <= 0)
    {
        _sigma2 = initial_sigma();
    }
    else
    {
        _sigma2 = sig2;
    }

    _beta = _K * _gamma / ((1-_gamma) * get_bbox_volume(_gmm_mean));

    _c_const_part = _beta * powf((2 * 3.1415926535897932), 1.5);

    _zeros_src1 = torch::zeros({_all_pts_num, 1}, at::kFloat).to(torch::kCUDA);
    _zeros_srcd = torch::zeros({_all_pts_num, 3}, at::kFloat).to(torch::kCUDA);
    _zeros_trg1 = torch::zeros({_K, 1}, at::kFloat).to(torch::kCUDA);
    _zeros_trgd = torch::zeros({_K, 3}, at::kFloat).to(torch::kCUDA);

    _ones_trg1 = torch::ones({_K, 1}, at::kFloat).to(torch::kCUDA);

    _vin0_k = torch::concatenate({_zeros_src1, _ones_trg1}, 0);

}

float EJRGFgpu::initial_sigma() {
    float ds_sum = 0;
    torch::Tensor trg_trans = _gmm_mean.unsqueeze(0);
    for (const torch::Tensor src : _src_pcd_list)
    {
        torch::Tensor src_trans = src.unsqueeze(1);
        torch::Tensor ds = (src_trans - trg_trans).pow(2).sum(2).squeeze();
        ds_sum += ds.sum().item<float>();
    }
    return ds_sum / (_all_pts_num * _K);
}

float EJRGFgpu::good_initial_sigma() {
    float ds_sum = 0;
    torch::Tensor trg_trans = _gmm_mean.unsqueeze(0);
    for (const torch::Tensor src : _src_pcd_list)
    {
        torch::Tensor src_trans = src.unsqueeze(1);
        torch::Tensor ds = (src_trans - trg_trans).pow(2).sum(2).squeeze();
        float mean_val = ds.mean().item<float>();
        auto mask = ds > (mean_val * 0.1);
        ds.masked_fill_(mask, 0);
        ds_sum += ds.sum().item<float>();
    }
    return ds_sum / (_all_pts_num * _K);
}


float EJRGFgpu::get_bbox_volume(torch::Tensor pcd)
{
    torch::Tensor min_point = std::get<0>(torch::min(pcd, 0));
    torch::Tensor max_point = std::get<0>(torch::max(pcd, 0));
    torch::Tensor diag = max_point - min_point;
    return (diag[0] * diag[1] * diag[2]).item<float>();
}

float EJRGFgpu::get_bbox_diag(torch::Tensor pcd)
{
    torch::Tensor min_point = std::get<0>(torch::min(pcd, 0));
    torch::Tensor max_point = std::get<0>(torch::max(pcd, 0));
    torch::Tensor diag = max_point - min_point;
    return sqrtf(powf(diag[0].item<float>(), 2) + powf(diag[1].item<float>(), 2) + powf(diag[2].item<float>(), 2));
}


std::vector<torch::Tensor> EJRGFgpu::Register() {
    float loglikelihood_prev = 0;
    float loglikelihood = 0;
    float sigma;
    torch::Tensor all_src = torch::concatenate({_src_pcd_list}, 0);
    torch::Tensor vin2_tmp_k = torch::ones_like(_vin0_k).to(torch::kCUDA);
    for (int i = 0; i < _num_iters; ++i) {
        loglikelihood_prev = loglikelihood;
        sigma = sqrtf(_sigma2);
        torch::Tensor fx = all_src / sigma;
        torch::Tensor fy = _gmm_mean / sigma;

        torch::Tensor fin = torch::concatenate({fx, fy}, 0);

        _vin1_k = torch::concatenate({_zeros_srcd, _gmm_mean}, 0);

        torch::Tensor vin_012_k = torch::concatenate({_vin0_k, _vin1_k, vin2_tmp_k}, 1);

        if (!fin.is_contiguous()) {
            std::cout << "not contiguous" << std::endl;
            fin = fin.contiguous();
        }

        PL_Filter ph(fin.flatten(), fin.size(0), fin.size(1), vin_012_k.size(1)+1, false);
        torch::Tensor M_ji_012 = ph.Filter(vin_012_k.flatten(), false).reshape({-1, 5});

        torch::Tensor M_ji_0 = M_ji_012.select(1, 0).index({torch::indexing::Slice(0, _all_pts_num)});
        torch::Tensor M_ji_1 = M_ji_012.index({torch::indexing::Slice(0, _all_pts_num), torch::indexing::Slice(1, 4)});

        torch::Tensor nonzero_idx = (M_ji_0 != 0).toType(torch::kBool);

        float c = _c_const_part * powf(sigma, 3);
        torch::Tensor M_ji_0_c = M_ji_0 + c;
        torch::Tensor weight_ji = M_ji_0 / M_ji_0_c;


        for (int j = 0; j < _Nj.size(); ++j) {
            torch::Tensor cur_nonzero_idx = nonzero_idx.index({torch::indexing::Slice(starter[j], starter[j] + _Nj[j])});
            torch::Tensor target = (M_ji_1.index({torch::indexing::Slice(starter[j], starter[j] + _Nj[j])}).index_select(0, cur_nonzero_idx.nonzero().squeeze()).t() /
                                    M_ji_0.index({torch::indexing::Slice(starter[j], starter[j] + _Nj[j])}).index_select(0, cur_nonzero_idx.nonzero().squeeze())).t();
            torch::Tensor weight = weight_ji.index({torch::indexing::Slice(starter[j], starter[j] + _Nj[j])});

            pose_solver(&R_list[j], &t_list[j], T_pcd_list[j].index_select(0, cur_nonzero_idx.nonzero().squeeze()), target, weight.index_select(0, cur_nonzero_idx.nonzero().squeeze()).reshape({-1, 1}));

            g_list[j] = torch::matmul(get_SE3(R_list[j], t_list[j]), g_list[j]);

            T_pcd_list[j] = torch::matmul(_src_pcd_list[j], g_list[j].index(
                    {torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}).t()) +
                            g_list[j].index({torch::indexing::Slice(0, 3), 3});
        }
        all_src = torch::concatenate(T_pcd_list, 0);
        torch::Tensor all_src_m0c = (all_src.t() / M_ji_0_c).t();
        torch::Tensor tttmp = torch::sum(torch::pow(all_src, 2), 1) / M_ji_0_c;

        _vin0_ji = torch::concatenate({(1 / M_ji_0_c).reshape({-1, 1}), _zeros_trg1}, 0);
        _vin1_ji = torch::concatenate({all_src_m0c, _zeros_trgd}, 0);
        _vin2_ji = torch::concatenate({tttmp.reshape({-1, 1}), _zeros_trg1}, 0);

        torch::Tensor vin_012_ji = torch::concatenate({_vin0_ji, _vin1_ji, _vin2_ji}, 1);

        torch::Tensor M_k_012 = ph.Filter(vin_012_ji.flatten(), false).reshape({-1, 5});

        torch::Tensor M_k_0 = M_k_012.select(1, 0).index({torch::indexing::Slice(_all_pts_num, _all_pts_num+_K)});
        torch::Tensor M_k_1 = M_k_012.index({torch::indexing::Slice(_all_pts_num, _all_pts_num+_K), torch::indexing::Slice(1, 4)});
        torch::Tensor M_k_2 = M_k_012.select(1, 4).index({torch::indexing::Slice(_all_pts_num, _all_pts_num+_K)});

        _Mk0 = M_k_0;

        M_k_0 += 1e-10;

        _gmm_mean = (M_k_1.t() / M_k_0).t();

        _vin2_k = torch::concatenate({_zeros_src1, torch::sum(torch::pow(_gmm_mean, 2), 1).reshape({-1, 1})}, 0);
        torch::Tensor vin012_tmp = _vin2_k.expand({_vin2_k.size(0), vin_012_k.size(1)});
        torch::Tensor M_ji_2 = ph.Filter(vin012_tmp.flatten(), false).reshape({-1, 5}).select(1, 4).index({torch::indexing::Slice(0, _all_pts_num)});
        float upper = torch::sum(M_k_2).item<float>() + torch::sum(M_ji_2 / M_ji_0_c).item<float>() -
                      2 * torch::matmul(M_k_1.flatten(), _gmm_mean.flatten()).item<float>();
        float lower = torch::sum(M_k_0).item<float>();

        float sigma2_tmp = upper / (3 * lower);
        _sigma2 = sigma2_tmp + _epsilon;
        if (sigma2_tmp < 0) {
            _sigma2 = 1e-10;
        }

        loglikelihood = upper;

        // if (fabs(loglikelihood_prev / loglikelihood - 1.0) < _threshold) {
        //    break;
        // }

    }
    return g_list;
}

void EJRGFgpu::pose_solver(torch::Tensor * R, torch::Tensor * t, torch::Tensor src, torch::Tensor trg, torch::Tensor weight)
{
    float tmp_N_P = torch::sum(weight).item<float>();
    torch::Tensor mu_src = torch::sum((src * weight), 0) / tmp_N_P;
    torch::Tensor mu_trg = torch::sum((trg * weight), 0) / tmp_N_P;
    torch::Tensor src_prime = src - mu_src;
    torch::Tensor trg_prime = trg - mu_trg;
    torch::Tensor A = torch::matmul(trg_prime.t(), (weight * src_prime));
    // 执行SVD分解
    torch::Tensor U, S, Vh;
    std::tie(U, S, Vh) = torch::svd(A);
    torch::Tensor C = torch::eye(3, at::kFloat).to(torch::kCUDA);
    C[2][2] = torch::det(torch::matmul(U, Vh.t()));
    *R = torch::matmul(U, torch::matmul(C, Vh.t()));
    *t = mu_trg - torch::matmul(mu_src, (*R).t());
}

torch::Tensor EJRGFgpu::get_SE3(torch::Tensor R, torch::Tensor t)
{
    torch::Tensor g = torch::eye(4, at::kFloat).to(torch::kCUDA);
    g.index({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}) = R;
    g.index({torch::indexing::Slice(0, 3), 3}) = t;
    return g;
}

torch::Tensor EJRGFgpu::get_gmm_mean() {
    return _gmm_mean;
}