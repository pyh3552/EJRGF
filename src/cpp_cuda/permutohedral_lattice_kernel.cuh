//
// Created by pyh on 24-7-4.
//
//#pragma once
#ifndef GPU_PERMUTO_PERMUTOHEDRAL_LATTICE_KERNEL_CUH
#define GPU_PERMUTO_PERMUTOHEDRAL_LATTICE_KERNEL_CUH

#define BLOCK_SIZE 256
#define pd 3
#define vd 6
#include <cuda_runtime.h>
#include <torch/torch.h>
//#include "pybind11/pybind11.h"
#include "DeviceMemoryAllocator.h"
#include "cuda_code_indexing.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <utility>
#include <iostream>
//#include <device_atomic_functions.hpp>
//extern void cudaErrorCheck();
//void cudaErrorCheck(){
//    auto code = cudaGetLastError();
//    if(cudaSuccess != code){
//        fprintf(stderr,"GPU Error: %s\n", cudaGetErrorString(code));
//        exit(code);
//    }
//}
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x "must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x "must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


//64 bit implementation not implemented for compute capability < 6.0
// none trivial performance cost for compute capability < 6.0
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif


struct MatrixEntry {
    int index;
    float weight;
};

class HashTableGPU{
public:
    int capacity;// 初始化为：n * (pd + 1)即position的数量 * （包围每个position的格点数）；含有冗余，某些position可能存在公共的格点
    float * values;
    short * keys;
    int * entries;
//    int pd;
//    int vd;

    HashTableGPU(int capacity_, DeviceMemoryAllocator* allocator): capacity(capacity_), values(nullptr), keys(nullptr), entries(nullptr){

        allocator->allocate_device_memory<float>((void**)&values, capacity * vd);
        allocator->memset<float>((void*)values, 0, capacity * vd);//values——代表被加权向量的数组。长度为position的数量 * （包围每个position的格点数）* 被加权向量的维度

        allocator->allocate_device_memory<int>((void**)&entries, capacity * 2);//entries——代表被加权向量的数组。长度为position的数量 * （包围每个position的格点数）* 2
        allocator->memset<int>((void*)entries, -1, capacity * 2);

        allocator->allocate_device_memory<short>((void**)&keys, capacity * pd);//keys——代表用于哈希计算的投影到超平面的position的前pd维数值。长度为position的数量 * （包围每个position的格点数）* pd
        allocator->memset<short>((void*)keys, 0, capacity * pd);
    }

    __device__ int modHash(unsigned int n){
        return(n % (2 * capacity));
    }

    __device__ unsigned int hash(short *key) {
        unsigned int k = 0;
        for (int i = 0; i < pd; i++) {
            k += key[i];
            k = k * 2531011;
        }
        return k;
    }

    __device__ int insert(short *key, unsigned int slot);


    __device__ int retrieve(short *key) {

        int h = modHash(hash(key));// 计算key的哈希值和在entries中的位置
        while (1) {
            int *e = entries + h;// 计算当前key对应entry的位置

            if (*e == -1)//理论上应该进不去这个if
                return -1;//如果没有值则直接返回-1

            bool match = true;
            for (int i = 0; i < pd && match; i++) {
                match = (keys[(*e)*pd+i] == key[i]);
            }
            if (match)
                return *e;

            h++;
            if (h == capacity*2)
                h = 0;
        }
    }
};


__global__ static void createLattice(const int n,
                                     const float *positions,
                                     const float *scaleFactor,
                                     MatrixEntry *matrix,
                                     HashTableGPU table);














/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/


class Permutohedral_Lattice {
public:
    int n;//
    float* scaleFactor;
    MatrixEntry* matrix;
    HashTableGPU hashTable;
    cudaStream_t stream;
    float * newValues; // auxiliary array for blur stage
    int M;// lattice number
    int *d_M; // 设备指针
    bool with_blur_ = true;
    void init_scaleFactor(DeviceMemoryAllocator* allocator){
        float hostScaleFactor[pd];
        float invStdDev = (with_blur_) ? (pd + 1) * sqrt(2.0f / 3) : sqrt(1.0f / 6.0f)*(pd+1);
        for (int i = 0; i < pd; i++) {
            hostScaleFactor[i] = 1.0f / (sqrt((float) (i + 1) * (i + 2))) * invStdDev;
        }
        allocator->allocate_device_memory<float>((void**)&scaleFactor, pd);
        allocator->memcpy<float>((void*)scaleFactor, (void*)hostScaleFactor, pd);
    }

    void init_matrix(DeviceMemoryAllocator* allocator){
        allocator->allocate_device_memory<MatrixEntry>((void**)&matrix, n * (pd + 1));
    }

    void init_newValues(DeviceMemoryAllocator* allocator){
        allocator->allocate_device_memory<float>((void**)&newValues,  n * (pd + 1) * vd);
        allocator->memset<float>((void *)newValues, 0, n * (pd + 1) * vd);
    }

    // 构造函数
    Permutohedral_Lattice(int n_, bool with_blur, DeviceMemoryAllocator* allocator, cudaStream_t stream_=0):
            n(n_),
            M(0),
            with_blur_(with_blur),
            scaleFactor(nullptr),
            matrix(nullptr),
            newValues(nullptr),
            hashTable(HashTableGPU(n * (pd + 1), allocator)),
            stream(stream_){

        if (n >= 65535 * BLOCK_SIZE) {
            printf("Not enough GPU memory (on x axis, you can change the code to use other grid dims)\n");
            //this should crash the program
        }

        // initialize device memory
        init_scaleFactor(allocator);
        init_matrix(allocator);
        init_newValues(allocator);
    }

    void test(){
        int a = 10;
        printf("a = %d", a);
    }

    // values and position must already be device pointers
//    void filter(float* output, const float* inputs, const float*  positions, bool reverse);
    void Initialization(const float*  positions);
    void Splat(const float* inputs);
    void Blur(bool reverse);
    void Slice(float* output, bool reverse);
    int get_lattice_size(){
        return M;
    }
};

class PL_Filter{
public:
    const int n_;
//    auto allocator = DeviceMemoryAllocator();
    DeviceMemoryAllocator allocator;
    Permutohedral_Lattice lattice;
    bool with_blur__;
//    const torch::Tensor input_;
//    long input_rows;
//    long input_cols;

    PL_Filter(const torch::Tensor positions, int n, int pd_, int vd_, bool with_blur=true):n_(n), lattice(n, with_blur, &allocator), with_blur__(with_blur){
        // 确保 Tensor 是 CPU Tensor 并且数据类型是 float 或 double
//        std::cout << input << std::endl;
//        std::cout << "input is on " << input.device() << std::endl;

//        if (input.device().type() == torch::kCPU && (input.dtype() == torch::kFloat || input.dtype() == torch::kDouble)) {
//            // 获取指向 Tensor 数据的指针
//            const float* data = input.data_ptr<float>();
//
//            // 假设我们知道 Tensor 的大小
//            int rows = input.size(0); // 假设 Tensor 是 2D 的，n 是第一维的大小
//            int cols = input.size(1); // pd 是第二维的大小
//            printf("rows = %d\n", rows);
//            printf("cols = %d\n", cols);
//            for (int i = 0; i < rows; i++) {
//                for (int j = 0; j < cols; j++) {
//                    // 打印 Tensor 中的元素
//                    std::cout << data[i * cols + j] << " ";
//                    printf("%f ", data[i * rows + j]);
//                }
//                printf("\n");
//            }
//        } else {
//            // 如果 Tensor 不在 CPU 或者数据类型不是 float/double，你需要进行相应的转换
//            std::cerr << "Tensor must be on CPU and of type float or double." << std::endl;
//        }


//        auto allocator = DeviceMemoryAllocator();
        //vd = image_channels + 1
//        std::cout << "pd = "<< pd << "and vd = " << vd << std::endl;
        if(pd != pd_ || vd != vd_){
            throw 1;
        }
        lattice.Initialization(positions.data_ptr<float>());
    }
//  input 被加权向量； positions 高维特征
    torch::Tensor Filter(const torch::Tensor input, bool reverse=false){

//        torch::Tensor out;
//        torch::Tensor output;
        auto output = torch::empty_like(input);
        lattice.Splat(input.data_ptr<float>());
        if(with_blur__){
            lattice.Blur(reverse);
        }
        lattice.Slice(output.data_ptr<float>(), reverse);
//        lattice.filter(output.data_ptr<float>(), input.data_ptr<float>(), positions.data_ptr<float>(), reverse);
        return output;
    }

    int get_lattice_number(){
        return lattice.get_lattice_size();
    }
};

#endif //GPU_PERMUTO_PERMUTOHEDRAL_LATTICE_KERNEL_CUH
