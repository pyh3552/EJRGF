//
// Created by pyh on 24-7-10.
//
#include "permutohedral_lattice_kernel.cuh"
#include "cuda_runtime.h"
#include <stdio.h>
__device__ int HashTableGPU::insert(short *key, unsigned int slot) {
    int h = modHash(hash(key));//使用哈希函数将key——投影后的position——的前pd维，计算为哈希值；随后将哈希值通过取余数映射为（0 - 2*capacity）之间的一个数值
    while (1) {
        int *e = entries + h;// 移动到entries数组中的位置h上

        // If the cell is empty (-1), lock it (-2)
        int contents = atomicCAS(e, -1, -2);//如果e中的数值为-1即没有被使用过，则将它变为-2

        if (contents == -2){
            //例如线程A已经锁定了一个没用过的entry（e），该e的数值已经是-2了，发生哈希冲突的线程B再次锁定它的话，由于本来e就是-2,那么得到的contents（旧值）也是-2
            //由于该e已经被锁定，线程B不能操作所以该if不做任何事情
            //直接到下面往后移动一格，这种情况可能引发其实没有发生哈希冲突即查询key和keys中的对应key是match的情况被当成了哈希冲突的情况处理，导致entries中的部分格子放了重复的内容
            // If it was locked already, move on to the next cell
        }else if (contents == -1) {
            // 如果该e原来的值为-1,则说明没有被使用
            // If it was empty, we successfully locked it. Write our key.
            for (int i = 0; i < pd; i++) {
                keys[slot * pd + i] = key[i];//保存key，即当前格点的前pd维数据
            }
            // Unlock
            atomicExch(e, slot);//将当前格点的总索引保存到e里，🔓解锁
            return h;//返回该key在entries中的索引
        } else {
            // 如果e是被解锁的，并且有相应的key在keys里，检查是否匹配
            // The cell is unlocked and has a key in it, check if it matches
            bool match = true;
            for (int i = 0; i < pd && match; i++) {
                match = (keys[contents*pd+i] == key[i]);// 根据e中已经保存的总索引，在keys中寻找对应的key，看是否和当前的key一样
            }
            if (match)//如果一样
                return h;//返回该key在entries中的索引
        }
        // increment the bucket with wraparound
        //哈希值一样，但是对应的key又不一样，则把指向entries的下一个位置。
        h++;
        if (h == capacity*2)
            h = 0;
    }
}

__global__ static void createLattice(const int n,
                                     const float *positions,
                                     const float *scaleFactor,
                                     MatrixEntry *matrix,
                                     HashTableGPU table) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;

    float elevated[pd + 1];
    const float *position = positions + idx * pd;
    int rem0[pd + 1];
    int rank[pd + 1];

    // embed position vector into the hyperplane
    // first rotate position into the (pd+1)-dimensional hyperplane
    // sm contains the sum of 1..n of our feature vector
    float sm = 0;
    for (int i = pd; i > 0; i--) {
        float cf = position[i - 1] * scaleFactor[i - 1];
        elevated[i] = sm - i * cf;
        sm += cf;
    }
    elevated[0] = sm;


    // Find the closest 0-colored simplex through rounding
    // greedily search for the closest zero-colored lattice point
    short sum = 0;
    for (int i = 0; i <= pd; i++) {
        float v = elevated[i] * (1.0 / (pd + 1));
        float up = ceil(v) * (pd + 1);
        float down = floor(v) * (pd + 1);
        if (up - elevated[i] < elevated[i] - down) {
            rem0[i] = (short) up;
        } else {
            rem0[i] = (short) down;
        }
        sum += rem0[i];
    }
    sum /= pd + 1;


    // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the features values)
    for (int i = 0; i <= pd; i++)
        rank[i] = 0;
    for (int i = 0; i < pd; i++) {
        double di = elevated[i] - rem0[i];
        for (int j = i + 1; j <= pd; j++)
            if (di < elevated[j] - rem0[j])
                rank[i]++;
            else
                rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= pd; i++) {
        rank[i] += sum;
        if (rank[i] < 0) {
            rank[i] += pd + 1;
            rem0[i] += pd + 1;
        } else if (rank[i] > pd) {
            rank[i] -= pd + 1;
            rem0[i] -= pd + 1;
        }
    }


    float barycentric[pd + 2];
    for (int i = 0; i < pd + 2; ++i) {
        barycentric[i] = 0.0f;
    }
    // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
    for (int i = 0; i <= pd; i++) {
        float delta = (elevated[i] - rem0[i]) * (1.0 / (pd + 1));
        barycentric[pd - rank[i]] += delta;
        barycentric[pd + 1 - rank[i]] -= delta;
    }
    // Wrap around
    barycentric[0] += 1.0 + barycentric[pd + 1];


    short key[pd];
    for (int remainder = 0; remainder <= pd; remainder++) {
        // Compute the location of the lattice point explicitly (all but
        // the last coordinate - it's redundant because they sum to zero)
        for (int i = 0; i < pd; i++) {
            key[i] = static_cast<short>(rem0[i] + remainder);
            if (rank[i] > pd - remainder)
                key[i] -= (pd + 1);
        }

        MatrixEntry r;
        unsigned int slot = static_cast<unsigned int>(idx * (pd + 1) + remainder);// 索引：保存了包围第idx个position的第remainder个格点
        r.index = table.insert(key, slot);//index保存当前key在哈希表entries中的索引
        r.weight = barycentric[remainder];//保存了当前格点关于当前position的重心插值的权重
        matrix[idx * (pd + 1) + remainder] = r;// 一个matrix的数组，保存了包围第idx个position的第remainder个格点在哈希表中的index和重心权重
        //备注：如果有1,2,3,三个数据点。它们每个都有3个格点。格点这些格点有公共的。对于公共的格点，insert的时候，entries里存的slot理论只保留最早insert的
    }
}

__global__ static void cleanHashTable(int n, HashTableGPU table, int * M) {
    // n = 2 * 总数据点数 * （pd+1）
    // pd+1为postion的维度+1
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;

    if (idx >= n)
        return;

    // entries的长度是2*capacity
    // capacity的长度是 数据点数 * (pd + 1)
    // 因此n才需要为2 * 数据点数 * (pd + 1)
    // find my hash table entry
    int *e = table.entries + idx;//对于entries中的所有值

    // Check if I created my own key in the previous phase
    if (*e >= 0) {// 对于所有完成插入了的e进行检查
        // Rehash my key and reset the pointer in order to merge with
        // any other pixel that created a different entry under the
        // same key. If the computation was serial this would never
        // happen, but sometimes race conditions can make the same key
        // be inserted twice. hashTableRetrieve always returns the
        // earlier, so it's no problem as long as we rehash now.
        // keys的总长度即为capacity * pd = 数据点数 * (pd + 1) * pd = 数据点数 * 包围一个position需要的格点数 * pd
        // *e就是当前格点在【数据点数 * 包围一个position需要的格点数】中的索引
        // 该数值乘以pd后就到了该格点的前pd维【也就是当前的key】在keys中的第一个值的地址
        // 重新进行哈希，解决了insert过程中分支-2中可能出现的问题——相同的key占用不同的格子，格子里面存了不同的slot
//        *e = table.retrieve(table.keys + *e * pd);
        int e_check = table.retrieve(table.keys + *e * pd);
        if (*e == e_check)
        {
            atomicAdd(M, 1);// 对于没有问题的，它们就是不重复的格点，利用atomicAdd计算格点总数
        }
        *e = e_check;// 重新存回第一个insert的那个slot
    }
}

__global__ static void update_matrix(const int n, MatrixEntry *matrix, HashTableGPU table) {
    // n个数据点， values代表低维输入，matrix代表每个数据点的每个remaider在哈希表中的索引和该数据点带来的重心插值系数， table代表哈希表
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;// 每个数据点并行
    const int color = blockIdx.y;// remainder个数层面的并行
    const bool outOfBounds = (idx >= n);//当前线程idx是否超过数据点点数上限

    if (!outOfBounds) {//对于没有越届的线程
        // convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
        matrix[idx * (pd + 1) + color].index = table.entries[matrix[idx * (pd + 1) +
                                                                    color].index];//之前进行过rehash因此进行对应的更新
    }
}
__global__ static void splatCache(const int n, const float *values, MatrixEntry *matrix, HashTableGPU table) {
// n个数据点， values代表低维输入，matrix代表每个数据点的每个remaider在哈希表中的索引和该数据点带来的重心插值系数， table代表哈希表
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;// 每个数据点并行
    const int threadId = threadIdx.x;//每个block中的线程索引
    const int color = blockIdx.y;// remainder个数层面的并行
    const bool outOfBounds = (idx >= n);//当前线程idx是否超过数据点点数上限

    __shared__ int sharedOffsets[BLOCK_SIZE];//存储每个线程计算的值将要更新的 values 数组中的位置偏移
    __shared__ float sharedValues[BLOCK_SIZE * vd];//这个BLOCK中所有低维输入
    int myOffset = -1;//当前线程计算的值将要更新的位置偏移，初始值为 -1 表示无效。
    float *myValue = sharedValues + threadId * vd;//指向当前BLOCK中某个低维输入的指针

    if (!outOfBounds) {//对于没有越届的线程

        float * value = const_cast<float *>(values + idx * (vd - 1));//当前线程指向对应的低维输入

        MatrixEntry r = matrix[idx * (pd + 1) + color];//取出当前数据点的第color个remainder

        // convert the matrix entry from a pointer into the entries array to a pointer into the keys/values array
//        matrix[idx * (pd + 1) + color].index = r.index = table.entries[r.index];//之前进行过rehash因此进行对应的更新
        // 从现在开始，r.index不再是entries的索引而是entries的里的值slot

        // record the offset into the keys/values array in shared space
        myOffset = sharedOffsets[threadId] = r.index * vd;//该remaider所对应的value在哈希表values中对应的索引

        for (int j = 0; j < vd - 1; j++) {//以rgb图片为例子vd=4,vd-1=3。
            myValue[j] = value[j] * r.weight;//当前数据点splat到它的第color个remainder上的颜色数值
        }
        myValue[vd - 1] = r.weight;//用最后一个维度（4）保存对于当前数据点和它的第color个remainder之间权重

    } else {
        sharedOffsets[threadId] = -1;
    }

    __syncthreads();

    // am I the first thread in this block to care about this key?
    if (outOfBounds)
        return;

    for (int i = 0; i < BLOCK_SIZE; i++) {
        if (i < threadId) {// myOffset是当前threadId所对应的值，代表了当前thread要操作的对象位置
            // 在这里检查同一个block下其他thread，它们所操作的对象是否和当前thread相同
            // 如果编号小于当前thread，那么它们的优先级比当前thread更高
            // 处理的又是同一个对象的情况下，当前线程应当退出。
            if (myOffset == sharedOffsets[i]) {
                // somebody else with higher priority cares about this key
                return;
            }
        } else if (i > threadId) {
            // 如果编号大于当前thread，那么它们的优先级比当前thread更低
            if (myOffset == sharedOffsets[i]) {
                // someone else with lower priority cares about this key, accumulate it into mine
                // 那么就将受到的splat累加
                for (int j = 0; j < vd; j++) {
                    sharedValues[threadId * vd + j] += sharedValues[i * vd + j];
                }
            }
        }
    }
    //经过上面for循环之后，应该留下和lattice点相同数量的线程。而且这些线程之前有着操作这些lattice的最优先等级
    // only the threads with something to write to main memory are still going
    float *val = table.values + myOffset;//完成了splat的格点所具备的低维输入值先被保存在共享内存上，之后通过如下的循环复制给哈希表的value
    for (int j = 0; j < vd; j++) {
        atomicAdd(val + j, myValue[j]);
    }
}

__global__ static void blur(int n, float *newValues, MatrixEntry *matrix, int color, HashTableGPU table) {
//这里n代表了所有数据点的各自d+1个格点。
    const int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y + threadIdx.x;
    if (idx >= n)
        return;

    // Check if I'm valid
    // table里存的就是slot
    if (matrix[idx].index != idx)//由于splat中的该语句：matrix[idx（这里的idx是数据点索引） * (pd + 1) + color].index = r.index = table.entries[r.index]
        return;


    // find my key and the keys of my neighbors
    short myKey[pd + 1];
    short np[pd + 1];
    short nm[pd + 1];


    for (int i = 0; i < pd; i++) {
        myKey[i] = table.keys[idx * pd + i];//取出当前thread对应的key
        np[i] = myKey[i] + 1;
        nm[i] = myKey[i] - 1;
    }
    np[color] -= pd + 1;
    nm[color] += pd + 1;
    //计算出它在color这个维度上的两个近邻

    // 找到计算出来的两个近邻在哈希表上的位置
    int offNp = table.retrieve(np);
    int offNm = table.retrieve(nm);

    float *valMe = table.values + vd * idx;//取出当前thread对应的数值
    float *valOut = newValues + vd * idx;//指向在newValues上和上面的对应位置

    //in case neighbours don't exist (lattice edges) offNp and offNm are -1
    float zeros[vd]{0};//对于边界上的lattice，使用全0数组
    float *valNp = zeros; //or valMe? for edges?
    float *valNm = zeros;
    if(offNp >= 0)
        valNp = table.values + vd * offNp;//指向Np这一近邻对应的数值
    if(offNm >= 0)
        valNm = table.values + vd * offNm;//指向Nm这一近邻对应的数值

//目前感觉虽然有重复操作的格点但是是可以进行覆盖的，因为newvalue存在另外的数组里面。
    for (int i = 0; i < vd; i++)
        valOut[i] = 0.25 * valNp[i] + 0.5 * valMe[i] + 0.25 * valNm[i];//完成blur
    //valOut[i] = 0.5f * valNp[i] + 1.0f * valMe[i] + 0.5f * valNm[i];
}

__global__ static void slice(const int n, float *values, MatrixEntry *matrix, HashTableGPU table) {
//对于每一个数据点
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n)
        return;

    float value[vd-1]{0};
    float weight = 0;

    for (int i = 0; i <= pd; i++) {
        MatrixEntry r = matrix[idx * (pd + 1) + i];
        float *val = table.values + r.index * vd;
        for (int j = 0; j < vd - 1; j++) {
            value[j] += r.weight * val[j];
        }
        weight += r.weight * val[vd - 1];
    }

    for (int j = 0; j < vd - 1; j++)
        values[idx * (vd - 1) + j] = value[j];
//    weight = 1.0 / weight;
//    weight = 1.0;
//    for (int j = 0; j < vd - 1; j++)
//        values[idx * (vd - 1) + j] = value[j] * weight;
}

// values and position must already be device pointers
//void Permutohedral_Lattice::filter(float* output, const float* inputs, const float*  positions, bool reverse){
//    dim3 blocks((n - 1) / BLOCK_SIZE + 1, 1, 1);
//    dim3 blockSize(BLOCK_SIZE, 1, 1);
//    int cleanBlockSize = 128;
//    dim3 cleanBlocks((n - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);
//
//    createLattice<<<blocks, blockSize, 0, stream>>>(n, positions, scaleFactor, matrix, hashTable);
//    auto code = cudaGetLastError();
//    if(cudaSuccess != code) {
//        fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
//        exit(code);
//    }
////    std::cout << "lattice gen" << std::endl;
//    cleanHashTable <<<cleanBlocks, cleanBlockSize, 0, stream>>>(2 * n * (pd + 1), hashTable);
//    code = cudaGetLastError();
//    if(cudaSuccess != code) {
//        fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
//        exit(code);
//    }
////    std::cout << "clean hash" << std::endl;
////        cudaErrorCheck();
////
//    blocks.y = pd + 1;
//    splatCache<<<blocks, blockSize, 0, stream>>>(n, inputs, matrix, hashTable, &M);
//    code = cudaGetLastError();
//    if(cudaSuccess != code) {
//        fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
//        exit(code);
//    }
////    std::cout << "splat cache" << std::endl;
////        cudaErrorCheck();
//
//    // 逐个维度进行blur
//    for (int remainder=reverse?pd:0; remainder >= 0 && remainder <= pd; reverse?remainder--:remainder++) {
//        blur<<<cleanBlocks, cleanBlockSize, 0, stream>>>(n * (pd + 1), newValues, matrix, remainder, hashTable);
//        code = cudaGetLastError();
//        if(cudaSuccess != code) {
//            fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
//            exit(code);
//        }
//        std::swap(hashTable.values, newValues);
//    }
////    std::cout << "blur over" << std::endl;
//    blockSize.y = 1;
//    slice<<<blocks, blockSize, 0, stream>>>(n, output, matrix, hashTable);
//    code = cudaGetLastError();
//    if(cudaSuccess != code) {
//        fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
//        exit(code);
//    }
////    std::cout << "slice over" << std::endl;
//}

void Permutohedral_Lattice::Initialization(const float*  positions){
    // 分配设备内存
    cudaMalloc((void **)&d_M, sizeof(int));
    // 从主机复制数据到设备
    cudaMemcpy(d_M, &M, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocks((n - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
    int cleanBlockSize = 128;
    dim3 cleanBlocks((n - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);

    createLattice<<<blocks, blockSize, 0, stream>>>(n, positions, scaleFactor, matrix, hashTable);
    auto code = cudaGetLastError();
    if(cudaSuccess != code) {
        fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
        exit(code);
    }
//    std::cout << "lattice gen" << std::endl;
    cleanHashTable <<<cleanBlocks, cleanBlockSize, 0, stream>>>(2 * n * (pd + 1), hashTable, d_M);
    code = cudaGetLastError();
    if(cudaSuccess != code) {
        fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
        exit(code);
    }
    // 从设备复制数据回主机
    cudaMemcpy(&M, d_M, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_M);

    blocks.y = pd + 1;
    update_matrix<<<blocks, blockSize, 0, stream>>>(n, matrix, hashTable);
//    std::cout << "clean hash" << std::endl;
//        cudaErrorCheck();
//

}

void Permutohedral_Lattice::Splat(const float *inputs) {
    dim3 blocks((n - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);

    blocks.y = pd + 1;
    splatCache<<<blocks, blockSize, 0, stream>>>(n, inputs, matrix, hashTable);
    // 同步设备
    cudaDeviceSynchronize();

    auto code = cudaGetLastError();
    if(cudaSuccess != code) {
        fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
        exit(code);
    }


}

void Permutohedral_Lattice::Blur(bool reverse) {
//    dim3 blocks((n - 1) / BLOCK_SIZE + 1, 1, 1);
//    dim3 blockSize(BLOCK_SIZE, 1, 1);
    int cleanBlockSize = 128;
    dim3 cleanBlocks((n - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);
    // 逐个维度进行blur
    for (int remainder=reverse?pd:0; remainder >= 0 && remainder <= pd; reverse?remainder--:remainder++) {
        blur<<<cleanBlocks, cleanBlockSize, 0, stream>>>(n * (pd + 1), newValues, matrix, remainder, hashTable);
        auto code = cudaGetLastError();
        if(cudaSuccess != code) {
            fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
            exit(code);
        }
        std::swap(hashTable.values, newValues);
    }
}

void Permutohedral_Lattice::Slice(float *output, bool reverse) {
    dim3 blocks((n - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);
//    int cleanBlockSize = 128;
//    dim3 cleanBlocks((n - 1) / cleanBlockSize + 1, 2 * (pd + 1), 1);

    blockSize.y = 1;
    slice<<<blocks, blockSize, 0, stream>>>(n, output, matrix, hashTable);
    auto code = cudaGetLastError();
    if(cudaSuccess != code) {
        fprintf(stderr, "GPU Error: %s\n", cudaGetErrorString(code));
        exit(code);
    }

    // 使用cudaMemset将数组设置为0
    cudaError_t status = cudaMemset(hashTable.values, 0, hashTable.capacity * vd * sizeof(float));

    // 检查是否有错误发生
    if (status != cudaSuccess) {
        // 处理错误
    }
}


