// This program executes a typical convolutional layer in regular CNNs
// 0233
#include "cnnConvLayer.h"
#include <iostream>
using namespace std;

// This is the CPU version, please don't modify it
void convLayerCPU()
{
    // declarations for bunch of indexing parameters
    int fn, sli, fmy, fmx, y, x;
    int sum, ifmy, ifmx, ofmy, ofmx;
    int filtIdx, inNeuIdx, outNeuIdx, outIdx;
    int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
    int filtArea = FILTSIZE * FILTSIZE;
    int fmArea = FMSIZE * FMSIZE;
    int outArea = FMSIZE / 2 * FMSIZE / 2;

    // Convolution
    for (fn = 0; fn < FILTNUM; ++fn) {
        for (fmy = 0; fmy < FMSIZE; fmy += STRIDE) {
            for (fmx = 0; fmx < FMSIZE; fmx += STRIDE) {
                sum = 0;
                for (sli = 0; sli < FMDEPTH; ++sli) {
                    for (y = 0; y < FILTSIZE; ++y) {
                        for (x = 0; x < FILTSIZE; ++x) {
                            ifmy = fmy - FILTSIZE / 2 + y;
                            ifmx = fmx - FILTSIZE / 2 + x;
                            filtIdx = fn * filtVol + sli * filtArea + y * FILTSIZE + x;
                            inNeuIdx = sli * fmArea + ifmy * FMSIZE + ifmx;
                            if (ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE) {
                                sum += filt[filtIdx] * inNeu[inNeuIdx];
                            }
                        }
                    }
                }

                // Activation - ReLU
                outNeuIdx = fn * fmArea + fmy * FMSIZE + fmx;
                if (sum <= 0) {
                    outNeu[outNeuIdx] = 0;
                } else {
                    outNeu[outNeuIdx] = sum;
                }
            }
        }
    }

    // Max Pooling with Window Size 2x2
    int max, tmpVal;
    for (sli = 0; sli < FILTNUM; ++sli) {
        for (fmy = 0; fmy < FMSIZE / 2; ++fmy) {
            for (fmx = 0; fmx < FMSIZE / 2; ++fmx) {
                outNeuIdx = sli * fmArea + fmy * 2 * FMSIZE + fmx * 2;
                max = outNeu[outNeuIdx];
                for (y = 0; y < 2; ++y) {
                    for (x = 0; x < 2; ++x) {
                        ofmy = fmy * 2 + y;
                        ofmx = fmx * 2 + x;
                        outNeuIdx = sli * fmArea + ofmy * FMSIZE + ofmx;
                        tmpVal = outNeu[outNeuIdx];
                        if (tmpVal > max) {
                            max = tmpVal;
                        }
                    }
                }
                outIdx = sli * outArea + fmy * FMSIZE / 2 + fmx;
                outCPU[outIdx] = max;
            }
        }
    }
}

/***   Implement your CUDA Kernel here   ***/
__global__ void convLayerGPU(const short* filtCooData, const short* filtCccIdx,
    const short* inNeuCooData, const short* inNeuCccIdx, int* devOut)
{
    __shared__ int sum[34][34];

    int inY(0), inX(0), offY(0), offX(0), max(0);
    const int BLOCKID(blockIdx.x), THREADID(threadIdx.x), outY(THREADID / 16), outX(THREADID % 16),
        outIdx(BLOCKID * 16 * 16 + outY * 16 + outX);

    // 1024 threads
    // sum[THREADID % 32 + 1][THREADID / 32 + 1] = 0;

    // 256 threads
    sum[THREADID % 16 + 1 + 0][THREADID / 16 + 1 + 0] = 0;
    sum[THREADID % 16 + 1 + 16][THREADID / 16 + 1 + 0] = 0;
    sum[THREADID % 16 + 1 + 0][THREADID / 16 + 1 + 16] = 0;
    sum[THREADID % 16 + 1 + 16][THREADID / 16 + 1 + 16] = 0;
    // no need to care the surroundings

    __syncthreads();

    for (int i(0); i < 512; ++i) {
        if (THREADID < 204) {
            offY = 1 - filtCccIdx[BLOCKID * 512 + i] / 3;
            offX = 1 - filtCccIdx[BLOCKID * 512 + i] % 3;
            // inX and inY range from -1 to 32
            inY = inNeuCccIdx[i * 204 + THREADID] / 32 + offY;
            inX = inNeuCccIdx[i * 204 + THREADID] % 32 + offX;
            sum[inY + 1][inX + 1]
                += filtCooData[BLOCKID * 512 + i] * inNeuCooData[i * 204 + THREADID];
        }
        __syncthreads();
    }

    __syncthreads();

    // Max Pooling
    if (THREADID < 256) {
        for (int i(0); i < 2; ++i) {
            for (int j(0); j < 2; ++j) {
                inY = outY * 2 + i + 1;  // ranges from 0 to 31
                inX = outX * 2 + j + 1;  // ranges from 0 to 31
                if (sum[inY][inX] > max) {
                    max = sum[inY][inX];
                }
            }
        }
        devOut[outIdx] = max;
    }
}
/***   Implement your CUDA Kernel here   ***/

int main()
{
    int convLayerCPUExecTime, convLayerGPUExecTime;
    init();
    initCoo();

    timespec time_begin, time_end;
    clock_gettime(CLOCK_REALTIME, &time_begin);

    convLayerCPU();

    clock_gettime(CLOCK_REALTIME, &time_end);
    convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
    cout << "CPU time for executing a typical convolutional layer = "
         << convLayerCPUExecTime / 1000 << "ms" << endl;

    ///////////////////////////////////////////////////////////////////////////
    // 512 filters,with only one nnz data per slice(512 in total)
    cudaMalloc(&devFiltCooData, sizeof(short) * 512 * 512);
    cudaMalloc(&devFiltCccIdx, sizeof(short) * 512 * 512);
    cudaMalloc(&devInNeuCooData, sizeof(short) * 204 * 512);
    cudaMalloc(&devInNeuCccIdx, sizeof(short) * 204 * 512);
    cudaMalloc(&devOut, sizeof(int) * 16 * 116 * 512);
    ///////////////////////////////////////////////////////////////////////////

    clock_gettime(CLOCK_REALTIME, &time_begin);

    ///////////////////////////////////////////////////////////////////////////
    cudaMemcpy(devFiltCooData, filtCooData, sizeof(short) * 512 * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(devFiltCccIdx, inFiltCccIdx, sizeof(short) * 512 * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(devInNeuCooData, inNeuCooData, sizeof(short) * 204 * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(devInNeuCccIdx, inNeuCccIdx, sizeof(short) * 204 * 512, cudaMemcpyHostToDevice);
    ///////////////////////////////////////////////////////////////////////////

    /***  Lunch your CUDA Kernel here  ***/
    convLayerGPU<<<512, 256>>>(devFiltCooData, devFiltCccIdx, devInNeuCooData, devInNeuCccIdx,
        devOut);  // Lunch the kernel, about 2600X
    cudaDeviceSynchronize();  // Do synchronization before clock_gettime()
    /***  Lunch your CUDA Kernel here  ***/
    clock_gettime(CLOCK_REALTIME, &time_end);

    cudaMemcpy(outGPU, devOut, sizeof(int) * 16 * 16 * 512, cudaMemcpyDeviceToHost);

    convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
    cout << "GPU time for executing a typical convolutional layer = "
         << convLayerGPUExecTime / 1000 << "ms" << endl;

    if (checker()) {
        cout << "Congratulations! You pass the check." << endl;
        cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
    } else {
        cout << "Sorry! Your result is wrong." << endl;
    }

    ending();

    return 0;
}
