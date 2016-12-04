// This program executes a typical convolutional layer in regular CNNs
#include <iostream>
#include "cnnConvLayer.h"

using namespace std;

// This is the CPU version, please don't modify it
void convLayerCPU()
{
    const int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
    const int filtArea = FILTSIZE * FILTSIZE;
    const int fmArea = FMSIZE * FMSIZE;
    const int outArea = FMSIZE / 2 * FMSIZE / 2;

    int fn(0), fmy(0), fmx(0), sum(0), sli(0), y(0), x(0), ifmy(0), ifmx(0), filtIdx(0), inNeuIdx(0), outNeuIdx(0), max(0), tmpVal(0), ofmy(0), ofmx(0), outIdx(0);

    // Convolution
    for (fn = 0; fn < FILTNUM; ++fn)
    {
        for (fmy = 0; fmy < FMSIZE; fmy += STRIDE)
        {
            for (fmx = 0; fmx < FMSIZE; fmx += STRIDE)
            {
                sum = 0;
                for (sli = 0; sli < FMDEPTH; ++sli)
                {
                    for (y = 0; y < FILTSIZE; ++y)
                    {
                        for (x = 0; x < FILTSIZE; ++x)
                        {
                            ifmy = fmy - FILTSIZE / 2 + y;
                            ifmx = fmx - FILTSIZE / 2 + x;
                            filtIdx = fn * filtVol + sli * filtArea + y * FILTSIZE + x;
                            inNeuIdx = sli * fmArea + ifmy * FMSIZE + ifmx;
                            if (ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
                            {
                                sum += filt[filtIdx] * inNeu[inNeuIdx];
                            }
                        }
                    }
                }
                // Activation - ReLU
                outNeuIdx = fn * fmArea + fmy * FMSIZE + fmx;
                if (sum <= 0)
                {
                    outNeu[outNeuIdx] = 0;
                }
                else
                {
                    outNeu[outNeuIdx] = sum;
                }
            }
        }
    }

    // Max Pooling with Window Size 2x2
    for (sli = 0; sli < FILTNUM; ++sli)
    {
        for (fmy = 0; fmy < FMSIZE / 2 ; ++fmy)
        {
            for (fmx = 0; fmx < FMSIZE / 2 ; ++fmx)
            {
                outNeuIdx = sli * fmArea + fmy * 2 * FMSIZE + fmx * 2;
                max = outNeu[outNeuIdx];
                for (y = 0; y < 2; ++y)
                {
                    for (x = 0; x < 2; ++x)
                    {
                        ofmy = fmy * 2 + y;
                        ofmx = fmx * 2 + x;
                        outNeuIdx = sli * fmArea + ofmy * FMSIZE + ofmx;
                        tmpVal = outNeu[outNeuIdx];
                        if (tmpVal > max)
                        {
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

/*** Implement your CUDA Kernel here ***/
__global__
void convLayerGPU(short * devInputNeuron, short * devInputFilter, int * devOutputNeuron, int * devOutput)
{
    const int filtVol = FMDEPTH * FILTSIZE * FILTSIZE;
    const int filtArea = FILTSIZE * FILTSIZE;
    const int fmArea = FMSIZE * FMSIZE;
    const int outArea = FMSIZE / 2 * FMSIZE / 2;

    int fn(blockIdx.x), fmy(0), fmx(0), sum(0), sli(0), y(0), x(0), ifmy(0), ifmx(0), filtIdx(0), inNeuIdx(0), outNeuIdx(0);

    for (fmy = 0; fmy < FMSIZE; fmy += STRIDE)
    {
        for (fmx = 0; fmx < FMSIZE; fmx += STRIDE)
        {
            sum = 0;
            for (sli = 0; sli < FMDEPTH; ++sli)
            {
                for (y = 0; y < FILTSIZE; ++y)
                {
                    for (x = 0; x < FILTSIZE; ++x)
                    {
                        ifmy = fmy - FILTSIZE / 2 + y;
                        ifmx = fmx - FILTSIZE / 2 + x;
                        filtIdx = fn * filtVol + sli * filtArea + y * FILTSIZE + x;
                        inNeuIdx = sli * fmArea + ifmy * FMSIZE + ifmx;
                        if (ifmy >= 0 && ifmy < FMSIZE && ifmx >= 0 && ifmx < FMSIZE)
                        {
                            sum += devInputFilter[filtIdx] * devInputNeuron[inNeuIdx];
                        }
                    }
                }
            }
            // Activation - ReLU
            outNeuIdx = fn * fmArea + fmy * FMSIZE + fmx;
            if (sum <= 0)
            {
                devOutputNeuron[outNeuIdx] = 0;
            }
            else
            {
                devOutputNeuron[outNeuIdx] = sum;
            }
        }
    }

    // Max Pooling with Window Size 2x2
    int max(0), tmpVal(0), ofmy(0), ofmx(0), outIdx(0);
    for (sli = 0; sli < FILTNUM; ++sli)
    {
        for (fmy = 0; fmy < FMSIZE / 2 ; ++fmy)
        {
            for (fmx = 0; fmx < FMSIZE / 2 ; ++fmx)
            {
                outNeuIdx = sli * fmArea + fmy * 2 * FMSIZE + fmx * 2;
                max = devOutputNeuron[outNeuIdx];
                for (y = 0; y < 2; ++y)
                {
                    for (x = 0; x < 2; ++x)
                    {
                        ofmy = fmy * 2 + y;
                        ofmx = fmx * 2 + x;
                        outNeuIdx = sli * fmArea + ofmy * FMSIZE + ofmx;
                        tmpVal = devOutputNeuron[outNeuIdx];
                        if (tmpVal > max)
                        {
                            max = tmpVal;
                        }
                    }
                }
                outIdx = sli * outArea + fmy * FMSIZE / 2 + fmx;
                devOutput[outIdx] = max;
            }
        }
    }
}
/*** Implement your CUDA Kernel here ***/

int main()
{
  int convLayerCPUExecTime, convLayerGPUExecTime;
    init(); // Initialize the data on host memory

  timespec time_begin, time_end;

  clock_gettime(CLOCK_REALTIME, &time_begin);
    convLayerCPU();
  clock_gettime(CLOCK_REALTIME, &time_end);
  convLayerCPUExecTime = timespec_diff_us(time_begin, time_end);
  cout << "CPU time for executing a typical convolutional layer = " <<  convLayerCPUExecTime / 1000 << "ms" << endl;

    // declare device pointer
    short * devInputNeuron;
    short * devInputFilter;
    int * devOutputNeuron;
    int * devOutput;

    // compute the size for allocating memory on device
    const int inputNeuronSize = sizeof(short) * FMSIZE * FMSIZE * FMDEPTH;
    const int filtersSize = sizeof(short) * FILTNUM * FILTSIZE * FILTSIZE * FMDEPTH;
    const int outputNeuronSize = sizeof(int) * FMSIZE * FMSIZE * FMDEPTH;
    const int outputSize = sizeof(int) * FMSIZE / 2 * FMSIZE / 2 * FMDEPTH;

    // allocate memory on device
    cudaMalloc(&devInputNeuron, inputNeuronSize);
    cudaMalloc(&devInputFilter, filtersSize);
    cudaMalloc(&devOutputNeuron, outputNeuronSize);
    cudaMalloc(&devOutput, outputSize);

    // copy data from host to deivce
    cudaMemcpy(devInputNeuron, inNeu, inputNeuronSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devInputFilter, filt, filtersSize, cudaMemcpyHostToDevice);

    /*** Lunch your CUDA Kernel here ***/
  clock_gettime(CLOCK_REALTIME, &time_begin);
    convLayerGPU<<<512, 1>>>(devInputNeuron, devInputFilter, devOutputNeuron, devOutput); // Lunch the kernel
    cudaDeviceSynchronize(); // Do synchronization before clock_gettime()
  clock_gettime(CLOCK_REALTIME, &time_end);
    /*** Lunch your CUDA Kernel here ***/

    // copy data from device back to host
    cudaMemcpy(outGPU, devOutput, outputSize, cudaMemcpyDeviceToHost);

    // free the allocated memory on device
    cudaFree(&devInputNeuron);
    cudaFree(&devOutputNeuron);
    cudaFree(&devOutput);
    cudaFree(&devInputFilter);

  convLayerGPUExecTime = timespec_diff_us(time_begin, time_end);
  cout << "GPU time for executing a typical convolutional layer = " << convLayerGPUExecTime / 1000 << "ms" << endl;

    if (checker())
    {
        cout << "Congratulations! You pass the check." << endl;
        cout << "Speedup: " << (float)convLayerCPUExecTime / convLayerGPUExecTime << endl;
    }
    else
    {
        cout << "Sorry! Your result is wrong." << endl;
    }

    ending();

    return 0;
}
