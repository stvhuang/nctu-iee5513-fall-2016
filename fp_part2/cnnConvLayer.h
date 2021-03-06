#ifndef __CNNCONVLAYER_H__
#define __CNNCONVLAYER_H__
// 0233
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
using namespace std;

#define FMSIZE 32
#define FMDEPTH 512
#define FILTSIZE 3
#define FILTNUM 512
#define STRIDE 1

short* filt;
short* inNeu;
int* outNeu;
int* outCPU;
int* outGPU;

short* filtCooNNZ;
short* filtCooData;
short* filtCooRow;
short* filtCooCol;

short* inNeuCooNNZ;
short* inNeuCooData;
short* inNeuCooRow;
short* inNeuCooCol;

///////////////////////////////////////////////////////////////////////////////
// host pointer
short* inFiltCccIdx;  // our format
short* inNeuCccIdx;  // our format

// device pointer
short* devFiltCooData;
short* devFiltCccIdx;  // our format
short* devInNeuCooData;
short* devInNeuCccIdx;  // ourformat
int* devOut;  // result

// convert filter's format
void cooToCccFilt(const int nnz, const short* cooRow, const short* cooCol, short* cccIdx)
{
    for (int k(0); k < 512; ++k) {
        for (int j(0); j < 512; ++j) {
            for (int i(0); i < nnz; ++i) {
                cccIdx[i + k * nnz * 512 + j * nnz] = cooRow[i + k * nnz * 512 + j * nnz] * 3
                    + cooCol[i + k * nnz * 512 + j * nnz];
            }
        }
    }
}

// convert neuron's format
void cooToCccNeu(const int nnz, const short* cooRow, const short* cooCol, short* cccIdx)
{
    int k(0);
    for (int j(0); j < 512; ++j) {
        k = j * nnz;
        for (int i(0); i < nnz; ++i) {
            cccIdx[i + k] = cooRow[i + k] * 32 + cooCol[i + k];
        }
    }
}
///////////////////////////////////////////////////////////////////////////////

void init()
{
    int i, j, k, l;
    string str;
    int tmp, inNeuIdx, filtIdx;
    int filtTensorVol = FILTNUM * FMDEPTH * FILTSIZE * FILTSIZE;
    int inNeuVol = FMDEPTH * FMSIZE * FMSIZE;
    int outNeuVol = FILTNUM * FMSIZE * FMSIZE;
    int outVol = FILTNUM * FMSIZE / 2 * FMSIZE / 2;

    fstream ifs;

    inNeu = new short[inNeuVol]();
    ifs.open("../data/inNeu.txt", ifstream::in);
    if (!ifs.is_open()) {
        cout << "Can not open the neurons input file\n";
        exit(-1);
    }
    for (i = 0; i < FMDEPTH; ++i) {
        ifs >> str;
        for (j = 0; j < FMSIZE; ++j) {
            for (k = 0; k < FMSIZE; ++k) {
                ifs >> tmp;
                inNeuIdx = i * FMSIZE * FMSIZE + j * FMSIZE + k;
                inNeu[inNeuIdx] = tmp;
            }
        }
    }
    ifs.close();

    filt = new short[filtTensorVol]();
    ifs.open("../data/filt.txt", ifstream::in);
    if (!ifs.is_open()) {
        cout << "Can not open the filters input file\n";
        exit(-1);
    }
    for (i = 0; i < FILTNUM; ++i) {
        ifs >> str;
        for (j = 0; j < FMDEPTH; ++j) {
            ifs >> str;
            for (k = 0; k < FILTSIZE; ++k) {
                for (l = 0; l < FILTSIZE; ++l) {
                    ifs >> tmp;
                    filtIdx = i * FMDEPTH * FILTSIZE * FILTSIZE + j * FILTSIZE * FILTSIZE
                        + k * FILTSIZE + l;
                    filt[filtIdx] = tmp;
                }
            }
        }
    }
    ifs.close();

    outNeu = new int[outNeuVol]();

    outCPU = new int[outVol]();
    outGPU = new int[outVol]();
}

void initCoo()
{
    int i, j, k, idx;
    short tmp, nnz;
    string str;

    fstream ifs;

    filtCooNNZ = new short[FILTNUM * FMDEPTH];

    ifs.open("../data/filt.coo", ifstream::in);
    if (!ifs.is_open()) {
        cout << "Can not open the filters input file\n";
        exit(-1);
    }
    for (i = 0; i < FILTNUM; ++i) {
        ifs >> str;
        for (j = 0; j < FMDEPTH; ++j) {
            ifs >> str;
            ifs >> str >> nnz;
            idx = i * FMDEPTH + j;
            filtCooNNZ[idx] = nnz;
            if (i == 0 && j == 0) {
                filtCooData = new short[FILTNUM * FMDEPTH * nnz];
                filtCooRow = new short[FILTNUM * FMDEPTH * nnz];
                filtCooCol = new short[FILTNUM * FMDEPTH * nnz];
            }
            for (k = 0; k < nnz; ++k) {
                ifs >> str >> tmp;
                idx = i * FMDEPTH * nnz + j * nnz + k;
                filtCooData[idx] = tmp;
            }
            for (k = 0; k < nnz; ++k) {
                ifs >> str >> tmp;
                idx = i * FMDEPTH * nnz + j * nnz + k;
                filtCooRow[idx] = tmp;
            }
            for (k = 0; k < nnz; ++k) {
                ifs >> str >> tmp;
                idx = i * FMDEPTH * nnz + j * nnz + k;
                filtCooCol[idx] = tmp;
            }
        }
    }
    ifs.close();

    inNeuCooNNZ = new short[FMDEPTH];

    ifs.open("../data/inNeu.coo", ifstream::in);
    if (!ifs.is_open()) {
        cout << "Can not open the neurons input file\n";
        exit(-1);
    }
    for (i = 0; i < FMDEPTH; ++i) {
        ifs >> str;
        ifs >> str >> nnz;
        inNeuCooNNZ[i] = nnz;
        if (i == 0) {
            inNeuCooData = new short[FMDEPTH * nnz];
            inNeuCooRow = new short[FMDEPTH * nnz];
            inNeuCooCol = new short[FMDEPTH * nnz];
        }

        ifs >> str;
        for (j = 0; j < nnz; ++j) {
            ifs >> tmp;
            idx = i * nnz + j;
            inNeuCooData[idx] = tmp;
        }
        ifs >> str;
        for (j = 0; j < nnz; ++j) {
            ifs >> tmp;
            idx = i * nnz + j;
            inNeuCooRow[idx] = tmp;
        }
        ifs >> str;
        for (j = 0; j < nnz; ++j) {
            ifs >> tmp;
            idx = i * nnz + j;
            inNeuCooCol[idx] = tmp;
        }
    }
    ifs.close();

    ///////////////////////////////////////////////////////////////////////////////
    inNeuCccIdx = new short[nnz * 512];
    cooToCccNeu(204, inNeuCooRow, inNeuCooCol, inNeuCccIdx);
    inFiltCccIdx = new short[nnz * 512 * 512];
    cooToCccFilt(1, filtCooRow, filtCooCol, inFiltCccIdx);
    ///////////////////////////////////////////////////////////////////////////////
}

void ending()
{
    delete[] filt;
    delete[] inNeu;
    delete[] outNeu;
    delete[] outCPU;
    delete[] outGPU;

    delete[] filtCooNNZ;
    delete[] filtCooData;
    delete[] filtCooRow;
    delete[] filtCooCol;

    delete[] inNeuCooNNZ;
    delete[] inNeuCooData;
    delete[] inNeuCooRow;
    delete[] inNeuCooCol;

    ///////////////////////////////////////////////////////////////////////////////
    delete[] inFiltCccIdx;
    delete[] inNeuCccIdx;

    cudaFree(&devFiltCooData);
    cudaFree(&devFiltCccIdx);

    cudaFree(&devInNeuCooData);
    cudaFree(&devInNeuCccIdx);

    cudaFree(&devOut);  // done
    ///////////////////////////////////////////////////////////////////////////////
}

bool checker()
{
    int i;
    int outVol = FILTNUM * FMSIZE / 2 * FMSIZE / 2;

    for (i = 0; i < outVol; ++i) {
        if (outCPU[i] != outGPU[i]) {
            cout << "The element: " << i << " is wrong!\n";
            cout << "outCPU[" << i << "] = " << outCPU[i] << endl;
            cout << "outGPU[" << i << "] = " << outGPU[i] << endl;
            return false;
        }
    }

    return true;
}

int timespec_diff_us(timespec& t1, timespec& t2)
{
    return (t2.tv_sec - t1.tv_sec) * 1e6 + (t2.tv_nsec - t1.tv_nsec) / 1e3;
}

#endif
