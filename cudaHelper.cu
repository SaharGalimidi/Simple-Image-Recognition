#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "helper.h"

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
// this gpuErrchk macro was taken from this link: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

__device__ void calculateMatch(int objectDimension, int pictureDimension, int* d_pictureColorsMatrix, int* d_objectSubColorsMatrix, int picrureRow, int pictureCol, double* res)
{
        for( int i = 0; i < objectDimension; i++)
        {
            for( int j = 0; j < objectDimension; j++)
            {
                int objectColor = d_objectSubColorsMatrix[i * objectDimension + j];
                int pictureColor = d_pictureColorsMatrix[(picrureRow + i) * pictureDimension + (pictureCol + j)];
                if (pictureColor != 0)
                    *(res) += (double)abs((pictureColor - objectColor)) / pictureColor;
            }
        }
}

/*
 * Kernel function for calculating the difference between the colors of the overlapping pixels of the Object and the Picture using the formula:abs((P - O) / P)
 * @param d_pictureColorsMatrix - the colors matrix of the Picture on the GPU
 * @param d_objectSubColorsMatrix - the sub colors matrix of the Object on the GPU
 * @param d_matchingValue - the matching value that will be returned to the host
 * @param d_objectDimension - the dimension of the Object
 * @param d_pictureDimension - the dimension of the Picture
 * @param d_upperLeftCorner - the index of the upper-left corner of the object in the picture
 */
__global__ void calculateMatching(int *d_pictureColorsMatrix, int *d_objectSubColorsMatrix, double *d_matchingThreshold, int *d_objectDimension, int *d_pictureDimension, int *d_upperLeftCorner)
{
    int globalThreadIndex = blockDim.x * blockIdx.x + threadIdx.x;

    if (globalThreadIndex < ((*d_pictureDimension) - (*d_objectDimension) + 1) * ((*d_pictureDimension) - (*d_objectDimension) + 1))
    {
        double res;
        int pictureRow = globalThreadIndex / ((*d_pictureDimension) - (*d_objectDimension) + 1);
        int pictureCol = globalThreadIndex % ((*d_pictureDimension) - (*d_objectDimension) + 1);
        if (pictureCol < 0 || pictureCol >= (*d_pictureDimension) - (*d_objectDimension) + 1 || pictureRow < 0 || pictureRow >= (*d_pictureDimension) - (*d_objectDimension) + 1)
            return;
        calculateMatch(*d_objectDimension, *d_pictureDimension, d_pictureColorsMatrix, d_objectSubColorsMatrix, pictureRow, pictureCol, &res);
        if (res / ((*d_objectDimension) * (*d_objectDimension)) < (*d_matchingThreshold))
            (*d_upperLeftCorner) = pictureRow * (*d_pictureDimension) + pictureCol;
    }
}

__host__ void calculateMatchingOnGPU(Picture *picture, Object *object, int *upperLeftCorner, double matchingThreshold)
{
    // Allocate and copy memory for the matchingThreshold on the GPU
    double *d_matchingThreshold;
    gpuErrchk(cudaMalloc((void **)&d_matchingThreshold, sizeof(double)));
    gpuErrchk(cudaMemcpy(d_matchingThreshold, &matchingThreshold, sizeof(double), cudaMemcpyHostToDevice));

    // Allocate memory and copy for the upper left corner on the GPU
    int *d_upperLeftCorner;
    gpuErrchk(cudaMalloc((void **)&d_upperLeftCorner, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_upperLeftCorner, upperLeftCorner, sizeof(int), cudaMemcpyHostToDevice));

    // Allocate memory and copy for the picture colors matrix on the GPU
    int *d_pictureColorsMatrix;
    gpuErrchk(cudaMalloc((void **)&d_pictureColorsMatrix, picture->dimension * picture->dimension * sizeof(int)));
    gpuErrchk(cudaMemcpy(d_pictureColorsMatrix, picture->colorsMatrix, picture->dimension * picture->dimension * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate memory and copy for the picture dimension on the GPU
    int *d_pictureDimension;
    gpuErrchk(cudaMalloc((void **)&d_pictureDimension, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_pictureDimension, &picture->dimension, sizeof(int), cudaMemcpyHostToDevice));

    // Allocate memory and copy for the object dimension on the GPU
    int *d_objectDimension;
    gpuErrchk(cudaMalloc((void **)&d_objectDimension, sizeof(int)));
    gpuErrchk(cudaMemcpy(d_objectDimension, &object->dimension, sizeof(int), cudaMemcpyHostToDevice));

    // Allocate memory and copy for the object sub colors matrix on the GPU
    int *d_objectSubColorsMatrix;
    gpuErrchk(cudaMalloc((void **)&d_objectSubColorsMatrix, (object->dimension * object->dimension * sizeof(int))));
    gpuErrchk(cudaMemcpy(d_objectSubColorsMatrix, object->subColorsMatrix, (object->dimension * object->dimension * sizeof(int)), cudaMemcpyHostToDevice));

    int size = (picture->dimension - object->dimension + 1) * (picture->dimension - object->dimension + 1);
    int blocksPerGrid = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // call the kernel function
    calculateMatching<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_pictureColorsMatrix, d_objectSubColorsMatrix, d_matchingThreshold, d_objectDimension, d_pictureDimension, d_upperLeftCorner);

    // check if the kernel function was called successfully
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // copy the upper left corner row from the GPU to the host
    gpuErrchk(cudaMemcpy(upperLeftCorner, d_upperLeftCorner, sizeof(int), cudaMemcpyDeviceToHost));

    // free the memory on the GPU
    cudaFree(d_matchingThreshold);
    cudaFree(d_pictureColorsMatrix);
    cudaFree(d_objectSubColorsMatrix);
    cudaFree(d_pictureDimension);
    cudaFree(d_objectDimension);
    cudaFree(d_upperLeftCorner);
}
