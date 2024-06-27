#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <random>
#include <cmath>
#include <chrono>

#define N 2048

void mmGPU(int* a, int* b, int* c, int n);
void mmCPU(int* a, int* b, int* c, int n);
void populateArray(int* arr, unsigned int size);
void printArray(int* arr, unsigned int size);

template<typename Func>
void elapsedTime(Func func, int* a, int* b, int* c, int n);

__global__ void mmKernel(int* a, int* b, int* c) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < N && col < N) {
		int sum = 0;

		for (int i = 0; i < N; i++) {
			sum += a[row * N + i] * b[i * N + col];
		}
		c[row * N + col] = sum;
	}
}


int main() {
	int size = N * N;
	int* a = new int[size];
	int* b = new int[size];
	int* cpuC = new int[size];
	int* gpuC = new int[size];

	populateArray(a, size);
	populateArray(b, size);

	//printArray(a, size);
	//printArray(b, size);

	elapsedTime(mmCPU, a, b, cpuC, N);
	printf("%d\n", cpuC[size - 1]);

	elapsedTime(mmGPU, a, b, gpuC, N);
	printf("%d\n", gpuC[size - 1]);

	delete[] a;
	delete[] b;
	delete[] cpuC;
	delete[] gpuC;

	return 0;
}

void populateArray(int* arr, unsigned int size) { //test with non-random?
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, 9);

	for (int i = 0; i < size; i++) {
		arr[i] = dis(gen);
	}
}

void printArray(int* arr, unsigned int size) {
	for (int i = 0; i < size; i++) {
		if (i > 0 && i % N == 0) {
			printf("\n");
		}
		printf("%d ", arr[i]);
	}
	printf("\n\n");
}

void mmCPU(int* a, int* b, int* c, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			int sum = 0;
			for (int k = 0; k < n; k++) {
				sum += a[i * n + k] * b[k * n + j];
			}
			c[i * n + j] = sum;
		}
	}
}

void mmGPU(int* a, int* b, int* c, int n) {
	int* d_a = 0;
	int* d_b = 0;
	int* d_c = 0;

	int alloSize = n * n * sizeof(int);

	cudaMalloc((void**)&d_a, alloSize);
	cudaMalloc((void**)&d_b, alloSize);
	cudaMalloc((void**)&d_c, alloSize);

	cudaMemcpy(d_a, a, alloSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, alloSize, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(32, 32);
	dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

	mmKernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c);

	cudaDeviceSynchronize();
	cudaMemcpy(c, d_c, alloSize, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}


template<typename Func>
void elapsedTime(Func func, int* a, int* b, int* c, int n) {
	auto start = std::chrono::high_resolution_clock::now();

	func(a, b, c, n);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> duration = end - start;

	printf("elapsed time: %f seconds\n", duration.count());
}