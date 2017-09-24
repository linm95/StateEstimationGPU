#pragma once
#include<math.h>
#include<cuda.h>
#include<device_launch_parameters.h>
#include "define.h"
#include "math.h"

__device__ my_type _dAtomicAdd(my_type* address, my_type val) {
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void check(my_type *d_dv, my_type *d_v, int n, bool *flag, int* Pinv) {
	int index = threadIdx.x;
	int stride = blockDim.x;
	int block = blockIdx.x;
	__shared__ bool l_flag;
	if (index == 0) l_flag = false;
	__syncthreads();
	for (int i = index; i < n; i += stride) {
		my_type dv = d_dv[Pinv[i]];
		d_v[i] += dv;
		if (!l_flag && fabs(dv) > toll)
			l_flag = true;
	}
	__syncthreads();
	if (index == 0) {
		*flag = l_flag;
	}
}

__global__ void getJ(my_type *d_dP, my_type *d_dQ, int m, my_type *J) {
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = gridDim.x*blockDim.x;
	int mm = m;
	__shared__ my_type lo_J;
	if (index == 0) {
		lo_J = 0;
	}
	__syncthreads();
	for (int i = index; i < 2 * mm; i += stride) {
		if (i < mm) {
			my_type temp = d_dP[i];
			temp = temp*temp;
			_dAtomicAdd(&lo_J, temp);
		}
		else {
			my_type temp = d_dQ[i - mm];
			temp = temp*temp;
			_dAtomicAdd(&lo_J, temp);
		}
	}
	__syncthreads();
	if (index == 0) {
		*J = lo_J;
	}
	__syncthreads();
}

__global__ void differ(int n, my_type *measureValue, my_type *calculatedValue, my_type *differnce) {
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = gridDim.x*blockDim.x;
	for (int i = index; i < n; i += stride) {
		differnce[i] = measureValue[i] - calculatedValue[i];
	}
}

void reorder(my_type *val, my_type *val2, int *rowPtr, int *colInd, int *P, int m, int n) {

	//Ñ¡ÔñÅÅÐò
	for (int i = 0; i < m; i++) {
		int offset = rowPtr[i];
		int length = rowPtr[i + 1] - rowPtr[i];
		for (int j = 0; j < length; j++) {
			colInd[j + offset] = P[colInd[j + offset]];
		}
		for (int j = 0; j < length - 1; j++) {
			int min = n;
			int indOfMin;
			for (int k = j; k < length; k++)
				if (min > colInd[k + offset]) {
					min = colInd[k + offset];
					indOfMin = k;
				}
			int tempC = colInd[j + offset];
			colInd[j + offset] = colInd[indOfMin + offset];
			colInd[indOfMin + offset] = tempC;
			my_type tempV = val[j + offset];
			val[j + offset] = val[indOfMin + offset];
			val[indOfMin + offset] = tempV;
			tempV = val2[j + offset];
			val2[j + offset] = val2[indOfMin + offset];
			val2[indOfMin + offset] = tempV;
		}
	}
}


