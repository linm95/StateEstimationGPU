#pragma once
#include<math.h>
#include<cuda_runtime.h>
#include<cuda.h>
#include<device_launch_parameters.h>
#include<device_functions.h>
#include<device_atomic_functions.h>
#include "define.h"


__global__ void measureQ(int n, my_type * valG, my_type * valB, int * rowPtrG, int * colIndG, my_type * vol, my_type * theta, my_type *Q, my_type *d_yc, int *idx2row, int *idx2idx) {

	long int index = blockIdx.x * blockDim.x + threadIdx.x;
	long int stride = gridDim.x*blockDim.x;
	int m = rowPtrG[n];
	for (int ii = index; ii < m; ii += stride) {
		int i = idx2idx[ii];
		int row = idx2row[i];
		int col = colIndG[i];
		if (row == col) {
			my_type temp = 0;
			int top = rowPtrG[row + 1];
			for (int j = rowPtrG[row]; j < top; j++) {
				int colj = colIndG[j];
				my_type dtheta = theta[row] - theta[colj];
				temp += vol[colj] * (valG[j] * sin(dtheta) - valB[j] * cos(dtheta));
			}
			temp *= vol[row];
			Q[i] = temp;
		}
		else {
			my_type dtheta = theta[row] - theta[col];
			my_type vi = vol[row];
			my_type vj = vol[col];
			Q[i] = vi*(valB[i] * (vi - vj*cos(dtheta)) - vi*d_yc[i] + vj*valG[i] * sin(dtheta));
		}
	}
}

__global__ void measureP(int n, my_type * valG, my_type * valB, int * rowPtrG, int * colIndG, my_type * vol, my_type * theta, my_type * P, int *idx2row, int *idx2idx) {

	long int index = blockIdx.x * blockDim.x + threadIdx.x;
	long int stride = gridDim.x*blockDim.x;
	int m = rowPtrG[n];
	for (int ii = index; ii < m; ii += stride) {
		int i = idx2idx[ii];
		int row = idx2row[i];
		int col = colIndG[i];
		if (row == col) {
			my_type temp = 0;
			int top = rowPtrG[row + 1];
			for (int j = rowPtrG[row]; j < top; j++) {
				int colj = colIndG[j];
				my_type dtheta = theta[row] - theta[colj];
				temp += vol[colj] * (valG[j] * cos(dtheta) + valB[j] * sin(dtheta));
			}
			temp *= vol[row];
			P[i] = temp;
		}
		else {
			my_type dtheta = theta[row] - theta[col];
			my_type vi = vol[row];
			my_type vj = vol[col];
			P[i] = vi*(valG[i] * (vj*cos(dtheta) - vi) + vj*valB[i] * sin(dtheta));
		}
	}

}

