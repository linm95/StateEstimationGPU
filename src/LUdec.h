#pragma once
#include "stdio.h"
#include "windows.h"
#include "nicslu.h"
#include "cusparse.h"
#include<cuda.h>
#include<cuda_runtime.h>
#include "define.h"

void GetG(my_type * valH, my_type *valHQ, int *colIndH, int *rowPtrH, my_type **valG, my_type **valGQ,
	int ** colIndG, int ** rowPtrG, int m, int n,double *stat, bool en) {

	cusparseStatus_t status;
	cusparseHandle_t handle = 0;


	status = cusparseCreate(&handle);


	cusparseMatDescr_t descrH = 0, descrHT = 0, descrM = 0;
	status = cusparseCreateMatDescr(&descrH);
	status = cusparseCreateMatDescr(&descrHT);
	status = cusparseCreateMatDescr(&descrM);


	cusparseOperation_t transH, transHT;
	transH = CUSPARSE_OPERATION_NON_TRANSPOSE;
	transHT = CUSPARSE_OPERATION_NON_TRANSPOSE;
	int nnzH = rowPtrH[m];
	int *d_colIndH, *d_rowPtrH, *d_colIndG, *d_rowPtrG, *d_colIndHT, *d_rowPtrHT;
	my_type *d_valH, *d_valG, *d_valHT, *d_valHQ, *d_valHQT, *d_valGQ;



	cudaMalloc((void**)&d_colIndHT, nnzH * sizeof(int));
	cudaMalloc((void**)&d_rowPtrHT, (n + 1) * sizeof(int));
	cudaMalloc((void**)&d_valHT, nnzH * sizeof(my_type));
	cudaMalloc((void**)&d_valH, nnzH * sizeof(my_type));
	cudaMalloc((void**)&d_valHQT, nnzH * sizeof(my_type));
	cudaMalloc((void**)&d_valHQ, nnzH * sizeof(my_type));
	cudaMalloc((void**)&d_colIndH, nnzH * sizeof(int));
	cudaMalloc((void**)&d_rowPtrH, (m + 1) * sizeof(int));

	cudaMemcpy(d_valHQ, valHQ, nnzH * sizeof(my_type), cudaMemcpyHostToHost);

	LARGE_INTEGER freq, start, stop;
	double exe_time;
	if (en) {
		QueryPerformanceFrequency(&freq);
		QueryPerformanceCounter(&start);
	}


	cudaMemcpy(d_valH, valH, nnzH * sizeof(my_type), cudaMemcpyHostToHost);
	cudaMemcpy(d_colIndH, colIndH, nnzH * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowPtrH, rowPtrH, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);

	if (en) {
	QueryPerformanceCounter(&stop);
	exe_time = 1e3*(stop.QuadPart - start.QuadPart) / freq.QuadPart;
		stat[14] += exe_time;
	}


	cusparseAction_t copyValues = CUSPARSE_ACTION_NUMERIC;
	cusparseIndexBase_t idx = CUSPARSE_INDEX_BASE_ZERO;

	status = cusparseDcsr2csc(handle, m, n, nnzH, d_valH, d_rowPtrH, d_colIndH, d_valHT, d_colIndHT, d_rowPtrHT, copyValues, idx);
	status = cusparseDcsr2csc(handle, m, n, nnzH, d_valHQ, d_rowPtrH, d_colIndH, d_valHQT, d_colIndHT, d_rowPtrHT, copyValues, idx);



	cudaMalloc((void**)&d_rowPtrG, (n + 1) * sizeof(int));

	int nnzM, baseM;
	int* d_nnzM = &nnzM;



	status = cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
	status = cusparseXcsrgemmNnz(handle, transHT, transH, n, n, m, descrHT, nnzH, d_rowPtrHT, d_colIndHT,
		descrH, nnzH, d_rowPtrH, d_colIndH, descrM, d_rowPtrG, d_nnzM);


	if (NULL != d_nnzM) {
		nnzM = *d_nnzM;
	}
	else {
		printf("Wrong!\n");
	}
	(*valG) = (my_type*)malloc(nnzM * sizeof(my_type));
	(*valGQ) = (my_type*)malloc(nnzM * sizeof(my_type));
	(*colIndG) = (int*)malloc(nnzM * sizeof(int));
	(*rowPtrG) = (int*)malloc((n + 1) * sizeof(int));

	

	cudaMalloc((void**)&d_colIndG, nnzM * sizeof(int));
	cudaMalloc((void**)&d_valG, nnzM * sizeof(my_type));
	cudaMalloc((void**)&d_valGQ, nnzM * sizeof(my_type));


	status = cusparseDcsrgemm(handle, transHT, transH, n, n, m, descrHT, nnzH, d_valHT, d_rowPtrHT, d_colIndHT,
		descrH, nnzH, d_valH, d_rowPtrH, d_colIndH, descrM, d_valG, d_rowPtrG, d_colIndG);
	status = cusparseDcsrgemm(handle, transHT, transH, n, n, m, descrHT, nnzH, d_valHQT, d_rowPtrHT, d_colIndHT,
		descrH, nnzH, d_valHQ, d_rowPtrH, d_colIndH, descrM, d_valGQ, d_rowPtrG, d_colIndG);

	cudaMemcpy(*valGQ, d_valGQ, nnzM * sizeof(my_type), cudaMemcpyDeviceToHost);

	if (en) {
		QueryPerformanceCounter(&start);
	}

	cudaMemcpy(*colIndG, d_colIndG, nnzM * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(*rowPtrG, d_rowPtrG, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(*valG, d_valG, nnzM * sizeof(my_type), cudaMemcpyDeviceToHost);
	

	if (en) {
		QueryPerformanceCounter(&stop);
		exe_time = 1e3*(stop.QuadPart - start.QuadPart) / freq.QuadPart;

		stat[15] += exe_time;
	}

	cudaFree(d_colIndH);
	cudaFree(d_colIndHT);
	cudaFree(d_colIndG);
	cudaFree(d_rowPtrH);
	cudaFree(d_rowPtrHT);
	cudaFree(d_rowPtrG);
	cudaFree(d_valH);
	cudaFree(d_valHT);
	cudaFree(d_valHQ);
	cudaFree(d_valHQT);
	cudaFree(d_valG);
	cudaFree(d_valGQ);

	cusparseDestroy(handle);
	cusparseDestroyMatDescr(descrH);
	cusparseDestroyMatDescr(descrHT);
	cusparseDestroyMatDescr(descrM);
}

void LUdecom(my_type *valG, my_type *valGQ, int *rowPtrG, int *colIndG, int **P, my_type **valL, int **colIndL, int **rowPtrL,
	my_type **valU, int **colIndU, int **rowPtrU, my_type **valLQ, my_type **valUQ, int m, int n) {


	//int *colIndG, *rowPtrG;
	//my_type *valG, *valGQ;

	//QueryPerformanceCounter(&start);

	//GetG(val, valHQ, colInd, rowPtr, &valG, &valGQ, &colIndG, &rowPtrG, m, n);
	int nnzG = rowPtrG[n];

	/*QueryPerformanceCounter(&stop);
	exe_time = 1e3*(stop.QuadPart - start.QuadPart) / freq.QuadPart;
	printf("get M takes %f ms\n", exe_time);*/


	//QueryPerformanceCounter(&start);

	_handle_t solver = NULL;
	_double_t *cfg;
	const _double_t *stat;
	NicsLU_Initialize(&solver, &cfg, &stat);
	cfg[3] = -2;

	NicsLU_CreateThreads(solver, 0);

	_uint_t *P1, *P2;
	P1 = (_uint_t*)malloc(n * sizeof(_uint_t));
	P2 = (_uint_t*)malloc(n * sizeof(_uint_t));
	NicsLU_Analyze(solver, n, valG, (_uint_t*)colIndG, (_uint_t*)rowPtrG, MATRIX_ROW_REAL, P1, P2);
	*P = (int*)malloc((n + 1) * sizeof(int));
	for (int i = 0; i < n; i++) {
		(*P)[P1[i]] = i;
	}
	(*P)[n] = n;
	NicsLU_Factorize(solver, valG, 1);
	int nnzL = stat[9], nnzU = stat[10];
	(*colIndL) = (int*)malloc(nnzL * sizeof(int));
	(*rowPtrL) = (int*)malloc((n + 1) * sizeof(int));
	(*valL) = (my_type*)malloc(nnzL * sizeof(my_type));
	(*colIndU) = (int*)malloc(nnzU * sizeof(int));
	(*rowPtrU) = (int*)malloc((n + 1) * sizeof(int));
	(*valU) = (my_type*)malloc(nnzU * sizeof(my_type));
	_size_t *lp, *up;
	lp = (_size_t*)malloc((n + 1) * sizeof(_size_t));
	up = (_size_t*)malloc((n + 1) * sizeof(_size_t));
	_uint_t *li, *ui;
	li = (_uint_t*)malloc(nnzL * sizeof(_uint_t));
	ui = (_uint_t*)malloc(nnzU * sizeof(_uint_t));
	NicsLU_GetFactors(solver, *valL, (_uint_t*)(*colIndL), lp, *valU, (_uint_t*)(*colIndU), up, true, NULL, NULL, NULL, NULL);
	for (int i = 0; i <= n; i++) {
		(*rowPtrL)[i] = lp[i];
		(*rowPtrU)[i] = up[i];
	}

	NicsLU_ReFactorize(solver, valGQ, 1);
	nnzL = stat[9], nnzU = stat[10];

	(*valLQ) = (my_type*)malloc(nnzL * sizeof(my_type));
	(*valUQ) = (my_type*)malloc(nnzU * sizeof(my_type));

	NicsLU_GetFactors(solver, *valLQ, li, lp, *valUQ, ui, up, true, NULL, NULL, NULL, NULL);

	//QueryPerformanceCounter(&stop);
	//exe_time = 1e3*(stop.QuadPart - start.QuadPart) / freq.QuadPart;
	//printf("LU refact takes %f ms\n", exe_time);

	NicsLU_Free(solver);
	free(P1);
	free(P2);
	free(li);
	free(ui);
	free(up);
	free(lp);
}
