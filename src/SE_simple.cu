#include "define.h"
#include "DataProcessing.h"
#include "Jacob.h"
#include "measure.h";
#include "LUdec.h"
#include "appendix.h"


int main() {
	LARGE_INTEGER freq, start, stop;
	double exe_time;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start);

	/*Step 1: read the data, including: mat B, mat G, vector P, vector Q, vector vol, vector theta*/
	printf("***The accelaration of State Estimation based on GPU***\n");
	printf("			writer:linm\n\n");
	printf("PreSE start\n	Reading and processing data...\n");
	struct point *data;
	FILE *mat;
	double SEtime;
	mat = fopen("case/case13659.txt", "r");
	if (mat == NULL) {
		printf("fail");
	}
	int nB, numOfData;
	fscanf(mat, "%d", &nB);
	fscanf(mat, "%d", &numOfData);
	char s[100];
	fgets(s, 100, mat);
	data = (point*)malloc(numOfData * sizeof(point));
	for (int i = 0; i < numOfData; i++) {
		fscanf(mat, "%d", &(data[i].rowInd));
		fscanf(mat, "%d", &(data[i].colInd));
		fscanf(mat, "%lf", &(data[i].G));
		fscanf(mat, "%lf", &(data[i].Br));
		fscanf(mat, "%lf", &(data[i].Ba));
		fscanf(mat, "%lf", &(data[i].P));
		fscanf(mat, "%lf", &(data[i].Q));
		fscanf(mat, "%lf", &(data[i].yc_half));
		fscanf(mat, "%lf", &(data[i].vol));
		fscanf(mat, "%lf", &(data[i].theta));
	}
	//get admittance mat B, power flow vector P,Q, yc
	int *idx2row, *idx2idx;
	my_type *valB, *valG, *P, *Q, *yc_half, *valBa;
	int *colIndB, *rowPtrB, maxPerRowB;
	dataProcessing(data, &idx2row, &idx2idx, &valB, &valG, &valBa, &P, &Q, &yc_half,
		&rowPtrB, &colIndB, nB, numOfData, &maxPerRowB);

	//set the initial value
	my_type *vol, *theta;
	vol = (my_type*)malloc(nB * sizeof(my_type));
	theta = (my_type*)calloc(nB, sizeof(my_type));
	for (int i = 0; i < nB; i++) {
		vol[i] = data[numOfData - 1].vol;
		theta[i] = 0;
	}

	//synchronize the gpu and cpu
	QueryPerformanceCounter(&stop);
	exe_time = 1e3*(stop.QuadPart - start.QuadPart) / freq.QuadPart;
	printf("	Data ready. Spent %f ms \n	Starting the GPU...\n",exe_time);
	SEtime = exe_time;
	QueryPerformanceCounter(&start);

	cudaThreadSynchronize();
	cusparseHandle_t handle;
	cusparseCreate(&handle);

	QueryPerformanceCounter(&stop);
	exe_time = 1e3*(stop.QuadPart - start.QuadPart) / freq.QuadPart;
	SEtime += exe_time;
	printf("	GPU ready. Spent %f ms\nPreSE done. Spent %f ms\nSE start\n	Initializing...\n",exe_time,SEtime);
	QueryPerformanceCounter(&start);

	//Get the jacob mat H
	my_type *valH, *valHQ;
	int *rowPtrH, *colIndH;

	GetJacob(valBa, valB, colIndB, rowPtrB, &valH, &valHQ, &rowPtrH, &colIndH, nB);


	//LU refact 
	my_type *valR, *valL, *valRQ, *valLQ;
	int *colIndR, *rowPtrR, *colIndL, *rowPtrL;
	int nH = nB - 1;
	int m = rowPtrB[nB];
	int *Pinv;
	LUdecom(valH, valHQ, rowPtrH, colIndH, &Pinv, &valL, &colIndL, &rowPtrL, &valR, &colIndR, &rowPtrR, &valLQ, &valRQ, m, nH);

	//reorder the jacob mat H

	reorder(valH, valHQ, rowPtrH, colIndH, Pinv, m, nH);

	

	/*step3: loop*/
	cusparseStatus_t status;
	/*copy L U mat*/
	int nnzR = rowPtrR[nH];
	int nnzL = rowPtrL[nH];
	my_type *d_valR, *d_valRQ, *d_valL, *d_valLQ;
	int *d_colIndR, *d_rowPtrR, *d_colIndL, *d_rowPtrL;
	cudaMalloc((void**)&d_valR, nnzR * sizeof(my_type));
	cudaMalloc((void**)&d_valRQ, nnzR * sizeof(my_type));
	cudaMalloc((void**)&d_colIndR, nnzR * sizeof(int));
	cudaMalloc((void**)&d_rowPtrR, (nH + 1) * sizeof(int));
	cudaMemcpy(d_valR, valR, nnzR * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_valRQ, valRQ, nnzR * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIndR, colIndR, nnzR * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowPtrR, rowPtrR, (nH + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_valL, nnzL * sizeof(my_type));
	cudaMalloc((void**)&d_valLQ, nnzL * sizeof(my_type));
	cudaMalloc((void**)&d_colIndL, nnzL * sizeof(int));
	cudaMalloc((void**)&d_rowPtrL, (nH + 1) * sizeof(int));
	cudaMemcpy(d_valL, valL, nnzL * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_valLQ, valLQ, nnzL * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIndL, colIndL, nnzL * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowPtrL, rowPtrL, (nH + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cusparseMatDescr_t descrR;
	cusparseCreateMatDescr(&descrR);
	cusparseSetMatType(descrR, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descrR, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatIndexBase(descrR, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatDiagType(descrR, CUSPARSE_DIAG_TYPE_UNIT);
	cusparseMatDescr_t descrL;
	cusparseCreateMatDescr(&descrL);
	cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER);
	cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT);

	/*copy H mat*/
	int nnzH = rowPtrH[m];
	my_type *d_valH, *d_valHQ;
	int *d_colIndH, *d_rowPtrH;
	cudaMalloc((void**)&d_valH, nnzH * sizeof(my_type));
	cudaMalloc((void**)&d_valHQ, nnzH * sizeof(my_type));
	cudaMalloc((void**)&d_colIndH, nnzH * sizeof(int));
	cudaMalloc((void**)&d_rowPtrH, (m + 1) * sizeof(int));
	cudaMemcpy(d_valH, valH, nnzH * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_valHQ, valHQ, nnzH * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIndH, colIndH, nnzH * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowPtrH, rowPtrH, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cusparseMatDescr_t descrH;
	cusparseCreateMatDescr(&descrH);

	/*copy B mat*/
	int nnzB = rowPtrB[nB];
	my_type *d_valB, *d_valG;
	int *d_colIndB, *d_rowPtrB;
	cudaMalloc((void**)&d_valG, nnzB * sizeof(my_type));
	cudaMalloc((void**)&d_valB, nnzB * sizeof(my_type));
	cudaMalloc((void**)&d_rowPtrB, (nB + 1) * sizeof(int));
	cudaMalloc((void**)&d_colIndB, nnzB * sizeof(int));
	cudaMemcpy(d_valG, valG, nnzB * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_valB, valB, nnzB * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowPtrB, rowPtrB, (nB + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIndB, colIndB, nnzB * sizeof(int), cudaMemcpyHostToDevice);

	/*copy P,Q,vol,theta,yc*/
	my_type *d_P, *d_Q, *d_vol, *d_theta, *d_yc_half;
	cudaMalloc((void**)&d_P, m * sizeof(my_type));
	cudaMalloc((void**)&d_Q, m * sizeof(my_type));
	cudaMemcpy(d_P, P, m * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Q, Q, m * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_vol, nB * sizeof(my_type));
	cudaMalloc((void**)&d_theta, nB * sizeof(my_type));
	cudaMemcpy(d_vol, vol, nB * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMemcpy(d_theta, theta, nB * sizeof(my_type), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_yc_half, m * sizeof(my_type));
	cudaMemcpy(d_yc_half, yc_half, m * sizeof(my_type), cudaMemcpyHostToDevice);

	/*prepare for calculating measurement function*/
	my_type *d_caledP, *d_caledQ;
	int *d_idx2row, *d_idx2idx;
	cudaMalloc((void**)&d_idx2row, m * sizeof(int));
	cudaMemcpy(d_idx2row, idx2row, m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_idx2idx, m * sizeof(int));
	cudaMemcpy(d_idx2idx, idx2idx, m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_caledP, m * sizeof(my_type));
	cudaMalloc((void**)&d_caledQ, m * sizeof(my_type));


	/*prepare for calculating difference vector*/
	my_type *d_dP, *d_dQ, *d_J;
	cudaMalloc((void**)&d_J, sizeof(my_type));
	cudaMalloc((void**)&d_dP, m * sizeof(my_type));
	cudaMalloc((void**)&d_dQ, m * sizeof(my_type));


	/*prepare for calculating the b vector*/
	my_type alpha = 1, beta = 0;
	my_type *d_bP, *d_bQ;
	cudaMalloc((void**)&d_bP, nH * sizeof(my_type));
	cudaMalloc((void**)&d_bQ, nH * sizeof(my_type));

	/*prepare for solving LU equation*/
	/*buffersize analysis*/
	int buffersizeL, buffersizeU, buffersizeLQ, buffersizeUQ;
	csrsv2Info_t infoL, infoU, infoLQ, infoUQ;
	cusparseCreateCsrsv2Info(&infoL);
	cusparseCreateCsrsv2Info(&infoU);
	cusparseCreateCsrsv2Info(&infoLQ);
	cusparseCreateCsrsv2Info(&infoUQ);
	status = cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzL, descrL, d_valL, d_rowPtrL, d_colIndL, infoL, &buffersizeL);
	status = cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, descrR, d_valR, d_rowPtrR, d_colIndR, infoU, &buffersizeU);
	status = cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzL, descrL, d_valLQ, d_rowPtrL, d_colIndL, infoLQ, &buffersizeLQ);
	status = cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, descrR, d_valRQ, d_rowPtrR, d_colIndR, infoUQ, &buffersizeUQ);
	void * buffer, *bufferQ;
	if (buffersizeL > buffersizeU)
		cudaMalloc((void**)&buffer, buffersizeL);
	else
		cudaMalloc((void**)&buffer, buffersizeU);
	if (buffersizeLQ > buffersizeUQ)
		cudaMalloc((void**)&bufferQ, buffersizeLQ);
	else
		cudaMalloc((void**)&bufferQ, buffersizeUQ);
	//the delta vector
	my_type *d_temp, *d_dvol, *d_dtheta;
	cudaMalloc((void**)&d_temp, nH * sizeof(my_type));
	cudaMalloc((void**)&d_dvol, nH * sizeof(my_type));
	cudaMalloc((void**)&d_dtheta, nH * sizeof(my_type));
	/*solving analysis*/
	status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzL, descrL, d_valL, d_rowPtrL, d_colIndL, infoL, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);
	status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, descrR, d_valR, d_rowPtrR, d_colIndR, infoU, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);
	status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzL, descrL, d_valLQ, d_rowPtrL, d_colIndL, infoLQ, CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferQ);
	status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, descrR, d_valRQ, d_rowPtrR, d_colIndR, infoUQ, CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferQ);

	/*prepare for checking the result*/
	int *d_Pinv;
	cudaMalloc((void**)&d_Pinv, nB * sizeof(int));
	cudaMemcpy(d_Pinv, Pinv, nB * sizeof(int), cudaMemcpyHostToDevice);
	//flag for loop
	bool *flagV, *d_flagV, *flagT, *d_flagT;
	flagV = (bool*)malloc(sizeof(bool));
	*flagV = true;
	cudaMalloc((void**)&d_flagV, sizeof(bool));
	flagT = (bool*)malloc(sizeof(bool));
	*flagT = true;
	cudaMalloc((void**)&d_flagT, sizeof(bool));

	QueryPerformanceCounter(&stop);
	exe_time = 1e3*(stop.QuadPart - start.QuadPart) / freq.QuadPart;
	SEtime = exe_time;
	printf("	Initialization done. Spent %f ms\n	Looping...\n", exe_time);
	QueryPerformanceCounter(&start);

	/*loop*/
	int loop = 0;
	while (*flagV || *flagT) {
		if (*flagT) {
			measureP << <BLOCK, THREAD >> >(nB, d_valG, d_valB, d_rowPtrB, d_colIndB, d_vol, d_theta,
				d_caledP, d_idx2row, d_idx2idx);
			differ << <BLOCK, THREAD >> >(m, d_P, d_caledP, d_dP);
			cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, m, nH, nnzH, &alpha, descrH, d_valH,
				d_rowPtrH, d_colIndH, d_dP, &beta, d_bP);
			status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzL, &alpha, descrL,
				d_valL, d_rowPtrL, d_colIndL, infoL, d_bP, d_temp, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);
			status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, &alpha, descrR, 
				d_valR, d_rowPtrR, d_colIndR, infoU, d_temp, d_dtheta, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);
			check << <1, 1024 >> >(d_dtheta, d_theta, nH, d_flagT, d_Pinv);
			cudaMemcpy(flagT, d_flagT, sizeof(bool), cudaMemcpyDeviceToHost);

		}
		if (*flagV) {
			measureQ << <BLOCK, THREAD >> >(nB, d_valG, d_valB, d_rowPtrB, d_colIndB, d_vol, 
				d_theta, d_caledQ, d_yc_half, d_idx2row, d_idx2idx);
			differ << <BLOCK, THREAD >> >(m, d_Q, d_caledQ, d_dQ);
			cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, m, nH, nnzH, &alpha, descrH, 
				d_valHQ, d_rowPtrH, d_colIndH, d_dQ, &beta, d_bQ);
			status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzL, &alpha, descrL,
				d_valLQ, d_rowPtrL, d_colIndL, infoLQ, d_bQ, d_temp, CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferQ);
			status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, &alpha, descrR,
				d_valRQ, d_rowPtrR, d_colIndR, infoUQ, d_temp, d_dvol, CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferQ);
			check <<<1, 1024 >>>(d_dvol, d_vol, nH, d_flagV, d_Pinv);
			cudaMemcpy(flagV, d_flagV, sizeof(bool), cudaMemcpyDeviceToHost);
		}
		loop++;
	}

	QueryPerformanceCounter(&stop);
	exe_time = 1e3*(stop.QuadPart - start.QuadPart) / freq.QuadPart;
	printf("	%d loops done. Spent %f ms\n",loop, exe_time);
	SEtime += exe_time;
	printf("SE done. Spent %f ms\n", SEtime);
	cudaMemcpy(vol, d_vol, nB * sizeof(my_type), cudaMemcpyDeviceToHost);
	cudaMemcpy(theta, d_theta, nB * sizeof(my_type), cudaMemcpyDeviceToHost);
	cudaMemcpy(P, d_caledP, m * sizeof(my_type), cudaMemcpyDeviceToHost);
	cudaMemcpy(Q, d_caledQ, m * sizeof(my_type), cudaMemcpyDeviceToHost);

	/*clear*/
	cudaFree(d_valL);
	cudaFree(d_valLQ);
	cudaFree(d_colIndL);
	cudaFree(d_rowPtrL);
	cudaFree(d_valR);
	cudaFree(d_valRQ);
	cudaFree(d_colIndR);
	cudaFree(d_rowPtrR);
	cudaFree(d_valB);
	cudaFree(d_colIndB);
	cudaFree(d_rowPtrB);
	cudaFree(d_valH);
	cudaFree(d_valHQ);
	cudaFree(d_colIndH);
	cudaFree(d_rowPtrH);
	cudaFree(d_valG);
	cudaFree(d_P);
	cudaFree(d_Q);
	cudaFree(d_vol);
	cudaFree(d_theta);
	cudaFree(d_caledP);
	cudaFree(d_caledQ);
	cudaFree(d_idx2row);
	cudaFree(d_idx2idx);
	cudaFree(d_J);
	cudaFree(d_dP);
	cudaFree(d_dQ);
	cudaFree(d_bP);
	cudaFree(d_bQ);
	cudaFree(d_temp);
	cudaFree(d_dvol);
	cudaFree(d_dtheta);
	cudaFree(d_yc_half);
	cudaFree(d_Pinv);
	cudaFree(d_flagT);
	cudaFree(d_flagV);
}