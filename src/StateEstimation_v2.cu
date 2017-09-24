#include<windows.h>
#include "jacob.cu";
#include "measure.cu";
#include "amd.h";
#include "LU_solve.cu";
#include "givens.cu";

#define BLOCK 80
#define THREAD 512



__global__ void check(double *d_dv,double *d_v, int n,bool *flag,int* Pinv) {
	int index = threadIdx.x ;
	int stride = blockDim.x;
	int block = blockIdx.x;
	__shared__ bool l_flag;
	if (index == 0) l_flag = false;
	__syncthreads();
	for (int i = index; i < n; i += stride) {
		double dv = d_dv[Pinv[i]];
		d_v[i] += dv;
		if (!l_flag && fabs(dv) > 1e-4)
			l_flag = true;
	}
	__syncthreads();
	if (index == 0 ) {
		*flag = l_flag;
	}
}

__global__ void getJ(double *d_dP, double *d_dQ, int m, double *J) {
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	int stride = gridDim.x*blockDim.x;
	int mm = m;
	__shared__ double lo_J;
	if (index == 0) {
		lo_J = 0;
	}
	__syncthreads();
	for (int i = index; i < 2*mm; i+=stride) {
		if (i < mm) {
			double temp = d_dP[i];
			temp = temp*temp;
			_dAtomicAdd(&lo_J, temp);
		}
		else {
			double temp = d_dQ[i - mm];
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

int main() {

	LARGE_INTEGER freq, start_all, stop_all, start_single, stop_single;
	double exe_time;
	QueryPerformanceFrequency(&freq);


	QueryPerformanceCounter(&start_all);
	/*Step 1: read the data, including: mat B, mat G, vector P, vector Q, vector vol, vector theta*/

	QueryPerformanceCounter(&start_single);

	struct point *data;
	FILE *mat;
	mat = fopen("case/case9241.txt", "r");
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

	QueryPerformanceCounter(&stop_single);
	exe_time = 1e3*(stop_single.QuadPart - start_single.QuadPart) / freq.QuadPart;
	printf("reading data takes %f ms\n", exe_time);

	QueryPerformanceCounter(&start_single);

	//get admittance mat B, power flow vector P,Q, yc
	int *idx2row, *idx2idx;
	double *valB,*valG,*P,*Q,*yc_half,*valBa;
	int *colIndB, *rowPtrB, maxPerRowB;
	cooToCsr(data, &idx2row,&idx2idx,&valB, &valG, &valBa, &P, &Q, &yc_half, &rowPtrB, &colIndB, nB, numOfData, &maxPerRowB);
	if (maxPerRowB > MaxItemsPerRowB) {
		printf("MaxItemPerRowB should be more than %d!", maxPerRowB);
		return;
	}

	//set the initial value
	double *vol, *theta;
	vol = (double*)malloc(nB * sizeof(double));
	theta = (double*)calloc(nB, sizeof(double));
	for (int i = 0; i < nB; i++) {
			vol[i] =data[numOfData-1].vol ;
			theta[i] = 0;
	}
	//Get the jacob mat H
	double *valH, *valHQ;
	int *rowPtrH, *colIndH,*rowPtrHQ,*colIndHQ;
	GetJacob(valBa, colIndB, rowPtrB, &valH, &rowPtrH, &colIndH, nB);
	GetJacob(valB, colIndB, rowPtrB, &valHQ, &rowPtrHQ, &colIndHQ, nB);

	QueryPerformanceCounter(&stop_single);
	exe_time = 1e3*(stop_single.QuadPart - start_single.QuadPart) / freq.QuadPart;
	printf("getting jacob takes %f ms\n", exe_time);
	
	
	
	QueryPerformanceCounter(&start_single);

	//get reoder vector Pvec
	double *valR, *valRQ;
	int *colIndR, *rowPtrR, *colIndRQ, *rowPtrRQ;
	int nH = nB - 1;
	int m = rowPtrB[nB];
	int *Pinv;
	//amd2(valH, valHQ, rowPtrH, colIndH, &Pinv, &valR, &colIndR, &rowPtrR, &valRQ, &colIndRQ, &rowPtrRQ, m, nH);
	//getReorderdH(valH, rowPtrH, colIndH, Pinv, m, nH);
	amd(valH, rowPtrH, colIndH, &Pinv, m, nH);
	getReorderdH(valHQ, rowPtrHQ, colIndHQ, Pinv, m, nH);


	QueryPerformanceCounter(&stop_single);
	exe_time = 1e3*(stop_single.QuadPart - start_single.QuadPart) / freq.QuadPart;
	printf("amd takes %f ms(including sync)\n", exe_time);

	QueryPerformanceCounter(&start_single);

	//get mat R, RQ
	FILE *out;
	out = fopen("e:/9241_H", "w");
	fprintf(out, "%d %d %d\n", m, nH, rowPtrH[m]);
	for (int i = 0; i < m; i++) {
		for (int j = rowPtrH[i]; j < rowPtrH[i + 1]; j++)
			fprintf(out, "%d %d %f\n", i, colIndH[j], valH[j]);
	}
	
	givens_cpu(valH, rowPtrH, colIndH, &valR, &rowPtrR, &colIndR, m, nH);
	givens_cpu(valHQ, rowPtrHQ, colIndHQ, &valRQ, &rowPtrRQ, &colIndRQ, m, nH);
	
	
	out = fopen("e:/9241_R", "w");
	fprintf(out, "%d %d %d\n", m, nH, rowPtrH[m]);
	for (int i = 0; i < nH; i++) {
	for (int j = rowPtrR[i]; j < rowPtrR[i + 1]; j++)
	fprintf(out, "%d %d %f\n", i, colIndR[j], valR[j]);
	}
	return;

//	givens_scheduler(valH, rowPtrH, colIndH, &valR, &rowPtrR, &colIndR, m, nH);
	//givens_scheduler(valHQ, rowPtrHQ, colIndHQ, &valRQ, &rowPtrRQ, &colIndRQ, m, nH);


	QueryPerformanceCounter(&stop_single);
	exe_time = 1e3*(stop_single.QuadPart - start_single.QuadPart) / freq.QuadPart;
	printf("givensRotation takes %f ms\n", exe_time);

	//target J
	double *J;
	J = (double*)calloc(1, sizeof(double));

	/*step3: loop*/
	cusparseStatus_t status;	

	QueryPerformanceCounter(&start_single);

	/*copy R mat*/
	int nnzR = rowPtrR[nH];
	double *d_valR, *d_valRQ;
	int *d_colIndR, *d_rowPtrR, *d_colIndRQ, *d_rowPtrRQ;
	cudaMalloc((void**)&d_valR, nnzR * sizeof(double));
	cudaMalloc((void**)&d_colIndR, nnzR * sizeof(int));
	cudaMalloc((void**)&d_rowPtrR, (nH + 1) * sizeof(int));
	cudaMemcpy(d_valR, valR, nnzR * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIndR, colIndR, nnzR * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowPtrR, rowPtrR, (nH + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_valRQ, nnzR * sizeof(double));
	cudaMalloc((void**)&d_colIndRQ, nnzR * sizeof(int));
	cudaMalloc((void**)&d_rowPtrRQ, (nH + 1) * sizeof(int));
	cudaMemcpy(d_valRQ, valRQ, nnzR * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIndRQ, colIndRQ, nnzR * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowPtrRQ, rowPtrRQ, (nH + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cusparseMatDescr_t descrR;
	cusparseCreateMatDescr(&descrR);
	cusparseSetMatType(descrR, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatFillMode(descrR, CUSPARSE_FILL_MODE_UPPER);
	cusparseSetMatIndexBase(descrR, CUSPARSE_INDEX_BASE_ZERO);
	cusparseSetMatDiagType(descrR, CUSPARSE_DIAG_TYPE_NON_UNIT);
	
	/*copy H mat*/
	int nnzH = rowPtrH[m];
	double *d_valH,*d_valHQ;
	int *d_colIndH, *d_rowPtrH, *d_colIndHQ, *d_rowPtrHQ;
	cudaMalloc((void**)&d_valH, nnzH * sizeof(double));
	cudaMalloc((void**)&d_colIndH, nnzH * sizeof(int));
	cudaMalloc((void**)&d_rowPtrH, (m + 1) * sizeof(int));
	cudaMemcpy(d_valH, valH, nnzH * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIndH, colIndH, nnzH * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowPtrH, rowPtrH, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_valHQ, nnzH * sizeof(double));
	cudaMalloc((void**)&d_colIndHQ, nnzH * sizeof(int));
	cudaMalloc((void**)&d_rowPtrHQ, (m + 1) * sizeof(int));
	cudaMemcpy(d_valHQ, valHQ, nnzH * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIndHQ, colIndHQ, nnzH * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowPtrHQ, rowPtrHQ, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cusparseMatDescr_t descrH;
	cusparseCreateMatDescr(&descrH);
	
	/*copy B mat*/
	int nnzB = rowPtrB[nB];
	double *d_valB,*d_valG;
	int *d_colIndB, *d_rowPtrB;
	cudaMalloc((void**)&d_valG, nnzB * sizeof(double));
	cudaMalloc((void**)&d_valB, nnzB * sizeof(double));
	cudaMalloc((void**)&d_rowPtrB, (nB + 1) * sizeof(int));
	cudaMalloc((void**)&d_colIndB, nnzB * sizeof(int));
	cudaMemcpy(d_valG, valG, nnzB * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_valB, valB, nnzB * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rowPtrB, rowPtrB, (nB + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_colIndB, colIndB, nnzB * sizeof(int), cudaMemcpyHostToDevice);
	
	/*copy P,Q,vol,theta,yc*/
	double *d_P, *d_Q, *d_vol, *d_theta,*d_yc_half;
	cudaMalloc((void**)&d_P, m * sizeof(double));
	cudaMalloc((void**)&d_Q, m * sizeof(double));
	cudaMemcpy(d_P, P, m * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Q, Q, m * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_vol, nB * sizeof(double));
	cudaMalloc((void**)&d_theta, nB * sizeof(double));
	cudaMemcpy(d_vol, vol, nB * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_theta, theta, nB * sizeof(double), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_yc_half, m * sizeof(double));
	cudaMemcpy(d_yc_half , yc_half, m * sizeof(double), cudaMemcpyHostToDevice);
	
	/*prepare for calculating measurement function*/
	double *d_caledP, *d_caledQ;
	int *d_idx2row,*d_idx2idx;
	cudaMalloc((void**)&d_idx2row, m * sizeof(int));
	cudaMemcpy(d_idx2row, idx2row, m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_idx2idx, m * sizeof(int));
	cudaMemcpy(d_idx2idx, idx2idx, m * sizeof(int), cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_caledP, m * sizeof(double));
	cudaMalloc((void**)&d_caledQ, m * sizeof(double));


	/*prepare for calculating difference vector*/
	double *d_dP, *d_dQ,*d_J;
	cudaMalloc((void**)&d_J, sizeof(double));	
	cudaMalloc((void**)&d_dP, m * sizeof(double));
	cudaMalloc((void**)&d_dQ, m * sizeof(double));

	
	/*prepare for calculating the b vector*/
	double alpha = 1, beta = 0;
	cusparseHandle_t handle;
	cusparseCreate(&handle);
	double *d_bP, *d_bQ;
	cudaMalloc((void**)&d_bP, nH * sizeof(double));
	cudaMalloc((void**)&d_bQ, nH * sizeof(double));		

	/*prepare for solving LU equation*/
	/*buffersize analysis*/
	int buffersizeL, buffersizeU, buffersizeLQ, buffersizeUQ;
	csrsv2Info_t infoL, infoU, infoLQ, infoUQ;
	cusparseCreateCsrsv2Info(&infoL);
	cusparseCreateCsrsv2Info(&infoU);
	cusparseCreateCsrsv2Info(&infoLQ);
	cusparseCreateCsrsv2Info(&infoUQ);
	status = cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, nH, nnzR, descrR, d_valR, d_rowPtrR, d_colIndR, infoL, &buffersizeL);
	status = cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, descrR, d_valR, d_rowPtrR, d_colIndR, infoU, &buffersizeU);
	status = cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE, nH, nnzR, descrR, d_valRQ, d_rowPtrRQ, d_colIndRQ, infoLQ, &buffersizeLQ);
	status = cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, descrR, d_valRQ, d_rowPtrRQ, d_colIndRQ, infoUQ, &buffersizeUQ);
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
	double *d_temp,*d_dvol,*d_dtheta;
	cudaMalloc((void**)&d_temp, nH * sizeof(double));
	cudaMalloc((void**)&d_dvol, nH * sizeof(double));
	cudaMalloc((void**)&d_dtheta, nH * sizeof(double));
	/*solving analysis*/
	status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE, nH, nnzR, descrR, d_valR, d_rowPtrR, d_colIndR, infoL, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);
	status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, descrR, d_valR, d_rowPtrR, d_colIndR, infoU, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);
	status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE, nH, nnzR, descrR, d_valRQ, d_rowPtrRQ, d_colIndRQ, infoLQ, CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferQ);
	status = cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, descrR, d_valRQ, d_rowPtrRQ, d_colIndRQ, infoUQ, CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferQ);
	
	/*prepare for checking the result*/
	int *d_Pinv;
	cudaMalloc((void**)&d_Pinv, nB * sizeof(int));
	cudaMemcpy(d_Pinv, Pinv, nB * sizeof(int), cudaMemcpyHostToDevice);
	//flag for loop
	bool *flagV,*d_flagV,*flagT,*d_flagT,*d_flagBuf;
	flagV = (bool*)malloc(sizeof(bool));
	*flagV = true;
	cudaMalloc((void**)&d_flagV, sizeof(bool));
	cudaMalloc((void**)&d_flagBuf, BLOCK * sizeof(bool));
	flagT = (bool*)malloc(sizeof(bool));
	*flagT = true;
	cudaMalloc((void**)&d_flagT, sizeof(bool));

	QueryPerformanceCounter(&stop_single);
	exe_time = 1e3*(stop_single.QuadPart - start_single.QuadPart) / freq.QuadPart;
	printf("data copy and analysis takes %f ms\n", exe_time);


	QueryPerformanceCounter(&stop_all);
	exe_time = 1e3*(stop_all.QuadPart - start_all.QuadPart) / freq.QuadPart;
	printf("preparing takes %f ms\n", exe_time);
	
	QueryPerformanceCounter(&start_all);
	double recP[6] = { 0 }, recQ[6] = { 0 };
	int loopP = 0, loopQ = 0;
	/*loop*/
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float time;
	int loop = 0;
	
	while (*flagV || *flagT) {
	//	printf("\n%d loop:--------------\n");
		if (*flagT) {
			cudaEventRecord(start, 0);

			measureP_gpu_v2 <<<BLOCK, THREAD >>>(nB, d_valG, d_valB, d_rowPtrB, d_colIndB, d_vol, d_theta, d_caledP,d_idx2row,d_idx2idx);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recP[0] += time;
			cudaEventRecord(start, 0);

			differ << <BLOCK, THREAD >> >(m, d_P, d_caledP, d_dP);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recP[1] += time;
			cudaEventRecord(start, 0);

			cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, m, nH, nnzH, &alpha, descrH, d_valH, d_rowPtrH, d_colIndH, d_dP, &beta, d_bP);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recP[2] += time;
			cudaEventRecord(start, 0);

			status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, nH, nnzR, &alpha, descrR, d_valR, d_rowPtrR, d_colIndR, infoL, d_bP, d_temp, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recP[3] += time;
			cudaEventRecord(start, 0);

			status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, &alpha, descrR, d_valR, d_rowPtrR, d_colIndR, infoU, d_temp, d_dtheta, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buffer);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recP[4] += time;
			cudaEventRecord(start, 0);

			check<<<1, 1024>>>(d_dtheta, d_theta, nH, d_flagT, d_Pinv);
			cudaMemcpy(flagT, d_flagT, sizeof(bool), cudaMemcpyDeviceToHost);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recP[5] += time;

			loopP++;
		}
		if (*flagV) {
			cudaEventRecord(start, 0);

			measureQ_gpu_v2 << <BLOCK, THREAD >> >(nB, d_valG, d_valB, d_rowPtrB, d_colIndB, d_vol, d_theta, d_caledQ, d_yc_half,d_idx2row,d_idx2idx);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recQ[0] += time;
			cudaEventRecord(start, 0);

			differ << <BLOCK, THREAD >> >(m, d_Q, d_caledQ, d_dQ);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recQ[1] += time;
			cudaEventRecord(start, 0);

			cusparseDcsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, m, nH, nnzH, &alpha, descrH, d_valHQ, d_rowPtrHQ, d_colIndHQ, d_dQ, &beta, d_bQ);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recQ[2] += time;
			cudaEventRecord(start, 0);

			status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_TRANSPOSE, nH, nnzR, &alpha, descrR, d_valRQ, d_rowPtrRQ, d_colIndRQ, infoLQ, d_bQ, d_temp, CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferQ);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recQ[3] += time;
			cudaEventRecord(start, 0);

			status = cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, nH, nnzR, &alpha, descrR, d_valRQ, d_rowPtrRQ, d_colIndRQ, infoUQ, d_temp, d_dvol, CUSPARSE_SOLVE_POLICY_NO_LEVEL, bufferQ);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recQ[4] += time;
			cudaEventRecord(start, 0);

			check<<<1, 1024 >>>(d_dvol, d_vol, nH, d_flagV, d_Pinv);
			
			cudaMemcpy(flagV, d_flagV, sizeof(bool), cudaMemcpyDeviceToHost);

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(start);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&time, start, stop);
			recQ[5] += time;
			loopQ++;
		}
		getJ<<<1, 1024 >>>(d_dP, d_dQ, m, d_J);
		cudaMemcpy(J, d_J, sizeof(double), cudaMemcpyDeviceToHost);
		loop++;
		printf("第%d次迭代的J函数：%f\n", loop, *J);
	}

	QueryPerformanceCounter(&stop_all);
	exe_time = 1e3*(stop_all.QuadPart - start_all.QuadPart) / freq.QuadPart;
	printf("%d loops take %f ms\n", loop, exe_time);
	printf("%d times measuring for P take average %f ms\n", loopP, recP[0] / loopP);
	printf("%d times getting diff vector for P take average %f ms\n", loopP, recP[1] / loopP);
	printf("%d times getting b vec for P take average %f ms\n", loopP, recP[2] / loopP);
	printf("%d times solving L for P take average %f ms\n", loopP, recP[3] / loopP);
	printf("%d times solving U for P take average %f ms\n", loopP, recP[4] / loopP);
	printf("%d times update an check for P take average %f ms\n", loopP, recP[5] / loopP);
	printf("%d times measuring for Q take average %f ms\n", loopQ, recQ[0] / loopQ);
	printf("%d times getting diff vector for Q take average %f ms\n", loopQ, recP[1] / loopQ);
	printf("%d times getting b vec for Q take average %f ms\n", loopQ, recQ[2] / loopQ);
	printf("%d times solving L for Q take average %f ms\n", loopQ, recQ[3] / loopQ);
	printf("%d times solving U for Q take average %f ms\n", loopQ, recQ[4] / loopQ);
	printf("%d times update an check for Q take average %f ms\n", loopQ, recQ[5] / loopQ);
	
	cudaMemcpy(vol, d_vol, nB * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(theta, d_theta, nB * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(P, d_caledP, m * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(Q, d_caledQ, m * sizeof(double), cudaMemcpyDeviceToHost);

	/*clear*/
	cudaFree(d_valR);
	cudaFree(d_colIndR);
	cudaFree(d_rowPtrR);
	cudaFree(d_valB);
	cudaFree(d_colIndB);
	cudaFree(d_rowPtrB);
	cudaFree(d_valH);
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
	cudaFree(d_J);
	cudaFree(d_dP);
	cudaFree(d_dQ);
	cudaFree(d_bP);
	cudaFree(d_bQ);
	cudaFree(d_temp);
	cudaFree(d_dvol);
	cudaFree(d_dtheta);
	cudaFree(d_Pinv);
	cudaFree(d_flagT);
	cudaFree(d_flagV);
}