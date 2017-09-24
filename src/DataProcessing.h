#pragma once
#include<stdlib.h>
#include "define.h"

struct point {
	int rowInd;
	int colInd;
	my_type G;
	my_type Br;
	my_type Ba;
	my_type P;
	my_type Q;
	my_type yc_half;
	my_type vol;
	my_type theta;
};

int comOfPoint(const void *p1, const void *p2) {
	if (((point*)p1)->rowInd != ((point*)p2)->rowInd)
		return ((point*)p1)->rowInd - ((point*)p2)->rowInd;
	return ((point*)p1)->colInd - ((point*)p2)->colInd;
}

void dataProcessing(point * cooB, int **idx2row, int **idx2idx, my_type **valB, my_type **valG, my_type **valBa, my_type **P, my_type **Q, my_type ** yc_half, int **rowPtrB, int **colIndB, int n, int numOfPoints, int *maxPerrRow) {
	qsort(cooB, numOfPoints, sizeof(point), comOfPoint);

	my_type *valB_buf, *valG_buf, *P_buf, *Q_buf, *yc_half_buf, *valBa_buf;
	int *colIndB_buf;
	valBa_buf = (my_type*)malloc(numOfPoints * sizeof(my_type));
	valB_buf = (my_type*)malloc(numOfPoints * sizeof(my_type));
	valG_buf = (my_type*)malloc(numOfPoints * sizeof(my_type));
	P_buf = (my_type*)malloc(numOfPoints * sizeof(my_type));
	Q_buf = (my_type*)malloc(numOfPoints * sizeof(my_type));
	colIndB_buf = (int*)malloc(numOfPoints * sizeof(int));
	yc_half_buf = (my_type*)malloc(numOfPoints * sizeof(my_type));
	(*rowPtrB) = (int*)malloc((n + 1) * sizeof(int));

	int cnt = 1, flag = 0, numOfRow = 1, row = 0;
	valBa_buf[0] = cooB[0].Ba;
	valB_buf[0] = cooB[0].Br;
	valG_buf[0] = cooB[0].G;
	P_buf[0] = cooB[0].P;
	Q_buf[0] = cooB[0].Q;
	yc_half_buf[0] = cooB[0].yc_half;
	colIndB_buf[0] = 0;
	(*rowPtrB)[0] = 0;
	int max = 0;
	for (int i = 1; i < numOfPoints; i++) {
		if (cooB[i].colInd == cooB[i - 1].colInd && cooB[i].rowInd == cooB[i - 1].rowInd) {
			valBa_buf[cnt - 1] += cooB[i].Ba;
			valB_buf[cnt - 1] += cooB[i].Br;
			valG_buf[cnt - 1] += cooB[i].G;
			P_buf[cnt - 1] += cooB[i].P;
			Q_buf[cnt - 1] += cooB[i].Q;
			yc_half_buf[cnt - 1] += cooB[i].yc_half;
		}
		else {
			if (flag == cooB[i].rowInd) {
				valBa_buf[cnt] = cooB[i].Ba;
				valB_buf[cnt] = cooB[i].Br;
				colIndB_buf[cnt] = cooB[i].colInd;
				valG_buf[cnt] = cooB[i].G;
				P_buf[cnt] = cooB[i].P;
				Q_buf[cnt] = cooB[i].Q;
				yc_half_buf[cnt] = cooB[i].yc_half;
				cnt++;
				numOfRow++;
			}
			else {
				valBa_buf[cnt] = cooB[i].Ba;
				valB_buf[cnt] = cooB[i].Br;
				colIndB_buf[cnt] = cooB[i].colInd;
				valG_buf[cnt] = cooB[i].G;
				P_buf[cnt] = cooB[i].P;
				Q_buf[cnt] = cooB[i].Q;
				yc_half_buf[cnt] = cooB[i].yc_half;
				(*rowPtrB)[row + 1] = (*rowPtrB)[row] + numOfRow;
				row++;
				if (numOfRow > max)
					max = numOfRow;
				numOfRow = 1;
				flag++;
				cnt++;
			}
		}
	}

	(*rowPtrB)[n] = cnt;
	(*idx2idx) = (int*)malloc(cnt * sizeof(int));
	(*idx2row) = (int*)malloc(cnt * sizeof(int));
	(*valB) = (my_type*)malloc(cnt * sizeof(my_type));
	(*valBa) = (my_type*)malloc(cnt * sizeof(my_type));
	(*yc_half) = (my_type*)malloc(cnt * sizeof(my_type));
	(*colIndB) = (int*)malloc(cnt * sizeof(int));
	(*valG) = (my_type*)malloc(cnt * sizeof(my_type));
	(*P) = (my_type*)malloc((cnt + 1) * sizeof(my_type));
	(*Q) = (my_type*)malloc((cnt + 1) * sizeof(my_type));
	*maxPerrRow = max;

	int non_diag_cnt = 0;
	for (int j = 0; j < n; j++) {
		for (int i = (*rowPtrB)[j]; i < (*rowPtrB)[j + 1]; i++) {
			(*valBa)[i] = valBa_buf[i];
			(*valB)[i] = valB_buf[i];
			(*colIndB)[i] = colIndB_buf[i];
			(*valG)[i] = valG_buf[i];
			(*P)[i] = P_buf[i];
			(*Q)[i] = Q_buf[i];
			(*yc_half)[i] = yc_half_buf[i];
			(*idx2row)[i] = j;
			if ((*colIndB)[i] == j)
				(*idx2idx)[j + cnt - n] = i;
			else {
				(*idx2idx)[non_diag_cnt] = i;
				non_diag_cnt++;
			}
		}
	}
	free(valBa_buf);
	free(valB_buf);
	free(valG_buf);
	free(P_buf);
	free(Q_buf);
	free(colIndB_buf);
	free(yc_half_buf);
}