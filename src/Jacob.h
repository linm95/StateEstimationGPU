#pragma once
#include<stdlib.h>
#include "define.h"

void jacob(my_type *valBa, my_type *valB, int *rowPtrB, int *colIndB, my_type *valH, my_type *valHQ, int  *rowPtrH, int *colIndH, int n) {
	int nnzH = 0;
	int refer = n - 1;
	int m = rowPtrB[n];
	rowPtrH[0] = 0;
	for (int row = 0; row < n; row++) {
		int offset = rowPtrB[row];
		int length = rowPtrB[row + 1] - rowPtrB[row];
		for (int i = offset; i < offset + length; i++) {
			if (colIndB[i] == row) {
				for (int j = rowPtrB[row]; j < rowPtrB[row + 1]; j++) {
					if (colIndB[j] != refer) {
						valHQ[nnzH] = -valB[j];
						valH[nnzH] = -valBa[j];
						colIndH[nnzH] = colIndB[j];
						nnzH++;
					}
				}
			}
			else {
				if (row<colIndB[i]) {
					if (row != refer) {
						valH[nnzH] = valBa[i];
						valHQ[nnzH] = valB[i];
						colIndH[nnzH] = row;
						nnzH++;
					}
					if (colIndB[i] != refer) {
						valH[nnzH] = -valBa[i];
						valHQ[nnzH] = -valB[i];
						colIndH[nnzH] = colIndB[i];
						nnzH++;
					}
				}
				else {
					if (colIndB[i] != refer) {
						valH[nnzH] = -valBa[i];
						valHQ[nnzH] = -valB[i];
						colIndH[nnzH] = colIndB[i];
						nnzH++;
					}
					if (row != refer) {
						valH[nnzH] = +valBa[i];
						valHQ[nnzH] = +valB[i];
						colIndH[nnzH] = row;
						nnzH++;
					}
				}
			}
			rowPtrH[i + 1] = nnzH;
		}
	}
}

void GetJacob(my_type *valBa, my_type *valB, int*colIndB, int*rowPtrB, my_type **valH, my_type ** valHQ,int **rowPtrH, int **colIndH, int n) {


	int rowsOfH = rowPtrB[n];
	int itemsOfH = rowsOfH + 2 * (rowsOfH - n);

	(*rowPtrH) = (int*)malloc((rowsOfH + 2) * sizeof(int));
	(*colIndH) = (int*)malloc((itemsOfH + 1) * sizeof(int));
	(*valH) = (my_type*)malloc((itemsOfH + 1) * sizeof(my_type));
	(*valHQ) = (my_type*)malloc((itemsOfH + 1) * sizeof(my_type));

	jacob(valBa,valB, rowPtrB, colIndB, *valH,*valHQ, *rowPtrH, *colIndH, n);




}