#include <stdlib.h>
#include <omp.h>
#include <time.h>

/*

This is the C-backend for the Python linear algebra class. It boasts:
    * A Matrix struct with a defined macro


Written by Erik Hjortsäter February 27th 2026.

*/

typedef struct {
    int m;
    int n;
    double* data;
} Matrix;

#define MAT(M, i, j) ((M)->data[(i) * (M)->n + (j)])

Matrix* matrix_create(int m, int n){
    Matrix* M = (Matrix*) malloc(sizeof(Matrix));
    if (!M)
        // Fatal malloc error occured, return immediately.
        return NULL;
    M->m = m;
    M->n = n;
    M->data = (double*) calloc(m*n,sizeof(double));
    if (!M->data){
        // Fatal calloc error occured, free & return immediately.
        free(M);
        return NULL;
    }
    return M;
}

void matrix_free(Matrix* M) {
    // Free specific matrix reference immediately.
    if (!M) return;
    if (M->data) free(M->data);
    free(M);
}

int matrix_rows(Matrix* M){
    if (!M) return 0;
    return M->m;
}

int matrix_cols(Matrix* M){
    if (!M) return 0;
    return M->n;
}

void matrix_set(Matrix* M, int i, int j, double value){
    if (!M || !M->data){
        // Fatal Matrix reference or data error, return immediately
        return;
    }
    MAT(M,i,j) = value;
}

double matrix_get(Matrix* M, int i, int j){
    if (!M || !M->data){
        // Fatal Matrix reference or data error, return immediately
        return 0.0;
    }
    return MAT(M,i,j);
}

void matrix_fill(Matrix* M, double value){
    if (!M || !M->data){
        // Fatal Matrix reference or data error, return immediately
        return;
    }
    int size = M->m * M->n;
    for (int i=0; i<size; i++){
        M->data[i]=value;
    }
    return;
}

void matrix_seed_random(unsigned int seed){
    srand(seed);
}

void matrix_fill_random(Matrix* M, double min, double max){
    if (!M || !M->data){
        // Fatal Matrix reference error, return immediately.
        return;
    }

    double range = max - min;
    for (int i = 0; i < M->m * M->n; i++){
        double r = (double)rand() / (double)RAND_MAX; // 0..1
        M->data[i] = min + r * range;
    }
}

double matrix_get_max(Matrix* M){
    if (!M || !M->data){
        // Fatal Matrix reference or data error, return immediately
    }
    int size = M->m * M-> n;
    double largest;
    for (int i=0; i<size; i++){
        double ith = M->data[i];
        if (ith > largest){
            largest = ith;
        }
    }
    return largest;
}

double matrix_get_min(Matrix* M){
    if (!M || !M->data){
        // Fatal Matrix reference or data error, return immediately
    }
    int size = M->m * M-> n;
    double smallest;
    for (int i=0; i<size; i++){
        double ith = M->data[i];
        if (ith < smallest){
            smallest = ith;
        }
    }
    return smallest;
}

Matrix* matrix_add(Matrix* A, Matrix* B, int multithreaded){
    if(!A || !B || A->m != B->m || A->n != B->n){
        // Fatal matrix reference or data error(s), return immediately
        return NULL;
    }
    Matrix* C = matrix_create(A->m, A->n);
    if(!C){
        // Unable to create new matrix, return immediately
        return NULL;
    }
    int size = A->m * A->n;

#if defined(_OPENMP)
    if(multithreaded){
        #pragma omp parallel for
        for(int i=0;i<size;i++)
            C->data[i] = A->data[i] + B->data[i];
        return C;
    }
#endif

    // Fallback sequential
    for(int i=0;i<size;i++)
        C->data[i] = A->data[i] + B->data[i];

    return C;
}