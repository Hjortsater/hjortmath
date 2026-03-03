#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <cblas.h>


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

Matrix* matrix_create_from_buffer(size_t m, size_t n, const double* data){
    Matrix* M = matrix_create(m, n);
    if (!M){
        // Fatal Matrix creation error, return immediately.
        return NULL;
    }

    memcpy(M->data, data, m * n * sizeof(double));
    return M;
}

void matrix_free(Matrix* M){
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

    matrix_seed_random(time(NULL));
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
    double largest = M->data[0];
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
    double smallest = M->data[0];
    for (int i=0; i<size; i++){
        double ith = M->data[i];
        if (ith < smallest){
            smallest = ith;
        }
    }
    return smallest;
}

Matrix* matrix_add(const Matrix* restrict A,
                   const Matrix* restrict B,
                   int multithreaded){
    if(!A || !B || A->m != B->m || A->n != B->n){
        // Fatal matrix reference or data error(s), return immediately
        return NULL;
    }

    Matrix* C = matrix_create(A->m, A->n);
    if(!C){
        // Unable to create new matrix, return immediately
        return NULL;
    }

    size_t size = (size_t)A->m * A->n;

    /* Cache data pointers locally to help optimizer */
    double* restrict a = A->data;
    double* restrict b = B->data;
    double* restrict c = C->data;

#if defined(_OPENMP)
    if(multithreaded){
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < size; i++)
            c[i] = a[i] + b[i];
        return C;
    }
#endif

    // Fallback sequential
    for(size_t i = 0; i < size; i++)
        c[i] = a[i] + b[i];

    return C;
}

Matrix* matrix_add_inplace(const Matrix* restrict A,
                           const Matrix* restrict B,
                           Matrix* restrict C,
                           int multithreaded) {

    /* This function is identical to add except it writes to an allocated memory buffer, restric C.*/
    if(!A || !B || !C || A->m != B->m || A->n != B->n || C->m != A->m || C->n != A->n)
        return NULL;

    size_t size = (size_t)A->m * A->n;
    double* restrict a = A->data;
    double* restrict b = B->data;
    double* restrict c = C->data;

#if defined(_OPENMP)
    if(multithreaded){
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < size; i++)
            c[i] = a[i] + b[i];
        return C;
    }
#endif

    for(size_t i = 0; i < size; i++)
        c[i] = a[i] + b[i];

    return C;
}

Matrix* matrix_sub(const Matrix* restrict A,
                   const Matrix* restrict B,
                   int multithreaded){
    if(!A || !B || A->m != B->m || A->n != B->n){
        // Fatal matrix reference or data error(s), return immediately
        return NULL;
    }

    Matrix* C = matrix_create(A->m, A->n);
    if(!C){
        // Unable to create new matrix, return immediately
        return NULL;
    }

    size_t size = (size_t)A->m * A->n;

    /* Cache data pointers locally to help optimizer */
    double* restrict a = A->data;
    double* restrict b = B->data;
    double* restrict c = C->data;

#if defined(_OPENMP)
    if(multithreaded){
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < size; i++)
            c[i] = a[i] - b[i];
        return C;
    }
#endif

    // Fallback sequential
    for(size_t i = 0; i < size; i++)
        c[i] = a[i] - b[i];

    return C;
}

Matrix* matrix_sub_inplace(const Matrix* restrict A,
                           const Matrix* restrict B,
                           Matrix* restrict C,
                           int multithreaded) {

    /* This function is identical to sub except it writes to an allocated memory buffer, restric C.*/
    if(!A || !B || !C || A->m != B->m || A->n != B->n || C->m != A->m || C->n != A->n)
        return NULL;

    size_t size = (size_t)A->m * A->n;
    double* restrict a = A->data;
    double* restrict b = B->data;
    double* restrict c = C->data;

#if defined(_OPENMP)
    if(multithreaded){
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < size; i++)
            c[i] = a[i] - b[i];
        return C;
    }
#endif

    for(size_t i = 0; i < size; i++)
        c[i] = a[i] - b[i];

    return C;
}


Matrix* matrix_mul(Matrix* A, Matrix* B, int multithreaded){

    /* Borrowed matrix multiplication BLAS implementation. Difficult to compete with its speed!*/
    if (!A || !B || A->n != B->m){
        // Fatal Matrix reference or dimension error, return immediately.
        return NULL;
    }
    
    Matrix* C = matrix_create(A->m, B->n);
    if (!C) return NULL;
    
    #ifdef _OPENMP
    if (multithreaded) {
        openblas_set_num_threads(omp_get_max_threads());
    } else {
        openblas_set_num_threads(1);
    }
    #endif
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A->m, B->n, A->n,
                1.0, A->data, A->n,
                B->data, B->n,
                0.0, C->data, B->n);
    
    return C;
}

Matrix* matrix_mul_inplace(Matrix* A, Matrix* B, Matrix* C, int multithreaded){

    /* Borrowed matrix multiplication BLAS implementation. Difficult to compete with its speed!*/
    if (!A || !B || !C || A->n != B->m ||
        C->m != A->m || C->n != B->n){
        // Fatal Matrix reference or dimension error, return immediately.
        return NULL;
    }
    
    #ifdef _OPENMP
    if (multithreaded) {
        openblas_set_num_threads(omp_get_max_threads());
    } else {
        openblas_set_num_threads(1);
    }
    #endif
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A->m, B->n, A->n,
                1.0, A->data, A->n,
                B->data, B->n,
                0.0, C->data, B->n);
    
    return C;
}

inline int matrix_lu_decompose(Matrix* M, int multithreaded); // Needed for determinant calculation


double matrix_determinant(const Matrix* M, int multithreaded) {
    if (!M || M->m != M->n)
        return 0.0;

    int n = M->n;

    /* Make a copy because LU modifies the matrix */
    Matrix temp;
    temp.m = n;
    temp.n = n;
    temp.data = malloc(sizeof(double) * n * n);
    if (!temp.data)
        return 0.0;

    for (int i = 0; i < n * n; i++)
        temp.data[i] = M->data[i];

    int swap_count = matrix_lu_decompose(&temp, multithreaded);

    if (swap_count < 0) {
        free(temp.data);
        return 0.0;
    }

    double det = 1.0;

    // Product of diagonal elements of U
    for (int i = 0; i < n; i++)
        det *= temp.data[i*n + i];

    // Adjust sign based on row swaps
    if (swap_count % 2 != 0)
        det = -det;

    free(temp.data);
    return det;
}

double matrix_log_determinant(const Matrix* M, int* sign) {
    if (!M || M->m != M->n)
        return -INFINITY;

    int n = M->n;

    Matrix temp;
    temp.m = n;
    temp.n = n;
    temp.data = malloc(sizeof(double) * n * n);
    if (!temp.data)
        return -INFINITY;

    for (int i = 0; i < n*n; i++)
        temp.data[i] = M->data[i];

    int swap_count = matrix_lu_decompose(&temp, 0);
    if (swap_count < 0) {
        free(temp.data);
        return -INFINITY;
    }

    double log_det = 0.0;
    int s = (swap_count % 2 == 0) ? 1 : -1;

    for (int i = 0; i < n; i++) {
        double diag = temp.data[i*n + i];
        if (diag < 0) {
            s *= -1;
            diag = -diag;
        }
        log_det += log(diag);
    }

    free(temp.data);

    if (sign)
        *sign = s;

    return log_det;
}

#define BLOCK_SIZE 64
#define EPS 1e-12

inline int matrix_lu_decompose(Matrix* M, int multithreaded) {
    if (!M || M->m != M->n)
        return -1;

    int n = M->n;
    double* a = M->data;
    int swap_count = 0;

    for (int k = 0; k < n; k += BLOCK_SIZE) {

        int bk = (k + BLOCK_SIZE > n) ? (n - k) : BLOCK_SIZE;

        /* 1. Factor diagonal block */
        for (int kk = 0; kk < bk; kk++) {

            int col = k + kk;

            // Pivoting
            int pivot = col;
            double max = fabs(a[col*n + col]);

            for (int i = col + 1; i < n; i++) {
                double val = fabs(a[i*n + col]);
                if (val > max) {
                    max = val;
                    pivot = i;
                }
            }

            if (max < EPS)
                return -1;

            if (pivot != col) {
                for (int j = 0; j < n; j++) {
                    double tmp = a[col*n + j];
                    a[col*n + j] = a[pivot*n + j];
                    a[pivot*n + j] = tmp;
                }
                swap_count++;
            }

            // Eliminate inside panel
            for (int i = col + 1; i < n; i++) {
                a[i*n + col] /= a[col*n + col];
                double factor = a[i*n + col];
                for (int j = col + 1; j < k + bk; j++)
                    a[i*n + j] -= factor * a[col*n + j];
            }
        }

        if (k + bk >= n)
            continue;

        /* Solve U12 block
           L11 * U12 = A12
           Use BLAS triangular solve */
        cblas_dtrsm(
            CblasRowMajor,
            CblasLeft,
            CblasLower,
            CblasNoTrans,
            CblasUnit,
            bk,
            n - k - bk,
            1.0,
            &a[k*n + k], n,
            &a[k*n + k + bk], n
        );

        /* Update trailing matrix
           A22 -= L21 * U12
           Use GEMM (parallelized I believe) */

        int rows = n - k - bk;
        int cols = n - k - bk;

        if (multithreaded) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < rows; i += BLOCK_SIZE) {
                int ib = (i + BLOCK_SIZE > rows) ? (rows - i) : BLOCK_SIZE;

                cblas_dgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    ib,
                    cols,
                    bk,
                    -1.0,
                    &a[(k + bk + i)*n + k], n,
                    &a[k*n + k + bk], n,
                    1.0,
                    &a[(k + bk + i)*n + k + bk], n
                );
            }
        } else {
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                rows,
                cols,
                bk,
                -1.0,
                &a[(k + bk)*n + k], n,
                &a[k*n + k + bk], n,
                1.0,
                &a[(k + bk)*n + k + bk], n
            );
        }
    }

    return swap_count;
}