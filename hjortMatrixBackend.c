#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <cblas.h>
#include <lapacke.h>


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
    if (!M){
        // Fatal malloc error occured, return immediately.
        return NULL;
    }
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

void matrix_copy(const Matrix* src, Matrix* dst){
    // Matrix copy function
    if(!src || !dst || src->m != dst->m || src->n != dst->n || !src->data || !dst->data){
        return;
    }
    size_t size = (size_t)(src->m) * (size_t)(src->n);
    memcpy(dst->data, src->data, size * sizeof(double));
}

Matrix* matrix_clone(const Matrix* src) {
    if (!src) return NULL;
    Matrix* dst = (Matrix*)malloc(sizeof(Matrix));
    if (!dst) return NULL;

    dst->m = src->m;
    dst->n = src->n;

    dst->data = (double*)malloc(src->m * src->n * sizeof(double));
    if (!dst->data) {
        free(dst);
        return NULL;
    }

    memcpy(dst->data, src->data, src->m * src->n * sizeof(double));
    return dst;
}

Matrix* matrix_create_from_buffer(size_t m, size_t n, const double* data){
    Matrix* M = matrix_create((int)m, (int)n);
    if (!M){
        // Fatal Matrix creation error, return immediately.
        return NULL;
    }

    memcpy(M->data, data, m * n * sizeof(double));
    return M;
}

void matrix_free(Matrix* M){
    // Free specific matrix reference immediately.
    if (!M){
        return;
    }
    if (M->data){
        free(M->data);
    }
    free(M);
}

int matrix_rows(Matrix* M){
    if (!M){
        return 0;
    }
    return M->m;
}

int matrix_cols(Matrix* M){
    if (!M){
        return 0;
    }
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

Matrix* matrix_zero(int m, int n){
    if (m <= 0 || n <= 0){
        // Fatal dimension error, return immediately
        return NULL;
    }
    Matrix* M = (Matrix*) malloc(sizeof(Matrix));
    if (!M){
        // Fatal Matrix memory allocation error, return immediately 
        return NULL;
    }
    M->m = m; M->n = n; 
    M->data = (double*) calloc(m*n, sizeof(double));
    if (!M->data){
        // Fatal Matrix memory allocation error, return immediately
        free(M);
        return NULL;
    }
    return M;
}

Matrix* matrix_identity(int m){
    if (m <= 0){
        // Fatal dimension error, return immediately
        return NULL;
    }

    Matrix* M = (Matrix*) malloc(sizeof(Matrix));
    if (!M){
        // Fatal Matrix memory allocation error, return immediately
        return NULL;
    }

    M->m = m;
    M->n = m;

    M->data = (double*) malloc(sizeof(double) * m * m);
    if (!M->data){
        // Fatal data allocation error, free matrix struct and return immediately
        free(M);
        return NULL;
    }

    for (int i = 0; i < m*m; i++){
        M->data[i] = (i % m == i / m) ? 1.0 : 0.0;
    }

    return M;
}

Matrix* matrix_random(int m, int n, double min, double max) {
    if (m <= 0 || n <= 0) return NULL;
    
    Matrix* M = matrix_create(m, n);
    if (!M) return NULL;
    
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned int seed = tv.tv_sec * 1000000 + tv.tv_usec;
    srand(seed);
    
    double range = max - min;
    for (int i = 0; i < m * n; i++) {
        double r = (double)rand() / (double)RAND_MAX;
        M->data[i] = min + r * range;
    }
    
    return M;
}

double matrix_get_max(Matrix* M){
    if (!M || !M->data){
        // Fatal Matrix reference or data error, return immediately
        return 0.0;
    }
    int size = M->m * M-> n;
    double largest = M->data[0];
    for (int i=1; i<size; i++){
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
        return 0.0;
    }
    int size = M->m * M-> n;
    double smallest = M->data[0];
    for (int i=1; i<size; i++){
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

    for(size_t i = 0; i < size; i++){
        c[i] = a[i] + b[i];
    }

    return C;
}

Matrix* matrix_add_inplace(const Matrix* restrict A,
                           const Matrix* restrict B,
                           Matrix* restrict C,
                           int multithreaded){

    /* This function is identical to add except it writes to an allocated memory buffer, restric C.*/
    if(!A || !B || !C || A->m != B->m || A->n != B->n || C->m != A->m || C->n != A->n){
        return NULL;
    }

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

    for(size_t i = 0; i < size; i++){
        c[i] = a[i] + b[i];
    }

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

    for(size_t i = 0; i < size; i++){
        c[i] = a[i] - b[i];
    }

    return C;
}

Matrix* matrix_sub_inplace(const Matrix* restrict A,
                           const Matrix* restrict B,
                           Matrix* restrict C,
                           int multithreaded){

    /* This function is identical to sub except it writes to an allocated memory buffer, restric C.*/
    if(!A || !B || !C || A->m != B->m || A->n != B->n || C->m != A->m || C->n != A->n){
        return NULL;
    }

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

    for(size_t i = 0; i < size; i++){
        c[i] = a[i] - b[i];
    }

    return C;
}

Matrix* matrix_mul(Matrix* A, Matrix* B, int multithreaded){

    /* Borrowed matrix multiplication BLAS implementation. Difficult to compete with its speed!*/
    if (!A || !B || A->n != B->m){
        // Fatal Matrix reference or dimension error, return immediately.
        return NULL;
    }
    
    Matrix* C = matrix_create(A->m, B->n);
    if (!C){
        return NULL;
    }
    
    #ifdef _OPENMP
    if (multithreaded){
        goto skip_threading_mul;
    }
    #endif

    skip_threading_mul:
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
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                A->m, B->n, A->n,
                1.0, A->data, A->n,
                B->data, B->n,
                0.0, C->data, B->n);
    
    return C;
}

Matrix* matrix_scalar_mul(Matrix* A, double scalar, int multithreaded) {
    if (!A) return NULL;
    
    Matrix* C = matrix_create(A->m, A->n);
    if (!C) return NULL;
    
    #ifdef _OPENMP
    #pragma omp parallel for if(multithreaded)
    #endif
    for (int i = 0; i < A->m * A->n; i++) {
        C->data[i] = A->data[i] * scalar;
    }
    
    return C;
}

int matrix_scalar_mul_inplace(Matrix* A, Matrix* C, double scalar, int multithreaded) {
    if (!A || !C) return 0;
    if (A->m != C->m || A->n != C->n) return 0;
    
    #ifdef _OPENMP
    #pragma omp parallel for if(multithreaded)
    #endif
    for (int i = 0; i < A->m * A->n; i++) {
        C->data[i] = A->data[i] * scalar;
    }
    
    return 1;
}



inline int matrix_lu_decompose(Matrix* M, int multithreaded); // Needed for determinant calculation

Matrix* matrix_inverse(const Matrix* A, int multithreaded){
    if (!A || A->m != A->n){
        return NULL;
    }
    int n = A->n;

    Matrix* inv = matrix_create_from_buffer((size_t)n, (size_t)n, A->data);
    if (!inv){
        return NULL;
    }

    int* ipiv = (int*)malloc(n * sizeof(int));
    if (!ipiv){
        matrix_free(inv);
        return NULL;
    }

    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, inv->data, n, ipiv);
    
    if (info == 0){
        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, inv->data, n, ipiv);
    }

    free(ipiv);

    if (info != 0){
        matrix_free(inv);
        return NULL;
    }

    return inv;
}

int matrix_inverse_inplace(const Matrix* A, Matrix* C, int multithreaded) {
    if (!A || !C || A->m != A->n || C->m != C->n || A->m != C->m) {
        return 0;
    }
    
    int n = A->n;
    
    // Copy input data into C to perform operations in-place
    memcpy(C->data, A->data, (size_t)(n * n * sizeof(double)));

    int* ipiv = (int*)malloc((size_t)n * sizeof(int));
    if (!ipiv) {
        return 0;
    }

    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, C->data, n, ipiv);
    
    if (info == 0) {
        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, C->data, n, ipiv);
    }

    free(ipiv);

    return (info == 0);
}



Matrix* matrix_solve(const Matrix* A, const Matrix* B) {
    if (!A || !B || B->m != B->n || B->n != A->n) return NULL;

    int n = B->n;
    int nrhs = A->m;

    Matrix* B_copy = matrix_create_from_buffer(n, n, B->data);
    Matrix* X = matrix_create_from_buffer(A->m, A->n, A->data);

    int* ipiv = (int*)malloc(n * sizeof(int));

    int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, nrhs, 
                              B_copy->data, n, ipiv, 
                              X->data, n);

    free(ipiv);
    matrix_free(B_copy);
    return (info == 0) ? X : (matrix_free(X), NULL);
}


// For symmetric positive definite matrices (faster)
Matrix* matrix_solve_spd(const Matrix* A, const Matrix* B, int multithreaded) {
    if (!A || !B || A->m != A->n || A->m != B->m) {
        return NULL;
    }
    
    int n = A->n;
    int nrhs = B->n;
    
    // Create copies
    Matrix* A_copy = matrix_create_from_buffer((size_t)n, (size_t)n, A->data);
    if (!A_copy) return NULL;
    
    Matrix* X = matrix_create_from_buffer((size_t)n, (size_t)nrhs, B->data);
    if (!X) {
        matrix_free(A_copy);
        return NULL;
    }
    
    // Solve using Cholesky (A must be symmetric positive definite)
    int info = LAPACKE_dposv(LAPACK_ROW_MAJOR, 'U', n, nrhs,
                              A_copy->data, n, X->data, nrhs);
    
    matrix_free(A_copy);
    
    if (info != 0) {
        matrix_free(X);
        return NULL;
    }
    
    return X;
}

int matrix_factor_lu(Matrix* A, int* ipiv, int multithreaded) {
    if (!A || A->m != A->n || !ipiv) return 0;
    
    int n = A->n;
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A->data, n, ipiv);
    return (info == 0);
}

Matrix* matrix_solve_factored(const Matrix* A_factored, const int* ipiv, 
                              const Matrix* B, int multithreaded) {
    if (!A_factored || !ipiv || !B || A_factored->m != B->m) {
        return NULL;
    }
    
    int n = A_factored->n;
    int nrhs = B->n;
    
    Matrix* X = matrix_create_from_buffer((size_t)n, (size_t)nrhs, B->data);
    if (!X) return NULL;
    
    int info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', n, nrhs,
                               A_factored->data, n, ipiv,
                               X->data, nrhs);
    
    if (info != 0) {
        matrix_free(X);
        return NULL;
    }
    
    return X;
}

Matrix* matrix_solve_lstsq(const Matrix* A, const Matrix* B, int multithreaded) {
    if (!A || !B || A->m != B->m) return NULL;
    
    int m = A->m;
    int n = A->n;
    int nrhs = B->n;
    
    // Create copies
    Matrix* A_copy = matrix_create_from_buffer((size_t)m, (size_t)n, A->data);
    if (!A_copy) return NULL;
    
    Matrix* X = matrix_create_from_buffer((size_t)m, (size_t)nrhs, B->data);
    if (!X) {
        matrix_free(A_copy);
        return NULL;
    }
    
    int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', m, n, nrhs,
                              A_copy->data, n, X->data, nrhs);
    
    matrix_free(A_copy);
    
    if (info != 0) {
        matrix_free(X);
        return NULL;
    }
    
    Matrix* result = matrix_create(n, nrhs);
    if (!result) {
        matrix_free(X);
        return NULL;
    }
    
    // Copy solution part
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < nrhs; j++) {
            MAT(result, i, j) = MAT(X, i, j);
        }
    }
    
    matrix_free(X);
    return result;
}

double matrix_determinant(const Matrix* M, int multithreaded){
    if (!M || M->m != M->n){
        perror("Determinant Matrix error");
        return 0.0;
    }

    int n = M->n;

    Matrix temp;
    temp.m = n;
    temp.n = n;
    temp.data = (double*)malloc(sizeof(double) * n * n);
    if (!temp.data){
        perror("Determinant Matrix data error");
        return 0.0;
    }

    for (int i = 0; i < n * n; i++){
        temp.data[i] = M->data[i];
    }

    int swap_count = matrix_lu_decompose(&temp, multithreaded);

    if (swap_count < 0){
        free(temp.data);
        return 0.0;
    }

    double det = 1.0;

    for (int i = 0; i < n; i++){
        det *= temp.data[i*n + i];
    }

    if (swap_count % 2 != 0){
        det = -det;
    }

    free(temp.data);
    return det;
}

double matrix_log_determinant(const Matrix* M, int* sign){
    if (!M || M->m != M->n){
        return -INFINITY;
    }

    int n = M->n;

    Matrix temp;
    temp.m = n;
    temp.n = n;
    temp.data = (double*)malloc(sizeof(double) * n * n);
    if (!temp.data){
        return -INFINITY;
    }

    for (int i = 0; i < n*n; i++){
        temp.data[i] = M->data[i];
    }

    int swap_count = matrix_lu_decompose(&temp, 0);
    if (swap_count < 0){
        free(temp.data);
        return -INFINITY;
    }

    double log_det = 0.0;
    int s = (swap_count % 2 == 0) ? 1 : -1;

    for (int i = 0; i < n; i++){
        double diag = temp.data[i*n + i];
        if (diag < 0){
            s *= -1;
            diag = -diag;
        }
        log_det += log(diag);
    }

    free(temp.data);

    if (sign){
        *sign = s;
    }

    return log_det;
}

#define BLOCK_SIZE 64
#define EPS 1e-12

inline int matrix_lu_decompose(Matrix* M, int multithreaded){
    if (!M || M->m != M->n){
        return -1;
    }

    int n = M->n;
    double* a = M->data;
    int swap_count = 0;

    for (int k = 0; k < n; k += BLOCK_SIZE){

        int bk = (k + BLOCK_SIZE > n) ? (n - k) : BLOCK_SIZE;

        for (int kk = 0; kk < bk; kk++){

            int col = k + kk;
            int pivot = col;
            double max_val = fabs(a[col*n + col]);

            for (int i = col + 1; i < n; i++){
                double val = fabs(a[i*n + col]);
                if (val > max_val){
                    max_val = val;
                    pivot = i;
                }
            }

            if (max_val < EPS){
                return -1;
            }

            if (pivot != col){
                for (int j = 0; j < n; j++){
                    double tmp = a[col*n + j];
                    a[col*n + j] = a[pivot*n + j];
                    a[pivot*n + j] = tmp;
                }
                swap_count++;
            }

            for (int i = col + 1; i < n; i++){
                a[i*n + col] /= a[col*n + col];
                double factor = a[i*n + col];
                for (int j = col + 1; j < k + bk; j++)
                    a[i*n + j] -= factor * a[col*n + j];
            }
        }

        if (k + bk >= n){
            continue;
        }

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

        int rows = n - k - bk;
        int cols = n - k - bk;

        if (multithreaded){
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < rows; i += BLOCK_SIZE){
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