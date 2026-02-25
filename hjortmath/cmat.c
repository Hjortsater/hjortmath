#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <math.h>

void mat_add(const double* A,
             const double* B,
             double* C,
             size_t size)
{
    for (size_t i = 0; i < size; i++)
        C[i] = A[i] + B[i];
}

void mat_sub(const double* A,
             const double* B,
             double* C,
             size_t size)
{
    for (size_t i = 0; i < size; i++)
        C[i] = A[i] - B[i];
}

void hadamard(const double* A,
              const double* B,
              double* C,
              size_t size)
{
    for (size_t i = 0; i < size; i++)
        C[i] = A[i] * B[i];
}

void mat_mul(const double* A, const double* B, double* C,
             size_t m, size_t n, size_t p)
{
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < p; j++)
            C[i*p + j] = 0.0;

    for (size_t i = 0; i < m; i++)
    {
        for (size_t k = 0; k < n; k++)
        {
            double a = A[i*n + k];
            for (size_t j = 0; j < p; j++)
            {
                C[i*p + j] += a * B[k*p + j];
            }
        }
    }
}

void scalar_mul(const double* A,
                double scalar,
                double* C,
                size_t size)
{
    for (size_t i = 0; i < size; i++)
        C[i] = A[i] * scalar;
}

double mat_det(const double* A, size_t n) {
    if (n == 0) return 0;
    if (n == 1) return A[0];

    double* temp = malloc(n * n * sizeof(double));
    memcpy(temp, A, n * n * sizeof(double));

    double det = 1.0;

    for (size_t i = 0; i < n; i++) {
        size_t pivot = i;
        for (size_t j = i + 1; j < n; j++) {
            if (fabs(temp[j * n + i]) > fabs(temp[pivot * n + i])) {
                pivot = j;
            }
        }

        if (pivot != i) {
            for (size_t k = 0; k < n; k++) {
                double swap = temp[i * n + k];
                temp[i * n + k] = temp[pivot * n + k];
                temp[pivot * n + k] = swap;
            }
            det *= -1.0;
        }

        if (fabs(temp[i * n + i]) < 1e-12) {
            free(temp);
            return 0.0;
        }

        det *= temp[i * n + i];

        for (size_t j = i + 1; j < n; j++) {
            double factor = temp[j * n + i] / temp[i * n + i];
            for (size_t k = i + 1; k < n; k++) {
                temp[j * n + k] -= factor * temp[i * n + k];
            }
        }
    }

    free(temp);
    return det;
}

#define IDX(i,j,n) ((i)*(n) + (j))

void mat_inv(const double* A, double* invA, int n)
{

    /* Fast LU Implentation */

    double* LU = (double*)malloc(n * n * sizeof(double));
    int* piv = (int*)malloc(n * sizeof(int));

    memcpy(LU, A, n * n * sizeof(double));

    for (int i = 0; i < n; i++)
        piv[i] = i;

    for (int k = 0; k < n; k++) {
        double max = fabs(LU[IDX(k,k,n)]);
        int pivot = k;

        for (int i = k + 1; i < n; i++) {
            double val = fabs(LU[IDX(i,k,n)]);
            if (val > max) {
                max = val;
                pivot = i;
            }
        }

        if (pivot != k) {
            for (int j = 0; j < n; j++) {
                double tmp = LU[IDX(k,j,n)];
                LU[IDX(k,j,n)] = LU[IDX(pivot,j,n)];
                LU[IDX(pivot,j,n)] = tmp;
            }
            int tmp = piv[k];
            piv[k] = piv[pivot];
            piv[pivot] = tmp;
        }

        double diag = LU[IDX(k,k,n)];

        for (int i = k + 1; i < n; i++) {
            LU[IDX(i,k,n)] /= diag;
            double mult = LU[IDX(i,k,n)];
            for (int j = k + 1; j < n; j++)
                LU[IDX(i,j,n)] -= mult * LU[IDX(k,j,n)];
        }
    }

    for (int col = 0; col < n; col++) {

        double* x = (double*)calloc(n, sizeof(double));
        x[col] = 1.0;

            double* x_perm = (double*)malloc(n * sizeof(double));
            for (int i = 0; i < n; i++)
                x_perm[i] = x[piv[i]];

            free(x);
            x = x_perm;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++)
                x[i] -= LU[IDX(i,j,n)] * x[j];
        }

        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++)
                x[i] -= LU[IDX(i,j,n)] * x[j];
            x[i] /= LU[IDX(i,i,n)];
        }

        for (int i = 0; i < n; i++)
            invA[IDX(i,col,n)] = x[i];

        free(x);
    }

    free(LU);
    free(piv);
}