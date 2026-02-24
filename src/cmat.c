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

    // Create a local copy so we don't mutate the original Matrix in Python
    double* temp = malloc(n * n * sizeof(double));
    memcpy(temp, A, n * n * sizeof(double));

    double det = 1.0;

    for (size_t i = 0; i < n; i++) {
        // --- Partial Pivoting ---
        size_t pivot = i;
        for (size_t j = i + 1; j < n; j++) {
            if (fabs(temp[j * n + i]) > fabs(temp[pivot * n + i])) {
                pivot = j;
            }
        }

        // Swap rows if we found a better pivot
        if (pivot != i) {
            for (size_t k = 0; k < n; k++) {
                double swap = temp[i * n + k];
                temp[i * n + k] = temp[pivot * n + k];
                temp[pivot * n + k] = swap;
            }
            det *= -1.0; // Swapping rows flips the determinant sign
        }

        // Check for singularity (can't divide by zero)
        if (fabs(temp[i * n + i]) < 1e-12) {
            free(temp);
            return 0.0;
        }

        // Multiply the diagonal into our determinant result
        det *= temp[i * n + i];

        // --- Elimination Step ---
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