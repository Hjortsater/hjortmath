#include <Python.h>
#include <stdlib.h>

#include "hjortLazyEvaluate.h"


/*
    Lazy evaluate function queues operations in the backend to minimise malloc calls
    and increase accuracy by performing algebraic simplifications.
*/

inline Matrix* lazy_deligated_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded);
inline MatrixOp* simplify_ops(MatrixOp* ops, int num_ops, int* out_count);

Matrix* hjort_lazy_evaluate(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded, int simplify_flag){
    Matrix* result = NULL;
    if (simplify_flag) {
        int new_count;
        MatrixOp* simplified_ops = simplify_ops(ops, num_ops, &new_count);

        result = lazy_deligated_evaluation(root, simplified_ops, new_count, multithreaded);
        free(simplified_ops);
    } else {
        result = lazy_deligated_evaluation(root, ops, num_ops, multithreaded);
    }
    return result;
}



inline MatrixOp* simplify_ops(MatrixOp* ops, int num_ops, int* out_count) {
    // Print all operations
    printf("=== simplify_ops received %d operations ===\n", num_ops);
    for (int i = 0; i < num_ops; i++) {
        printf("  %d: type=%d, operand=%p, version=%d\n", 
               i, ops[i].op_type, ops[i].operand, ops[i].version);
    }




    // Just return a copy of the original ops (no simplification)
    MatrixOp* new_ops = malloc(sizeof(MatrixOp) * num_ops);
    for (int i = 0; i < num_ops; i++) {
        if (ops[i].op_type == 4) {
            // Need to copy scalar for SML ops
            double* scalar_copy = malloc(sizeof(double));
            *scalar_copy = *(double*)ops[i].operand;
            new_ops[i].op_type = 4;
            new_ops[i].operand = scalar_copy;
            new_ops[i].version = ops[i].version;
        } else {
            // Just copy the operation as-is
            new_ops[i] = ops[i];
        }
    }
    
    *out_count = num_ops;
    return new_ops;
}



inline Matrix* lazy_deligated_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded){

    Matrix* res = matrix_clone(root);
    if (!res) return NULL;

    for (int i = 0; i < num_ops; i++) {
        switch (ops[i].op_type) {
            case 0: // ADD
                if (!matrix_add_inplace(res, ops[i].operand, res, multithreaded)) goto error;
                break;
            case 1: // SUB
                if (!matrix_sub_inplace(res, ops[i].operand, res, multithreaded)) goto error;
                break;
            case 2: // RML
            {
                Matrix* next_res = matrix_mul(res, ops[i].operand, multithreaded);
                if (!next_res) goto error;
                matrix_free(res);
                res = next_res;
                break;
            }
            case 3: // LML
            {
                Matrix* next_res = matrix_mul(ops[i].operand, res, multithreaded);
                if (!next_res) goto error;
                matrix_free(res);
                res = next_res;
                break;
            }
            case 4: // SML (scalar multiplication)
            {
                double scalar = *(double*)ops[i].operand;  // make sure Python sends a pointer to double
                if (!matrix_scalar_mul_inplace(res, res, scalar, multithreaded)) goto error;
                break;
            }
            default:
                goto error;
        }
    }
    return res;

error:
    matrix_free(res);
    return NULL;
}
