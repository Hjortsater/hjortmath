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
        printf("Simplified operation queue (%d ops):\n", new_count);
        for (int i = 0; i < new_count; i++) {
            printf("  [%d] op_type=%d, operand=%p, version=%d\n",
                   i,
                   simplified_ops[i].op_type,
                   simplified_ops[i].operand,
                   simplified_ops[i].version);
        }
        result = lazy_deligated_evaluation(root, simplified_ops, new_count, multithreaded);
        free(simplified_ops);
    } else {
        result = lazy_deligated_evaluation(root, ops, num_ops, multithreaded);
    }
    return result;
}


typedef struct {
    int* ops;
    void** operands;
    int* versions;
    int count;
} SimplifiedOps;


inline MatrixOp* simplify_ops(MatrixOp* ops, int num_ops, int* out_count) {
    MatrixOp* new_ops = malloc(sizeof(MatrixOp) * num_ops);
    int new_count = 0;

    for (int i = 0; i < num_ops; i++) {
        MatrixOp current = ops[i];
        int found = 0;

        for (int j = 0; j < new_count; j++) {
            if (new_ops[j].operand == current.operand &&
                new_ops[j].version == current.version) {
                int delta = (current.op_type == 0 ? 1 : -1);
                int existing_op = new_ops[j].op_type;
                if (existing_op == 0) {
                    delta += 1;
                    if (delta == 0) {
                        new_count--;
                        new_ops[j] = new_ops[new_count];
                    } else {
                        new_ops[j].op_type = 0;
                    }
                }
                found = 1;
                break;
            }
        }

        if (!found) {
            new_ops[new_count++] = current;
        }
    }

    *out_count = new_count;
    return new_ops;
}





inline Matrix* lazy_deligated_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded){

    /* Use a standard matrix clone as the starting point */
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
            case 2: // MUL (Standard non-inplace multiplication)
            {
                Matrix* next_res = matrix_mul(res, ops[i].operand, multithreaded);
                if (!next_res) goto error;
                
                // Free the intermediate result and point to the new one
                matrix_free(res);
                res = next_res;
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