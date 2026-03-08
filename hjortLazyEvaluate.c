#include <Python.h>
#include <stdlib.h>

#include "hjortLazyEvaluate.h"


/*
    Lazy evaluate function queues operations in the backend to minimise malloc calls
    and increase accuracy by performing algebraic simplifications.
*/

inline Matrix* lazy_deligated_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded);


Matrix* hjort_lazy_evaluate(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded){
    

    return lazy_deligated_evaluation(root, ops, num_ops, multithreaded);
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