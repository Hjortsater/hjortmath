#include <Python.h>
#include <stdlib.h>

#include "hjortLazyEvaluate.h"


/*
    Lazy evaluate function queues operations in the backend to minimise malloc calls
    and increase accuracy by performing algebraic simplifications.
*/


Matrix* hjort_lazy_evaluate(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded)
{
    Matrix* res = matrix_clone(root);
    if (!res) return NULL;

    for (int i = 0; i < num_ops; i++) {
        switch (ops[i].op_type) {
            case 0:
                if (!matrix_add_inplace(res, ops[i].operand, res, multithreaded)) goto error;
                break;
            case 1:
                if (!matrix_sub_inplace(res, ops[i].operand, res, multithreaded)) goto error;
                break;
            case 2:
                if (!matrix_mul_inplace(res, ops[i].operand, res, multithreaded)) goto error;
                break;
            default:
                goto error;
        }
    }
    return res;

error:
    matrix_free(res);
    return NULL;
}