#ifndef HJORT_LAZY_EVALUATE_H
#define HJORT_LAZY_EVALUATE_H

#include <Python.h>
#include "hjortMatrixBackend.h"

typedef struct {
    int op_type;
    int version;
    union {
        Matrix* mat;
        double scalar;
    } operand;
} MatrixOp;

Matrix* hjort_lazy_evaluate(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded, int simplify_flag);
MatrixOp* simplify_ops(Matrix* root, MatrixOp* ops, int num_ops, int* out_count);
Matrix* lazy_deligated_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded);

#endif