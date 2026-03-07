#ifndef HJORT_LAZY_EVALUATE_H
#define HJORT_LAZY_EVALUATE_H

#include <Python.h>
#include "hjortMatrixBackend.h"

Matrix* hjort_lazy_evaluate(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded);

#endif