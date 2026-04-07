#ifndef HJORT_LAZY_EVALUATE_H
#define HJORT_LAZY_EVALUATE_H

#include "hjortMatrixBackend.h"
#include <stdlib.h>

typedef struct MatrixNode MatrixNode;

struct MatrixNode {
    int type;
    union {
        struct {
            int is_scalar;
            union {
                Matrix* mat;
                double scalar;
            } value;
        } leaf;
        struct {
            int op;
            MatrixNode* left;
            MatrixNode* right;
        } binop;
    } data;
};

typedef struct {
    int op_type;
    int version;
    int is_scalar;
    union {
        Matrix* mat;
        double scalar;
    } operand;
} MatrixOp;

Matrix* hjort_lazy_evaluate(
    Matrix* root,
    MatrixOp* ops,
    int num_ops,
    int multithreaded,
    int simplify_flag
);

#endif