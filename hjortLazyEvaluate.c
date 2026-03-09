#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include "hjortLazyEvaluate.h"

inline Matrix* lazy_deligated_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded);
inline MatrixOp* simplify_ops(Matrix* root, MatrixOp* ops, int num_ops, int* out_count);
void debug_print_ops(const char* label, MatrixOp* ops, int count);

Matrix* hjort_lazy_evaluate(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded, int simplify_flag){
    Matrix* result = NULL;
    if (simplify_flag) {
        debug_print_ops("BEFORE SIMPLIFY", ops, num_ops);
        
        int new_count;
        MatrixOp* simplified_ops = simplify_ops(root, ops, num_ops, &new_count);
        
        debug_print_ops("AFTER SIMPLIFY", simplified_ops, new_count);

        result = lazy_deligated_evaluation(root, simplified_ops, new_count, multithreaded);
        free(simplified_ops);
    } else {
        result = lazy_deligated_evaluation(root, ops, num_ops, multithreaded);
    }
    return result;
}

inline MatrixOp* simplify_ops(Matrix* root, MatrixOp* ops, int num_ops, int* out_count) {
    typedef struct {
        Matrix* mat;
        int version;
        double coeff;
    } LinearTerm;

    LinearTerm terms[256];
    int term_count = 0;
    double root_coeff = 1.0;

    MatrixOp* out = malloc(sizeof(MatrixOp) * (num_ops * 64));
    int out_i = 0;

    for (int i = 0; i < num_ops; i++) {
        int type = ops[i].op_type;

        if (type == 0 || type == 1 || type == 4) {
            if (type == 0 || type == 1) {
                double sign = (type == 0) ? 1.0 : -1.0;
                if (ops[i].operand.mat == root) {
                    root_coeff += sign;
                } else {
                    int found = 0;
                    for (int j = 0; j < term_count; j++) {
                        if (terms[j].mat == ops[i].operand.mat && terms[j].version == ops[i].version) {
                            terms[j].coeff += sign;
                            found = 1;
                            break;
                        }
                    }
                    if (!found) {
                        terms[term_count].mat = ops[i].operand.mat;
                        terms[term_count].version = ops[i].version;
                        terms[term_count].coeff = sign;
                        term_count++;
                    }
                }
            } else if (type == 4) {
                double factor = ops[i].operand.scalar;
                root_coeff *= factor;
                for (int j = 0; j < term_count; j++) {
                    terms[j].coeff *= factor;
                }
            }
        } else {
            if (root_coeff != 1.0) {
                out[out_i].op_type = 4;
                out[out_i].operand.scalar = root_coeff;
                out[out_i].version = 0;
                out_i++;
            }
            for (int j = 0; j < term_count; j++) {
                int c = (int)terms[j].coeff;
                if (c == 0) continue;
                for (int k = 0; k < abs(c); k++) {
                    out[out_i].op_type = (c > 0) ? 0 : 1;
                    out[out_i].operand.mat = terms[j].mat;
                    out[out_i].version = terms[j].version;
                    out_i++;
                }
            }
            
            out[out_i++] = ops[i];
            
            root_coeff = 1.0;
            term_count = 0;
        }
    }

    if (root_coeff != 1.0) {
        out[out_i].op_type = 4;
        out[out_i].operand.scalar = root_coeff;
        out[out_i].version = 0;
        out_i++;
    }
    for (int j = 0; j < term_count; j++) {
        int c = (int)terms[j].coeff;
        if (c == 0) continue;
        for (int k = 0; k < abs(c); k++) {
            out[out_i].op_type = (c > 0) ? 0 : 1;
            out[out_i].operand.mat = terms[j].mat;
            out[out_i].version = terms[j].version;
            out_i++;
        }
    }

    *out_count = out_i;
    return out;
}


inline Matrix* lazy_deligated_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded){
    Matrix* res = matrix_clone(root);
    if (!res) return NULL;

    for (int i = 0; i < num_ops; i++) {
        switch (ops[i].op_type) {
            case 0:
                if (!matrix_add_inplace(res, ops[i].operand.mat, res, multithreaded)) goto error;
                break;
            case 1:
                if (!matrix_sub_inplace(res, ops[i].operand.mat, res, multithreaded)) goto error;
                break;
            case 2:
            {
                Matrix* next_res = matrix_mul(res, ops[i].operand.mat, multithreaded);
                if (!next_res) goto error;
                matrix_free(res);
                res = next_res;
                break;
            }
            case 3:
            {
                Matrix* next_res = matrix_mul(ops[i].operand.mat, res, multithreaded);
                if (!next_res) goto error;
                matrix_free(res);
                res = next_res;
                break;
            }
            case 4:
            {
                double val = ops[i].operand.scalar;
                if (!matrix_scalar_mul_inplace(res, res, val, multithreaded)) goto error;
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


void debug_print_ops(const char* label, MatrixOp* ops, int count) {
    printf("--- %s (%d ops) ---\n", label, count);
    for (int i = 0; i < count; i++) {
        printf("[%d] Type: %d | ", i, ops[i].op_type);
        if (ops[i].op_type == 4) {
            printf("Value: %f\n", ops[i].operand.scalar);
        } else {
            printf("Ptr: %p | Ver: %d\n", (void*)ops[i].operand.mat, ops[i].version);
        }
    }
    printf("--------------------------\n");
}