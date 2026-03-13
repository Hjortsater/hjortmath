#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include "hjortLazyEvaluate.h"

inline Matrix* lazy_deligated_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded);
inline Matrix* lazy_fused_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded);
inline MatrixOp* simplify_ops(Matrix* root, MatrixOp* ops, int num_ops, int* out_count);

Matrix* hjort_lazy_evaluate(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded, int simplify_flag) {
    if (!root || !ops) return NULL;

    Matrix* result = NULL;
    MatrixOp* active_ops = ops;
    int active_count = num_ops;

    if (simplify_flag) {
        int new_count = 0;
        active_ops = simplify_ops(root, ops, num_ops, &new_count);
        if (!active_ops) return NULL;
        active_count = new_count;
    }

    result = lazy_fused_evaluation(root, active_ops, active_count, multithreaded);

    if (!result) {
        result = lazy_deligated_evaluation(root, active_ops, active_count, multithreaded);
    }

    if (simplify_flag && active_ops != ops) {
        free(active_ops);
    }

    return result;
}

inline MatrixOp* simplify_ops(Matrix* root, MatrixOp* ops, int num_ops, int* out_count) {
    typedef struct {
        Matrix* mat;
        int version;
        double coeff;
    } LinearTerm;

    int term_capacity = 1024;
    LinearTerm* terms = malloc(sizeof(LinearTerm) * term_capacity);
    if (!terms) return NULL;

    int term_count = 0;
    double root_coeff = 1.0;

    int out_capacity = num_ops > 128 ? num_ops * 2 : 256;
    MatrixOp* out = malloc(sizeof(MatrixOp) * out_capacity);
    if (!out) {
        free(terms);
        return NULL;
    }
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
                        if (term_count >= term_capacity) {
                            term_capacity *= 2;
                            LinearTerm* new_terms = realloc(terms, sizeof(LinearTerm) * term_capacity);
                            if (!new_terms) {
                                free(terms);
                                free(out);
                                return NULL;
                            }
                            terms = new_terms;
                        }
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
            if (out_i + term_count * 2 + 5 >= out_capacity) {
                out_capacity = out_capacity * 2 + term_count * 2;
                MatrixOp* new_out = realloc(out, sizeof(MatrixOp) * out_capacity);
                if (!new_out) {
                    free(terms);
                    free(out);
                    return NULL;
                }
                out = new_out;
            }

            if (root_coeff != 1.0) {
                out[out_i].op_type = 4;
                out[out_i].operand.scalar = root_coeff;
                out[out_i++].version = 0;
            }
            for (int j = 0; j < term_count; j++) {
                if (terms[j].coeff == 0.0) continue;
                out[out_i].op_type = 4;
                out[out_i].operand.scalar = terms[j].coeff;
                out[out_i++].version = 0;
                out[out_i].op_type = 0;
                out[out_i].operand.mat = terms[j].mat;
                out[out_i++].version = terms[j].version;
            }

            out[out_i++] = ops[i];
            root_coeff = 1.0;
            term_count = 0;
        }
    }

    if (out_i + term_count * 2 + 2 >= out_capacity) {
        out_capacity += (term_count * 2 + 2);
        MatrixOp* new_out = realloc(out, sizeof(MatrixOp) * out_capacity);
        if (!new_out) {
            free(terms);
            free(out);
            return NULL;
        }
        out = new_out;
    }

    if (root_coeff != 1.0) {
        out[out_i].op_type = 4;
        out[out_i].operand.scalar = root_coeff;
        out[out_i++].version = 0;
    }
    for (int j = 0; j < term_count; j++) {
        if (terms[j].coeff == 0.0) continue;
        out[out_i].op_type = 4;
        out[out_i].operand.scalar = terms[j].coeff;
        out[out_i++].version = 0;
        out[out_i].op_type = 0;
        out[out_i].operand.mat = terms[j].mat;
        out[out_i++].version = terms[j].version;
    }

    free(terms);
    *out_count = out_i;
    return out;
}

inline Matrix* lazy_fused_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded) {
    for (int i = 0; i < num_ops; i++) {
        if (ops[i].op_type == 2 || ops[i].op_type == 3) return NULL;
    }

    Matrix* res = matrix_clone(root);
    if (!res) return NULL;

    double current_scalar = 1.0;

    for (int i = 0; i < num_ops; i++) {
        int type = ops[i].op_type;
        if (type == 0 || type == 1) {
            if (current_scalar != 1.0) {
                Matrix* tmp = matrix_scalar_mul(ops[i].operand.mat, current_scalar, multithreaded);
                if (!tmp) goto error;
                if (type == 0) matrix_add_inplace(res, tmp, res, multithreaded);
                else matrix_sub_inplace(res, tmp, res, multithreaded);
                matrix_free(tmp);
            } else {
                if (type == 0) matrix_add_inplace(res, ops[i].operand.mat, res, multithreaded);
                else matrix_sub_inplace(res, ops[i].operand.mat, res, multithreaded);
            }
        } else if (type == 4) {
            if (!matrix_scalar_mul_inplace(res, res, ops[i].operand.scalar, multithreaded)) goto error;
        }
    }

    return res;

error:
    matrix_free(res);
    return NULL;
}

inline Matrix* lazy_deligated_evaluation(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded) {
    Matrix* res = matrix_clone(root);
    if (!res) return NULL;

    for (int i = 0; i < num_ops; i++) {
        Matrix* next = NULL;
        int type = ops[i].op_type;
        switch (type) {
            case 0:
                if (!matrix_add_inplace(res, ops[i].operand.mat, res, multithreaded)) goto error;
                break;
            case 1:
                if (!matrix_sub_inplace(res, ops[i].operand.mat, res, multithreaded)) goto error;
                break;
            case 2:
                next = matrix_mul(res, ops[i].operand.mat, multithreaded);
                matrix_free(res);
                res = next;
                break;
            case 3:
                next = matrix_mul(ops[i].operand.mat, res, multithreaded);
                matrix_free(res);
                res = next;
                break;
            case 4:
                if (!matrix_scalar_mul_inplace(res, res, ops[i].operand.scalar, multithreaded)) goto error;
                break;
        }
        if (!res) return NULL;
    }
    return res;

error:
    if (res) matrix_free(res);
    return NULL;
}