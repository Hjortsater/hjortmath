#include <Python.h>
#include <stdlib.h>
#include "hjortLazyEvaluate.h"
#include <stdio.h>

static MatrixNode* create_leaf_matrix(Matrix* mat) {
    MatrixNode* node = (MatrixNode*)malloc(sizeof(MatrixNode));
    if (!node) return NULL;
    node->type = 0;
    node->data.leaf.is_scalar = 0;
    node->data.leaf.value.mat = mat;
    return node;
}

static MatrixNode* create_leaf_scalar(double scalar) {
    MatrixNode* node = (MatrixNode*)malloc(sizeof(MatrixNode));
    if (!node) return NULL;
    node->type = 0;
    node->data.leaf.is_scalar = 1;
    node->data.leaf.value.scalar = scalar;
    return node;
}

static MatrixNode* create_binop(int op, MatrixNode* left, MatrixNode* right) {
    MatrixNode* node = (MatrixNode*)malloc(sizeof(MatrixNode));
    if (!node) return NULL;
    node->type = 1;
    node->data.binop.op = op;
    node->data.binop.left = left;
    node->data.binop.right = right;
    return node;
}

static void free_tree(MatrixNode* node) {
    if (!node) return;
    if (node->type == 1) {
        free_tree(node->data.binop.left);
        free_tree(node->data.binop.right);
    }
    free(node);
}

static MatrixNode* simplify_tree(MatrixNode* node) {
    if (!node) return NULL;
    if (node->type == 0) {
        return node; // leaf, no simplification
    }

    // Simplify children first
    MatrixNode* l = simplify_tree(node->data.binop.left);
    MatrixNode* r = simplify_tree(node->data.binop.right);

    int op = node->data.binop.op;

    if (op == 0) { // ADD
        // Check if both are leaves with same matrix
        if (l->type == 0 && r->type == 0 && !l->data.leaf.is_scalar && !r->data.leaf.is_scalar &&
            l->data.leaf.value.mat == r->data.leaf.value.mat) {
            MatrixNode* new_node = create_binop(4, create_leaf_scalar(2.0), create_leaf_matrix(l->data.leaf.value.mat));
            free_tree(l);
            free_tree(r);
            free(node);
            return new_node;
        }

        // Check if left is leaf, right is SML with same matrix
        if (l->type == 0 && !l->data.leaf.is_scalar && r->type == 1 && r->data.binop.op == 4 &&
            r->data.binop.left->type == 0 && r->data.binop.left->data.leaf.is_scalar &&
            r->data.binop.right->type == 0 && !r->data.binop.right->data.leaf.is_scalar &&
            l->data.leaf.value.mat == r->data.binop.right->data.leaf.value.mat) {
            double new_scalar = 1.0 + r->data.binop.left->data.leaf.value.scalar;
            MatrixNode* new_node = create_binop(4, create_leaf_scalar(new_scalar), create_leaf_matrix(l->data.leaf.value.mat));
            free_tree(l);
            free_tree(r);
            free(node);
            return new_node;
        }

        // Check if right is SML, left is same matrix
        if (r->type == 1 && r->data.binop.op == 4 &&
            r->data.binop.left->type == 0 && r->data.binop.left->data.leaf.is_scalar &&
            r->data.binop.right->type == 0 && !r->data.binop.right->data.leaf.is_scalar &&
            l->type == 0 && !l->data.leaf.is_scalar &&
            l->data.leaf.value.mat == r->data.binop.right->data.leaf.value.mat) {
            double new_scalar = r->data.binop.left->data.leaf.value.scalar + 1.0;
            MatrixNode* new_node = create_binop(4, create_leaf_scalar(new_scalar), create_leaf_matrix(l->data.leaf.value.mat));
            free_tree(l);
            free_tree(r);
            free(node);
            return new_node;
        }

        // Check if both are SML with same matrix
        if (l->type == 1 && l->data.binop.op == 4 &&
            r->type == 1 && r->data.binop.op == 4 &&
            l->data.binop.left->type == 0 && l->data.binop.left->data.leaf.is_scalar &&
            r->data.binop.left->type == 0 && r->data.binop.left->data.leaf.is_scalar &&
            l->data.binop.right->type == 0 && !l->data.binop.right->data.leaf.is_scalar &&
            r->data.binop.right->type == 0 && !r->data.binop.right->data.leaf.is_scalar &&
            l->data.binop.right->data.leaf.value.mat == r->data.binop.right->data.leaf.value.mat) {
            double new_scalar = l->data.binop.left->data.leaf.value.scalar + r->data.binop.left->data.leaf.value.scalar;
            MatrixNode* new_node = create_binop(4, create_leaf_scalar(new_scalar), create_leaf_matrix(l->data.binop.right->data.leaf.value.mat));
            free_tree(l);
            free_tree(r);
            free(node);
            return new_node;
        }

        // Check if left is leaf, right is SML, left == right.right
        if (l->type == 0 && !l->data.leaf.is_scalar && r->type == 1 && r->data.binop.op == 4 &&
            r->data.binop.left->type == 0 && r->data.binop.left->data.leaf.is_scalar &&
            r->data.binop.right->type == 0 && !r->data.binop.right->data.leaf.is_scalar &&
            l->data.leaf.value.mat == r->data.binop.right->data.leaf.value.mat) {
            double new_scalar = 1.0 + r->data.binop.left->data.leaf.value.scalar;
            MatrixNode* new_node = create_binop(4, create_leaf_scalar(new_scalar), create_leaf_matrix(l->data.leaf.value.mat));
            free_tree(l);
            free_tree(r);
            free(node);
            return new_node;
        }

    } else if (op == 1) { // SUB
        // Check if both are leaves with same matrix: A - A = 0*A
        if (l->type == 0 && r->type == 0 && !l->data.leaf.is_scalar && !r->data.leaf.is_scalar &&
            l->data.leaf.value.mat == r->data.leaf.value.mat) {
            MatrixNode* new_node = create_binop(4, create_leaf_scalar(0.0), create_leaf_matrix(l->data.leaf.value.mat));
            free_tree(l);
            free_tree(r);
            free(node);
            return new_node;
        }

        // Check if left is SML, right is leaf, same matrix: k*A - A = (k-1)*A
        if (l->type == 1 && l->data.binop.op == 4 &&
            l->data.binop.left->type == 0 && l->data.binop.left->data.leaf.is_scalar &&
            l->data.binop.right->type == 0 && !l->data.binop.right->data.leaf.is_scalar &&
            r->type == 0 && !r->data.leaf.is_scalar &&
            l->data.binop.right->data.leaf.value.mat == r->data.leaf.value.mat) {
            double new_scalar = l->data.binop.left->data.leaf.value.scalar - 1.0;
            MatrixNode* new_node = create_binop(4, create_leaf_scalar(new_scalar), create_leaf_matrix(r->data.leaf.value.mat));
            free_tree(l);
            free_tree(r);
            free(node);
            return new_node;
        }

        // Check if right is SML, left is leaf, same matrix: A - k*A = (1-k)*A
        if (r->type == 1 && r->data.binop.op == 4 &&
            r->data.binop.left->type == 0 && r->data.binop.left->data.leaf.is_scalar &&
            r->data.binop.right->type == 0 && !r->data.binop.right->data.leaf.is_scalar &&
            l->type == 0 && !l->data.leaf.is_scalar &&
            r->data.binop.right->data.leaf.value.mat == l->data.leaf.value.mat) {
            double new_scalar = 1.0 - r->data.binop.left->data.leaf.value.scalar;
            MatrixNode* new_node = create_binop(4, create_leaf_scalar(new_scalar), create_leaf_matrix(l->data.leaf.value.mat));
            free_tree(l);
            free_tree(r);
            free(node);
            return new_node;
        }

        // Check if both are SML with same matrix: k*A - m*A = (k-m)*A
        if (l->type == 1 && l->data.binop.op == 4 &&
            r->type == 1 && r->data.binop.op == 4 &&
            l->data.binop.left->type == 0 && l->data.binop.left->data.leaf.is_scalar &&
            r->data.binop.left->type == 0 && r->data.binop.left->data.leaf.is_scalar &&
            l->data.binop.right->type == 0 && !l->data.binop.right->data.leaf.is_scalar &&
            r->data.binop.right->type == 0 && !r->data.binop.right->data.leaf.is_scalar &&
            l->data.binop.right->data.leaf.value.mat == r->data.binop.right->data.leaf.value.mat) {
            double new_scalar = l->data.binop.left->data.leaf.value.scalar - r->data.binop.left->data.leaf.value.scalar;
            MatrixNode* new_node = create_binop(4, create_leaf_scalar(new_scalar), create_leaf_matrix(l->data.binop.right->data.leaf.value.mat));
            free_tree(l);
            free_tree(r);
            free(node);
            return new_node;
        }

        // Check if left is leaf, right is SML, left == right.right: A - k*A = (1-k)*A
        if (l->type == 0 && !l->data.leaf.is_scalar && r->type == 1 && r->data.binop.op == 4 &&
            r->data.binop.left->type == 0 && r->data.binop.left->data.leaf.is_scalar &&
            r->data.binop.right->type == 0 && !r->data.binop.right->data.leaf.is_scalar &&
            l->data.leaf.value.mat == r->data.binop.right->data.leaf.value.mat) {
            double new_scalar = 1.0 - r->data.binop.left->data.leaf.value.scalar;
            MatrixNode* new_node = create_binop(4, create_leaf_scalar(new_scalar), create_leaf_matrix(l->data.leaf.value.mat));
            free_tree(l);
            free_tree(r);
            free(node);
            return new_node;
        }

        // (X + Y) - Y = X
        if (l->type == 1 && l->data.binop.op == 0) {
            if (l->data.binop.right->type == 0 && !l->data.binop.right->data.leaf.is_scalar &&
                r->type == 0 && !r->data.leaf.is_scalar &&
                l->data.binop.right->data.leaf.value.mat == r->data.leaf.value.mat) {
                MatrixNode* result = l->data.binop.left;
                free_tree(l->data.binop.right);
                free_tree(r);
                free(node);
                return result;
            }
            if (l->data.binop.left->type == 0 && !l->data.binop.left->data.leaf.is_scalar &&
                r->type == 0 && !r->data.leaf.is_scalar &&
                l->data.binop.left->data.leaf.value.mat == r->data.leaf.value.mat) {
                MatrixNode* result = l->data.binop.right;
                free_tree(l->data.binop.left);
                free_tree(r);
                free(node);
                return result;
            }
        }
    }

    // No simplification, update node with simplified children
    node->data.binop.left = l;
    node->data.binop.right = r;
    return node;
}

static Matrix* evaluate_tree(MatrixNode* node, int multithreaded) {
    if (node->type == 0) {
        if (node->data.leaf.is_scalar) {
            return NULL;
        } else {
            return matrix_clone(node->data.leaf.value.mat);
        }
    } else {
        Matrix* result = NULL;
        if (node->data.binop.op == 4) {
            if (node->data.binop.left->type == 0 && node->data.binop.left->data.leaf.is_scalar) {
                double scalar = node->data.binop.left->data.leaf.value.scalar;
                if (node->data.binop.right->type == 1) {
                    int subop = node->data.binop.right->data.binop.op;
                    if (subop == 0 || subop == 1) {
                        Matrix* left = evaluate_tree(node->data.binop.right->data.binop.left, multithreaded);
                        Matrix* right = evaluate_tree(node->data.binop.right->data.binop.right, multithreaded);
                        if (subop == 0) {
                            result = matrix_scalar_mul_add(left, right, scalar, multithreaded);
                        } else {
                            result = matrix_scalar_mul_sub(left, right, scalar, multithreaded);
                        }
                        matrix_free(left);
                        matrix_free(right);
                        return result;
                    }
                }
                Matrix* right = evaluate_tree(node->data.binop.right, multithreaded);
                result = matrix_scalar_mul(right, scalar, multithreaded);
                matrix_free(right);
            } else {
                result = NULL;
            }
        } else {
            Matrix* left = evaluate_tree(node->data.binop.left, multithreaded);
            Matrix* right = evaluate_tree(node->data.binop.right, multithreaded);
            switch (node->data.binop.op) {
                case 0:
                    result = matrix_add(left, right, multithreaded);
                    break;
                case 1:
                    result = matrix_sub(left, right, multithreaded);
                    break;
                case 2:
                case 3:
                    result = matrix_mul(left, right, multithreaded);
                    break;
                case 5:
                    result = matrix_solve(left, right);
                    break;
                case 6:
                    result = matrix_elementwise_mul(left, right, multithreaded);
                    break;
                default:
                    result = NULL;
            }
            matrix_free(left);
            matrix_free(right);
        }
        return result;
    }
}

Matrix* hjort_lazy_evaluate(Matrix* root, MatrixOp* ops, int num_ops, int multithreaded, int simplify_flag) {
    MatrixNode** stack = (MatrixNode**)malloc(sizeof(MatrixNode*) * num_ops);
    if (!stack) return NULL;
    int stack_size = 0;

    for (int i = 0; i < num_ops; i++) {
        if (ops[i].op_type == -1) {
            MatrixNode* node = create_leaf_matrix(ops[i].operand.mat);
            if (!node) goto error;
            stack[stack_size++] = node;
        } else if (ops[i].is_scalar) {
            MatrixNode* node = create_leaf_scalar(ops[i].operand.scalar);
            if (!node) goto error;
            stack[stack_size++] = node;
        } else {
            if (stack_size < 2) goto error;
            MatrixNode* right = stack[--stack_size];
            MatrixNode* left = stack[--stack_size];
            MatrixNode* node = create_binop(ops[i].op_type, left, right);
            if (!node) goto error;
            stack[stack_size++] = node;
        }
    }

    if (stack_size != 1) goto error;

    MatrixNode* root_node = stack[0];
    if (simplify_flag) {
        root_node = simplify_tree(root_node);
    }
    Matrix* result = evaluate_tree(root_node, multithreaded);
    free_tree(root_node);
    free(stack);
    return result;

error:
    for (int i = 0; i < stack_size; i++) {
        free_tree(stack[i]);
    }
    free(stack);
    return NULL;
}