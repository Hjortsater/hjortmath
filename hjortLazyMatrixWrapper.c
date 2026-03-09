#include <Python.h>
#include "hjortMatrixBackend.h"
#include "hjortLazyEvaluate.h"


static PyObject* py_matrix_evaluate_kernel(PyObject* self, PyObject* args) {
    PyObject *root_capsule, *ops_list;
    int multithreaded;
    int simplify_flag;

    if (!PyArg_ParseTuple(args, "OOii", &root_capsule, &ops_list, &multithreaded, &simplify_flag)) return NULL;

    Matrix* root = PyCapsule_GetPointer(root_capsule, "hjortMatrixWrapper.Matrix");
    Py_ssize_t num_ops = PyList_Size(ops_list);

    MatrixOp* ops = malloc(sizeof(MatrixOp) * num_ops);
    if (!ops) return PyErr_NoMemory();

    for (Py_ssize_t i = 0; i < num_ops; i++) {
        PyObject* item = PyList_GetItem(ops_list, i);
        ops[i].op_type = (int)PyLong_AsLong(PyTuple_GetItem(item, 0));
        PyObject* operand_obj = PyTuple_GetItem(item, 1);
        ops[i].version = (int)PyLong_AsLong(PyTuple_GetItem(item, 2));

        if (ops[i].op_type == 4) {
            double* scalar = malloc(sizeof(double));
            if (!scalar) { free(ops); return PyErr_NoMemory(); }
            *scalar = PyFloat_AsDouble(operand_obj);
            ops[i].operand = scalar;
        } else {
            ops[i].operand = PyCapsule_GetPointer(operand_obj, "hjortMatrixWrapper.Matrix");
        }
    }

    Matrix* result = NULL;
    MatrixOp* ops_to_free = NULL;

    if (simplify_flag) {
        int new_count;
        MatrixOp* simplified_ops = simplify_ops(ops, (int)num_ops, &new_count);
        result = lazy_deligated_evaluation(root, simplified_ops, new_count, multithreaded);
        ops_to_free = simplified_ops;
    } else {
        result = lazy_deligated_evaluation(root, ops, (int)num_ops, multithreaded);
        ops_to_free = ops;
    }

    for (int i = 0; i < (simplify_flag ? num_ops : (int)num_ops); i++) {
        if (ops_to_free[i].op_type == 4) free(ops_to_free[i].operand);
    }
    free(ops_to_free);
    
    if (simplify_flag) {
        for (Py_ssize_t i = 0; i < num_ops; i++) {
            if (ops[i].op_type == 4) free(ops[i].operand);
        }
        free(ops);
    }

    if (!result) return PyErr_NoMemory();
    return PyCapsule_New(result, "hjortMatrixWrapper.Matrix", NULL);
}


static PyMethodDef HjortLazyMethods[] = {
    {"matrix_evaluate_kernel", py_matrix_evaluate_kernel, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef HjortLazyModule = {
    PyModuleDef_HEAD_INIT, "hjortLazyMatrixWrapper", "", -1, HjortLazyMethods
};

PyMODINIT_FUNC PyInit_hjortLazyMatrixWrapper(void) {
    return PyModule_Create(&HjortLazyModule);
}