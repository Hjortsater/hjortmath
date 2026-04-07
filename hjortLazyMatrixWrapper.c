#include <Python.h>
#include <stdlib.h>
#include "hjortLazyEvaluate.h"

static PyObject* wrap_matrix(Matrix* M) {
    return PyCapsule_New(M, "hjortMatrixWrapper.Matrix", NULL);
}

static PyObject* py_matrix_lazy_evaluate(PyObject* self, PyObject* args, PyObject* kwargs) {

    PyObject *root_capsule, *ops_list;

    int multithreaded = 1;
    int simplify_flag = 1;

    static char *kwlist[] = {"root", "ops", "multithreaded", "simplify", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|pp", kwlist,
                                     &root_capsule, &ops_list,
                                     &multithreaded, &simplify_flag))
        return NULL;

    Matrix* root = NULL;
    if (root_capsule != Py_None && PyCapsule_CheckExact(root_capsule)) {
        root = PyCapsule_GetPointer(root_capsule, "hjortMatrixWrapper.Matrix");
        if (!root) {
            PyErr_SetString(PyExc_ValueError, "Invalid root matrix capsule.");
            return NULL;
        }
    }

    if (!PyList_Check(ops_list)) {
        PyErr_SetString(PyExc_TypeError, "ops must be a list.");
        return NULL;
    }

    int num_ops = (int)PyList_Size(ops_list);

    MatrixOp* ops = (MatrixOp*)malloc(sizeof(MatrixOp) * num_ops);
    if (!ops)
        return PyErr_NoMemory();

    for (int i = 0; i < num_ops; i++) {

        PyObject* tuple = PyList_GetItem(ops_list, i);

        if (!PyTuple_Check(tuple) || PyTuple_Size(tuple) != 3) {
            free(ops);
            PyErr_SetString(PyExc_ValueError, "Each op must be (op_type, operand, version).");
            return NULL;
        }

        PyObject* py_op_type = PyTuple_GetItem(tuple, 0);
        PyObject* py_operand = PyTuple_GetItem(tuple, 1);
        PyObject* py_version = PyTuple_GetItem(tuple, 2);

        ops[i].op_type = (int)PyLong_AsLong(py_op_type);
        ops[i].version = (int)PyLong_AsLong(py_version);

        if (PyCapsule_CheckExact(py_operand)) {

            Matrix* M = PyCapsule_GetPointer(py_operand, "hjortMatrixWrapper.Matrix");

            if (!M) {
                free(ops);
                PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule in ops.");
                return NULL;
            }

            ops[i].is_scalar = 0;
            ops[i].operand.mat = M;

        } else if (PyFloat_Check(py_operand) || PyLong_Check(py_operand)) {

            ops[i].is_scalar = 1;
            ops[i].operand.scalar = PyFloat_AsDouble(py_operand);

        } else if (py_operand == Py_None) {

            ops[i].is_scalar = 0;
            ops[i].operand.mat = NULL;

        } else {

            free(ops);
            PyErr_SetString(PyExc_TypeError, "Operand must be matrix capsule, scalar, or None.");
            return NULL;
        }
    }

    Matrix* result = NULL;

    Py_BEGIN_ALLOW_THREADS
    result = hjort_lazy_evaluate(
        root,
        ops,
        num_ops,
        multithreaded,
        simplify_flag
    );
    Py_END_ALLOW_THREADS

    free(ops);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Lazy matrix evaluation failed.");
        return NULL;
    }

    return wrap_matrix(result);
}


static PyMethodDef HjortLazyEvaluateMethods[] = {
    {"matrix_lazy_evaluate", (PyCFunction)py_matrix_lazy_evaluate, METH_VARARGS | METH_KEYWORDS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef HjortLazyEvaluateModule = {
    PyModuleDef_HEAD_INIT,
    "hjortLazyEvaluateWrapper",
    "",
    -1,
    HjortLazyEvaluateMethods
};

PyMODINIT_FUNC PyInit_hjortLazyMatrixWrapper(void) {
    return PyModule_Create(&HjortLazyEvaluateModule);
}