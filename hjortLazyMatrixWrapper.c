#include <Python.h>
#include "hjortMatrixBackend.h"
#include "hjortLazyEvaluate.h"

static PyObject* py_matrix_evaluate_kernel(PyObject* self, PyObject* args) {
    PyObject *root_capsule, *ops_list;
    int multithreaded;

    if (!PyArg_ParseTuple(args, "OOi", &root_capsule, &ops_list, &multithreaded)) return NULL;

    Matrix* root = PyCapsule_GetPointer(root_capsule, "hjortMatrixWrapper.Matrix");
    Py_ssize_t num_ops = PyList_Size(ops_list);
    MatrixOp* ops = malloc(sizeof(MatrixOp) * num_ops);

    for (Py_ssize_t i = 0; i < num_ops; i++) {
        PyObject* item = PyList_GetItem(ops_list, i);
        ops[i].op_type = (int)PyLong_AsLong(PyTuple_GetItem(item, 0));
        ops[i].operand = PyCapsule_GetPointer(PyTuple_GetItem(item, 1), "hjortMatrixWrapper.Matrix");
    }

    Matrix* result = hjort_lazy_evaluate(root, ops, (int)num_ops, multithreaded);
    free(ops);

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