#include <Python.h>
#include <stdlib.h>

#include "hjortMatrixBackend.h"
#include "hjortLazyEvaluate.h"


static PyObject* wrap_matrix(Matrix* M) {
    return PyCapsule_New(M, "hjortMatrixWrapper.Matrix", NULL);
}


static PyObject* py_matrix_evaluate_kernel(PyObject* self, PyObject* args) {

    PyObject *root_capsule;
    PyObject *ops_list;
    int multithreaded;

    if (!PyArg_ParseTuple(args, "OOi", &root_capsule, &ops_list, &multithreaded))
        return NULL;

    Matrix* root = PyCapsule_GetPointer(root_capsule, "hjortMatrixWrapper.Matrix");
    if (!root) return NULL;

    Matrix* result = hjort_lazy_evaluate(root, ops_list, multithreaded);

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Lazy evaluation failed");
        return NULL;
    }

    return wrap_matrix(result);
}



static PyMethodDef HjortLazyMethods[] = {
    {"matrix_evaluate_kernel", py_matrix_evaluate_kernel, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef HjortLazyModule = {
    PyModuleDef_HEAD_INIT,
    "hjortLazyMatrixWrapper",
    "",
    -1,
    HjortLazyMethods
};


PyMODINIT_FUNC PyInit_hjortLazyMatrixWrapper(void) {
    return PyModule_Create(&HjortLazyModule);
}