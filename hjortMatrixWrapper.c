#include <Python.h>
#include <stdlib.h>
#include "hjortMatrixBackend.c"

static PyObject* wrap_matrix(Matrix* M) { return PyCapsule_New(M, "hjortMatrixWrapper.Matrix", NULL); }

static PyObject* py_matrix_create(PyObject* self, PyObject* args) {
    int m, n;
    if (!PyArg_ParseTuple(args, "ii", &m, &n)) return NULL;
    Matrix* M = matrix_create(m, n);
    if (!M) return PyErr_NoMemory();
    return wrap_matrix(M);
}

static PyObject* py_matrix_free(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (M) matrix_free(M);
    Py_RETURN_NONE;
}

static PyObject* py_matrix_set(PyObject* self, PyObject* args) {
    PyObject* capsule;
    int i, j;
    double value;
    if (!PyArg_ParseTuple(args, "Oiid", &capsule, &i, &j, &value)) return NULL;
    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (M) matrix_set(M, i, j, value);
    Py_RETURN_NONE;
}

static PyObject* py_matrix_get(PyObject* self, PyObject* args) {
    PyObject* capsule;
    int i, j;
    if (!PyArg_ParseTuple(args, "Oii", &capsule, &i, &j)) return NULL;
    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!M) return PyFloat_FromDouble(0.0);
    return PyFloat_FromDouble(matrix_get(M, i, j));
}

static PyObject* py_matrix_fill(PyObject* self, PyObject* args) {
    PyObject* capsule;
    double value;
    if (!PyArg_ParseTuple(args, "Od", &capsule, &value)) return NULL;
    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (M) matrix_fill(M, value);
    Py_RETURN_NONE;
}

static PyObject* py_matrix_rows(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!M) return PyLong_FromLong(0);
    return PyLong_FromLong(matrix_rows(M));
}

static PyObject* py_matrix_cols(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!M) return PyLong_FromLong(0);
    return PyLong_FromLong(matrix_cols(M));
}

static PyObject* py_matrix_add(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_b;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "B", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", kwlist, &capsule_a, &capsule_b, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* B = PyCapsule_GetPointer(capsule_b, "hjortMatrixWrapper.Matrix");

    Matrix* C = matrix_add(A, B, multithreaded);
    if(!C) Py_RETURN_NONE;

    return wrap_matrix(C);
}

static PyObject* py_matrix_seed_random(PyObject* self, PyObject* args) {
    unsigned int seed;
    if (!PyArg_ParseTuple(args, "I", &seed)) return NULL;
    matrix_seed_random(seed);
    Py_RETURN_NONE;
}

static PyObject* py_matrix_fill_random(PyObject* self, PyObject* args) {
    PyObject* capsule;
    double min, max;
    if (!PyArg_ParseTuple(args, "Odd", &capsule, &min, &max)) return NULL;
    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (M) matrix_fill_random(M, min, max);
    Py_RETURN_NONE;
}

static PyObject* py_matrix_get_max(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!M) return PyFloat_FromDouble(0.0);
    return PyFloat_FromDouble(matrix_get_max(M));
}

static PyObject* py_matrix_get_min(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!M) return PyFloat_FromDouble(0.0);
    return PyFloat_FromDouble(matrix_get_min(M));
}

static PyMethodDef HjortMatrixWrapperMethods[] = {
    {"matrix_create", py_matrix_create, METH_VARARGS, ""},
    {"matrix_free", py_matrix_free, METH_VARARGS, ""},
    {"matrix_set", py_matrix_set, METH_VARARGS, ""},
    {"matrix_get", py_matrix_get, METH_VARARGS, ""},
    {"matrix_fill", py_matrix_fill, METH_VARARGS, ""},
    {"matrix_rows", py_matrix_rows, METH_VARARGS, ""},
    {"matrix_cols", py_matrix_cols, METH_VARARGS, ""},
    {"matrix_add", (PyCFunction)py_matrix_add, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_seed_random", py_matrix_seed_random, METH_VARARGS, ""},
    {"matrix_fill_random", py_matrix_fill_random, METH_VARARGS, ""},
    {"matrix_get_max", py_matrix_get_max, METH_VARARGS, ""},
    {"matrix_get_min", py_matrix_get_min, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef HjortMatrixWrapperModule = {
    PyModuleDef_HEAD_INIT,
    "hjortMatrixWrapper",
    "",
    -1,
    HjortMatrixWrapperMethods
};

PyMODINIT_FUNC PyInit_hjortMatrixWrapper(void) {
    return PyModule_Create(&HjortMatrixWrapperModule);
}