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

static PyObject* py_matrix_create_from_buffer(PyObject* self, PyObject* args) {
    PyObject* obj;
    int m, n;

    if (!PyArg_ParseTuple(args, "Oii", &obj, &m, &n))
        return NULL;

    Py_buffer view;

    if (PyObject_GetBuffer(obj, &view, PyBUF_CONTIG_RO) < 0)
        return NULL;

    if (view.itemsize != sizeof(double)) {
        PyErr_SetString(PyExc_TypeError, "Buffer must contain double precision floats.");
        PyBuffer_Release(&view);
        return NULL;
    }

    if (view.len != (Py_ssize_t)(m * n * sizeof(double))) {
        PyErr_SetString(PyExc_ValueError, "Buffer size does not match matrix dimensions.");
        PyBuffer_Release(&view);
        return NULL;
    }

    Matrix* M = matrix_create(m, n);
    if (!M) {
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }

    memcpy(M->data, view.buf, view.len);

    PyBuffer_Release(&view);

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

static PyObject* py_matrix_add_inplace(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_b, *capsule_c;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "B", "C", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|p", kwlist,
                                     &capsule_a, &capsule_b, &capsule_c, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* B = PyCapsule_GetPointer(capsule_b, "hjortMatrixWrapper.Matrix");
    Matrix* C = PyCapsule_GetPointer(capsule_c, "hjortMatrixWrapper.Matrix");

    if(!matrix_add_inplace(A, B, C, multithreaded))
        Py_RETURN_FALSE;

    Py_RETURN_TRUE;
}

static PyObject* py_matrix_sub(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_b;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "B", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", kwlist, &capsule_a, &capsule_b, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* B = PyCapsule_GetPointer(capsule_b, "hjortMatrixWrapper.Matrix");

    Matrix* C = matrix_sub(A, B, multithreaded);
    if(!C) Py_RETURN_NONE;

    return wrap_matrix(C);
}

static PyObject* py_matrix_sub_inplace(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_b, *capsule_c;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "B", "C", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|p", kwlist,
                                     &capsule_a, &capsule_b, &capsule_c, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* B = PyCapsule_GetPointer(capsule_b, "hjortMatrixWrapper.Matrix");
    Matrix* C = PyCapsule_GetPointer(capsule_c, "hjortMatrixWrapper.Matrix");

    if(!matrix_sub_inplace(A, B, C, multithreaded))
        Py_RETURN_FALSE;

    Py_RETURN_TRUE;
}


static PyObject* py_matrix_mul(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_b;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "B", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", kwlist, &capsule_a, &capsule_b, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* B = PyCapsule_GetPointer(capsule_b, "hjortMatrixWrapper.Matrix");

    Matrix* C = matrix_mul(A, B, multithreaded);
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

static PyObject* py_matrix_determinant(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule))
        return NULL;
    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!M)
        return NULL;
    double det = matrix_determinant(M);
    return PyFloat_FromDouble(det);
}


static PyObject* py_matrix_to_list(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule))
        return NULL;

    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!M)
        return NULL;

    int m = M->m;
    int n = M->n;

    PyObject* outer = PyList_New(m);
    if (!outer) return NULL;

    for (int i = 0; i < m; ++i) {
        PyObject* row = PyList_New(n);
        if (!row) return NULL;

        for (int j = 0; j < n; ++j) {
            double val = MAT(M, i, j);
            PyObject* num = PyFloat_FromDouble(val);
            PyList_SET_ITEM(row, j, num);  // steals reference
        }

        PyList_SET_ITEM(outer, i, row);  // steals reference
    }

    return outer;
}

static PyMethodDef HjortMatrixWrapperMethods[] = {
    {"matrix_create", py_matrix_create, METH_VARARGS, ""},
    {"matrix_create_from_buffer", py_matrix_create_from_buffer, METH_VARARGS, ""},
    {"matrix_free", py_matrix_free, METH_VARARGS, ""},
    {"matrix_set", py_matrix_set, METH_VARARGS, ""},
    {"matrix_get", py_matrix_get, METH_VARARGS, ""},
    {"matrix_fill", py_matrix_fill, METH_VARARGS, ""},
    {"matrix_rows", py_matrix_rows, METH_VARARGS, ""},
    {"matrix_cols", py_matrix_cols, METH_VARARGS, ""},
    {"matrix_add", (PyCFunction)py_matrix_add, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_add_inplace", (PyCFunction)py_matrix_add_inplace, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_sub", (PyCFunction)py_matrix_sub, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_sub_inplace", (PyCFunction)py_matrix_sub_inplace, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_mul", (PyCFunction)py_matrix_mul, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_seed_random", py_matrix_seed_random, METH_VARARGS, ""},
    {"matrix_fill_random", py_matrix_fill_random, METH_VARARGS, ""},
    {"matrix_get_max", py_matrix_get_max, METH_VARARGS, ""},
    {"matrix_get_min", py_matrix_get_min, METH_VARARGS, ""},
    {"matrix_determinant", py_matrix_determinant, METH_VARARGS, ""},
    {"matrix_to_list", py_matrix_to_list, METH_VARARGS, ""},
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