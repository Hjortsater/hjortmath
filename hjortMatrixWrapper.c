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

static PyObject* py_matrix_copy(PyObject* self, PyObject* args) {
    PyObject *capsule_src, *capsule_dst;
    if (!PyArg_ParseTuple(args, "OO", &capsule_src, &capsule_dst))
        return NULL;

    Matrix* src = PyCapsule_GetPointer(capsule_src, "hjortMatrixWrapper.Matrix");
    Matrix* dst = PyCapsule_GetPointer(capsule_dst, "hjortMatrixWrapper.Matrix");
    if (!src || !dst) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule.");
        return NULL;
    }

    matrix_copy(src, dst);
    Py_RETURN_NONE;
}

static PyObject* py_matrix_clone(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) return NULL;
    Matrix* src = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    Matrix* clone = matrix_clone(src);
    if (!clone) return PyErr_NoMemory();
    return wrap_matrix(clone);
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

static PyObject* py_matrix_zero(PyObject* self, PyObject* args){
    int m, n;
    if (!PyArg_ParseTuple(args, "ii", &m, &n)) return NULL;

    Matrix* M = matrix_zero(m, n);
    if (!M) return PyErr_NoMemory();

    return wrap_matrix(M);
}

static PyObject* py_matrix_identity(PyObject* self, PyObject* args){
    int n;
    if (!PyArg_ParseTuple(args, "i", &n)) return NULL;

    Matrix* M = matrix_identity(n);
    if (!M) return PyErr_NoMemory();

    return wrap_matrix(M);
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

    #ifdef _OPENMP
    int prev_num_threads = omp_get_max_threads();
    if (!multithreaded) {
        omp_set_num_threads(1);
    }
    #endif

    Matrix* C = matrix_mul(A, B, multithreaded);
    
    #ifdef _OPENMP
    if (!multithreaded) {
        omp_set_num_threads(prev_num_threads);
    }
    #endif
    
    if(!C) Py_RETURN_NONE;
    return wrap_matrix(C);
}

static PyObject* py_matrix_mul_inplace(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_b, *capsule_c;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "B", "C", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|p", kwlist,
                                     &capsule_a, &capsule_b, &capsule_c, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* B = PyCapsule_GetPointer(capsule_b, "hjortMatrixWrapper.Matrix");
    Matrix* C = PyCapsule_GetPointer(capsule_c, "hjortMatrixWrapper.Matrix");

    #ifdef _OPENMP
    int prev_num_threads = omp_get_max_threads();
    if (!multithreaded) {
        omp_set_num_threads(1);
    }
    #endif

    int success = matrix_mul_inplace(A, B, C, multithreaded) != NULL;
    
    #ifdef _OPENMP
    if (!multithreaded) {
        omp_set_num_threads(prev_num_threads);
    }
    #endif

    if(!success) Py_RETURN_FALSE;
    Py_RETURN_TRUE;
}


static PyObject* py_matrix_scalar_mul(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* capsule;
    double scalar;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "scalar", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od|p", kwlist, &capsule, &scalar, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!A) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule.");
        return NULL;
    }

    Matrix* C = matrix_scalar_mul(A, scalar, multithreaded);
    if (!C) {
        PyErr_SetString(PyExc_RuntimeError, "Scalar multiplication failed.");
        return NULL;
    }

    return wrap_matrix(C);
}

static PyObject* py_matrix_scalar_mul_inplace(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_c;
    double scalar;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "C", "scalar", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOd|p", kwlist,
                                     &capsule_a, &capsule_c, &scalar, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* C = PyCapsule_GetPointer(capsule_c, "hjortMatrixWrapper.Matrix");

    if (!A || !C) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule.");
        return NULL;
    }

    if (!matrix_scalar_mul_inplace(A, C, scalar, multithreaded))
        Py_RETURN_FALSE;

    Py_RETURN_TRUE;
}

static PyObject* py_matrix_random(PyObject* self, PyObject* args) {
    int m, n;
    double min, max;
    if (!PyArg_ParseTuple(args, "iidd", &m, &n, &min, &max)) return NULL;
    
    Matrix* M = matrix_random(m, n, min, max);
    if (!M) return PyErr_NoMemory();
    
    return wrap_matrix(M);
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

static PyObject* py_matrix_inverse(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* capsule;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist, &capsule, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!A) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule.");
        return NULL;
    }

    Matrix* inv = NULL;

    Py_BEGIN_ALLOW_THREADS
    inv = matrix_inverse(A, multithreaded);
    Py_END_ALLOW_THREADS

    if (!inv) {
        PyErr_SetString(PyExc_RuntimeError, "Matrix inversion failed (possibly singular).");
        return NULL;
    }

    return wrap_matrix(inv);
}

static PyObject* py_matrix_inverse_inplace(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_c;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "C", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", kwlist,
                                     &capsule_a, &capsule_c, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* C = PyCapsule_GetPointer(capsule_c, "hjortMatrixWrapper.Matrix");

    if (!A || !C) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule.");
        return NULL;
    }

    if (!matrix_inverse_inplace(A, C, multithreaded))
        Py_RETURN_FALSE;

    Py_RETURN_TRUE;
}

static PyObject* py_matrix_solve(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_b;
    static char *kwlist[] = {"A", "B", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", kwlist, &capsule_a, &capsule_b))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* B = PyCapsule_GetPointer(capsule_b, "hjortMatrixWrapper.Matrix");
    if (!A || !B) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule.");
        return NULL;
    }

    Matrix* X = NULL;

    Py_BEGIN_ALLOW_THREADS
    X = matrix_solve(A, B);
    Py_END_ALLOW_THREADS

    if (!X) {
        PyErr_SetString(PyExc_RuntimeError, "Matrix solve failed (possibly singular).");
        return NULL;
    }

    return wrap_matrix(X);
}


static PyObject* py_matrix_solve_spd(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_b;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "B", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", kwlist, &capsule_a, &capsule_b, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* B = PyCapsule_GetPointer(capsule_b, "hjortMatrixWrapper.Matrix");
    if (!A || !B) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule.");
        return NULL;
    }

    Matrix* X = NULL;

    Py_BEGIN_ALLOW_THREADS
    X = matrix_solve_spd(A, B, multithreaded);
    Py_END_ALLOW_THREADS

    if (!X) {
        PyErr_SetString(PyExc_RuntimeError, "Matrix solve failed (matrix not SPD or singular).");
        return NULL;
    }

    return wrap_matrix(X);
}

static PyObject* py_matrix_factor_lu(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_ipiv;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "ipiv", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", kwlist, &capsule_a, &capsule_ipiv, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    if (!A) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule.");
        return NULL;
    }

    Py_buffer ipiv_view;
    if (PyObject_GetBuffer(capsule_ipiv, &ipiv_view, PyBUF_WRITABLE) < 0)
        return NULL;

    if (ipiv_view.itemsize != sizeof(int)) {
        PyErr_SetString(PyExc_TypeError, "Buffer must contain ints.");
        PyBuffer_Release(&ipiv_view);
        return NULL;
    }

    if (ipiv_view.len != (Py_ssize_t)(A->n * sizeof(int))) {
        PyErr_SetString(PyExc_ValueError, "ipiv buffer size must match matrix dimension.");
        PyBuffer_Release(&ipiv_view);
        return NULL;
    }

    int success = matrix_factor_lu(A, (int*)ipiv_view.buf, multithreaded);
    PyBuffer_Release(&ipiv_view);

    if (!success) {
        PyErr_SetString(PyExc_RuntimeError, "LU factorization failed (possibly singular).");
        return NULL;
    }

    Py_RETURN_TRUE;
}

static PyObject* py_matrix_solve_factored(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_ipiv, *capsule_b;
    int multithreaded = 1;
    static char *kwlist[] = {"A_factored", "ipiv", "B", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|p", kwlist, 
                                     &capsule_a, &capsule_ipiv, &capsule_b, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* B = PyCapsule_GetPointer(capsule_b, "hjortMatrixWrapper.Matrix");
    if (!A || !B) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule.");
        return NULL;
    }

    Py_buffer ipiv_view;
    if (PyObject_GetBuffer(capsule_ipiv, &ipiv_view, PyBUF_SIMPLE) < 0)
        return NULL;

    if (ipiv_view.itemsize != sizeof(int)) {
        PyErr_SetString(PyExc_TypeError, "Buffer must contain ints.");
        PyBuffer_Release(&ipiv_view);
        return NULL;
    }

    Matrix* X = NULL;

    Py_BEGIN_ALLOW_THREADS
    X = matrix_solve_factored(A, (const int*)ipiv_view.buf, B, multithreaded);
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&ipiv_view);

    if (!X) {
        PyErr_SetString(PyExc_RuntimeError, "Solve from factored LU failed.");
        return NULL;
    }

    return wrap_matrix(X);
}

static PyObject* py_matrix_solve_lstsq(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *capsule_a, *capsule_b;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "B", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|p", kwlist, &capsule_a, &capsule_b, &multithreaded))
        return NULL;

    Matrix* A = PyCapsule_GetPointer(capsule_a, "hjortMatrixWrapper.Matrix");
    Matrix* B = PyCapsule_GetPointer(capsule_b, "hjortMatrixWrapper.Matrix");
    if (!A || !B) {
        PyErr_SetString(PyExc_ValueError, "Invalid matrix capsule.");
        return NULL;
    }

    Matrix* X = NULL;

    Py_BEGIN_ALLOW_THREADS
    X = matrix_solve_lstsq(A, B, multithreaded);
    Py_END_ALLOW_THREADS

    if (!X) {
        PyErr_SetString(PyExc_RuntimeError, "Least squares solve failed.");
        return NULL;
    }

    return wrap_matrix(X);
}

static PyObject* py_matrix_determinant(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* capsule;
    int multithreaded = 1;
    static char *kwlist[] = {"A", "multithreaded", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist,
                                     &capsule, &multithreaded))
        return NULL;

    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!M)
        return NULL;

    double det;

    Py_BEGIN_ALLOW_THREADS
    det = matrix_determinant(M, multithreaded);
    Py_END_ALLOW_THREADS

    if (isnan(det)) {
        PyErr_SetString(PyExc_RuntimeError, "Determinant calculation failed (possibly singular matrix).");
        return NULL;
    }

    return PyFloat_FromDouble(det);
}

static PyObject* py_matrix_log_determinant(PyObject* self, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule))
        return NULL;

    Matrix* M = PyCapsule_GetPointer(capsule, "hjortMatrixWrapper.Matrix");
    if (!M)
        return NULL;

    int sign = 0;
    double logdet = matrix_log_determinant(M, &sign);

    if (isinf(logdet) && sign == 0) {
        PyErr_SetString(PyExc_RuntimeError, "Log determinant calculation failed.");
        return NULL;
    }

    return PyFloat_FromDouble(sign * logdet);
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
        if (!row) {
            Py_DECREF(outer);
            return NULL;
        }

        for (int j = 0; j < n; ++j) {
            double val = MAT(M, i, j);
            PyObject* num = PyFloat_FromDouble(val);
            PyList_SET_ITEM(row, j, num);
        }

        PyList_SET_ITEM(outer, i, row);
    }

    return outer;
}

static PyMethodDef HjortMatrixWrapperMethods[] = {
    {"matrix_create", py_matrix_create, METH_VARARGS, ""},
    {"matrix_copy", py_matrix_copy, METH_VARARGS, ""},
    {"matrix_clone", py_matrix_clone, METH_VARARGS, ""},
    {"matrix_create_from_buffer", py_matrix_create_from_buffer, METH_VARARGS, ""},
    {"matrix_free", py_matrix_free, METH_VARARGS, ""},
    {"matrix_set", py_matrix_set, METH_VARARGS, ""},
    {"matrix_get", py_matrix_get, METH_VARARGS, ""},
    {"matrix_fill", py_matrix_fill, METH_VARARGS, ""},
    {"matrix_zero", py_matrix_zero, METH_VARARGS, ""},
    {"matrix_identity", py_matrix_identity, METH_VARARGS, ""},
    {"matrix_rows", py_matrix_rows, METH_VARARGS, ""},
    {"matrix_cols", py_matrix_cols, METH_VARARGS, ""},
    {"matrix_add", (PyCFunction)py_matrix_add, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_add_inplace", (PyCFunction)py_matrix_add_inplace, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_sub", (PyCFunction)py_matrix_sub, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_sub_inplace", (PyCFunction)py_matrix_sub_inplace, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_mul", (PyCFunction)py_matrix_mul, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_mul_inplace", (PyCFunction)py_matrix_mul_inplace, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_scalar_mul", (PyCFunction)py_matrix_scalar_mul, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_scalar_mul_inplace", (PyCFunction)py_matrix_scalar_mul_inplace, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_random", py_matrix_random, METH_VARARGS, ""},
    {"matrix_get_max", py_matrix_get_max, METH_VARARGS, ""},
    {"matrix_get_min", py_matrix_get_min, METH_VARARGS, ""},
    {"matrix_inverse", (PyCFunction)py_matrix_inverse, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_inverse_inplace", (PyCFunction)py_matrix_inverse_inplace, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_solve", (PyCFunction)py_matrix_solve, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_solve_spd", (PyCFunction)py_matrix_solve_spd, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_factor_lu", (PyCFunction)py_matrix_factor_lu, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_solve_factored", (PyCFunction)py_matrix_solve_factored, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_solve_lstsq", (PyCFunction)py_matrix_solve_lstsq, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_determinant", (PyCFunction)py_matrix_determinant, METH_VARARGS | METH_KEYWORDS, ""},
    {"matrix_log_determinant", py_matrix_log_determinant, METH_VARARGS, ""},
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