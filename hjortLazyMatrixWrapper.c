#include <Python.h>
#include "hjortMatrixBackend.h"
#include "hjortLazyEvaluate.h"


static PyObject* py_matrix_evaluate_kernel(PyObject* self, PyObject* args) {
    PyObject *root_capsule, *ops_list;
    int multithreaded, simplify_flag;

    if (!PyArg_ParseTuple(args, "OOii", &root_capsule, &ops_list, &multithreaded, &simplify_flag)) 
        return NULL;

    Matrix* root = PyCapsule_GetPointer(root_capsule, "hjortMatrixWrapper.Matrix");
    if (!root) return NULL;

    Py_ssize_t num_ops = PyList_Size(ops_list);
    MatrixOp* ops = malloc(sizeof(MatrixOp) * num_ops);
    if (!ops) return PyErr_NoMemory();

    for (Py_ssize_t i = 0; i < num_ops; i++) {
        PyObject* item = PyList_GetItem(ops_list, i);
        
        PyObject* op_type_obj = PyTuple_GetItem(item, 0);
        PyObject* operand_obj = PyTuple_GetItem(item, 1);
        PyObject* version_obj = PyTuple_GetItem(item, 2);

        ops[i].op_type = (int)PyLong_AsLong(op_type_obj);
        ops[i].version = (int)PyLong_AsLong(version_obj);

        if (ops[i].op_type == 4) { // SML (Scalar Multiply)
            ops[i].operand.scalar = PyFloat_AsDouble(operand_obj);
        } else {
            ops[i].operand.mat = PyCapsule_GetPointer(operand_obj, "hjortMatrixWrapper.Matrix");
        }

        if (PyErr_Occurred()) {
            free(ops);
            return NULL;
        }
    }
    
    Matrix* result = hjort_lazy_evaluate(root, ops, (int)num_ops, multithreaded, simplify_flag);
    free(ops);

    if (!result) {
        // Only set NoMemory if the backend didn't already set a specific errors
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "C backend evaluation failed");
        return NULL;
    }

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