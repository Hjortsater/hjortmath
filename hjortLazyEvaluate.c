#include <Python.h>
#include <stdlib.h>

#include "hjortLazyEvaluate.h"


/*
    Lazy evaluate function queues operations in the backend to minimise malloc calls
    and increase accuracy by performing algebraic simplifications.
*/

Matrix* hjort_lazy_evaluate(Matrix* root, PyObject* ops_list, int multithreaded)
{

    /* clone root as starting point */
    Matrix* res = matrix_clone(root);

    if (!res)
        return NULL;

    Py_ssize_t num_ops = PyList_Size(ops_list);

    for (Py_ssize_t i = 0; i < num_ops; i++) {

        PyObject* item = PyList_GetItem(ops_list, i);

        int op_type = (int)PyLong_AsLong(PyTuple_GetItem(item, 0));
        PyObject* operand_capsule = PyTuple_GetItem(item, 1);

        Matrix* operand =
            PyCapsule_GetPointer(operand_capsule, "hjortMatrixWrapper.Matrix");

        if (!operand)
            return NULL;

        switch (op_type)
        {

            case 0: /* ADD */
                break;

            case 1: /* SUB */
                break;

            case 2: /* RML */
                break;

            case 3: /* LML */
                break;

            case 4: /* DIV */
                break;

            default:
                return NULL;
        }
    }

    return res;
}