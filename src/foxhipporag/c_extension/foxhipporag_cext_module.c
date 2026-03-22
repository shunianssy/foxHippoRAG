/**
 * foxhipporag C extension - Python binding module
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "foxhipporag_cext.h"

static PyObject* py_cosine_similarity(PyObject* self, PyObject* args) {
    PyArrayObject *query_obj, *matrix_obj;
    
    if (!PyArg_ParseTuple(args, "O!O!", 
                          &PyArray_Type, &query_obj,
                          &PyArray_Type, &matrix_obj)) {
        return NULL;
    }
    
    PyArrayObject *query = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)query_obj, NPY_FLOAT32, 1, 1);
    PyArrayObject *matrix = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)matrix_obj, NPY_FLOAT32, 2, 2);
    
    if (!query || !matrix) {
        Py_XDECREF(query);
        Py_XDECREF(matrix);
        PyErr_SetString(PyExc_ValueError, "Input must be numeric arrays");
        return NULL;
    }
    
    size_t dim = PyArray_DIM(query, 0);
    size_t n = PyArray_DIM(matrix, 0);
    size_t matrix_dim = PyArray_DIM(matrix, 1);
    
    if (dim != matrix_dim) {
        Py_DECREF(query);
        Py_DECREF(matrix);
        PyErr_SetString(PyExc_ValueError, "Query vector and matrix dimensions do not match");
        return NULL;
    }
    
    npy_intp dims[1] = {(npy_intp)n};
    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    
    if (!result) {
        Py_DECREF(query);
        Py_DECREF(matrix);
        return NULL;
    }
    
    int ret = cosine_similarity_single(
        (float*)PyArray_DATA(query),
        (float*)PyArray_DATA(matrix),
        (float*)PyArray_DATA(result),
        n, dim
    );
    
    Py_DECREF(query);
    Py_DECREF(matrix);
    
    if (ret != 0) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_RuntimeError, "Cosine similarity calculation failed");
        return NULL;
    }
    
    return (PyObject*)result;
}

static PyObject* py_cosine_similarity_batch(PyObject* self, PyObject* args) {
    PyArrayObject *queries_obj, *keys_obj;
    
    if (!PyArg_ParseTuple(args, "O!O!", 
                          &PyArray_Type, &queries_obj,
                          &PyArray_Type, &keys_obj)) {
        return NULL;
    }
    
    PyArrayObject *queries = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)queries_obj, NPY_FLOAT32, 2, 2);
    PyArrayObject *keys = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)keys_obj, NPY_FLOAT32, 2, 2);
    
    if (!queries || !keys) {
        Py_XDECREF(queries);
        Py_XDECREF(keys);
        PyErr_SetString(PyExc_ValueError, "Input must be 2D numeric arrays");
        return NULL;
    }
    
    size_t m = PyArray_DIM(queries, 0);
    size_t dim = PyArray_DIM(queries, 1);
    size_t n = PyArray_DIM(keys, 0);
    size_t key_dim = PyArray_DIM(keys, 1);
    
    if (dim != key_dim) {
        Py_DECREF(queries);
        Py_DECREF(keys);
        PyErr_SetString(PyExc_ValueError, "Query vectors and key vectors dimensions do not match");
        return NULL;
    }
    
    npy_intp dims[2] = {(npy_intp)m, (npy_intp)n};
    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    
    if (!result) {
        Py_DECREF(queries);
        Py_DECREF(keys);
        return NULL;
    }
    
    int ret = cosine_similarity_batch(
        (float*)PyArray_DATA(queries),
        (float*)PyArray_DATA(keys),
        (float*)PyArray_DATA(result),
        m, n, dim
    );
    
    Py_DECREF(queries);
    Py_DECREF(keys);
    
    if (ret != 0) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_RuntimeError, "Batch cosine similarity calculation failed");
        return NULL;
    }
    
    return (PyObject*)result;
}

static PyObject* py_top_k_indices(PyObject* self, PyObject* args) {
    PyArrayObject *scores_obj;
    int k;
    
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &scores_obj, &k)) {
        return NULL;
    }
    
    PyArrayObject *scores = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)scores_obj, NPY_FLOAT32, 1, 1);
    
    if (!scores) {
        PyErr_SetString(PyExc_ValueError, "Input must be 1D numeric array");
        return NULL;
    }
    
    size_t n = PyArray_DIM(scores, 0);
    k = (k > (int)n) ? (int)n : k;
    
    npy_intp dims[1] = {k};
    PyArrayObject *indices = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT64);
    PyArrayObject *top_scores = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    
    if (!indices || !top_scores) {
        Py_XDECREF(indices);
        Py_XDECREF(top_scores);
        Py_DECREF(scores);
        return NULL;
    }
    
    int ret = top_k_indices_1d(
        (float*)PyArray_DATA(scores),
        (int64_t*)PyArray_DATA(indices),
        (float*)PyArray_DATA(top_scores),
        n, k
    );
    
    Py_DECREF(scores);
    
    if (ret != 0) {
        Py_DECREF(indices);
        Py_DECREF(top_scores);
        PyErr_SetString(PyExc_RuntimeError, "Top-K selection failed");
        return NULL;
    }
    
    return Py_BuildValue("(OO)", indices, top_scores);
}

static PyObject* py_top_k_indices_2d(PyObject* self, PyObject* args) {
    PyArrayObject *scores_obj;
    int k;
    
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &scores_obj, &k)) {
        return NULL;
    }
    
    PyArrayObject *scores = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)scores_obj, NPY_FLOAT32, 2, 2);
    
    if (!scores) {
        PyErr_SetString(PyExc_ValueError, "Input must be 2D numeric array");
        return NULL;
    }
    
    size_t m = PyArray_DIM(scores, 0);
    size_t n = PyArray_DIM(scores, 1);
    k = (k > (int)n) ? (int)n : k;
    
    npy_intp dims[2] = {(npy_intp)m, k};
    PyArrayObject *indices = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT64);
    PyArrayObject *top_scores = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    
    if (!indices || !top_scores) {
        Py_XDECREF(indices);
        Py_XDECREF(top_scores);
        Py_DECREF(scores);
        return NULL;
    }
    
    int ret = top_k_indices_2d(
        (float*)PyArray_DATA(scores),
        (int64_t*)PyArray_DATA(indices),
        (float*)PyArray_DATA(top_scores),
        m, n, k
    );
    
    Py_DECREF(scores);
    
    if (ret != 0) {
        Py_DECREF(indices);
        Py_DECREF(top_scores);
        PyErr_SetString(PyExc_RuntimeError, "Batch Top-K selection failed");
        return NULL;
    }
    
    return Py_BuildValue("(OO)", indices, top_scores);
}

static PyObject* py_l2_normalize(PyObject* self, PyObject* args) {
    PyArrayObject *vector_obj;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &vector_obj)) {
        return NULL;
    }
    
    PyArrayObject *vector = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)vector_obj, NPY_FLOAT32, 1, 2);
    
    if (!vector) {
        PyErr_SetString(PyExc_ValueError, "Input must be numeric array");
        return NULL;
    }
    
    int ndim = PyArray_NDIM(vector);
    PyObject *result;
    
    if (ndim == 1) {
        size_t dim = PyArray_DIM(vector, 0);
        npy_intp dims[1] = {(npy_intp)dim};
        result = PyArray_SimpleNew(1, dims, NPY_FLOAT32);
        
        if (!result) {
            Py_DECREF(vector);
            return NULL;
        }
        
        l2_normalize(
            (float*)PyArray_DATA(vector),
            (float*)PyArray_DATA((PyArrayObject*)result),
            dim
        );
    } else {
        size_t m = PyArray_DIM(vector, 0);
        size_t n = PyArray_DIM(vector, 1);
        npy_intp dims[2] = {(npy_intp)m, (npy_intp)n};
        result = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
        
        if (!result) {
            Py_DECREF(vector);
            return NULL;
        }
        
        batch_l2_normalize(
            (float*)PyArray_DATA(vector),
            (float*)PyArray_DATA((PyArrayObject*)result),
            m, n
        );
    }
    
    Py_DECREF(vector);
    return result;
}

static PyObject* py_min_max_normalize(PyObject* self, PyObject* args) {
    PyArrayObject *values_obj;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &values_obj)) {
        return NULL;
    }
    
    PyArrayObject *values = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)values_obj, NPY_FLOAT32, 1, 1);
    
    if (!values) {
        PyErr_SetString(PyExc_ValueError, "Input must be 1D numeric array");
        return NULL;
    }
    
    size_t n = PyArray_DIM(values, 0);
    npy_intp dims[1] = {(npy_intp)n};
    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    
    if (!result) {
        Py_DECREF(values);
        return NULL;
    }
    
    min_max_normalize(
        (float*)PyArray_DATA(values),
        (float*)PyArray_DATA(result),
        n
    );
    
    Py_DECREF(values);
    return (PyObject*)result;
}

static PyObject* py_knn_search(PyObject* self, PyObject* args) {
    PyArrayObject *query_obj, *index_vectors_obj;
    int k;
    
    if (!PyArg_ParseTuple(args, "O!O!i", 
                          &PyArray_Type, &query_obj,
                          &PyArray_Type, &index_vectors_obj, &k)) {
        return NULL;
    }
    
    PyArrayObject *query = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)query_obj, NPY_FLOAT32, 1, 1);
    PyArrayObject *index_vectors = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)index_vectors_obj, NPY_FLOAT32, 2, 2);
    
    if (!query || !index_vectors) {
        Py_XDECREF(query);
        Py_XDECREF(index_vectors);
        PyErr_SetString(PyExc_ValueError, "Input must be numeric arrays");
        return NULL;
    }
    
    size_t dim = PyArray_DIM(query, 0);
    size_t n = PyArray_DIM(index_vectors, 0);
    size_t index_dim = PyArray_DIM(index_vectors, 1);
    
    if (dim != index_dim) {
        Py_DECREF(query);
        Py_DECREF(index_vectors);
        PyErr_SetString(PyExc_ValueError, "Query vector and index vectors dimensions do not match");
        return NULL;
    }
    
    k = (k > (int)n) ? (int)n : k;
    
    npy_intp dims[1] = {k};
    PyArrayObject *indices = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_INT64);
    PyArrayObject *scores = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    
    if (!indices || !scores) {
        Py_XDECREF(indices);
        Py_XDECREF(scores);
        Py_DECREF(query);
        Py_DECREF(index_vectors);
        return NULL;
    }
    
    int ret = knn_search(
        (float*)PyArray_DATA(query),
        (float*)PyArray_DATA(index_vectors),
        (int64_t*)PyArray_DATA(indices),
        (float*)PyArray_DATA(scores),
        n, dim, k
    );
    
    Py_DECREF(query);
    Py_DECREF(index_vectors);
    
    if (ret != 0) {
        Py_DECREF(indices);
        Py_DECREF(scores);
        PyErr_SetString(PyExc_RuntimeError, "KNN search failed");
        return NULL;
    }
    
    return Py_BuildValue("(OO)", indices, scores);
}

static PyObject* py_knn_search_batch(PyObject* self, PyObject* args) {
    PyArrayObject *queries_obj, *index_vectors_obj;
    int k;
    
    if (!PyArg_ParseTuple(args, "O!O!i", 
                          &PyArray_Type, &queries_obj,
                          &PyArray_Type, &index_vectors_obj, &k)) {
        return NULL;
    }
    
    PyArrayObject *queries = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)queries_obj, NPY_FLOAT32, 2, 2);
    PyArrayObject *index_vectors = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)index_vectors_obj, NPY_FLOAT32, 2, 2);
    
    if (!queries || !index_vectors) {
        Py_XDECREF(queries);
        Py_XDECREF(index_vectors);
        PyErr_SetString(PyExc_ValueError, "Input must be 2D numeric arrays");
        return NULL;
    }
    
    size_t m = PyArray_DIM(queries, 0);
    size_t dim = PyArray_DIM(queries, 1);
    size_t n = PyArray_DIM(index_vectors, 0);
    size_t index_dim = PyArray_DIM(index_vectors, 1);
    
    if (dim != index_dim) {
        Py_DECREF(queries);
        Py_DECREF(index_vectors);
        PyErr_SetString(PyExc_ValueError, "Query vectors and index vectors dimensions do not match");
        return NULL;
    }
    
    k = (k > (int)n) ? (int)n : k;
    
    npy_intp dims[2] = {(npy_intp)m, k};
    PyArrayObject *indices = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_INT64);
    PyArrayObject *scores = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    
    if (!indices || !scores) {
        Py_XDECREF(indices);
        Py_XDECREF(scores);
        Py_DECREF(queries);
        Py_DECREF(index_vectors);
        return NULL;
    }
    
    int ret = knn_search_batch(
        (float*)PyArray_DATA(queries),
        (float*)PyArray_DATA(index_vectors),
        (int64_t*)PyArray_DATA(indices),
        (float*)PyArray_DATA(scores),
        m, n, dim, k
    );
    
    Py_DECREF(queries);
    Py_DECREF(index_vectors);
    
    if (ret != 0) {
        Py_DECREF(indices);
        Py_DECREF(scores);
        PyErr_SetString(PyExc_RuntimeError, "Batch KNN search failed");
        return NULL;
    }
    
    return Py_BuildValue("(OO)", indices, scores);
}

static PyObject* py_fuse_scores(PyObject* self, PyObject* args) {
    PyArrayObject *scores1_obj, *scores2_obj;
    float weight1, weight2;
    
    if (!PyArg_ParseTuple(args, "O!O!ff", 
                          &PyArray_Type, &scores1_obj,
                          &PyArray_Type, &scores2_obj,
                          &weight1, &weight2)) {
        return NULL;
    }
    
    PyArrayObject *scores1 = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)scores1_obj, NPY_FLOAT32, 1, 1);
    PyArrayObject *scores2 = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)scores2_obj, NPY_FLOAT32, 1, 1);
    
    if (!scores1 || !scores2) {
        Py_XDECREF(scores1);
        Py_XDECREF(scores2);
        PyErr_SetString(PyExc_ValueError, "Input must be 1D numeric arrays");
        return NULL;
    }
    
    size_t n1 = PyArray_DIM(scores1, 0);
    size_t n2 = PyArray_DIM(scores2, 0);
    
    if (n1 != n2) {
        Py_DECREF(scores1);
        Py_DECREF(scores2);
        PyErr_SetString(PyExc_ValueError, "Score arrays must have the same length");
        return NULL;
    }
    
    npy_intp dims[1] = {(npy_intp)n1};
    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    
    if (!result) {
        Py_DECREF(scores1);
        Py_DECREF(scores2);
        return NULL;
    }
    
    fuse_scores(
        (float*)PyArray_DATA(scores1),
        (float*)PyArray_DATA(scores2),
        (float*)PyArray_DATA(result),
        weight1, weight2, n1
    );
    
    Py_DECREF(scores1);
    Py_DECREF(scores2);
    return (PyObject*)result;
}

static PyObject* py_multiplicative_fuse(PyObject* self, PyObject* args) {
    PyArrayObject *scores1_obj, *scores2_obj;
    float alpha = 0.5f;
    
    if (!PyArg_ParseTuple(args, "O!O!|f", 
                          &PyArray_Type, &scores1_obj,
                          &PyArray_Type, &scores2_obj,
                          &alpha)) {
        return NULL;
    }
    
    PyArrayObject *scores1 = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)scores1_obj, NPY_FLOAT32, 1, 1);
    PyArrayObject *scores2 = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)scores2_obj, NPY_FLOAT32, 1, 1);
    
    if (!scores1 || !scores2) {
        Py_XDECREF(scores1);
        Py_XDECREF(scores2);
        PyErr_SetString(PyExc_ValueError, "Input must be 1D numeric arrays");
        return NULL;
    }
    
    size_t n1 = PyArray_DIM(scores1, 0);
    size_t n2 = PyArray_DIM(scores2, 0);
    
    if (n1 != n2) {
        Py_DECREF(scores1);
        Py_DECREF(scores2);
        PyErr_SetString(PyExc_ValueError, "Score arrays must have the same length");
        return NULL;
    }
    
    npy_intp dims[1] = {(npy_intp)n1};
    PyArrayObject *result = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    
    if (!result) {
        Py_DECREF(scores1);
        Py_DECREF(scores2);
        return NULL;
    }
    
    multiplicative_fuse(
        (float*)PyArray_DATA(scores1),
        (float*)PyArray_DATA(scores2),
        (float*)PyArray_DATA(result),
        alpha, n1
    );
    
    Py_DECREF(scores1);
    Py_DECREF(scores2);
    return (PyObject*)result;
}

static PyObject* py_get_version(PyObject* self, PyObject* args) {
    return PyUnicode_FromString(get_version());
}

static PyObject* py_has_simd_support(PyObject* self, PyObject* args) {
    return PyBool_FromLong(has_simd_support());
}

static PyMethodDef foxhipporag_cext_methods[] = {
    {"cosine_similarity", py_cosine_similarity, METH_VARARGS, "Calculate cosine similarity"},
    {"cosine_similarity_batch", py_cosine_similarity_batch, METH_VARARGS, "Batch cosine similarity"},
    {"top_k_indices", py_top_k_indices, METH_VARARGS, "Select Top-K indices"},
    {"top_k_indices_2d", py_top_k_indices_2d, METH_VARARGS, "Batch Top-K selection"},
    {"l2_normalize", py_l2_normalize, METH_VARARGS, "L2 normalize vector"},
    {"min_max_normalize", py_min_max_normalize, METH_VARARGS, "Min-Max normalize"},
    {"knn_search", py_knn_search, METH_VARARGS, "KNN search"},
    {"knn_search_batch", py_knn_search_batch, METH_VARARGS, "Batch KNN search"},
    {"fuse_scores", py_fuse_scores, METH_VARARGS, "Fuse scores with weights"},
    {"multiplicative_fuse", py_multiplicative_fuse, METH_VARARGS, "Multiplicative fuse scores"},
    {"get_version", py_get_version, METH_NOARGS, "Get version info"},
    {"has_simd_support", py_has_simd_support, METH_NOARGS, "Check SIMD support"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef foxhipporag_cext_module = {
    PyModuleDef_HEAD_INIT,
    "foxhipporag_cext",
    "foxhipporag C extension module",
    -1,
    foxhipporag_cext_methods
};

PyMODINIT_FUNC PyInit_foxhipporag_cext(void) {
    import_array();
    return PyModule_Create(&foxhipporag_cext_module);
}
