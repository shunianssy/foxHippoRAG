/**
 * foxhipporag C extension - Core computation functions implementation
 */

#include "foxhipporag_cext.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define FOXHIPPORAG_VERSION "1.0.0"
#define EPSILON 1e-10f

int cosine_similarity_single(
    const float* query,
    const float* matrix,
    float* result,
    size_t n,
    size_t dim
) {
    size_t i, j;
    float query_norm = 0.0f;
    
    if (!query || !matrix || !result || n == 0 || dim == 0) {
        return -1;
    }
    
    for (j = 0; j < dim; j++) {
        query_norm += query[j] * query[j];
    }
    query_norm = sqrtf(query_norm);
    
    if (query_norm < EPSILON) {
        memset(result, 0, n * sizeof(float));
        return 0;
    }
    
    for (i = 0; i < n; i++) {
        const float* vec = matrix + i * dim;
        float dot = 0.0f;
        float norm = 0.0f;
        
        for (j = 0; j < dim; j++) {
            dot += query[j] * vec[j];
            norm += vec[j] * vec[j];
        }
        norm = sqrtf(norm);
        
        result[i] = (norm < EPSILON) ? 0.0f : (dot / (query_norm * norm));
    }
    
    return 0;
}

int cosine_similarity_batch(
    const float* queries,
    const float* keys,
    float* result,
    size_t m,
    size_t n,
    size_t dim
) {
    size_t i, j, k;
    float* key_norms;
    
    if (!queries || !keys || !result || m == 0 || n == 0 || dim == 0) {
        return -1;
    }
    
    key_norms = (float*)malloc(n * sizeof(float));
    if (!key_norms) return -1;
    
    for (i = 0; i < n; i++) {
        float norm = 0.0f;
        const float* key = keys + i * dim;
        for (j = 0; j < dim; j++) {
            norm += key[j] * key[j];
        }
        key_norms[i] = sqrtf(norm);
    }
    
    for (i = 0; i < m; i++) {
        const float* query = queries + i * dim;
        float* result_row = result + i * n;
        float query_norm = 0.0f;
        
        for (j = 0; j < dim; j++) {
            query_norm += query[j] * query[j];
        }
        query_norm = sqrtf(query_norm);
        
        if (query_norm < EPSILON) {
            memset(result_row, 0, n * sizeof(float));
            continue;
        }
        
        for (j = 0; j < n; j++) {
            const float* key = keys + j * dim;
            float dot = 0.0f;
            
            for (k = 0; k < dim; k++) {
                dot += query[k] * key[k];
            }
            
            result_row[j] = (key_norms[j] < EPSILON) ? 0.0f : (dot / (query_norm * key_norms[j]));
        }
    }
    
    free(key_norms);
    return 0;
}

static size_t partition(float* arr, int64_t* idx, size_t left, size_t right) {
    float pivot = arr[right];
    size_t i = left;
    size_t j;
    
    for (j = left; j < right; j++) {
        if (arr[j] >= pivot) {
            float temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            
            {
                int64_t temp_idx = idx[i];
                idx[i] = idx[j];
                idx[j] = temp_idx;
            }
            i++;
        }
    }
    
    {
        float temp = arr[i];
        arr[i] = arr[right];
        arr[right] = temp;
        
        {
            int64_t temp_idx = idx[i];
            idx[i] = idx[right];
            idx[right] = temp_idx;
        }
    }
    
    return i;
}

static void quickselect(float* arr, int64_t* idx, size_t left, size_t right, size_t k) {
    if (left >= right) return;
    
    {
        size_t pivot_idx = partition(arr, idx, left, right);
        
        if (pivot_idx < k) {
            quickselect(arr, idx, pivot_idx + 1, right, k);
        } else if (pivot_idx > k) {
            quickselect(arr, idx, left, pivot_idx - 1, k);
        }
    }
}

int top_k_indices_1d(
    const float* scores,
    int64_t* indices,
    float* top_scores,
    size_t n,
    size_t k
) {
    size_t i, j;
    float* scores_copy;
    int64_t* idx_copy;
    
    if (!scores || !indices || !top_scores || n == 0 || k == 0) return -1;
    
    k = (k < n) ? k : n;
    
    scores_copy = (float*)malloc(n * sizeof(float));
    idx_copy = (int64_t*)malloc(n * sizeof(int64_t));
    
    if (!scores_copy || !idx_copy) {
        free(scores_copy);
        free(idx_copy);
        return -1;
    }
    
    memcpy(scores_copy, scores, n * sizeof(float));
    for (i = 0; i < n; i++) {
        idx_copy[i] = (int64_t)i;
    }
    
    quickselect(scores_copy, idx_copy, 0, n - 1, k - 1);
    
    for (i = 0; i < k; i++) {
        for (j = i + 1; j < k; j++) {
            if (scores_copy[i] < scores_copy[j]) {
                float temp = scores_copy[i];
                scores_copy[i] = scores_copy[j];
                scores_copy[j] = temp;
                
                {
                    int64_t temp_idx = idx_copy[i];
                    idx_copy[i] = idx_copy[j];
                    idx_copy[j] = temp_idx;
                }
            }
        }
    }
    
    memcpy(indices, idx_copy, k * sizeof(int64_t));
    memcpy(top_scores, scores_copy, k * sizeof(float));
    
    free(scores_copy);
    free(idx_copy);
    
    return 0;
}

int top_k_indices_2d(
    const float* scores,
    int64_t* indices,
    float* top_scores,
    size_t m,
    size_t n,
    size_t k
) {
    size_t i;
    
    if (!scores || !indices || !top_scores || m == 0 || n == 0 || k == 0) return -1;
    
    k = (k < n) ? k : n;
    
    for (i = 0; i < m; i++) {
        const float* row = scores + i * n;
        int64_t* row_indices = indices + i * k;
        float* row_scores = top_scores + i * k;
        
        top_k_indices_1d(row, row_indices, row_scores, n, k);
    }
    
    return 0;
}

int l2_normalize(
    const float* vector,
    float* result,
    size_t dim
) {
    size_t i;
    float norm = 0.0f;
    
    if (!vector || !result || dim == 0) return -1;
    
    for (i = 0; i < dim; i++) {
        norm += vector[i] * vector[i];
    }
    norm = sqrtf(norm);
    
    if (norm < EPSILON) {
        memset(result, 0, dim * sizeof(float));
    } else {
        for (i = 0; i < dim; i++) {
            result[i] = vector[i] / norm;
        }
    }
    
    return 0;
}

int batch_l2_normalize(
    const float* matrix,
    float* result,
    size_t m,
    size_t n
) {
    size_t i, j;
    
    if (!matrix || !result || m == 0 || n == 0) return -1;
    
    for (i = 0; i < m; i++) {
        const float* row = matrix + i * n;
        float* result_row = result + i * n;
        float norm = 0.0f;
        
        for (j = 0; j < n; j++) {
            norm += row[j] * row[j];
        }
        norm = sqrtf(norm);
        
        if (norm < EPSILON) {
            memset(result_row, 0, n * sizeof(float));
        } else {
            for (j = 0; j < n; j++) {
                result_row[j] = row[j] / norm;
            }
        }
    }
    
    return 0;
}

int min_max_normalize(
    const float* values,
    float* result,
    size_t n
) {
    size_t i;
    float min_val, max_val, range_val;
    
    if (!values || !result || n == 0) return -1;
    
    min_val = values[0];
    max_val = values[0];
    
    for (i = 1; i < n; i++) {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
    }
    
    range_val = max_val - min_val;
    
    if (range_val < EPSILON) {
        for (i = 0; i < n; i++) {
            result[i] = 1.0f;
        }
    } else {
        for (i = 0; i < n; i++) {
            result[i] = (values[i] - min_val) / range_val;
        }
    }
    
    return 0;
}

int knn_search(
    const float* query,
    const float* index_vectors,
    int64_t* indices,
    float* scores,
    size_t n,
    size_t dim,
    size_t k
) {
    float* similarities;
    int ret;
    
    if (!query || !index_vectors || !indices || !scores || n == 0 || dim == 0 || k == 0) {
        return -1;
    }
    
    k = (k < n) ? k : n;
    
    similarities = (float*)malloc(n * sizeof(float));
    if (!similarities) return -1;
    
    ret = cosine_similarity_single(query, index_vectors, similarities, n, dim);
    if (ret != 0) {
        free(similarities);
        return -1;
    }
    
    ret = top_k_indices_1d(similarities, indices, scores, n, k);
    
    free(similarities);
    return ret;
}

int knn_search_batch(
    const float* queries,
    const float* index_vectors,
    int64_t* indices,
    float* scores,
    size_t m,
    size_t n,
    size_t dim,
    size_t k
) {
    float* similarity_matrix;
    int ret;
    
    if (!queries || !index_vectors || !indices || !scores || m == 0 || n == 0 || dim == 0 || k == 0) {
        return -1;
    }
    
    k = (k < n) ? k : n;
    
    similarity_matrix = (float*)malloc(m * n * sizeof(float));
    if (!similarity_matrix) return -1;
    
    ret = cosine_similarity_batch(queries, index_vectors, similarity_matrix, m, n, dim);
    if (ret != 0) {
        free(similarity_matrix);
        return -1;
    }
    
    ret = top_k_indices_2d(similarity_matrix, indices, scores, m, n, k);
    
    free(similarity_matrix);
    return ret;
}

int fuse_scores(
    const float* scores1,
    const float* scores2,
    float* result,
    float weight1,
    float weight2,
    size_t n
) {
    size_t i;
    
    if (!scores1 || !scores2 || !result || n == 0) return -1;
    
    for (i = 0; i < n; i++) {
        result[i] = weight1 * scores1[i] + weight2 * scores2[i];
    }
    
    return 0;
}

int multiplicative_fuse(
    const float* scores1,
    const float* scores2,
    float* result,
    float alpha,
    size_t n
) {
    size_t i;
    
    if (!scores1 || !scores2 || !result || n == 0) return -1;
    
    for (i = 0; i < n; i++) {
        float multiplicative = scores1[i] * scores2[i];
        float weighted = alpha * scores1[i] + (1.0f - alpha) * scores2[i];
        result[i] = 0.3f * multiplicative + 0.7f * weighted;
    }
    
    return 0;
}

const char* get_version(void) {
    return FOXHIPPORAG_VERSION;
}

int has_simd_support(void) {
#ifdef __AVX2__
    return 1;
#else
    return 0;
#endif
}
