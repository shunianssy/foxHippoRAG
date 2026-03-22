/**
 * foxhipporag C extension - Core computation functions header file
 */

#ifndef FOXHIPPORAG_CEXT_H
#define FOXHIPPORAG_CEXT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Cosine Similarity Calculation */
int cosine_similarity_single(
    const float* query,
    const float* matrix,
    float* result,
    size_t n,
    size_t dim
);

int cosine_similarity_batch(
    const float* queries,
    const float* keys,
    float* result,
    size_t m,
    size_t n,
    size_t dim
);

/* Top-K Selection Algorithm */
int top_k_indices_1d(
    const float* scores,
    int64_t* indices,
    float* top_scores,
    size_t n,
    size_t k
);

int top_k_indices_2d(
    const float* scores,
    int64_t* indices,
    float* top_scores,
    size_t m,
    size_t n,
    size_t k
);

/* Vector Normalization */
int l2_normalize(
    const float* vector,
    float* result,
    size_t dim
);

int batch_l2_normalize(
    const float* matrix,
    float* result,
    size_t m,
    size_t n
);

int min_max_normalize(
    const float* values,
    float* result,
    size_t n
);

/* KNN Search */
int knn_search(
    const float* query,
    const float* index_vectors,
    int64_t* indices,
    float* scores,
    size_t n,
    size_t dim,
    size_t k
);

int knn_search_batch(
    const float* queries,
    const float* index_vectors,
    int64_t* indices,
    float* scores,
    size_t m,
    size_t n,
    size_t dim,
    size_t k
);

/* Score Fusion */
int fuse_scores(
    const float* scores1,
    const float* scores2,
    float* result,
    float weight1,
    float weight2,
    size_t n
);

int multiplicative_fuse(
    const float* scores1,
    const float* scores2,
    float* result,
    float alpha,
    size_t n
);

/* Utility Functions */
const char* get_version(void);
int has_simd_support(void);

#ifdef __cplusplus
}
#endif

#endif /* FOXHIPPORAG_CEXT_H */
