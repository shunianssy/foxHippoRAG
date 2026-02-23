"""
算法优化模块

包含多种高级算法优化：
1. 优化的向量检索算法（KNN优化、FAISS优化）
2. 图算法优化（近似PPR、图剪枝）
3. 检索融合策略优化
4. 查询优化（查询重写、扩展）
5. 自适应剪枝和动态调整
"""

import numpy as np
import logging
import time
import hashlib
from typing import List, Tuple, Dict, Any, Optional, Set, Callable
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

logger = logging.getLogger(__name__)


class OptimizedVectorRetrieval:
    """优化的向量检索算法
    
    主要优化：
    1. 分层KNN检索 - 先粗排再精排
    2. 乘积量化优化
    3. 局部敏感哈希（LSH）近似检索
    4. 批量检索优化
    """
    
    def __init__(
        self,
        index_type: str = 'hierarchical',
        coarse_k: int = 100,
        fine_k: int = 10,
        use_lsh: bool = False,
        num_hash_tables: int = 8,
        num_hash_bits: int = 16
    ):
        """
        初始化优化的向量检索器
        
        Args:
            index_type: 索引类型 ['hierarchical', 'flat', 'lsh']
            coarse_k: 粗排阶段检索数量
            fine_k: 精排阶段检索数量
            use_lsh: 是否使用局部敏感哈希
            num_hash_tables: LSH哈希表数量
            num_hash_bits: LSH哈希位数
        """
        self.index_type = index_type
        self.coarse_k = coarse_k
        self.fine_k = fine_k
        self.use_lsh = use_lsh
        self.num_hash_tables = num_hash_tables
        self.num_hash_bits = num_hash_bits
        
        self.key_vectors = None
        self.key_ids = None
        self.lsh_tables = []
        self.lsh_projections = []
        self._is_indexed = False
    
    def build_index(self, key_ids: List[str], key_vectors: np.ndarray) -> None:
        """
        构建检索索引
        
        Args:
            key_ids: 键ID列表
            key_vectors: 键向量矩阵 [n_samples, n_dim]
        """
        logger.info(f"Building optimized vector index for {len(key_ids)} vectors")
        
        self.key_ids = key_ids
        self.key_vectors = key_vectors.astype(np.float32)
        
        if self.use_lsh:
            self._build_lsh_index()
        
        self._is_indexed = True
        logger.info("Vector index built successfully")
    
    def _build_lsh_index(self) -> None:
        """构建局部敏感哈希索引"""
        logger.info(f"Building LSH index with {self.num_hash_tables} tables")
        
        n_dim = self.key_vectors.shape[1]
        
        for table_idx in range(self.num_hash_tables):
            random_projections = np.random.randn(n_dim, self.num_hash_bits)
            self.lsh_projections.append(random_projections)
            
            hashes = np.sign(np.dot(self.key_vectors, random_projections))
            hash_table = defaultdict(list)
            
            for vec_idx, h in enumerate(hashes):
                hash_key = tuple(h.astype(int))
                hash_table[hash_key].append(vec_idx)
            
            self.lsh_tables.append(hash_table)
    
    def retrieve(
        self,
        query_ids: List[str],
        query_vectors: np.ndarray,
        k: int = 10
    ) -> Dict[str, Tuple[List[str], List[float]]]:
        """
        批量检索
        
        Args:
            query_ids: 查询ID列表
            query_vectors: 查询向量矩阵 [n_queries, n_dim]
            k: 要检索的数量
            
        Returns:
            查询ID到(键ID列表, 分数列表)的映射
        """
        if not self._is_indexed:
            raise ValueError("Index not built. Call build_index first.")
        
        logger.info(f"Retrieving for {len(query_ids)} queries with k={k}")
        
        results = {}
        
        for q_idx, (q_id, q_vec) in enumerate(zip(query_ids, query_vectors)):
            if self.index_type == 'hierarchical':
                retrieved_ids, scores = self._hierarchical_retrieve(q_vec, k)
            elif self.use_lsh:
                retrieved_ids, scores = self._lsh_retrieve(q_vec, k)
            else:
                retrieved_ids, scores = self._flat_retrieve(q_vec, k)
            
            results[q_id] = (
                [self.key_ids[i] for i in retrieved_ids],
                scores.tolist()
            )
        
        return results
    
    def _flat_retrieve(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """暴力检索"""
        scores = np.dot(self.key_vectors, query_vector)
        sorted_indices = np.argsort(scores)[::-1][:k]
        return sorted_indices, scores[sorted_indices]
    
    def _hierarchical_retrieve(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """分层检索：先粗排再精排"""
        coarse_k = min(self.coarse_k, len(self.key_vectors))
        fine_k = max(self.fine_k, k)
        
        coarse_scores = np.dot(self.key_vectors, query_vector)
        coarse_indices = np.argpartition(coarse_scores, -coarse_k)[-coarse_k:]
        
        fine_vectors = self.key_vectors[coarse_indices]
        fine_scores = np.dot(fine_vectors, query_vector)
        
        fine_sorted_idx = np.argsort(fine_scores)[::-1][:fine_k]
        final_indices = coarse_indices[fine_sorted_idx]
        
        return final_indices[:k], fine_scores[fine_sorted_idx][:k]
    
    def _lsh_retrieve(self, query_vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """LSH近似检索"""
        candidates = set()
        
        for table_idx in range(self.num_hash_tables):
            projections = self.lsh_projections[table_idx]
            hash_key = tuple(np.sign(np.dot(query_vector, projections)).astype(int))
            table_candidates = self.lsh_tables[table_idx].get(hash_key, [])
            candidates.update(table_candidates)
        
        if not candidates:
            return self._flat_retrieve(query_vector, k)
        
        candidate_indices = np.array(list(candidates))
        candidate_vectors = self.key_vectors[candidate_indices]
        scores = np.dot(candidate_vectors, query_vector)
        
        sorted_idx = np.argsort(scores)[::-1][:k]
        return candidate_indices[sorted_idx], scores[sorted_idx]


class OptimizedGraphPPR:
    """优化的个性化PageRank算法
    
    主要优化：
    1. 近似PPR算法 - 使用幂迭代的早期终止
    2. 图剪枝 - 移除低权重边
    3. 节点重要性预计算
    4. 批处理PPR计算
    """
    
    def __init__(
        self,
        use_approximate: bool = True,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        prune_threshold: float = 0.01,
        use_cache: bool = True
    ):
        """
        初始化优化的PPR计算器
        
        Args:
            use_approximate: 是否使用近似算法
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
            prune_threshold: 边权重剪枝阈值
            use_cache: 是否使用缓存
        """
        self.use_approximate = use_approximate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.prune_threshold = prune_threshold
        self.use_cache = use_cache
        
        self.adjacency_matrix = None
        self.node_indices = None
        self.node_importance = None
        self.pruned_edges = None
        self._cache = {}
    
    def build_graph(
        self,
        node_names: List[str],
        edges: List[Tuple[str, str, float]]
    ) -> None:
        """
        构建图数据结构
        
        Args:
            node_names: 节点名称列表
            edges: 边列表 [(source, target, weight)]
        """
        logger.info(f"Building optimized graph with {len(node_names)} nodes and {len(edges)} edges")
        
        self.node_indices = {name: idx for idx, name in enumerate(node_names)}
        n_nodes = len(node_names)
        
        adjacency = [[] for _ in range(n_nodes)]
        self.pruned_edges = []
        
        for source, target, weight in edges:
            if weight < self.prune_threshold:
                self.pruned_edges.append((source, target, weight))
                continue
            
            if source in self.node_indices and target in self.node_indices:
                s_idx = self.node_indices[source]
                t_idx = self.node_indices[target]
                adjacency[s_idx].append((t_idx, weight))
        
        self.adjacency_matrix = adjacency
        self._precompute_node_importance()
        
        logger.info(f"Graph built. Pruned {len(self.pruned_edges)} low-weight edges")
    
    def _precompute_node_importance(self) -> None:
        """预计算节点重要性"""
        if self.adjacency_matrix is None:
            return
        
        n_nodes = len(self.adjacency_matrix)
        degrees = np.array([len(edges) for edges in self.adjacency_matrix])
        
        self.node_importance = degrees / (np.sum(degrees) + 1e-8)
    
    def compute_ppr_batch(
        self,
        reset_probs_list: List[np.ndarray],
        damping: float = 0.5,
        passage_node_indices: Optional[List[int]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        批量计算PPR
        
        Args:
            reset_probs_list: 重置概率列表
            damping: 阻尼因子
            passage_node_indices: 段落节点索引（可选，用于只返回段落得分）
            
        Returns:
            PPR结果列表，每个元素为(排序后的索引, 得分)
        """
        logger.info(f"Computing PPR for {len(reset_probs_list)} queries in batch")
        
        results = []
        
        for reset_probs in reset_probs_list:
            cache_key = self._compute_cache_key(reset_probs, damping)
            
            if self.use_cache and cache_key in self._cache:
                results.append(self._cache[cache_key])
                continue
            
            ppr_result = self._compute_single_ppr(reset_probs, damping, passage_node_indices)
            
            if self.use_cache:
                self._cache[cache_key] = ppr_result
            
            results.append(ppr_result)
        
        return results
    
    def _compute_single_ppr(
        self,
        reset_probs: np.ndarray,
        damping: float,
        passage_node_indices: Optional[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算单个PPR"""
        n_nodes = len(self.adjacency_matrix) if self.adjacency_matrix else len(reset_probs)
        
        if self.use_approximate:
            return self._approximate_ppr(reset_probs, damping, passage_node_indices)
        else:
            return self._exact_ppr(reset_probs, damping, passage_node_indices)
    
    def _approximate_ppr(
        self,
        reset_probs: np.ndarray,
        damping: float,
        passage_node_indices: Optional[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """近似PPR：幂迭代 + 早期终止"""
        n_nodes = len(self.adjacency_matrix)
        
        ppr = reset_probs.copy()
        prev_ppr = np.zeros_like(ppr)
        
        for iteration in range(self.max_iterations):
            delta = np.max(np.abs(ppr - prev_ppr))
            
            if delta < self.tolerance:
                logger.debug(f"PPR converged after {iteration} iterations")
                break
            
            prev_ppr = ppr.copy()
            
            new_ppr = (1 - damping) * reset_probs
            
            for node_idx in range(n_nodes):
                neighbors = self.adjacency_matrix[node_idx]
                if not neighbors:
                    continue
                
                total_weight = sum(w for _, w in neighbors)
                if total_weight == 0:
                    continue
                
                for neighbor_idx, weight in neighbors:
                    new_ppr[neighbor_idx] += damping * ppr[node_idx] * (weight / total_weight)
            
            ppr = new_ppr
        
        return self._sort_ppr_results(ppr, passage_node_indices)
    
    def _exact_ppr(
        self,
        reset_probs: np.ndarray,
        damping: float,
        passage_node_indices: Optional[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """精确PPR（用于小图）"""
        n_nodes = len(self.adjacency_matrix)
        
        transition_matrix = np.zeros((n_nodes, n_nodes))
        
        for node_idx in range(n_nodes):
            neighbors = self.adjacency_matrix[node_idx]
            if not neighbors:
                transition_matrix[node_idx, node_idx] = 1.0
                continue
            
            total_weight = sum(w for _, w in neighbors)
            for neighbor_idx, weight in neighbors:
                transition_matrix[node_idx, neighbor_idx] = weight / total_weight
        
        identity = np.eye(n_nodes)
        A = identity - damping * transition_matrix
        b = (1 - damping) * reset_probs
        
        ppr = np.linalg.solve(A, b)
        
        return self._sort_ppr_results(ppr, passage_node_indices)
    
    def _sort_ppr_results(
        self,
        ppr: np.ndarray,
        passage_node_indices: Optional[List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """排序PPR结果"""
        if passage_node_indices is not None:
            ppr_scores = ppr[passage_node_indices]
            sorted_indices = np.argsort(ppr_scores)[::-1]
            final_indices = np.array(passage_node_indices)[sorted_indices]
            final_scores = ppr_scores[sorted_indices]
        else:
            sorted_indices = np.argsort(ppr)[::-1]
            final_indices = sorted_indices
            final_scores = ppr[sorted_indices]
        
        return final_indices, final_scores
    
    def _compute_cache_key(self, reset_probs: np.ndarray, damping: float) -> str:
        """计算缓存键"""
        content = f"{damping}:{np.array2string(reset_probs, precision=4)}"
        return hashlib.md5(content.encode()).hexdigest()


class AdaptiveRetrievalFusion:
    """自适应检索融合策略
    
    主要优化：
    1. 动态权重调整 - 根据检索质量自适应调整DPR和图检索的权重
    2. 置信度评分 - 评估每种检索方法的置信度
    3. 结果重排序 - 融合后再排序
    4. 早期终止 - 当置信度足够高时提前终止
    """
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        max_confidence: float = 0.95,
        enable_early_termination: bool = True,
        fusion_method: str = 'weighted'
    ):
        """
        初始化自适应融合器
        
        Args:
            min_confidence: 最小置信度阈值
            max_confidence: 最大置信度阈值
            enable_early_termination: 是否启用早期终止
            fusion_method: 融合方法 ['weighted', 'reciprocal_rank', 'borda']
        """
        self.min_confidence = min_confidence
        self.max_confidence = max_confidence
        self.enable_early_termination = enable_early_termination
        self.fusion_method = fusion_method
        
        self.history = []
    
    def fuse(
        self,
        dpr_results: Tuple[List[str], List[float]],
        graph_results: Optional[Tuple[List[str], List[float]]] = None,
        query: Optional[str] = None
    ) -> Tuple[List[str], List[float]]:
        """
        融合检索结果
        
        Args:
            dpr_results: DPR检索结果 (doc_ids, scores)
            graph_results: 图检索结果 (doc_ids, scores)，可选
            query: 查询文本，可选
            
        Returns:
            融合后的结果 (doc_ids, scores)
        """
        if graph_results is None:
            return dpr_results
        
        dpr_ids, dpr_scores = dpr_results
        graph_ids, graph_scores = graph_results
        
        dpr_confidence = self._calculate_confidence(dpr_scores)
        graph_confidence = self._calculate_confidence(graph_scores)
        
        should_terminate_early = (
            self.enable_early_termination and 
            (dpr_confidence > self.max_confidence or graph_confidence > self.max_confidence)
        )
        
        if should_terminate_early:
            if dpr_confidence > graph_confidence:
                return dpr_results
            else:
                return graph_results
        
        weights = self._calculate_adaptive_weights(dpr_confidence, graph_confidence)
        
        if self.fusion_method == 'weighted':
            fused_ids, fused_scores = self._weighted_fusion(
                dpr_ids, dpr_scores, graph_ids, graph_scores, weights
            )
        elif self.fusion_method == 'reciprocal_rank':
            fused_ids, fused_scores = self._reciprocal_rank_fusion(
                dpr_ids, graph_ids, weights
            )
        elif self.fusion_method == 'borda':
            fused_ids, fused_scores = self._borda_fusion(
                dpr_ids, graph_ids, weights
            )
        else:
            fused_ids, fused_scores = self._weighted_fusion(
                dpr_ids, dpr_scores, graph_ids, graph_scores, weights
            )
        
        self._record_history(query, dpr_confidence, graph_confidence, weights)
        
        return fused_ids, fused_scores
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """计算检索结果的置信度"""
        if len(scores) < 2:
            return 0.5
        
        scores_arr = np.array(scores)
        
        gap = scores_arr[0] - scores_arr[1]
        normalized_gap = gap / (scores_arr[0] + 1e-8)
        
        variance = np.var(scores_arr[:min(10, len(scores_arr))])
        
        confidence = 0.5 + 0.5 * normalized_gap
        confidence = min(max(confidence, 0.0), 1.0)
        
        return confidence
    
    def _calculate_adaptive_weights(self, dpr_confidence: float, graph_confidence: float) -> Tuple[float, float]:
        """计算自适应权重"""
        total_confidence = dpr_confidence + graph_confidence
        
        if total_confidence < 1e-8:
            return 0.5, 0.5
        
        dpr_weight = dpr_confidence / total_confidence
        graph_weight = graph_confidence / total_confidence
        
        dpr_weight = max(self.min_confidence * dpr_weight, 0.1)
        graph_weight = max(self.min_confidence * graph_weight, 0.1)
        
        total = dpr_weight + graph_weight
        dpr_weight /= total
        graph_weight /= total
        
        return dpr_weight, graph_weight
    
    def _weighted_fusion(
        self,
        dpr_ids: List[str],
        dpr_scores: List[float],
        graph_ids: List[str],
        graph_scores: List[float],
        weights: Tuple[float, float]
    ) -> Tuple[List[str], List[float]]:
        """加权融合"""
        dpr_weight, graph_weight = weights
        
        dpr_score_map = {doc_id: score for doc_id, score in zip(dpr_ids, dpr_scores)}
        graph_score_map = {doc_id: score for doc_id, score in zip(graph_ids, graph_scores)}
        
        all_doc_ids = set(dpr_ids) | set(graph_ids)
        
        fused_scores = []
        for doc_id in all_doc_ids:
            dpr_s = dpr_score_map.get(doc_id, 0.0)
            graph_s = graph_score_map.get(doc_id, 0.0)
            fused_s = dpr_weight * dpr_s + graph_weight * graph_s
            fused_scores.append((doc_id, fused_s))
        
        fused_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [x[0] for x in fused_scores], [x[1] for x in fused_scores]
    
    def _reciprocal_rank_fusion(
        self,
        dpr_ids: List[str],
        graph_ids: List[str],
        weights: Tuple[float, float]
    ) -> Tuple[List[str], List[float]]:
        """倒数排名融合"""
        dpr_weight, graph_weight = weights
        
        rank_map = defaultdict(float)
        
        for rank, doc_id in enumerate(dpr_ids):
            rank_map[doc_id] += dpr_weight / (rank + 1)
        
        for rank, doc_id in enumerate(graph_ids):
            rank_map[doc_id] += graph_weight / (rank + 1)
        
        sorted_docs = sorted(rank_map.items(), key=lambda x: x[1], reverse=True)
        
        return [doc_id for doc_id, score in sorted_docs], [score for doc_id, score in sorted_docs]
    
    def _borda_fusion(
        self,
        dpr_ids: List[str],
        graph_ids: List[str],
        weights: Tuple[float, float]
    ) -> Tuple[List[str], List[float]]:
        """Borda计数融合"""
        dpr_weight, graph_weight = weights
        n_docs = len(dpr_ids) + len(graph_ids)
        
        borda_map = defaultdict(float)
        
        for rank, doc_id in enumerate(dpr_ids):
            borda_map[doc_id] += dpr_weight * (n_docs - rank)
        
        for rank, doc_id in enumerate(graph_ids):
            borda_map[doc_id] += graph_weight * (n_docs - rank)
        
        sorted_docs = sorted(borda_map.items(), key=lambda x: x[1], reverse=True)
        
        return [doc_id for doc_id, score in sorted_docs], [score for doc_id, score in sorted_docs]
    
    def _record_history(
        self,
        query: Optional[str],
        dpr_confidence: float,
        graph_confidence: float,
        weights: Tuple[float, float]
    ):
        """记录历史用于后续分析"""
        self.history.append({
            'query': query,
            'dpr_confidence': dpr_confidence,
            'graph_confidence': graph_confidence,
            'dpr_weight': weights[0],
            'graph_weight': weights[1],
            'timestamp': time.time()
        })


class QueryOptimizer:
    """查询优化器
    
    主要优化：
    1. 查询重写 - 优化查询表述
    2. 查询扩展 - 增加相关术语
    3. 查询分解 - 复杂查询分解为子查询
    4. 查询缓存 - 缓存相似查询的结果
    """
    
    def __init__(
        self,
        enable_rewriting: bool = True,
        enable_expansion: bool = True,
        enable_decomposition: bool = False,
        cache_similarity_threshold: float = 0.9
    ):
        """
        初始化查询优化器
        
        Args:
            enable_rewriting: 是否启用查询重写
            enable_expansion: 是否启用查询扩展
            enable_decomposition: 是否启用查询分解
            cache_similarity_threshold: 缓存相似度阈值
        """
        self.enable_rewriting = enable_rewriting
        self.enable_expansion = enable_expansion
        self.enable_decomposition = enable_decomposition
        self.cache_similarity_threshold = cache_similarity_threshold
        
        self.query_cache = {}
        self.stop_words = self._load_stop_words()
    
    def optimize(self, query: str) -> Dict[str, Any]:
        """
        优化查询
        
        Args:
            query: 原始查询
            
        Returns:
            优化结果字典
        """
        result = {
            'original_query': query,
            'optimized_queries': [],
            'from_cache': False,
            'cache_hit': None
        }
        
        cached_result = self._check_cache(query)
        if cached_result is not None:
            result['from_cache'] = True
            result['cache_hit'] = cached_result
            result['optimized_queries'] = cached_result['optimized_queries']
            return result
        
        optimized_queries = []
        
        if self.enable_rewriting:
            rewritten = self._rewrite_query(query)
            if rewritten != query:
                optimized_queries.append(rewritten)
        
        if self.enable_expansion:
            expanded = self._expand_query(query)
            if expanded:
                optimized_queries.extend(expanded)
        
        if self.enable_decomposition:
            decomposed = self._decompose_query(query)
            if decomposed:
                optimized_queries.extend(decomposed)
        
        if not optimized_queries:
            optimized_queries = [query]
        
        result['optimized_queries'] = optimized_queries
        
        self._add_to_cache(query, result)
        
        return result
    
    def _load_stop_words(self) -> Set[str]:
        """加载停用词"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'shall', 'should', 'can', 'could', 'may',
            'might', 'must', 'what', 'which', 'who', 'whom', 'whose',
            'where', 'when', 'why', 'how', 'this', 'that', 'these', 'those'
        }
    
    def _check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """检查缓存"""
        if query in self.query_cache:
            return self.query_cache[query]
        return None
    
    def _add_to_cache(self, query: str, result: Dict[str, Any]) -> None:
        """添加到缓存"""
        self.query_cache[query] = {
            'optimized_queries': result['optimized_queries'],
            'timestamp': time.time()
        }
    
    def _rewrite_query(self, query: str) -> str:
        """重写查询"""
        words = query.lower().split()
        
        filtered_words = [w for w in words if w not in self.stop_words]
        
        if len(filtered_words) < len(words) * 0.5:
            return query
        
        return ' '.join(filtered_words)
    
    def _expand_query(self, query: str) -> List[str]:
        """扩展查询"""
        expanded = []
        
        words = query.split()
        
        if len(words) > 3:
            for i in range(len(words)):
                expanded_phrase = ' '.join(words[:i+1])
                if expanded_phrase != query:
                    expanded.append(expanded_phrase)
        
        return expanded[:3]
    
    def _decompose_query(self, query: str) -> List[str]:
        """分解查询"""
        decomposed = []
        
        conjunctions = [' and ', ' or ', ' but ', ' however ']
        for conj in conjunctions:
            if conj in query.lower():
                parts = query.lower().split(conj)
                decomposed.extend([p.strip() for p in parts if p.strip()])
                break
        
        return decomposed[:3]


class BatchOptimizer:
    """批量优化器
    
    主要优化：
    1. 智能批处理大小调整
    2. 任务调度优化
    3. 负载均衡
    4. 失败重试策略
    """
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        max_batch_size: int = 256,
        min_batch_size: int = 8,
        max_workers: int = 8,
        max_retries: int = 3
    ):
        """
        初始化批量优化器
        
        Args:
            initial_batch_size: 初始批处理大小
            max_batch_size: 最大批处理大小
            min_batch_size: 最小批处理大小
            max_workers: 最大工作线程数
            max_retries: 最大重试次数
        """
        self.batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        self.success_times = []
        self.failure_count = 0
    
    def process_batch(
        self,
        items: List[Any],
        process_fn: Callable[[List[Any]], List[Any]],
        desc: str = "Processing"
    ) -> List[Any]:
        """
        批量处理项目
        
        Args:
            items: 待处理项目列表
            process_fn: 批处理函数
            desc: 进度条描述
            
        Returns:
            处理结果列表
        """
        from tqdm import tqdm
        
        if not items:
            return []
        
        batches = self._create_batches(items)
        
        results = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            start_idx = 0
            
            for batch in batches:
                future = executor.submit(self._process_with_retry, batch, process_fn)
                futures[future] = (start_idx, len(batch))
                start_idx += len(batch)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                start_idx, batch_size = futures[future]
                
                try:
                    batch_results = future.result()
                    results[start_idx:start_idx+batch_size] = batch_results
                    self._record_success()
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    self._record_failure()
        
        return results
    
    def _create_batches(self, items: List[Any]) -> List[List[Any]]:
        """创建批次"""
        batches = []
        for i in range(0, len(items), self.batch_size):
            batches.append(items[i:i+self.batch_size])
        return batches
    
    def _process_with_retry(
        self,
        batch: List[Any],
        process_fn: Callable[[List[Any]], List[Any]]
    ) -> List[Any]:
        """带重试的处理"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return process_fn(batch)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 0.5
                    time.sleep(wait_time)
        
        raise last_exception
    
    def _record_success(self) -> None:
        """记录成功"""
        if len(self.success_times) > 100:
            self.success_times.pop(0)
        
        self.success_times.append(time.time())
        
        if len(self.success_times) >= 5:
            self._adjust_batch_size(increase=True)
    
    def _record_failure(self) -> None:
        """记录失败"""
        self.failure_count += 1
        self._adjust_batch_size(increase=False)
    
    def _adjust_batch_size(self, increase: bool) -> None:
        """动态调整批处理大小"""
        if increase:
            self.batch_size = min(self.batch_size * 1.1, self.max_batch_size)
        else:
            self.batch_size = max(self.batch_size * 0.9, self.min_batch_size)
        
        self.batch_size = int(self.batch_size)


class AlgorithmOptimizationPipeline:
    """算法优化管道
    
    整合所有优化算法的管道
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化优化管道
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        self.vector_retrieval = OptimizedVectorRetrieval(
            index_type=self.config.get('index_type', 'hierarchical'),
            coarse_k=self.config.get('coarse_k', 100),
            fine_k=self.config.get('fine_k', 10),
            use_lsh=self.config.get('use_lsh', False)
        )
        
        self.graph_ppr = OptimizedGraphPPR(
            use_approximate=self.config.get('use_approximate_ppr', True),
            max_iterations=self.config.get('ppr_max_iterations', 50),
            tolerance=self.config.get('ppr_tolerance', 1e-4),
            prune_threshold=self.config.get('graph_prune_threshold', 0.01)
        )
        
        self.retrieval_fusion = AdaptiveRetrievalFusion(
            min_confidence=self.config.get('min_confidence', 0.5),
            max_confidence=self.config.get('max_confidence', 0.95),
            fusion_method=self.config.get('fusion_method', 'weighted')
        )
        
        self.query_optimizer = QueryOptimizer(
            enable_rewriting=self.config.get('enable_query_rewriting', True),
            enable_expansion=self.config.get('enable_query_expansion', True)
        )
        
        self.batch_optimizer = BatchOptimizer(
            initial_batch_size=self.config.get('initial_batch_size', 32),
            max_workers=self.config.get('max_workers', 8)
        )
        
        logger.info("Algorithm optimization pipeline initialized")
    
    def get_optimizer(self, name: str) -> Any:
        """获取指定的优化器"""
        optimizers = {
            'vector_retrieval': self.vector_retrieval,
            'graph_ppr': self.graph_ppr,
            'retrieval_fusion': self.retrieval_fusion,
            'query_optimizer': self.query_optimizer,
            'batch_optimizer': self.batch_optimizer
        }
        return optimizers.get(name)
