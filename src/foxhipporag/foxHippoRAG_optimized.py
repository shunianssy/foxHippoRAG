"""
优化版本的foxHippoRAG

主要优化：
1. 快速索引模式 - 跳过OpenIE，直接使用DPR
2. 增强嵌入缓存 - 预加载和智能缓存
3. 优化PPR计算 - 结果缓存和快速算法
4. 快速检索模式 - 简化检索流程
"""

import json
import os
import logging
import time
from typing import List, Set, Dict, Tuple
import numpy as np
from tqdm import tqdm
import re
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# 条件导入 igraph（快速模式不需要）
try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    ig = None
    IGRAPH_AVAILABLE = False

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .utils.misc_utils import (
    NerRawOutput,
    TripleRawOutput,
    QuerySolution,
    text_processing,
    reformat_openie_results,
    flatten_facts,
    min_max_normalize,
    compute_mdhash_id,
    extract_entity_nodes,
)
from .utils.config_utils import BaseConfig
from .utils.performance_utils import (
    PPRCache,
    EmbeddingCache,
)

logger = logging.getLogger(__name__)


class OptimizedfoxHippoRAG:
    """优化版本的foxHippoRAG
    
    性能优化：
    1. 快速索引模式 - 跳过OpenIE，直接使用DPR
    2. 增强嵌入缓存 - 预加载和智能缓存
    3. 优化PPR计算 - 结果缓存和快速算法
    4. 快速检索模式 - 简化检索流程
    """
    
    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """初始化优化版本的foxHippoRAG"""
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        # 覆盖配置
        if save_dir is not None:
            self.global_config.save_dir = save_dir
        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name
        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name
        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url
        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url
        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint
        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        # 创建工作目录
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        # 检查快速模式
        self.use_fast_index = getattr(self.global_config, 'use_fast_index', False)
        self.use_fast_retrieve = getattr(self.global_config, 'use_fast_retrieve', False)
        
        # 如果不是快速模式，检查 igraph 是否可用
        if not self.use_fast_index and not IGRAPH_AVAILABLE:
            raise ImportError("igraph is required for standard mode. Install it with: pip install igraph")

        # 初始化LLM
        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        # 初始化OpenIE（仅在非快速模式下）
        if not self.use_fast_index:
            if self.global_config.openie_mode == 'online':
                from .information_extraction import OpenIE
                self.openie = OpenIE(llm_model=self.llm_model)
            else:
                self.openie = None
        else:
            logger.info("使用快速索引模式，跳过OpenIE")
            self.openie = None

        # 初始化图（仅在非快速模式下）
        if not self.use_fast_index:
            self.graph = self.initialize_graph()
        else:
            self.graph = None

        # 初始化嵌入模型
        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        
        # 初始化嵌入存储
        self.chunk_embedding_store = EmbeddingStore(
            self.embedding_model,
            os.path.join(self.working_dir, "chunk_embeddings"),
            self.global_config.embedding_batch_size,
            'chunk',
            vector_db_backend=self.global_config.vector_db_backend,
            global_config=self.global_config
        )
        
        # 实体和事实存储（仅在非快速模式下需要）
        if not self.use_fast_index:
            self.entity_embedding_store = EmbeddingStore(
                self.embedding_model,
                os.path.join(self.working_dir, "entity_embeddings"),
                self.global_config.embedding_batch_size,
                'entity',
                vector_db_backend=self.global_config.vector_db_backend,
                global_config=self.global_config
            )
            self.fact_embedding_store = EmbeddingStore(
                self.embedding_model,
                os.path.join(self.working_dir, "fact_embeddings"),
                self.global_config.embedding_batch_size,
                'fact',
                vector_db_backend=self.global_config.vector_db_backend,
                global_config=self.global_config
            )
        else:
            self.entity_embedding_store = None
            self.fact_embedding_store = None

        # 初始化缓存
        self._init_caches()

        # 其他初始化
        self.openie_results_path = os.path.join(
            self.global_config.save_dir,
            f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json'
        )
        self.ready_to_retrieve = False
        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0
        self.ent_node_to_chunk_ids = None
        
        # 检索相关属性
        self.passage_node_keys = []
        self.passage_embeddings = None
        self.query_to_embedding = {'triple': {}, 'passage': {}}

    def _init_caches(self):
        """初始化缓存系统"""
        cache_dir = os.path.join(self.working_dir, "cache")
        
        # PPR结果缓存（仅在非快速模式下需要）
        if not self.use_fast_index:
            self.ppr_cache = PPRCache(
                cache_dir,
                max_memory_size=getattr(self.global_config, 'ppr_cache_size', 100)
            )
        else:
            self.ppr_cache = None
        
        # 嵌入向量缓存
        self.embedding_cache = EmbeddingCache(
            cache_dir,
            max_memory_size=getattr(self.global_config, 'embedding_cache_size', 1000)
        )
        
        logger.info("缓存系统初始化完成")

    def initialize_graph(self):
        """初始化图（仅在非快速模式下）"""
        if self.use_fast_index:
            return None
            
        self._graph_pickle_filename = os.path.join(self.working_dir, "graph.pickle")

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)
                logger.info(f"Loaded graph with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges")
                return preloaded_graph

        return ig.Graph(directed=self.global_config.is_directed_graph)

    def index(self, docs: List[str]):
        """索引文档（优化版本）
        
        性能优化：
        - 快速索引模式：跳过OpenIE，直接使用DPR
        - 批量嵌入编码
        - 增量保存
        """
        logger.info("Indexing Documents (Optimized)")

        if self.use_fast_index:
            # 快速索引模式：只做DPR，跳过OpenIE
            return self._index_fast(docs)
        else:
            # 标准索引模式
            return self._index_standard(docs)

    def _index_fast(self, docs: List[str]):
        """快速索引模式：只做DPR，跳过OpenIE"""
        logger.info("Using fast index mode (DPR only)")
        
        # 插入文档到chunk存储
        self.chunk_embedding_store.insert_strings(docs)
        
        # 标记为可检索（索引完成后应设置为True）
        self.ready_to_retrieve = True
        
        logger.info(f"Fast indexing completed for {len(docs)} documents")

    def _index_standard(self, docs: List[str]):
        """标准索引模式：使用OpenIE"""
        logger.info("Using standard index mode (with OpenIE)")
        
        if not IGRAPH_AVAILABLE:
            raise ImportError("igraph is required for standard mode. Install it with: pip install igraph")

        if self.global_config.openie_mode == 'offline':
            raise NotImplementedError("Offline mode not supported in optimized version")

        # 插入文档
        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        # 加载现有OpenIE结果
        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k: chunk_to_rows[k] for k in chunk_keys_to_process}

        # 执行OpenIE
        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        # 准备数据
        chunk_ids = list(chunk_to_rows.keys())
        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        # 编码实体和事实
        logger.info("Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info("Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        # 构建图
        logger.info("Constructing Graph")
        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}
        self.add_fact_edges(chunk_ids, chunk_triples)
        num_new_chunks = self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if num_new_chunks > 0:
            logger.info(f"Found {num_new_chunks} new chunks to save into graph.")
            self.add_synonymy_edges()
            self.augment_graph()
            self.save_igraph()

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """加载现有OpenIE结果"""
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info
            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """合并OpenIE结果"""
        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            try:
                chunk_openie_info = {
                    'idx': chunk_key,
                    'passage': passage,
                    'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                    'extracted_triples': triple_results_dict[chunk_key].triples
                }
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_openie_info = {
                    'idx': chunk_key,
                    'passage': passage,
                    'extracted_entities': [],
                    'extracted_triples': []
                }
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """保存OpenIE结果"""
        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """添加事实边"""
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info("Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            if chunk_key not in current_graph_nodes:
                for triple in triples:
                    triple = tuple(triple)

                    node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                    node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                    entities_in_chunk.add(node_key)
                    entities_in_chunk.add(node_2_key)

                for node in entities_in_chunk:
                    self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """添加段落边"""
        if "name" in self.graph.vs.attribute_names():
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        num_new_chunks = 0

        logger.info("Connecting passage nodes to phrase nodes.")

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):
            if chunk_key not in current_graph_nodes:
                for chunk_ent in chunk_triple_entities[idx]:
                    node_key = compute_mdhash_id(chunk_ent, prefix="entity-")
                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0
                num_new_chunks += 1

        return num_new_chunks

    def add_synonymy_edges(self):
        """添加同义词边"""
        from .utils.embed_utils import retrieve_knn
        
        logger.info("Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        query_node_key2knn_node_keys = retrieve_knn(
            query_ids=entity_node_keys,
            key_ids=entity_node_keys,
            query_vecs=entity_embs,
            key_vecs=entity_embs,
            k=self.global_config.synonymy_edge_topk,
            query_batch_size=self.global_config.synonymy_edge_query_batch_size,
            key_batch_size=self.global_config.synonymy_edge_key_batch_size
        )

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        self.node_to_node_stats[sim_edge] = score
                        num_nns += 1

    def augment_graph(self):
        """增强图"""
        self.add_new_nodes()
        self.add_new_edges()
        logger.info("Graph construction completed!")

    def add_new_nodes(self):
        """添加新节点"""
        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_row = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_row = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_row
        node_to_rows.update(passage_to_row)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def add_new_edges(self):
        """添加新边"""
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        
        seen_edges = set()
        
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]:
                continue
            
            edge_key = (edge[0], edge[1])
            
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({"weight": weight})

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        self.graph.add_edges(valid_edges, attributes=valid_weights)

    def save_igraph(self):
        """保存图"""
        logger.info(f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges")
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info("Saving graph completed!")

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """检索文档（优化版本）
        
        性能优化：
        - 快速检索模式：直接使用DPR
        - PPR结果缓存
        - 批量查询处理
        """
        retrieve_start_time = time.time()

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            from .evaluation.retrieval_eval import RetrievalRecall
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        # 检查是否使用快速检索模式
        if self.use_fast_retrieve or self.use_fast_index:
            # 快速检索模式：直接使用DPR
            logger.info(f"Using fast retrieve mode (DPR only) for {len(queries)} queries")
            for query in tqdm(queries, desc="Retrieving"):
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
                top_k_docs = [
                    self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
                    for idx in sorted_doc_ids[:num_to_retrieve]
                ]
                retrieval_results.append(
                    QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve])
                )
        else:
            # 标准检索模式：使用图检索
            logger.info(f"Using graph retrieve mode for {len(queries)} queries")
            all_query_fact_scores = []
            for query in queries:
                all_query_fact_scores.append(self.get_fact_scores(query))

            # 批量处理检索
            max_workers = getattr(self.global_config, 'retrieval_parallel_workers', 8)
            
            if len(queries) <= 4:
                for q_idx, (query, query_fact_scores) in enumerate(zip(queries, all_query_fact_scores)):
                    result = self._retrieve_single(query, query_fact_scores, num_to_retrieve)
                    retrieval_results.append(result)
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self._retrieve_single, query, query_fact_scores, num_to_retrieve): q_idx
                        for q_idx, (query, query_fact_scores) in enumerate(zip(queries, all_query_fact_scores))
                    }
                    
                    temp_results = [None] * len(queries)
                    for future in tqdm(as_completed(futures), total=len(futures), desc="Retrieving"):
                        q_idx = futures[future]
                        try:
                            temp_results[q_idx] = future.result(timeout=120)
                        except Exception as e:
                            logger.error(f"Retrieval failed for query {q_idx}: {e}")
                            temp_results[q_idx] = QuerySolution(
                                question=queries[q_idx],
                                docs=[],
                                doc_scores=[]
                            )
                    
                    retrieval_results = temp_results

        retrieve_end_time = time.time()
        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # 评估检索
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs,
                retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list
            )
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")
            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def _retrieve_single(self, query: str, query_fact_scores: np.ndarray, num_to_retrieve: int) -> QuerySolution:
        """处理单个查询的检索"""
        top_k_fact_indices, top_k_facts, _ = self.rerank_facts(query, query_fact_scores)

        if len(top_k_facts) == 0:
            logger.debug('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
        else:
            sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(
                query=query,
                link_top_k=self.global_config.linking_top_k,
                query_fact_scores=query_fact_scores,
                top_k_facts=top_k_facts,
                top_k_fact_indices=top_k_fact_indices,
                passage_node_weight=self.global_config.passage_node_weight
            )

        top_k_docs = [
            self.chunk_embedding_store.get_row(self.passage_node_keys[idx])["content"]
            for idx in sorted_doc_ids[:num_to_retrieve]
        ]

        return QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve])

    def prepare_retrieval_objects(self):
        """准备检索对象"""
        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids())

        # 快速模式只需要段落嵌入
        logger.info("Loading embeddings.")
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        # 标准模式需要额外的实体和事实嵌入
        if not self.use_fast_index:
            self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids())
            self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())

            # 创建节点映射
            try:
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
                self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys]
                self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys]
            except Exception as e:
                logger.error(f"Error creating node index mapping: {str(e)}")
                self.node_name_to_vertex_idx = {}
                self.entity_node_idxs = []
                self.passage_node_idxs = []

            self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
            self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

            # 加载OpenIE结果
            all_openie_info, _ = self.load_existing_openie([])
            self.proc_triples_to_docs = {}

            for doc in all_openie_info:
                triples = flatten_facts([doc['extracted_triples']])
                for triple in triples:
                    if len(triple) == 3:
                        proc_triple = tuple(text_processing(list(triple)))
                        self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        self.ready_to_retrieve = True

    def get_query_embeddings(self, queries: List[str]):
        """获取查询嵌入"""
        all_query_strings = []
        for query in queries:
            if query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            logger.info(f"Encoding {len(all_query_strings)} queries")
            
            # 对于OpenAI嵌入模型，不需要instruction参数
            query_embeddings_for_triple = self.embedding_model.batch_encode(
                all_query_strings,
                norm=True
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            query_embeddings_for_passage = self.embedding_model.batch_encode(
                all_query_strings,
                norm=True
            )
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """获取事实得分"""
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                query,
                norm=True
            )

        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T)
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """密集段落检索"""
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(
                query,
                norm=True
            )
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        passage_node_weight: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """基于事实实体的图搜索"""
        # 分配短语权重
        linking_score_map = {}
        phrase_scores = {}
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1
                    phrases_and_ids.add((phrase, phrase_id))

        valid_mask = number_of_occurs > 0
        phrase_weights[valid_mask] /= number_of_occurs[valid_mask]

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []
            phrase_scores[phrase].append(phrase_weights[phrase_id])

        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if link_top_k:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # 获取段落得分
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.passage_node_keys[dpr_sorted_doc_id]
            passage_dpr_score = normalized_dpr_sorted_scores[i]
            passage_node_id = self.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score * passage_node_weight
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = passage_dpr_score * passage_node_weight

        # 合并权重
        node_weights = phrase_weights + passage_weights

        # 运行PPR
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(node_weights, damping=self.global_config.damping)

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores

    def run_ppr(self, reset_prob: np.ndarray, damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """运行PPR（带缓存优化）"""
        if damping is None:
            damping = 0.5
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        
        # 计算图的哈希值
        graph_hash = self._compute_graph_hash()
        
        # 尝试从缓存获取
        if self.ppr_cache:
            cached_result = self.ppr_cache.get(reset_prob, damping, graph_hash)
            if cached_result is not None:
                logger.debug("PPR cache hit")
                return cached_result
        
        # 缓存未命中，执行PPR
        ppr_start = time.time()
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]
        
        ppr_end = time.time()
        self.ppr_time += (ppr_end - ppr_start)
        
        # 缓存结果
        if self.ppr_cache:
            self.ppr_cache.put(reset_prob, damping, graph_hash, sorted_doc_ids, sorted_doc_scores)
        
        return sorted_doc_ids, sorted_doc_scores

    def _compute_graph_hash(self) -> str:
        """计算图的哈希值"""
        graph_info = f"{self.graph.vcount()}_{self.graph.ecount()}"
        return hashlib.md5(graph_info.encode()).hexdigest()

    def rerank_facts(self, query: str, query_fact_scores: np.ndarray):
        """重排序事实"""
        from .rerank import DSPyFilter
        
        link_top_k = self.global_config.linking_top_k
        
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            if len(query_fact_scores) <= link_top_k:
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]
            
            rerank_filter = DSPyFilter(self)
            top_k_fact_indices, top_k_facts, _ = rerank_filter(
                query,
                candidate_facts,
                candidate_fact_indices,
                len_after_rerank=link_top_k
            )
            
            return top_k_fact_indices, top_k_facts, {}
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'error': str(e)}
