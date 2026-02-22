import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."}
    )
    embedding_api_key: str = field(
        default=None,
        metadata={"help": "API key for the embedding model, if none, uses the same as LLM API key."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the LLM model, if none, uses OPENAI service directly."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the OpenAI embedding model, if none, uses OPENAI service directly."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    
    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )

    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )
    
    # Vector Database specific attributes
    vector_db_backend: Literal['parquet', 'faiss', 'milvus', 'chroma'] = field(
        default='parquet',
        metadata={"help": "Vector database backend to use. Options: parquet (default), faiss, milvus, chroma."}
    )
    vector_db_path: str = field(
        default=None,
        metadata={"help": "Path for local vector database storage. If None, uses save_dir/vdb."}
    )
    
    # FAISS specific attributes
    faiss_index_type: Literal['Flat', 'IVF', 'IVFFlat', 'IVFPQ', 'HNSW'] = field(
        default='Flat',
        metadata={"help": "FAISS index type. Flat is simplest, IVF types need training."}
    )
    faiss_nlist: int = field(
        default=100,
        metadata={"help": "Number of clusters for IVF index types in FAISS."}
    )
    faiss_nprobe: int = field(
        default=10,
        metadata={"help": "Number of clusters to probe during search in FAISS."}
    )
    faiss_use_gpu: bool = field(
        default=False,
        metadata={"help": "Whether to use GPU for FAISS operations."}
    )
    
    # Milvus specific attributes
    milvus_host: str = field(
        default='localhost',
        metadata={"help": "Milvus server host address."}
    )
    milvus_port: int = field(
        default=19530,
        metadata={"help": "Milvus server port."}
    )
    milvus_user: str = field(
        default='',
        metadata={"help": "Milvus username for authentication."}
    )
    milvus_password: str = field(
        default='',
        metadata={"help": "Milvus password for authentication."}
    )
    milvus_db_name: str = field(
        default='default',
        metadata={"help": "Milvus database name."}
    )
    milvus_collection_name: str = field(
        default=None,
        metadata={"help": "Milvus collection name. If None, uses foxhipporag_{namespace}."}
    )
    milvus_index_type: Literal['FLAT', 'IVF_FLAT', 'IVF_PQ', 'HNSW', 'ANNOY'] = field(
        default='IVF_FLAT',
        metadata={"help": "Milvus index type."}
    )
    milvus_metric_type: Literal['L2', 'IP', 'COSINE'] = field(
        default='COSINE',
        metadata={"help": "Distance metric type for Milvus."}
    )
    milvus_nlist: int = field(
        default=1024,
        metadata={"help": "Number of clusters for IVF index in Milvus."}
    )
    milvus_nprobe: int = field(
        default=16,
        metadata={"help": "Number of clusters to probe during search in Milvus."}
    )
    
    # Chroma specific attributes
    chroma_persistent: bool = field(
        default=True,
        metadata={"help": "Whether to use persistent storage for Chroma."}
    )
    chroma_distance_metric: Literal['cosine', 'l2', 'ip'] = field(
        default='cosine',
        metadata={"help": "Distance metric for Chroma vector search."}
    )
    
    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')
    
    
    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )
    
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )
    
    
    
    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    
    
    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    
    # Performance optimization attributes
    llm_parallel_workers: int = field(
        default=32,
        metadata={"help": "Number of parallel workers for LLM inference."}
    )
    retrieval_parallel_workers: int = field(
        default=8,
        metadata={"help": "Number of parallel workers for retrieval."}
    )
    embedding_parallel_workers: int = field(
        default=8,
        metadata={"help": "Number of parallel workers for embedding encoding."}
    )
    embedding_max_batch_size: int = field(
        default=256,
        metadata={"help": "Maximum batch size for embedding encoding."}
    )
    enable_parallel_qa: bool = field(
        default=True,
        metadata={"help": "Enable parallel QA inference."}
    )
    enable_parallel_retrieval: bool = field(
        default=True,
        metadata={"help": "Enable parallel retrieval."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation', 
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )
    
    
    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
        
        # 设置默认的向量数据库路径
        if self.vector_db_path is None:
            self.vector_db_path = os.path.join(self.save_dir, 'vdb')
            logger.debug(f"Initializing vector_db_path to be {self.vector_db_path}")
