# foxHippoRAG 开发文档

## 项目概述

foxHippoRAG 是一个基于图的知识检索增强生成(RAG)框架，从原始 HippoRAG 项目重命名而来。本文档记录了框架的重命名过程、Windows兼容性修复以及AI管家演示程序的开发。

---

## 一、框架重命名

### 1.1 重命名内容

| 原名称 | 新名称 |
|--------|--------|
| hipporag | foxHippoRAG |
| HippoRAG | foxHippoRAG |
| HippoRAG.py | foxHippoRAG.py |

### 1.2 修改的文件

#### 核心文件
- `setup.py` - 更新包名、版本号、作者和GitHub URL
- `src/foxhipporag/__init__.py` - 更新导入语句
- `src/foxhipporag/foxHippoRAG.py` - 重命名类和文件
- `src/foxhipporag/rerank.py` - 修复变量名引用
- `src/foxhipporag/prompts/prompt_template_manager.py` - 更新模块路径

#### 演示文件
- `demo.py`, `demo_openai.py`, `demo_local.py`, `demo_azure.py`, `demo_bedrock.py`
- `main.py`, `main_azure.py`, `main_dpr.py`
- `tests_openai.py`, `tests_local.py`, `tests_azure.py`, `test_transformers.py`

#### 文档文件
- `README.md` - 更新所有引用和安装说明
- `CONTRIBUTING.md` - 更新贡献指南

### 1.3 分发配置

```python
# setup.py 关键配置
setuptools.setup(
    name="foxHippoRAG",
    version="1.0.0",
    author="shunianssy",
    url="https://github.com/shunianssy/foxHippoRAG",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.10",
)
```

---

## 二、Windows兼容性修复

### 2.1 多进程问题修复

**问题**: Windows上`multiprocessing.Manager()`在模块导入时初始化会导致错误。

**修复文件**: `src/foxhipporag/embedding_model/base.py`

**修复方案**: 延迟初始化

```python
class EmbeddingCache:
    """A multiprocessing-safe global cache for storing embeddings."""
    
    _manager = None
    _cache = None
    _lock = threading.Lock()
    
    @classmethod
    def _ensure_initialized(cls):
        """延迟初始化，避免Windows多进程问题"""
        if cls._manager is None:
            cls._manager = multiprocessing.Manager()
            cls._cache = cls._manager.dict()

    @classmethod
    def get(cls, content):
        cls._ensure_initialized()
        return cls._cache.get(content)
    
    # ... 其他方法类似
```

### 2.2 vllm不可用问题修复

**问题**: vllm库依赖Unix系统的`resource`模块，在Windows上不可用。

**修复文件**: `src/foxhipporag/foxHippoRAG.py`

**修复方案**: 条件导入

```python
# vllm在Windows上不支持，使用条件导入
try:
    from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
    VLLM_AVAILABLE = True
except ImportError:
    VLLMOfflineOpenIE = None
    VLLM_AVAILABLE = False

try:
    from .information_extraction.openie_transformers_offline import TransformersOfflineOpenIE
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TransformersOfflineOpenIE = None
    TRANSFORMERS_AVAILABLE = False
```

### 2.3 嵌入模型支持扩展

**修复文件**: `src/foxhipporag/embedding_model/__init__.py`

**修复方案**: 添加对更多嵌入模型的支持

```python
def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    # ... 原有判断 ...
    
    # 支持更多OpenAI兼容的嵌入模型
    elif "bge" in embedding_model_name.lower():
        return OpenAIEmbeddingModel
    elif "embedding" in embedding_model_name.lower():
        return OpenAIEmbeddingModel
    # 默认使用OpenAI兼容模式
    else:
        logger.warning(f"Unknown embedding model name: {embedding_model_name}, using OpenAI compatible mode")
        return OpenAIEmbeddingModel
```

### 2.4 独立嵌入API密钥支持

**新增配置**: `src/foxhipporag/utils/config_utils.py`

```python
embedding_api_key: str = field(
    default=None,
    metadata={"help": "API key for the embedding model, if none, uses the same as LLM API key."}
)
```

**修复文件**: `src/foxhipporag/embedding_model/OpenAI.py`

```python
if self.global_config.azure_embedding_endpoint is None:
    # 使用embedding_api_key，如果没有则使用环境变量OPENAI_API_KEY
    api_key = self.global_config.embedding_api_key or os.getenv("OPENAI_API_KEY")
    self.client = OpenAI(
        api_key=api_key,
        base_url=self.global_config.embedding_base_url
    )
```

---

## 三、AI管家演示程序

### 3.1 创建的文件

| 文件名 | 描述 |
|--------|------|
| `demo_ai_assistant_hipporag.py` | 使用foxHippoRAG框架的AI管家演示 |
| `demo_ai_assistant_rag.py` | 独立知识库实现的AI管家演示 |
| `demo_ai_assistant_final.py` | 简化版AI管家演示 |
| `.env` | 环境变量配置文件 |
| `.env.example` | 环境变量配置示例 |

### 3.2 AI管家功能

1. **智能信息提取** - 使用LLM从用户输入中提取三元组格式的事实
2. **知识图谱存储** - 使用foxHippoRAG框架存储和检索知识
3. **主动回忆** - 对话时自动检索相关历史信息
4. **持久化存储** - 知识图谱自动保存到磁盘

### 3.3 配置说明

```env
# .env 配置文件

# LLM API配置
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE_URL=https://api.stepfun.com/v1
OPENAI_MODEL=step-3.5-flash

# 嵌入模型配置（可独立配置）
EMBEDDING_API_KEY=your_embedding_api_key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
```

### 3.4 使用方法

```bash
# 安装依赖
pip install openai python-dotenv

# 运行演示
python demo_ai_assistant_hipporag.py

# 命令
# 输入 '退出' 结束对话
# 输入 '知识' 查看知识库内容
```

---

## 四、项目结构

```
HippoRAG-sn/
├── src/
│   └── foxhipporag/
│       ├── __init__.py
│       ├── foxHippoRAG.py          # 主类
│       ├── StandardRAG.py
│       ├── embedding_store.py
│       ├── rerank.py               # 已修复变量名
│       ├── embedding_model/
│       │   ├── __init__.py         # 已扩展模型支持
│       │   ├── base.py             # 已修复多进程问题
│       │   ├── OpenAI.py           # 已添加独立API密钥支持
│       │   └── ...
│       ├── information_extraction/
│       ├── llm/
│       ├── prompts/
│       ├── evaluation/
│       └── utils/
│           └── config_utils.py     # 已添加embedding_api_key
├── demo_ai_assistant_hipporag.py   # AI管家演示
├── demo_ai_assistant_rag.py
├── demo_ai_assistant_final.py
├── .env                            # 环境变量配置
├── .env.example
├── setup.py                        # 包配置
└── README.md
```

---

## 五、已知问题

### 5.1 OpenIE三元组提取

某些LLM模型（如step-3.5-flash）可能无法正确理解OpenIE的JSON格式提示，导致三元组提取为空。建议使用GPT-4或更强大的模型以获得最佳效果。

### 5.2 Windows限制

- vllm离线模式不可用
- Transformers离线模式可能需要额外配置

---

## 六、版本信息

- **版本**: 1.0.0
- **作者**: shunianssy
- **GitHub**: https://github.com/shunianssy/foxHippoRAG
- **Python要求**: >=3.10

---

## 七、更新日志

### 2026-02-21

1. 完成框架重命名（HippoRAG → foxHippoRAG）
2. 修复Windows多进程兼容性问题
3. 添加vllm条件导入支持
4. 扩展嵌入模型支持（bge等）
5. 添加独立嵌入API密钥配置
6. 创建AI管家演示程序
7. 创建源码分发包
