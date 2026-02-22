"""
AI管家演示 - 使用foxHippoRAG框架（性能优化版）

这个演示展示了如何使用foxHippoRAG框架构建一个智能AI管家：
1. 使用RAG框架进行知识存储和检索
2. 主动解析用户信息并存储到知识图谱
3. 对话时主动回忆之前的内容
4. 持久化存储（foxHippoRAG自动处理）

性能优化版本：
- 延迟初始化：按需加载组件，加快启动速度
- 并行初始化：多线程并行初始化独立组件
- 并行信息提取和检索
- 本地缓存优化
- 批量处理优化
- 内存管理优化
"""

import os
import logging
import multiprocessing
import json
import hashlib
import sqlite3
import threading
import time
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    multiprocessing.freeze_support()


class LRUCache:
    """线程安全的LRU缓存实现"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class SQLiteCache:
    """基于SQLite的持久化缓存"""
    
    def __init__(self, cache_dir: str, cache_name: str = "assistant_cache.sqlite"):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, cache_name)
        self.lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)")
            conn.commit()
            conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute("SELECT value FROM cache WHERE key = ?", (key,))
            row = c.fetchone()
            conn.close()
            if row:
                return json.loads(row[0])
            return None
    
    def put(self, key: str, value: Any) -> None:
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False))
            )
            conn.commit()
            conn.close()
    
    def clear_old(self, days: int = 30) -> int:
        """清理旧缓存"""
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute(
                "DELETE FROM cache WHERE created_at < datetime('now', ?)",
                (f'-{days} days',)
            )
            deleted = c.rowcount
            conn.commit()
            conn.close()
            return deleted


class AIAssistantOptimized:
    """AI管家类（性能优化版）
    
    性能优化：
    1. 延迟初始化：按需加载foxHippoRAG，加快启动速度
    2. 并行初始化：多线程并行初始化独立组件
    3. 并行信息提取和检索
    4. 多级缓存（内存LRU + SQLite持久化）
    5. 批量处理优化
    6. 内存管理优化
    """
    
    _hippo_init_lock = threading.Lock()
    _hippo_instance = None
    
    def __init__(self, save_dir: str = "outputs/ai_assistant_hippo_optimized"):
        """初始化AI管家（快速启动版本）
        
        采用延迟初始化策略，只初始化必要的轻量级组件，
        foxHippoRAG在第一次使用时才加载。
        
        Args:
            save_dir: 存储目录
        """
        init_start = time.time()
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AIAssistantOptimized')
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", self.api_key)
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            raise ValueError("请在.env文件中设置OPENAI_API_KEY")
        
        self.save_dir = save_dir
        
        self.config = None
        self._hippo = None
        self._hippo_initialized = False
        
        self.conversation_history = []
        
        self.memory_cache = LRUCache(max_size=500)
        self.persistent_cache = SQLiteCache(save_dir)
        
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        self.system_prompt = """你是一个智能AI管家。你的任务是：

1. **信息提取**：从用户输入中提取关键信息
2. **智能记忆**：记住用户告诉你的重要信息
3. **主动回忆**：在回答时，主动参考之前对话中用户告诉你的信息
4. **个性化回应**：根据用户的历史信息提供定制化的回答

请用友好、自然的语气与用户交流。"""
        
        self._init_openai_client()
        
        init_elapsed = time.time() - init_start
        self.logger.info(f"AI管家快速初始化完成！耗时: {init_elapsed:.2f}s")
        self.logger.info("foxHippoRAG将在首次使用时加载...")
    
    @property
    def hippo(self):
        """延迟初始化foxHippoRAG"""
        if not self._hippo_initialized:
            with self._hippo_init_lock:
                if not self._hippo_initialized:
                    self._init_hippo()
                    self._hippo_initialized = True
        return self._hippo
    
    def _init_hippo(self):
        """初始化foxHippoRAG（延迟加载）"""
        from src.foxhipporag import foxHippoRAG
        from src.foxhipporag.utils.config_utils import BaseConfig
        
        init_start = time.time()
        self.logger.info("正在初始化foxHippoRAG...")
        
        self.config = BaseConfig()
        self.config.save_dir = self.save_dir
        self.config.llm_name = self.model
        self.config.llm_base_url = self.base_url
        self.config.embedding_model_name = self.embedding_model
        self.config.embedding_base_url = self.embedding_base_url
        self.config.embedding_api_key = self.embedding_api_key
        self.config.openie_mode = "online"
        self.config.force_index_from_scratch = False
        self.config.llm_parallel_workers = 16
        self.config.retrieval_parallel_workers = 8
        
        self._hippo = foxHippoRAG(global_config=self.config)
        
        init_elapsed = time.time() - init_start
        self.logger.info(f"foxHippoRAG初始化完成！耗时: {init_elapsed:.2f}s")
    
    def preload_hippo(self):
        """预加载foxHippoRAG（可选，用于后台预热）"""
        if not self._hippo_initialized:
            future = self.executor.submit(self._init_hippo)
            return future
        return None
    
    def is_hippo_ready(self) -> bool:
        """检查foxHippoRAG是否已初始化"""
        return self._hippo_initialized
    
    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        from openai import OpenAI
        
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.openai_client = OpenAI(**client_kwargs)
    
    def _get_cache_key(self, text: str, prefix: str = "") -> str:
        """生成缓存键"""
        content = f"{prefix}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def index_user_info(self, info: str):
        """将用户信息索引到知识库"""
        self.logger.info(f"索引用户信息: {info}")
        self.hippo.index(docs=[info])
        self.memory_cache.clear()
    
    def index_user_info_batch(self, info_list: List[str]):
        """批量索引用户信息（性能优化）"""
        self.logger.info(f"批量索引 {len(info_list)} 条用户信息")
        self.hippo.index(docs=info_list)
        self.memory_cache.clear()
    
    def retrieve_relevant_info(self, query: str, top_k: int = 5) -> list:
        """检索相关信息（带缓存）"""
        cache_key = self._get_cache_key(query, f"retrieve_{top_k}")
        
        cached = self.memory_cache.get(cache_key)
        if cached is not None:
            self.logger.debug(f"缓存命中: {query}")
            return cached
        
        cached_persistent = self.persistent_cache.get(cache_key)
        if cached_persistent is not None:
            self.logger.debug(f"持久化缓存命中: {query}")
            self.memory_cache.put(cache_key, cached_persistent)
            return cached_persistent
        
        try:
            results = self.hippo.retrieve(queries=[query], num_to_retrieve=top_k)
            if results:
                docs = results[0].docs
                self.memory_cache.put(cache_key, docs)
                self.persistent_cache.put(cache_key, docs)
                return docs
            return []
        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            return []
    
    def retrieve_batch(self, queries: List[str], top_k: int = 5) -> List[List[str]]:
        """批量检索（并行优化）"""
        if not queries:
            return []
        
        self.logger.info(f"批量检索 {len(queries)} 个查询")
        
        uncached_queries = []
        uncached_indices = []
        results = [None] * len(queries)
        
        for i, query in enumerate(queries):
            cache_key = self._get_cache_key(query, f"retrieve_{top_k}")
            cached = self.memory_cache.get(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_queries.append(query)
                uncached_indices.append(i)
        
        if uncached_queries:
            try:
                batch_results = self.hippo.retrieve(queries=uncached_queries, num_to_retrieve=top_k)
                
                for i, (query, result) in enumerate(zip(uncached_queries, batch_results)):
                    docs = result.docs if result else []
                    original_idx = uncached_indices[i]
                    results[original_idx] = docs
                    
                    cache_key = self._get_cache_key(query, f"retrieve_{top_k}")
                    self.memory_cache.put(cache_key, docs)
                    
            except Exception as e:
                self.logger.error(f"批量检索失败: {e}")
                for i in uncached_indices:
                    results[i] = []
        
        return results
    
    def chat_completion(self, messages: list) -> str:
        """调用LLM生成回复"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"调用LLM失败: {e}")
            return f"抱歉，处理请求时出错: {str(e)}"
    
    def chat_completion_batch(self, batch_messages: List[list]) -> List[str]:
        """批量调用LLM（并行优化）"""
        if not batch_messages:
            return []
        
        self.logger.info(f"批量生成 {len(batch_messages)} 个回复")
        
        results = [None] * len(batch_messages)
        
        futures = {
            self.executor.submit(self.chat_completion, messages): i
            for i, messages in enumerate(batch_messages)
        }
        
        for future in as_completed(futures):
            i = futures[future]
            try:
                results[i] = future.result(timeout=60)
            except Exception as e:
                self.logger.error(f"批量生成回复失败 [{i}]: {e}")
                results[i] = f"抱歉，处理请求时出错: {str(e)}"
        
        return results
    
    def extract_and_store_info(self, user_input: str) -> list:
        """从用户输入中提取并存储信息"""
        cache_key = self._get_cache_key(user_input, "extract")
        
        cached = self.memory_cache.get(cache_key)
        if cached is not None:
            self.logger.debug(f"信息提取缓存命中")
            return cached
        
        try:
            extraction_prompt = f"""请从以下用户输入中提取关键信息，以JSON格式返回。

用户输入：{user_input}

提取规则：
1. 只提取有价值的信息，忽略问候语和一般性问题
2. 格式为JSON数组，每个元素是一个三元组：{{"subject": "主体", "predicate": "关系", "object": "客体"}}
3. 主体通常是"用户"或具体的人名/物名
4. 谓词描述关系（如"喜欢"、"名字是"、"职业是"等）
5. 客体是具体值

示例：
输入："我叫小明，喜欢打篮球"
输出：[{{"subject": "用户", "predicate": "名字是", "object": "小明"}}, {{"subject": "用户", "predicate": "喜欢", "object": "打篮球"}}]

输入："你好"
输出：[]

请直接返回JSON数组，不要有其他内容："""

            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个信息提取助手，只返回JSON格式的数据。"},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            extracted_info = json.loads(result_text)
            
            if extracted_info and isinstance(extracted_info, list):
                docs_to_index = []
                for info in extracted_info:
                    subject = info.get('subject', '')
                    predicate = info.get('predicate', '')
                    obj = info.get('object', '')
                    if subject and predicate and obj:
                        doc = f"{subject}{predicate}{obj}。"
                        docs_to_index.append(doc)
                        self.logger.info(f"存储信息: {doc}")
                
                if docs_to_index:
                    self.index_user_info_batch(docs_to_index)
            
            self.memory_cache.put(cache_key, extracted_info)
            return extracted_info if isinstance(extracted_info, list) else []
            
        except Exception as e:
            self.logger.error(f"信息提取失败: {e}")
            return []
    
    def extract_batch(self, user_inputs: List[str]) -> List[list]:
        """批量提取信息（并行优化）"""
        if not user_inputs:
            return []
        
        self.logger.info(f"批量提取 {len(user_inputs)} 条输入的信息")
        
        results = [None] * len(user_inputs)
        uncached_indices = []
        
        for i, user_input in enumerate(user_inputs):
            cache_key = self._get_cache_key(user_input, "extract")
            cached = self.memory_cache.get(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
        
        if uncached_indices:
            uncached_inputs = [user_inputs[i] for i in uncached_indices]
            
            futures = {
                self.executor.submit(self.extract_and_store_info, inp): i
                for i, inp in zip(uncached_indices, uncached_inputs)
            }
            
            for future in as_completed(futures):
                original_i = futures[future]
                try:
                    results[original_i] = future.result(timeout=60)
                except Exception as e:
                    self.logger.error(f"批量提取失败 [{original_i}]: {e}")
                    results[original_i] = []
        
        return results
    
    def process_user_input(self, user_input: str) -> str:
        """处理用户输入（并行优化版本）"""
        self.logger.info(f"处理用户输入: {user_input}")
        
        start_time = time.time()
        
        extract_future = self.executor.submit(self.extract_and_store_info, user_input)
        
        retrieve_future = self.executor.submit(self.retrieve_relevant_info, user_input, 3)
        
        extracted_info = extract_future.result()
        relevant_docs = retrieve_future.result()
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if relevant_docs:
            context = "\n".join([f"- {doc}" for doc in relevant_docs])
            messages.append({
                "role": "assistant",
                "content": f"让我回忆一下...\n根据我的记忆：\n{context}"
            })
        
        messages.append({"role": "user", "content": user_input})
        
        response = self.chat_completion(messages)
        
        self.conversation_history.append((user_input, response))
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        elapsed = time.time() - start_time
        self.logger.info(f"处理完成，耗时: {elapsed:.2f}s")
        
        return response
    
    def process_batch(self, user_inputs: List[str]) -> List[str]:
        """批量处理用户输入（并行优化）"""
        if not user_inputs:
            return []
        
        self.logger.info(f"批量处理 {len(user_inputs)} 条输入")
        start_time = time.time()
        
        self.extract_batch(user_inputs)
        
        all_relevant_docs = self.retrieve_batch(user_inputs, 3)
        
        batch_messages = []
        for user_input, relevant_docs in zip(user_inputs, all_relevant_docs):
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            if relevant_docs:
                context = "\n".join([f"- {doc}" for doc in relevant_docs])
                messages.append({
                    "role": "assistant",
                    "content": f"让我回忆一下...\n根据我的记忆：\n{context}"
                })
            
            messages.append({"role": "user", "content": user_input})
            batch_messages.append(messages)
        
        responses = self.chat_completion_batch(batch_messages)
        
        for user_input, response in zip(user_inputs, responses):
            self.conversation_history.append((user_input, response))
        
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
        
        elapsed = time.time() - start_time
        self.logger.info(f"批量处理完成，耗时: {elapsed:.2f}s，平均: {elapsed/len(user_inputs):.2f}s/条")
        
        return responses
    
    def show_knowledge(self):
        """显示当前知识库内容"""
        print("\n=== 知识库内容 ===")
        
        try:
            graph = self.hippo.graph
            print(f"\n【知识图谱】")
            print(f"  节点数量: {graph.vcount()}")
            print(f"  边数量: {graph.ecount()}")
            
            if graph.vcount() > 0:
                print(f"\n  部分节点:")
                for i, vertex in enumerate(graph.vs[:10]):
                    print(f"    {i+1}. {vertex['name'] if 'name' in vertex.attributes() else vertex.index}")
            
            if graph.ecount() > 0:
                print(f"\n  部分关系:")
                for i, edge in enumerate(graph.es[:10]):
                    source = graph.vs[edge.source]['name'] if 'name' in graph.vs[edge.source].attributes() else edge.source
                    target = graph.vs[edge.target]['name'] if 'name' in graph.vs[edge.target].attributes() else edge.target
                    print(f"    {i+1}. {source} -> {target}")
                    
        except Exception as e:
            print(f"  获取知识图谱失败: {e}")
        
        print(f"\n【缓存统计】")
        stats = self.memory_cache.get_stats()
        print(f"  内存缓存: {stats['size']}/{stats['max_size']} 条")
        print(f"  命中率: {stats['hit_rate']:.2%}")
        
        print(f"\n【对话历史】")
        if self.conversation_history:
            for user, assistant in self.conversation_history[-5:]:
                print(f"  用户: {user}")
                print(f"  助手: {assistant[:50]}...")
                print()
        else:
            print("  暂无对话记录")
    
    def clear_cache(self):
        """清除所有缓存"""
        self.memory_cache.clear()
        self.logger.info("缓存已清除")
    
    def shutdown(self):
        """关闭资源"""
        self.executor.shutdown(wait=True)
        self.logger.info("线程池已关闭")


def main():
    """主函数"""
    print("AI管家演示（性能优化版 - 快速启动）")
    print()
    
    print("正在初始化AI管家...")
    try:
        assistant = AIAssistantOptimized()
        print("AI管家初始化完成！（快速启动模式）")
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    print("\n正在后台预加载foxHippoRAG...")
    preload_future = assistant.preload_hippo()
    
    print("\n=== 开始对话 ===")
    print("命令:")
    print("  '退出' - 结束对话")
    print("  '知识' - 显示知识库")
    print("  '缓存' - 显示缓存统计")
    print("  '清缓存' - 清除缓存")
    print("  '状态' - 检查foxHippoRAG加载状态")
    print()
    print("提示: 您可以立即开始对话，foxHippoRAG正在后台加载中...")
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '退出':
                print("AI管家: 再见！您的信息已保存到知识图谱中。")
                assistant.shutdown()
                break
            
            if user_input == '知识':
                if not assistant.is_hippo_ready():
                    print("提示: foxHippoRAG正在加载中，请稍后再试...")
                    continue
                assistant.show_knowledge()
                continue
            
            if user_input == '缓存':
                stats = assistant.memory_cache.get_stats()
                print(f"\n缓存统计:")
                print(f"  大小: {stats['size']}/{stats['max_size']}")
                print(f"  命中: {stats['hits']}, 未命中: {stats['misses']}")
                print(f"  命中率: {stats['hit_rate']:.2%}")
                continue
            
            if user_input == '清缓存':
                assistant.clear_cache()
                print("缓存已清除")
                continue
            
            if user_input == '状态':
                if assistant.is_hippo_ready():
                    print("foxHippoRAG已加载完成，可以正常使用所有功能。")
                else:
                    print("foxHippoRAG正在后台加载中...")
                    if preload_future:
                        print(f"加载状态: {'进行中' if not preload_future.done() else '已完成'}")
                continue
            
            if not assistant.is_hippo_ready():
                print("提示: foxHippoRAG正在首次加载中，请稍候...")
                print("（您也可以先输入简单问题，系统会在后台完成加载后处理）")
            
            response = assistant.process_user_input(user_input)
            print(f"AI管家: {response}")
            
        except KeyboardInterrupt:
            print("\n\nAI管家: 再见！")
            assistant.shutdown()
            break
        except Exception as e:
            print(f"发生错误: {e}")
            logging.error(f"主循环错误: {e}")


if __name__ == "__main__":
    main()
