"""
AI管家演示 - 使用foxHippoRAG框架

这个演示展示了如何使用foxHippoRAG框架构建一个智能AI管家：
1. 使用RAG框架进行知识存储和检索
2. 主动解析用户信息并存储到知识图谱
3. 对话时主动回忆之前的内容
4. 持久化存储（foxHippoRAG自动处理）
"""

import os
import logging
import multiprocessing
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

# Windows多进程支持
if __name__ == '__main__':
    multiprocessing.freeze_support()

# 导入foxHippoRAG
from src.foxhipporag import foxHippoRAG
from src.foxhipporag.utils.config_utils import BaseConfig


class AIAssistant:
    """AI管家类，使用foxHippoRAG框架
    
    功能：
    1. 使用RAG框架存储和检索知识
    2. 主动解析用户信息
    3. 对话时主动回忆
    4. 持久化存储
    """
    
    def __init__(self, save_dir: str = "outputs/ai_assistant_hippo"):
        """初始化AI管家
        
        Args:
            save_dir: 存储目录
        """
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AIAssistant')
        
        # 从环境变量获取配置
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # 嵌入模型配置（可以单独设置）
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY", self.api_key)
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            raise ValueError("请在.env文件中设置OPENAI_API_KEY")
        
        # 初始化foxHippoRAG
        self.logger.info("正在初始化foxHippoRAG...")
        
        # 创建配置
        self.config = BaseConfig()
        self.config.save_dir = save_dir
        self.config.llm_name = self.model
        self.config.llm_base_url = self.base_url
        self.config.embedding_model_name = self.embedding_model
        self.config.embedding_base_url = self.embedding_base_url
        self.config.embedding_api_key = self.embedding_api_key  # 嵌入模型API密钥
        self.config.openie_mode = "online"  # 使用在线模式
        self.config.force_index_from_scratch = False  # 不强制重建索引
        
        # 初始化foxHippoRAG实例
        self.hippo = foxHippoRAG(global_config=self.config)
        
        self.logger.info("foxHippoRAG初始化完成！")
        
        # 对话历史（用于上下文）
        self.conversation_history = []
        
        # 系统提示
        self.system_prompt = """你是一个智能AI管家。你的任务是：

1. **信息提取**：从用户输入中提取关键信息
2. **智能记忆**：记住用户告诉你的重要信息
3. **主动回忆**：在回答时，主动参考之前对话中用户告诉你的信息
4. **个性化回应**：根据用户的历史信息提供定制化的回答

请用友好、自然的语气与用户交流。"""
    
    def index_user_info(self, info: str):
        """将用户信息索引到知识库
        
        Args:
            info: 用户信息文本
        """
        self.logger.info(f"索引用户信息: {info}")
        self.hippo.index(docs=[info])
    
    def retrieve_relevant_info(self, query: str, top_k: int = 5) -> list:
        """检索相关信息
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
        
        Returns:
            相关文档列表
        """
        try:
            results = self.hippo.retrieve(queries=[query], num_to_retrieve=top_k)
            if results:
                return results[0].docs
            return []
        except Exception as e:
            self.logger.error(f"检索失败: {e}")
            return []
    
    def chat_completion(self, messages: list) -> str:
        """调用LLM生成回复
        
        Args:
            messages: 消息列表
        
        Returns:
            生成的回复
        """
        try:
            from openai import OpenAI
            
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            client = OpenAI(**client_kwargs)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"调用LLM失败: {e}")
            return f"抱歉，处理请求时出错: {str(e)}"
    
    def extract_and_store_info(self, user_input: str) -> list:
        """从用户输入中提取并存储信息
        
        Args:
            user_input: 用户输入
        
        Returns:
            提取的信息列表
        """
        try:
            from openai import OpenAI
            
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            client = OpenAI(**client_kwargs)
            
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

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个信息提取助手，只返回JSON格式的数据。"},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1
            )
            
            import json
            result_text = response.choices[0].message.content.strip()
            
            # 移除可能的markdown代码块标记
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            extracted_info = json.loads(result_text)
            
            # 将提取的信息转换为文档并索引
            if extracted_info and isinstance(extracted_info, list):
                for info in extracted_info:
                    subject = info.get('subject', '')
                    predicate = info.get('predicate', '')
                    obj = info.get('object', '')
                    if subject and predicate and obj:
                        # 转换为自然语言句子
                        doc = f"{subject}{predicate}{obj}。"
                        self.index_user_info(doc)
                        self.logger.info(f"存储信息: {doc}")
            
            return extracted_info if isinstance(extracted_info, list) else []
            
        except Exception as e:
            self.logger.error(f"信息提取失败: {e}")
            return []
    
    def process_user_input(self, user_input: str) -> str:
        """处理用户输入
        
        Args:
            user_input: 用户输入
        
        Returns:
            助手的回应
        """
        self.logger.info(f"处理用户输入: {user_input}")
        
        # 1. 提取并存储信息
        self.extract_and_store_info(user_input)
        
        # 2. 检索相关信息
        relevant_docs = self.retrieve_relevant_info(user_input, top_k=3)
        
        # 3. 构建消息
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # 添加检索到的上下文
        if relevant_docs:
            context = "\n".join([f"- {doc}" for doc in relevant_docs])
            messages.append({
                "role": "assistant",
                "content": f"让我回忆一下...\n根据我的记忆：\n{context}"
            })
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        # 4. 生成回复
        response = self.chat_completion(messages)
        
        # 5. 保存对话历史
        self.conversation_history.append((user_input, response))
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return response
    
    def show_knowledge(self):
        """显示当前知识库内容"""
        print("\n=== 知识库内容 ===")
        
        # 获取图中的节点和边
        try:
            graph = self.hippo.graph
            print(f"\n【知识图谱】")
            print(f"  节点数量: {graph.vcount()}")
            print(f"  边数量: {graph.ecount()}")
            
            # 显示部分节点
            if graph.vcount() > 0:
                print(f"\n  部分节点:")
                for i, vertex in enumerate(graph.vs[:10]):
                    print(f"    {i+1}. {vertex['name'] if 'name' in vertex.attributes() else vertex.index}")
            
            # 显示部分边
            if graph.ecount() > 0:
                print(f"\n  部分关系:")
                for i, edge in enumerate(graph.es[:10]):
                    source = graph.vs[edge.source]['name'] if 'name' in graph.vs[edge.source].attributes() else edge.source
                    target = graph.vs[edge.target]['name'] if 'name' in graph.vs[edge.target].attributes() else edge.target
                    print(f"    {i+1}. {source} -> {target}")
                    
        except Exception as e:
            print(f"  获取知识图谱失败: {e}")
        
        # 显示对话历史
        print(f"\n【对话历史】")
        if self.conversation_history:
            for user, assistant in self.conversation_history[-5:]:
                print(f"  用户: {user}")
                print(f"  助手: {assistant[:50]}...")
                print()
        else:
            print("  暂无对话记录")


def main():
    """主函数"""
    print("AI管家演示（使用foxHippoRAG框架）")
    print()
    
    # 初始化AI管家
    print("正在初始化AI管家...")
    try:
        assistant = AIAssistant()
        print("AI管家初始化完成！")
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    print("\n=== 开始对话 ===")
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '退出':
                print("AI管家: 再见！您的信息已保存到知识图谱中。")
                break
            
            if user_input == '知识':
                assistant.show_knowledge()
                continue
            
            # 处理用户输入
            response = assistant.process_user_input(user_input)
            print(f"AI管家: {response}")
            
        except KeyboardInterrupt:
            print("\n\nAI管家: 再见！")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            logging.error(f"主循环错误: {e}")


if __name__ == "__main__":
    main()
