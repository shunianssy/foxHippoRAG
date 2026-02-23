"""
简单性能测试

使用较少文档数量测试基本功能
"""

import os
import sys
import time
import logging
import multiprocessing
from typing import List
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    multiprocessing.freeze_support()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SimpleTest')


def test_fast_mode_basic():
    """测试快速模式基本功能"""
    logger.info("=" * 60)
    logger.info("快速模式基本功能测试")
    logger.info("=" * 60)
    
    try:
        from src.foxhipporag.foxHippoRAG_optimized import OptimizedfoxHippoRAG
        from src.foxhipporag.utils.config_utils import BaseConfig
        
        # 配置快速索引模式
        config = BaseConfig()
        config.save_dir = "outputs/simple_test"
        config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
        config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
        config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        config.openie_mode = "online"
        config.force_index_from_scratch = True
        
        # 启用快速模式
        config.use_fast_index = True
        config.use_fast_retrieve = True
        config.embedding_batch_size = 8  # 减小批次大小避免速率限制
        
        # 测试文档（少量）
        test_docs = [
            "张三是一名软件工程师，他在北京工作。",
            "李四是一名医生，他在上海工作。",
            "王五是一名教师，他在广州工作。",
        ]
        
        logger.info(f"测试文档数量: {len(test_docs)}")
        
        # 初始化
        init_start = time.time()
        hippo = OptimizedfoxHippoRAG(global_config=config)
        init_time = time.time() - init_start
        logger.info(f"初始化耗时: {init_time:.2f}s")
        
        # 索引
        index_start = time.time()
        hippo.index(docs=test_docs)
        index_time = time.time() - index_start
        logger.info(f"索引耗时: {index_time:.2f}s")
        
        # 检索
        test_queries = ["谁在北京工作？", "谁是医生？"]
        
        retrieve_start = time.time()
        results = hippo.retrieve(queries=test_queries, num_to_retrieve=2)
        retrieve_time = time.time() - retrieve_start
        logger.info(f"检索耗时: {retrieve_time:.2f}s")
        
        # 显示结果
        logger.info("\n检索结果:")
        for i, result in enumerate(results):
            logger.info(f"  查询: {result.question}")
            if result.docs:
                logger.info(f"  结果: {result.docs[0]}")
        
        logger.info("\n✓ 测试通过！快速模式工作正常")
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_fast_mode_basic()
    sys.exit(0 if success else 1)
