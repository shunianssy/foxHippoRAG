# 延迟导入主模块，避免启动时加载重型依赖
# from .foxHippoRAG import foxHippoRAG

def get_foxHippoRAG():
    """延迟导入 foxHippoRAG 以提高启动速度"""
    from .foxHippoRAG import foxHippoRAG
    return foxHippoRAG

__all__ = ["foxHippoRAG", "get_foxHippoRAG"]
