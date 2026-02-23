"""
拟人脑遗忘机制模块

实现类似人脑的遗忘机制：
1. 对使用少的记忆片段进行压缩，减少空间占用和token使用
2. 设计模糊化阈值，当记忆片段的使用频率低于阈值时，会被压缩
3. 当记忆发生遗忘时，相应节点/边不会立即被删除，而是被标记为"遗忘"，
   当某次记忆构建再次检索到该节点时，节点会被重新激活并被赋予更高的权重
4. 使用LRU序列对记忆节点和边进行管理，定期剪枝
5. 设计遗忘系数，随图谱的膨胀而增大
6. 设计保护阈值，当图谱的节点数超过阈值时，会自动触发已遗忘节点的释放
"""

import time
import json
import hashlib
import logging
import asyncio
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import numpy as np

logger = logging.getLogger('ForgettingMechanism')


class MemoryState(Enum):
    """记忆状态枚举"""
    ACTIVE = "active"           # 活跃状态
    COMPRESSED = "compressed"   # 压缩状态
    FORGOTTEN = "forgotten"     # 遗忘状态
    REACTIVATED = "reactivated" # 重新激活状态


@dataclass
class MemoryNode:
    """记忆节点"""
    node_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    state: MemoryState = MemoryState.ACTIVE
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    weight: float = 1.0
    compressed_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def access(self):
        """访问节点，更新访问统计"""
        self.access_count += 1
        self.last_access_time = time.time()
        if self.state == MemoryState.FORGOTTEN:
            self.state = MemoryState.REACTIVATED
            self.weight *= 1.5  # 重新激活时增加权重
            logger.info(f"节点 {self.node_id} 被重新激活，权重增至 {self.weight:.2f}")
    
    def compress(self, summary: str = None):
        """压缩节点"""
        if self.state == MemoryState.ACTIVE:
            self.state = MemoryState.COMPRESSED
            self.compressed_content = summary or self._generate_summary()
            logger.debug(f"节点 {self.node_id} 已压缩")
    
    def forget(self):
        """遗忘节点"""
        if self.state in [MemoryState.ACTIVE, MemoryState.COMPRESSED]:
            self.state = MemoryState.FORGOTTEN
            logger.debug(f"节点 {self.node_id} 已标记为遗忘")
    
    def _generate_summary(self) -> str:
        """生成压缩摘要"""
        if len(self.content) <= 50:
            return self.content
        return self.content[:50] + "..."


@dataclass
class MemoryEdge:
    """记忆边"""
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    state: MemoryState = MemoryState.ACTIVE
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    weight: float = 1.0
    
    def access(self):
        """访问边"""
        self.access_count += 1
        self.last_access_time = time.time()


class ForgettingConfig:
    """遗忘机制配置"""
    
    def __init__(
        self,
        # 模糊化阈值
        compression_threshold: float = 0.3,      # 使用频率低于此值时压缩
        forgetting_threshold: float = 0.1,       # 使用频率低于此值时遗忘
        
        # 保护阈值
        max_nodes: int = 10000,                  # 最大节点数
        max_edges: int = 50000,                  # 最大边数
        protected_threshold: float = 0.8,        # 节点数达到上限的此比例时触发清理
        
        # 遗忘系数
        base_forgetting_coefficient: float = 0.01,  # 基础遗忘系数
        max_forgetting_coefficient: float = 0.5,    # 最大遗忘系数
        
        # LRU配置
        lru_capacity: int = 5000,                # LRU容量
        pruning_interval: float = 3600.0,        # 剪枝间隔（秒）
        
        # 时间衰减
        time_decay_factor: float = 0.95,         # 时间衰减因子
        reactivation_weight_boost: float = 1.5,  # 重新激活权重提升
    ):
        self.compression_threshold = compression_threshold
        self.forgetting_threshold = forgetting_threshold
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.protected_threshold = protected_threshold
        self.base_forgetting_coefficient = base_forgetting_coefficient
        self.max_forgetting_coefficient = max_forgetting_coefficient
        self.lru_capacity = lru_capacity
        self.pruning_interval = pruning_interval
        self.time_decay_factor = time_decay_factor
        self.reactivation_weight_boost = reactivation_weight_boost


class ForgettingMechanism:
    """拟人脑遗忘机制"""
    
    def __init__(self, config: ForgettingConfig = None):
        """初始化遗忘机制"""
        self.config = config or ForgettingConfig()
        
        # 记忆存储
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges: Dict[str, MemoryEdge] = {}
        
        # LRU缓存
        self._node_lru: OrderedDict[str, float] = OrderedDict()
        self._edge_lru: OrderedDict[str, float] = OrderedDict()
        
        # 统计信息
        self.total_access_count = 0
        self.last_pruning_time = time.time()
        
        # 状态计数
        self._state_counts = {
            MemoryState.ACTIVE: 0,
            MemoryState.COMPRESSED: 0,
            MemoryState.FORGOTTEN: 0,
            MemoryState.REACTIVATED: 0,
        }
        
        logger.info("遗忘机制初始化完成")
    
    def _compute_forgetting_coefficient(self) -> float:
        """计算遗忘系数（随图谱膨胀而增大）"""
        node_ratio = len(self.nodes) / self.config.max_nodes
        edge_ratio = len(self.edges) / self.config.max_edges
        expansion_ratio = max(node_ratio, edge_ratio)
        
        coefficient = self.config.base_forgetting_coefficient + \
                     (self.config.max_forgetting_coefficient - self.config.base_forgetting_coefficient) * expansion_ratio
        
        return min(coefficient, self.config.max_forgetting_coefficient)
    
    def _compute_access_frequency(self, item: MemoryNode | MemoryEdge) -> float:
        """计算访问频率"""
        if self.total_access_count == 0:
            return 0.0
        
        time_since_creation = time.time() - item.creation_time if hasattr(item, 'creation_time') else 1.0
        if time_since_creation == 0:
            time_since_creation = 1.0
        
        frequency = item.access_count / (self.total_access_count * time_since_creation / 3600)
        return min(frequency, 1.0)
    
    def add_node(
        self,
        node_id: str,
        content: str,
        embedding: np.ndarray = None,
        metadata: Dict = None
    ) -> MemoryNode:
        """添加记忆节点"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.access()
            self._update_lru('node', node_id)
            return node
        
        node = MemoryNode(
            node_id=node_id,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        
        self.nodes[node_id] = node
        self._update_lru('node', node_id)
        self._state_counts[MemoryState.ACTIVE] += 1
        
        self._check_capacity()
        
        return node
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str
    ) -> MemoryEdge:
        """添加记忆边"""
        edge_id = f"{source_id}_{relation}_{target_id}"
        
        if edge_id in self.edges:
            edge = self.edges[edge_id]
            edge.access()
            self._update_lru('edge', edge_id)
            return edge
        
        edge = MemoryEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=relation
        )
        
        self.edges[edge_id] = edge
        self._update_lru('edge', edge_id)
        
        self._check_capacity()
        
        return edge
    
    def access_node(self, node_id: str) -> Optional[MemoryNode]:
        """访问节点"""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        node.access()
        self.total_access_count += 1
        self._update_lru('node', node_id)
        
        if node.state == MemoryState.FORGOTTEN:
            self._state_counts[MemoryState.FORGOTTEN] -= 1
            self._state_counts[MemoryState.REACTIVATED] += 1
        
        return node
    
    def access_edge(self, edge_id: str) -> Optional[MemoryEdge]:
        """访问边"""
        if edge_id not in self.edges:
            return None
        
        edge = self.edges[edge_id]
        edge.access()
        self.total_access_count += 1
        self._update_lru('edge', edge_id)
        
        return edge
    
    def _update_lru(self, item_type: str, item_id: str):
        """更新LRU缓存"""
        lru = self._node_lru if item_type == 'node' else self._edge_lru
        
        if item_id in lru:
            lru.move_to_end(item_id)
        else:
            lru[item_id] = time.time()
    
    def _check_capacity(self):
        """检查容量并触发清理"""
        node_ratio = len(self.nodes) / self.config.max_nodes
        edge_ratio = len(self.edges) / self.config.max_edges
        
        if node_ratio > self.config.protected_threshold or edge_ratio > self.config.protected_threshold:
            logger.warning(f"接近容量上限，触发清理: 节点={len(self.nodes)}, 边={len(self.edges)}")
            asyncio.create_task(self.prune())
    
    async def prune(self):
        """定期剪枝"""
        logger.info("开始剪枝...")
        start_time = time.time()
        
        forgetting_coeff = self._compute_forgetting_coefficient()
        logger.info(f"当前遗忘系数: {forgetting_coeff:.4f}")
        
        # 处理节点
        nodes_to_compress = []
        nodes_to_forget = []
        
        for node_id, node in self.nodes.items():
            if node.state == MemoryState.FORGOTTEN:
                continue
            
            frequency = self._compute_access_frequency(node)
            
            if frequency < self.config.forgetting_threshold:
                nodes_to_forget.append(node_id)
            elif frequency < self.config.compression_threshold:
                nodes_to_compress.append(node_id)
        
        # 压缩节点
        for node_id in nodes_to_compress:
            node = self.nodes[node_id]
            if node.state == MemoryState.ACTIVE:
                node.compress()
                self._state_counts[MemoryState.ACTIVE] -= 1
                self._state_counts[MemoryState.COMPRESSED] += 1
        
        # 遗忘节点
        for node_id in nodes_to_forget:
            node = self.nodes[node_id]
            old_state = node.state
            node.forget()
            self._state_counts[old_state] -= 1
            self._state_counts[MemoryState.FORGOTTEN] += 1
        
        # 清理LRU中过期的遗忘节点
        if len(self.nodes) > self.config.lru_capacity:
            forgotten_nodes = [
                nid for nid, node in self.nodes.items()
                if node.state == MemoryState.FORGOTTEN
            ]
            
            for nid in forgotten_nodes[:len(self.nodes) - self.config.lru_capacity]:
                del self.nodes[nid]
                self._node_lru.pop(nid, None)
                self._state_counts[MemoryState.FORGOTTEN] -= 1
        
        # 处理边
        edges_to_forget = []
        for edge_id, edge in self.edges.items():
            if edge.state == MemoryState.FORGOTTEN:
                continue
            
            frequency = self._compute_access_frequency(edge)
            if frequency < self.config.forgetting_threshold:
                edges_to_forget.append(edge_id)
        
        for edge_id in edges_to_forget:
            edge = self.edges[edge_id]
            edge.state = MemoryState.FORGOTTEN
        
        # 清理边
        if len(self.edges) > self.config.max_edges:
            forgotten_edges = [
                eid for eid, edge in self.edges.items()
                if edge.state == MemoryState.FORGOTTEN
            ]
            
            for eid in forgotten_edges[:len(self.edges) - self.config.max_edges]:
                del self.edges[eid]
                self._edge_lru.pop(eid, None)
        
        self.last_pruning_time = time.time()
        elapsed = time.time() - start_time
        
        logger.info(
            f"剪枝完成，耗时 {elapsed:.2f}s: "
            f"压缩 {len(nodes_to_compress)} 节点, "
            f"遗忘 {len(nodes_to_forget)} 节点, "
            f"遗忘 {len(edges_to_forget)} 边"
        )
    
    def get_active_nodes(self) -> List[MemoryNode]:
        """获取活跃节点"""
        return [
            node for node in self.nodes.values()
            if node.state in [MemoryState.ACTIVE, MemoryState.REACTIVATED]
        ]
    
    def get_active_edges(self) -> List[MemoryEdge]:
        """获取活跃边"""
        return [
            edge for edge in self.edges.values()
            if edge.state == MemoryState.ACTIVE
        ]
    
    def search_nodes(
        self,
        query_embedding: np.ndarray = None,
        query_text: str = None,
        top_k: int = 10,
        include_forgotten: bool = False
    ) -> List[Tuple[MemoryNode, float]]:
        """搜索节点"""
        results = []
        
        for node in self.nodes.values():
            if not include_forgotten and node.state == MemoryState.FORGOTTEN:
                continue
            
            score = node.weight
            
            # 嵌入相似度计算（如果有嵌入）
            if query_embedding is not None and node.embedding is not None:
                similarity = np.dot(query_embedding, node.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding) + 1e-8
                )
                score *= (0.5 + 0.5 * similarity)  # 相似度权重
            
            # 文本匹配（关键词匹配）
            if query_text:
                query_lower = query_text.lower()
                content_lower = node.content.lower()
                
                # 完全包含匹配
                if query_lower in content_lower:
                    score *= 2.0
                # 关键词匹配（分词后匹配）
                else:
                    query_words = set(query_lower.split())
                    content_words = set(content_lower.split())
                    common_words = query_words & content_words
                    if common_words:
                        word_match_ratio = len(common_words) / max(len(query_words), 1)
                        score *= (1.0 + word_match_ratio)
                
                # 元数据匹配
                if node.metadata:
                    for key, value in node.metadata.items():
                        if isinstance(value, str) and query_lower in value.lower():
                            score *= 1.5
            
            results.append((node, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'active_nodes': self._state_counts[MemoryState.ACTIVE],
            'compressed_nodes': self._state_counts[MemoryState.COMPRESSED],
            'forgotten_nodes': self._state_counts[MemoryState.FORGOTTEN],
            'reactivated_nodes': self._state_counts[MemoryState.REACTIVATED],
            'forgetting_coefficient': self._compute_forgetting_coefficient(),
            'total_access_count': self.total_access_count,
            'lru_size': len(self._node_lru),
            'last_pruning': self.last_pruning_time,
        }
    
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            'nodes': {
                nid: {
                    'node_id': node.node_id,
                    'content': node.content,
                    'state': node.state.value,
                    'access_count': node.access_count,
                    'last_access_time': node.last_access_time,
                    'weight': node.weight,
                    'compressed_content': node.compressed_content,
                }
                for nid, node in self.nodes.items()
            },
            'edges': {
                eid: {
                    'edge_id': edge.edge_id,
                    'source_id': edge.source_id,
                    'target_id': edge.target_id,
                    'relation': edge.relation,
                    'state': edge.state.value,
                    'access_count': edge.access_count,
                    'weight': edge.weight,
                }
                for eid, edge in self.edges.items()
            },
            'stats': self.get_stats(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict, config: ForgettingConfig = None) -> 'ForgettingMechanism':
        """从字典反序列化"""
        mechanism = cls(config)
        
        for nid, ndata in data.get('nodes', {}).items():
            node = MemoryNode(
                node_id=ndata['node_id'],
                content=ndata['content'],
                state=MemoryState(ndata['state']),
                access_count=ndata['access_count'],
                last_access_time=ndata['last_access_time'],
                weight=ndata['weight'],
                compressed_content=ndata.get('compressed_content'),
            )
            mechanism.nodes[nid] = node
            mechanism._state_counts[node.state] += 1
        
        for eid, edata in data.get('edges', {}).items():
            edge = MemoryEdge(
                edge_id=edata['edge_id'],
                source_id=edata['source_id'],
                target_id=edata['target_id'],
                relation=edata['relation'],
                state=MemoryState(edata['state']),
                access_count=edata['access_count'],
                weight=edata['weight'],
            )
            mechanism.edges[eid] = edge
        
        return mechanism


class AsyncForgettingMechanism(ForgettingMechanism):
    """异步遗忘机制"""
    
    def __init__(self, config: ForgettingConfig = None):
        super().__init__(config)
        self._lock = asyncio.Lock()
        self._pruning_task: Optional[asyncio.Task] = None
    
    async def async_add_node(
        self,
        node_id: str,
        content: str,
        embedding: np.ndarray = None,
        metadata: Dict = None
    ) -> MemoryNode:
        """异步添加节点"""
        async with self._lock:
            return self.add_node(node_id, content, embedding, metadata)
    
    async def async_add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str
    ) -> MemoryEdge:
        """异步添加边"""
        async with self._lock:
            return self.add_edge(source_id, target_id, relation)
    
    async def async_access_node(self, node_id: str) -> Optional[MemoryNode]:
        """异步访问节点"""
        async with self._lock:
            return self.access_node(node_id)
    
    async def async_search_nodes(
        self,
        query_embedding: np.ndarray = None,
        query_text: str = None,
        top_k: int = 10,
        include_forgotten: bool = False
    ) -> List[Tuple[MemoryNode, float]]:
        """异步搜索节点"""
        async with self._lock:
            return self.search_nodes(query_embedding, query_text, top_k, include_forgotten)
    
    async def start_auto_pruning(self):
        """启动自动剪枝任务"""
        async def pruning_loop():
            while True:
                await asyncio.sleep(self.config.pruning_interval)
                try:
                    await self.prune()
                except Exception as e:
                    logger.error(f"自动剪枝失败: {e}")
        
        self._pruning_task = asyncio.create_task(pruning_loop())
        logger.info("自动剪枝任务已启动")
    
    async def stop_auto_pruning(self):
        """停止自动剪枝任务"""
        if self._pruning_task:
            self._pruning_task.cancel()
            try:
                await self._pruning_task
            except asyncio.CancelledError:
                pass
            logger.info("自动剪枝任务已停止")
