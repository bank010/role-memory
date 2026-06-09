"""存储后端接口（仓储模式）。

只负责"原始持久化"，不含任何业务编排（向量化、去重、归一化、缓存都在上层）。
向量统一以 numpy.ndarray 进出，由后端自行决定如何落盘（SQLite=blob / Postgres=vector）。

这样设计的收益：
- 业务层 app.memory.stores 面向接口编程，换后端零改动；
- 三维打分 / 情节去重等算法只写一份，两种后端共用；
- 新后端（如 Qdrant、MySQL）只要实现这套接口即可接入。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class BaseStore(ABC):
    # ---- 生命周期 ----
    @abstractmethod
    def init(self) -> None:
        """建表 / 建扩展 / 建索引。幂等。"""

    def close(self) -> None:  # 可选
        pass

    # ---- 原始对话日志 turns（真相源，append-only）----
    @abstractmethod
    def append_turn(self, session: str, user_msg: str, ai_reply: str,
                    user_id: str = "", role_id: str = "") -> int:
        """追加一轮，返回该轮的 turn 序号（会话内自增）。user_id/role_id 用于独立列隔离。"""

    @abstractmethod
    def recent_turns(self, session: str, n: int) -> List[Dict]:
        """最近 n 轮，按 turn 升序。"""

    @abstractmethod
    def turns_after(self, session: str, after_turn: int) -> List[Dict]:
        ...

    @abstractmethod
    def max_turn(self, session: str) -> int:
        ...

    # ---- 语义记忆 facts ----
    @abstractmethod
    def upsert_fact(self, session: str, key: str, value: str,
                    confidence: float, turn: int, vec: Optional[np.ndarray] = None,
                    user_id: str = "", role_id: str = "") -> None:
        """写入/更新事实。vec 为实体向量（语义合并用），单一属性可为 None。"""

    @abstractmethod
    def all_facts(self, session: str) -> List[Dict]:
        """读路径用，不含向量。"""

    @abstractmethod
    def facts_with_vec(self, session: str) -> List[Dict]:
        """语义合并用，每条含 'vec'(np.ndarray，可能 size=0 表示无向量)。"""

    # ---- 情节记忆 episodes（带向量）----
    @abstractmethod
    def insert_episode(self, session: str, event: str, emotion: str,
                       importance: int, ts: float, turn: int, vec: np.ndarray,
                       sensitive: bool = False, user_id: str = "", role_id: str = "") -> None:
        ...

    @abstractmethod
    def update_episode(self, episode_id: int, importance: int, turn: int, ts: float) -> None:
        """情节去重命中后的合并更新。"""

    @abstractmethod
    def all_episodes(self, session: str) -> List[Dict]:
        """返回情节，每条含 'vec'(np.ndarray)。"""

    @abstractmethod
    def candidate_episodes(self, session: str, qvec: np.ndarray, limit: int) -> List[Dict]:
        """召回候选集：postgres 用 pgvector 原生 KNN 预筛 top-N；
        sqlite 直接返回全部（单会话量级小）。上层再做三维重排。"""

    @abstractmethod
    def mark_recalled(self, ids: List[int], turn: int) -> None:
        ...

    @abstractmethod
    def count_episodes(self, session: str) -> int:
        """返回会话内情节总数，用于判断是否需要淘汰。"""

    @abstractmethod
    def delete_episodes(self, ids: List[int]) -> None:
        """按 id 列表批量删除情节（遗忘淘汰）。"""

    # ---- 逐字记忆 chunks（带向量 + 归一化文本）----
    @abstractmethod
    def insert_chunk(self, session: str, turn: int, role: str, text: str,
                     norm: str, ts: float, vec: np.ndarray,
                     user_id: str = "", role_id: str = "") -> None:
        ...

    @abstractmethod
    def all_chunks(self, session: str) -> List[Dict]:
        """返回逐字片段，每条含 'vec'(np.ndarray)。"""

    @abstractmethod
    def count_chunks(self, session: str) -> int:
        """返回会话内 chunk 总数，用于判断是否需要淘汰。"""

    @abstractmethod
    def evict_oldest_chunks(self, session: str, keep_n: int) -> None:
        """保留按 turn 最新的 keep_n 条，删除其余（遗忘淘汰）。"""

    # ---- 关系/情绪状态 ----
    @abstractmethod
    def get_relationship(self, session: str) -> Dict:
        """不存在则创建默认行后返回。"""

    @abstractmethod
    def save_relationship(self, session: str, intimacy: float, trust: float,
                          stage: str, mood: str, summary: str, turn: int,
                          user_id: str = "", role_id: str = "") -> None:
        """以最终值整行覆盖（clamp/取舍逻辑在上层完成）。"""

    # ---- 加工进度 ----
    @abstractmethod
    def get_last_processed(self, session: str) -> int:
        ...

    @abstractmethod
    def set_last_processed(self, session: str, turn: int,
                           user_id: str = "", role_id: str = "") -> None:
        ...

    # ---- 清理 ----
    @abstractmethod
    def reset_session(self, session: str) -> None:
        ...
