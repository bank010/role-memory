"""MemoryBox — 角色扮演长期记忆引擎的统一对外接口。

所有记忆相关操作（读/写/查/重置/重加工）都通过此类完成。
内部模块（config/embeddings/rerank/cache/store/assembler/pipeline 等）
全部自包含在 memory_box 包内，调用方不需要了解任何内部实现。
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from . import assembler, cache, config, embeddings, pipeline, rerank, stores
from .session import make_session

log = logging.getLogger("memory_box")

_bg_tasks: set = set()


class MemoryBox:
    """记忆引擎的唯一对外接口。一个实例对应一个服务生命周期。"""

    async def init(self) -> None:
        """启动时调用一次：建表、连接池、缓存预热。"""
        await stores.init()
        log.info("MemoryBox 初始化完成 | store=%s | cache=%s | embed=%s | rerank=%s",
                 config.STORE_BACKEND, cache.enabled(),
                 config.EMBED_REAL, rerank.enabled())

    async def close(self) -> None:
        """优雅停机：排空后台任务，关闭所有连接。"""
        if _bg_tasks:
            await asyncio.gather(*list(_bg_tasks), return_exceptions=True)
        await embeddings.aclose()
        await rerank.aclose()
        await cache.aclose()
        await stores.close()

    # ──────────── 读路径 ────────────

    async def build_prompt(
        self,
        user_id: str,
        role_id: str,
        persona: str,
        user_msg: str,
        char_name: str = "Character",
        user_name: Optional[str] = None,
        language: Optional[str] = None,
        session: Optional[str] = None,
    ) -> Tuple[List[Dict], Dict]:
        """拼装完整的 LLM messages（含记忆注入），返回 (messages, debug)。

        messages 可直接传给任意 OpenAI 兼容 LLM。
        debug 包含检索命中、facts、耗时等可视化信息。
        """
        sid, _, _ = make_session(user_id, role_id, session)
        return await assembler.build_context(
            sid, persona, user_msg, char_name, user_name, language
        )

    # ──────────── 写路径 ────────────

    async def save_turn(
        self,
        user_id: str,
        role_id: str,
        user_msg: str,
        ai_reply: str,
        char_name: str = "Character",
        user_name: str = "",
        session: Optional[str] = None,
    ) -> int:
        """存储本轮对话并触发后台异步记忆加工，返回 turn 序号。"""
        sid, uid, rid = make_session(user_id, role_id, session)
        turn = await stores.append_turn(sid, user_msg, ai_reply, uid, rid)

        async def _index_and_process():
            try:
                await stores.index_chunk(sid, turn, "user", user_msg, uid, rid)
                await stores.index_chunk(sid, turn, "assistant", ai_reply, uid, rid)
            except Exception as e:
                log.error("逐字索引失败 session=%s err=%r", sid, e)
            try:
                await pipeline.maybe_process(sid, uid, rid, char_name, user_name)
            except Exception as e:
                log.error("后台记忆加工失败 session=%s err=%r", sid, e)

        _bg_tasks.add(t := asyncio.create_task(_index_and_process()))
        t.add_done_callback(_bg_tasks.discard)
        return turn

    # ──────────── 查询 ────────────

    async def get_memory(
        self,
        user_id: str,
        role_id: str,
        session: Optional[str] = None,
    ) -> Dict:
        """返回当前会话的完整记忆状态。"""
        sid, _, _ = make_session(user_id, role_id, session)
        facts, eps, rel, mt, lp = await asyncio.gather(
            stores.all_facts(sid),
            stores.all_episodes(sid),
            stores.get_relationship(sid),
            stores.max_turn(sid),
            stores.get_last_processed(sid),
        )
        for e in eps:
            e.pop("vec", None)
        eps.sort(key=lambda x: x["turn"], reverse=True)
        return {
            "facts": facts,
            "episodes": eps,
            "relationship": rel,
            "max_turn": mt,
            "last_processed": lp,
        }

    async def get_history(
        self,
        user_id: str,
        role_id: str,
        n: int = 40,
        session: Optional[str] = None,
    ) -> List[Dict]:
        """返回最近 n 轮对话记录。"""
        sid, _, _ = make_session(user_id, role_id, session)
        return await stores.recent_turns(sid, n)

    # ──────────── 管理 ────────────

    async def reset(
        self,
        user_id: str,
        role_id: str,
        session: Optional[str] = None,
    ) -> None:
        """清空该会话的全部记忆（不可逆）。"""
        sid, _, _ = make_session(user_id, role_id, session)
        await stores.reset_session(sid)

    async def reprocess(
        self,
        user_id: str,
        role_id: str,
        char_name: str = "Character",
        user_name: str = "",
        session: Optional[str] = None,
    ) -> Dict:
        """重置加工进度，从头重新抽取所有记忆。"""
        sid, uid, rid = make_session(user_id, role_id, session)
        if not user_name:
            facts = await stores.all_facts(sid)
            user_name = assembler.memory_user_name(facts)
        await stores.set_last_processed(sid, 0)
        await pipeline.maybe_process(sid, uid, rid, char_name, user_name)
        return {
            "max_turn": await stores.max_turn(sid),
            "last_processed": await stores.get_last_processed(sid),
        }
