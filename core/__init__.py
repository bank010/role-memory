"""memory_box — 角色扮演长期记忆引擎（自包含包）。

对外只暴露 MemoryBox 类和少量辅助函数，内部所有模块用相对 import，
不依赖 app.* 任何东西——可以独立拔出去复用到任何项目。

用法：
    from app.memory_box import MemoryBox

    box = MemoryBox()
    await box.init()

    messages, debug = await box.build_prompt(...)
    turn = await box.save_turn(...)
    mem = await box.get_memory(...)
"""

from .box import MemoryBox

__all__ = ["MemoryBox"]
