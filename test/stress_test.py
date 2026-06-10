"""
压力测试：200 并发用户同时请求 /api/chat
用法：
    python stress_test.py                  # 默认 200 并发
    python stress_test.py --concurrency 50 # 自定义并发数
    python stress_test.py --rounds 3       # 每个用户连续发 3 轮
"""

import argparse
import asyncio
import random
import statistics
import time

import httpx

BASE_URL = "http://localhost:8011"

PERSONA = (
    "You are a friendly chatbot named Tester. "
    "Reply in 1-2 short sentences. Keep it casual."
)

MESSAGES = [
    "hi", "hello there", "what's up?", "how are you?",
    "tell me a joke", "what do you like?", "nice weather today",
    "do you remember me?", "I like pizza", "my name is test_user",
    "what's your favorite color?", "I'm feeling great",
    "let's talk about movies", "any recommendations?",
    "goodbye", "see you later", "thanks for chatting",
]


async def single_request(client: httpx.AsyncClient, user_idx: int, round_idx: int):
    """发送一次聊天请求，返回 (延迟ms, 状态码, 错误信息)"""
    payload = {
        "user_id": f"stress_user_{user_idx}",
        "role_id": "stress_role",
        "message": random.choice(MESSAGES),
        "persona_text": PERSONA,
        "char_name": "Tester",
        "user_name": f"User{user_idx}",
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{BASE_URL}/api/chat", json=payload)
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            data = resp.json()
            reply_len = len(data.get("reply", ""))
            timing = data.get("debug", {}).get("timing_ms", {})
            return {
                "ok": True,
                "latency": latency,
                "status": 200,
                "reply_len": reply_len,
                "mem_ms": timing.get("context"),
                "llm_ms": timing.get("llm"),
            }
        else:
            return {"ok": False, "latency": latency, "status": resp.status_code, "error": resp.text[:200]}
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return {"ok": False, "latency": latency, "status": 0, "error": str(e)[:200]}


async def run_user(client: httpx.AsyncClient, user_idx: int, rounds: int, results: list):
    """模拟单个用户连续发多轮"""
    for r in range(rounds):
        res = await single_request(client, user_idx, r)
        res["user"] = user_idx
        res["round"] = r
        results.append(res)


async def main(concurrency: int, rounds: int):
    print(f"\n{'='*60}")
    print(f"  压力测试: {concurrency} 并发用户 × {rounds} 轮 = {concurrency * rounds} 总请求")
    print(f"  目标: {BASE_URL}")
    print(f"{'='*60}\n")

    # 先检查服务是否可用
    async with httpx.AsyncClient(timeout=5) as check:
        try:
            r = await check.get(f"{BASE_URL}/api/health")
            info = r.json()
            print(f"  服务状态: {'MOCK' if info.get('mock_mode') else 'REAL'} 模式")
            print(f"  模型: {info.get('chat_model', '?')}")
            print(f"  存储: {info.get('store_backend', '?')}")
            print()
        except Exception as e:
            print(f"  ❌ 服务不可用: {e}")
            return

    results = []
    t_start = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30, read=120, write=30, pool=30),
        limits=httpx.Limits(max_connections=concurrency + 50, max_keepalive_connections=concurrency),
    ) as client:
        tasks = [run_user(client, i, rounds, results) for i in range(concurrency)]
        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - t_start

    # ── 统计 ──
    ok_results = [r for r in results if r["ok"]]
    fail_results = [r for r in results if not r["ok"]]
    latencies = [r["latency"] for r in ok_results]
    mem_times = [r["mem_ms"] for r in ok_results if r.get("mem_ms") is not None]
    llm_times = [r["llm_ms"] for r in ok_results if r.get("llm_ms") is not None]

    print(f"{'='*60}")
    print(f"  测试结果")
    print(f"{'='*60}")
    print(f"  总耗时:         {total_time:.1f}s")
    print(f"  总请求:         {len(results)}")
    print(f"  成功:           {len(ok_results)}")
    print(f"  失败:           {len(fail_results)}")
    print(f"  吞吐量(QPS):    {len(ok_results) / total_time:.1f}")
    print()

    if latencies:
        latencies.sort()
        print(f"  ── 端到端延迟 (ms) ──")
        print(f"  最小:    {min(latencies):>10.0f}")
        print(f"  平均:    {statistics.mean(latencies):>10.0f}")
        print(f"  中位数:  {statistics.median(latencies):>10.0f}")
        print(f"  P90:     {latencies[int(len(latencies)*0.9)]:>10.0f}")
        print(f"  P95:     {latencies[int(len(latencies)*0.95)]:>10.0f}")
        print(f"  P99:     {latencies[int(len(latencies)*0.99)]:>10.0f}")
        print(f"  最大:    {max(latencies):>10.0f}")
        print()

    if mem_times:
        print(f"  ── 记忆读路径 (ms) ──")
        print(f"  平均:    {statistics.mean(mem_times):>10.0f}")
        print(f"  P95:     {sorted(mem_times)[int(len(mem_times)*0.95)]:>10.0f}")
        print(f"  最大:    {max(mem_times):>10.0f}")
        print()

    if llm_times:
        print(f"  ── LLM 生成 (ms) ──")
        print(f"  平均:    {statistics.mean(llm_times):>10.0f}")
        print(f"  P95:     {sorted(llm_times)[int(len(llm_times)*0.95)]:>10.0f}")
        print(f"  最大:    {max(llm_times):>10.0f}")
        print()

    if fail_results:
        print(f"  ── 失败详情 ──")
        error_counts = {}
        for r in fail_results:
            key = f"HTTP {r['status']}" if r["status"] else "连接错误"
            error_counts[key] = error_counts.get(key, 0) + 1
        for k, v in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v} 次")
        sample = fail_results[0]
        print(f"    示例: {sample.get('error', '')[:100]}")
        print()

    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="角色记忆系统压力测试")
    parser.add_argument("-c", "--concurrency", type=int, default=200, help="并发用户数 (默认 200)")
    parser.add_argument("-r", "--rounds", type=int, default=1, help="每用户请求轮数 (默认 1)")
    args = parser.parse_args()
    asyncio.run(main(args.concurrency, args.rounds))
