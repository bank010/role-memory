const $ = (id) => document.getElementById(id);

let busy = false;

// ====================== 角色（提示词）本地管理 ======================
// 角色完全由用户在前端创建，存浏览器 localStorage。每个角色：{id, name, charName, prompt}
const ROLES_KEY = "rm_roles";
const CUR_ROLE_KEY = "rm_current_role";

function loadRoles() {
  try {
    return JSON.parse(localStorage.getItem(ROLES_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveRoles(roles) {
  localStorage.setItem(ROLES_KEY, JSON.stringify(roles));
}

function getCurrentRoleId() {
  return localStorage.getItem(CUR_ROLE_KEY) || "";
}

function setCurrentRoleId(id) {
  localStorage.setItem(CUR_ROLE_KEY, id);
}

function getCurrentRole() {
  const roles = loadRoles();
  return roles.find((r) => r.id === getCurrentRoleId()) || roles[0] || null;
}

function newRoleId() {
  return "role_" + Math.random().toString(36).slice(2, 9);
}

// 渲染角色下拉
function renderRoleSelect() {
  const sel = $("persona");
  const roles = loadRoles();
  if (!roles.length) {
    sel.innerHTML = `<option value="">（无角色，点“＋ 新建角色”）</option>`;
    return;
  }
  const cur = getCurrentRoleId();
  sel.innerHTML = roles
    .map((r) => `<option value="${r.id}" ${r.id === cur ? "selected" : ""}>${escapeHtml(r.name)}</option>`)
    .join("");
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// 记忆隔离维度：user_id（持久化在本地，可手填）× role_id（=当前角色）
function getUserId() {
  let u = localStorage.getItem("rm_user_id");
  if (!u) {
    u = "user_" + Math.random().toString(36).slice(2, 8);
    localStorage.setItem("rm_user_id", u);
  }
  const el = $("user-id");
  if (el && el.value.trim()) u = el.value.trim();
  return u;
}

function getRoleId() {
  return getCurrentRoleId() || "default";
}

function memParams() {
  return `user_id=${encodeURIComponent(getUserId())}&role_id=${encodeURIComponent(getRoleId())}`;
}

async function health() {
  const r = await fetch("/api/health").then((x) => x.json());
  const tag = $("mode-tag");
  if (r.mock_mode) {
    tag.textContent = "MOCK 模式（多语言归一化需真实模型）";
    tag.classList.add("mock");
  } else {
    tag.textContent = r.chat_model + " · 记忆基准语言 EN";
  }
  $("pe").textContent = r.process_every;
  renderRoleSelect();
}

function addMsg(role, text, timeMeta) {
  const div = document.createElement("div");
  div.className = "msg " + role;
  div.textContent = text;
  if (timeMeta) {
    const t = document.createElement("div");
    t.className = "msg-time";
    t.textContent = timeMeta;
    div.appendChild(t);
  }
  $("messages").appendChild(div);
  $("messages").scrollTop = $("messages").scrollHeight;
}

// 耗时标签：总耗时 + 分段（记忆读路径 / LLM 生成），一眼看出慢在哪
function fmtTiming(timing, clientMs) {
  if (!timing) return clientMs != null ? `${(clientMs / 1000).toFixed(1)}s` : "";
  const sec = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`);
  const parts = [];
  if (timing.total != null) parts.push(`总 ${sec(timing.total)}`);
  if (timing.context != null) parts.push(`记忆 ${sec(timing.context)}`);
  if (timing.llm != null) parts.push(`生成 ${sec(timing.llm)}`);
  if (clientMs != null && timing.total != null && clientMs - timing.total > 300) {
    parts.push(`网络 ${sec(clientMs - timing.total)}`);
  }
  return parts.join(" · ");
}

async function send() {
  if (busy) return;
  const role = getCurrentRole();
  if (!role) {
    addMsg("sys", "请先点右上角「＋ 新建角色」创建一个角色。");
    return;
  }
  const text = $("input").value.trim();
  if (!text) return;
  busy = true;
  $("input").value = "";
  addMsg("user", text);
  const tSend = performance.now();

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: getUserId(),
        role_id: role.id,
        message: text,
        persona_text: role.prompt || "",
        char_name: role.charName || null,
        user_name: $("user-name").value.trim() || null,
      }),
    });

    // 容错解析：服务异常时可能返回非 JSON（如裸 500 文本），避免 res.json() 直接崩
    const raw = await resp.text();
    let res;
    try {
      res = JSON.parse(raw);
    } catch {
      addMsg("sys", `服务异常（HTTP ${resp.status}）：${raw.slice(0, 200)}`);
      return;
    }

    if (!resp.ok || res.error) {
      addMsg("sys", "出错了：" + (res.message || res.detail || `HTTP ${resp.status}`));
      return;
    }

    const dbg = res.debug || {};
    addMsg("ai", res.reply || "(空回复)", fmtTiming(dbg.timing_ms, performance.now() - tSend));
    renderRetrieved(dbg.retrieved_episodes || []);
    renderVerbatim(dbg.retrieved_verbatim || []);
    $("sysprompt").textContent = dbg.system_prompt || "";
    await refreshMemory();
  } catch (e) {
    addMsg("sys", "网络错误：" + e);
  } finally {
    busy = false;
  }
}

function bar(label, val) {
  const pct = Math.round((val || 0) * 100);
  return `<div class="bar-row"><span class="lbl">${label}</span><div class="bar"><i style="width:${pct}%"></i></div><span class="c">${val.toFixed(2)}</span></div>`;
}

function renderRelationship(rel) {
  if (!rel) return;
  $("relationship").innerHTML =
    bar("亲密度", rel.intimacy) +
    bar("信任度", rel.trust) +
    `<div class="meta">阶段：<b>${rel.stage || "—"}</b>　情绪：<b>${rel.mood || "—"}</b></div>` +
    (rel.summary ? `<div class="meta">摘要：${rel.summary}</div>` : "");
}

function renderFacts(facts) {
  const el = $("facts");
  if (!facts || !facts.length) {
    el.className = "facts empty";
    el.textContent = "还没有抽取到事实";
    return;
  }
  el.className = "facts";
  el.innerHTML = facts
    .map(
      (f) =>
        `<div class="fact"><div><span class="k">${f.key}</span><br>${f.value}</div><span class="c">${(f.confidence||0).toFixed(1)}</span></div>`
    )
    .join("");
}

function epHtml(ep, withScore) {
  const insight = (ep.event || "").startsWith("[insight]") || (ep.event || "").startsWith("[洞察]");
  const meta = [
    `<span class="pill imp">重要度 ${ep.importance}</span>`,
    `<span class="pill">第${ep.turn}轮</span>`,
    ep.emotion ? `<span class="pill">${ep.emotion}</span>` : "",
  ];
  if (withScore) {
    meta.push(`<span class="pill score">score ${ep.score}</span>`);
    meta.push(`<span class="c">rel ${ep.relevance} · rec ${ep.recency}</span>`);
  }
  return `<div class="ep ${insight ? "insight" : ""}"><div class="ev">${ep.event}</div><div class="meta">${meta.join("")}</div></div>`;
}

function renderRetrieved(eps) {
  const el = $("retrieved");
  if (!eps || !eps.length) {
    el.className = "retrieved empty";
    el.textContent = "本轮未召回（记忆库为空或无相关情节）";
    return;
  }
  el.className = "retrieved";
  el.innerHTML = eps.map((e) => epHtml(e, true)).join("");
}

function renderVerbatim(items) {
  const el = $("verbatim");
  if (!items || !items.length) {
    el.className = "retrieved empty";
    el.textContent = "本轮未召回原话（窗口外无相关原话）";
    return;
  }
  el.className = "retrieved";
  el.innerHTML = items
    .map((v) => {
      const who = v.role === "user" ? "对方" : "角色";
      return `<div class="ep"><div class="ev">「${v.text}」</div><div class="meta"><span class="pill">第${v.turn}轮·${who}</span><span class="pill score">score ${v.score}</span><span class="c">向量 ${v.vec} · 关键词 ${v.lex}</span></div></div>`;
    })
    .join("");
}

function renderEpisodes(eps) {
  const el = $("episodes");
  if (!eps || !eps.length) {
    el.className = "episodes empty";
    el.textContent = "还没有情节记忆";
    return;
  }
  el.className = "episodes";
  el.innerHTML = eps.map((e) => epHtml(e, false)).join("");
}

async function refreshMemory() {
  const m = await fetch(`/api/memory?${memParams()}`).then((x) => x.json());
  renderRelationship(m.relationship);
  renderFacts(m.facts);
  renderEpisodes(m.episodes);
}

async function reset() {
  await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: getUserId(), role_id: getRoleId() }),
  });
  $("messages").innerHTML = "";
  $("retrieved").className = "retrieved empty";
  $("retrieved").textContent = "本轮未召回";
  $("verbatim").className = "retrieved empty";
  $("verbatim").textContent = "本轮未召回原话";
  $("sysprompt").textContent = "";
  await refreshMemory();
  addMsg("sys", "记忆已重置，可以重新开始。");
}

async function reprocess() {
  const btn = $("reprocess-btn");
  btn.disabled = true;
  btn.textContent = "加工中…";
  try {
    const role = getCurrentRole();
    await fetch("/api/reprocess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: getUserId(),
        role_id: getRoleId(),
        char_name: role ? role.charName || null : null,
        user_name: $("user-name").value.trim() || null,
      }),
    });
    await refreshMemory();
    addMsg("sys", "记忆重新加工完毕，右侧面板已更新。");
  } catch (e) {
    addMsg("sys", "重新加工失败：" + e);
  } finally {
    btn.disabled = false;
    btn.textContent = "重新加工记忆";
  }
}

// 切换用户/角色 = 切换到另一段记忆：清空聊天，重新加载该 (user×role) 的记忆和历史
async function switchScope() {
  const u = $("user-id").value.trim();
  if (u) localStorage.setItem("rm_user_id", u);
  $("messages").innerHTML = "";
  await refreshMemory();
  await loadHistory();
}

// 选择角色（下拉）→ 切到该角色的记忆
async function onSelectRole() {
  setCurrentRoleId($("persona").value);
  await switchScope();
}

// ====================== 角色编辑弹窗 ======================
let editingRoleId = null; // null=新建

function openRoleModal(roleId) {
  editingRoleId = roleId;
  const role = roleId ? loadRoles().find((r) => r.id === roleId) : null;
  $("role-modal-title").textContent = role ? "编辑角色" : "新建角色";
  $("role-char-name").value = role ? role.charName || "" : "";
  $("role-prompt").value = role ? role.prompt || "" : "";
  $("role-delete-btn").hidden = !role;
  $("role-modal").hidden = false;
  $("role-char-name").focus();
}

function closeRoleModal() {
  $("role-modal").hidden = true;
  editingRoleId = null;
}

async function saveRole() {
  const charName = $("role-char-name").value.trim();
  const prompt = $("role-prompt").value.trim();
  if (!prompt) {
    alert("请填写角色提示词");
    return;
  }
  const name = charName || "未命名角色";
  let roles = loadRoles();
  if (editingRoleId) {
    roles = roles.map((r) =>
      r.id === editingRoleId ? { ...r, name, charName, prompt } : r
    );
    setCurrentRoleId(editingRoleId);
  } else {
    const id = newRoleId();
    roles.push({ id, name, charName, prompt });
    setCurrentRoleId(id);
  }
  saveRoles(roles);
  renderRoleSelect();
  closeRoleModal();
  await switchScope();
}

async function deleteRole() {
  if (!editingRoleId) return;
  if (!confirm("删除此角色？（不会删除已存的聊天记忆，但下拉里会消失）")) return;
  saveRoles(loadRoles().filter((r) => r.id !== editingRoleId));
  const remain = loadRoles();
  setCurrentRoleId(remain[0] ? remain[0].id : "");
  renderRoleSelect();
  closeRoleModal();
  await switchScope();
}

$("send-btn").onclick = send;
$("reset-btn").onclick = reset;
$("reprocess-btn").onclick = reprocess;
$("persona").addEventListener("change", onSelectRole);
$("user-id").addEventListener("change", switchScope);
$("new-role-btn").onclick = () => openRoleModal(null);
$("edit-role-btn").onclick = () => {
  if (!getCurrentRole()) return openRoleModal(null);
  openRoleModal(getCurrentRoleId());
};
$("role-save-btn").onclick = saveRole;
$("role-cancel-btn").onclick = closeRoleModal;
$("role-delete-btn").onclick = deleteRole;
$("role-modal").addEventListener("click", (e) => {
  if (e.target === $("role-modal")) closeRoleModal();
});
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

async function loadHistory() {
  const r = await fetch(`/api/history?${memParams()}&n=60`).then((x) => x.json());
  const turns = r.turns || [];
  if (turns.length === 0) {
    addMsg("sys", "开始聊天吧。告诉它一些关于你的事，多聊几轮，再回头问它还记不记得。");
    return;
  }
  addMsg("sys", `— 已恢复 ${turns.length} 轮历史记录 —`);
  turns.forEach((t) => {
    if (t.user_msg) addMsg("user", t.user_msg);
    if (t.ai_reply) addMsg("ai", t.ai_reply);
  });
}

(async () => {
  // 先填好 user-id 输入框（持久化值），再渲染角色下拉，最后按当前 (user×role) 加载记忆
  const el = $("user-id");
  if (el) el.value = localStorage.getItem("rm_user_id") || getUserId();
  await health();
  // 确保有一个“当前角色”被选中
  const roles = loadRoles();
  if (roles.length && !getCurrentRole()) setCurrentRoleId(roles[0].id);
  renderRoleSelect();

  if (!roles.length) {
    addMsg("sys", "还没有角色。点右上角「＋ 新建角色」写一个角色提示词，就能开始聊了。");
    return;
  }
  await refreshMemory();
  await loadHistory();
})();
