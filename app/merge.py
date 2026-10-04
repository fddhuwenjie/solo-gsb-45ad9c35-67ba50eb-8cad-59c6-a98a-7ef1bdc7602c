"""把版本差异审计结果「应用到当前草稿」：旧版为基线、新版为候选的三方合并。

审计只读两份不可变版本；本模块在 **旧版（base）/ 新版（candidate）/ 当前草稿
（draft）** 之间做逐条合并：

- 用户按条款逐条勾选要带入的 **正文**、**父子顺序 / 层级（同父兄弟相对次序）**
  以及 **删除 / 恢复 / 新增** 变更；未勾选项保留当前草稿；
- 若当前草稿相对旧版已经改动了同一条款或其父级（正文、删除状态或兄弟次序），
  该可选项标为 **冲突**，禁止覆盖；
- 引用改写与术语绑定 **不参与拼接**：合并只产出新的 正文 / 顺序 / 删除状态，
  随后在合并结果上重新解析、级联删除、重排、引用改写与术语终审；
- 与签发一致，引用问题（无法解析 / 目标已删除 / 循环引用）与未忽略的术语问题
  均为阻塞项；存在冲突或阻塞时整体拒绝，不写部分结果。

结构恒不变：条款以解析时的原路径稳定 ID 为身份，UI 只允许同父兄弟换序，因此
「层级」变化等价于同父兄弟相对次序变化；顺序合并按节点（含原文未单列的虚父
节点）在 O / N / D 三方序列中做「候选驱动 + 草稿锚定」的确定性合并，再交
``references.analyze`` 规整为合法树先序。
"""
from __future__ import annotations

from typing import Any

from .audit import _Side, _side_snapshot, _sibling_ranks, _load_side
from .issuance import snapshot_fingerprint
from .parser import parse_contract
from .references import (
    analyze,
    cascade_deleted,
    _build_tree,
)
from .terms import review_terms

# 可勾选维度
BODY = "body"
ORDER = "order"
LIFECYCLE = "lifecycle"
ADD = "add"
_DIMS = (BODY, ORDER, LIFECYCLE, ADD)


# ---------------------------------------------------------------- 三方装载

def _draft_side(row: dict) -> _Side:
    """把当前草稿快照当作一个审计侧（重新解析 + 引用 / 术语分析）。"""
    import json

    content = row["content"]
    order = json.loads(row["order_json"])
    deleted = json.loads(row["deleted_json"])
    ignores = json.loads(row["term_ignores_json"] or "[]")

    preamble, clauses = parse_contract(content)
    known = {c.id for c in clauses}
    order = [cid for cid in order if cid in known]
    deleted = [cid for cid in deleted if cid in known]
    ref_result = analyze(clauses, order, deleted)
    ref_result["preamble"] = preamble
    term_result = review_terms(
        preamble, clauses, cascade_deleted(clauses, deleted))
    side = _Side(
        version_id=0, version_no=None, revision=row["revision"],
        title=row["title"], issued_at=0, content=content,
        ref_result=ref_result, term_result=term_result)
    side.ignores = ignores
    side.row_deleted_json = row["deleted_json"]
    return side


def _label(side: _Side, cid: str) -> str:
    c = side.by_id.get(cid)
    if not c:
        return cid
    return side.issued_label(cid) or c.label


def _side_alive_path(side: _Side, cid: str) -> bool:
    """该条款在该侧快照中的存活状态（不存在视为 False）。"""
    return cid in side.by_id and cid in side.alive


# ---------------------------------------------------------------- 冲突判定

def _ancestors(side: _Side, cid: str) -> list[str]:
    c = side.by_id.get(cid)
    if not c:
        return []
    out = []
    for k in range(1, len(c.path)):
        pid = "c-" + "-".join(str(p) for p in c.path[:k])
        if pid in side.by_id:
            out.append(pid)
    return out


def _parent_reasons(d: _Side, o: _Side, cid: str,
                    r_d: dict[str, int], r_o: dict[str, int]) -> list[dict]:
    """旧版基线下，cid 的父级在草稿中的改动（正文 / 删除状态 / 次序）。"""
    c = o.by_id.get(cid)
    if not c:
        return []
    out: list[dict] = []
    for k in range(1, len(c.path)):
        pid = "c-" + "-".join(str(p) for p in c.path[:k])
        oc = o.by_id.get(pid)
        if oc is None:
            continue
        dc = d.by_id.get(pid)
        lab = _label(o, pid)
        if dc is None:
            out.append({
                "type": "parent_missing", "clause_id": pid, "label": lab,
                "message": f"父级条款 {lab} 在当前草稿中已不存在，禁止覆盖",
            })
            continue
        if dc.text != oc.text:
            out.append({
                "type": "parent_body", "clause_id": pid, "label": lab,
                "message": f"父级条款 {lab} 的正文已在当前草稿中修改，禁止覆盖",
            })
            continue
        o_alive, d_alive = pid in o.alive, pid in d.alive
        if o_alive != d_alive:
            what = "删除" if o_alive and not d_alive else "恢复"
            out.append({
                "type": "parent_lifecycle", "clause_id": pid, "label": lab,
                "message": f"父级条款 {lab} 在当前草稿中已被{what}，禁止覆盖",
            })
            continue
        if d_alive and r_d.get(pid) != r_o.get(pid):
            out.append({
                "type": "parent_order", "clause_id": pid, "label": lab,
                "message": f"父级条款 {lab} 的兄弟顺序已在当前草稿中调整，禁止覆盖",
            })
    return out


def _dim_state(reasons: list[dict], *, selectable: bool,
               already: bool = False) -> dict:
    if reasons:
        return {"status": "conflict", "selectable": False, "reasons": reasons}
    if already:
        return {"status": "already", "selectable": False, "reasons": []}
    if selectable:
        return {"status": "selectable", "selectable": True, "reasons": []}
    return {"status": "none", "selectable": False, "reasons": []}


def _self_conflict_reasons(d: _Side, o: _Side, cid: str,
                           r_d: dict[str, int], r_o: dict[str, int]
                           ) -> list[dict]:
    """条款级冲突：当前草稿相对旧版对该条款自身的任一改动（正文 / 顺序 / 生命周期）。

    规格要求「草稿相对旧版已改动**同一条款**或其父级」即禁止该项覆盖——
    与具体维度无关：草稿改正文后，新版对该条款的顺序 / 删除带入同样危险
    （会丢失草稿改动或使其错位），因此所有可带入维度统一冲突。
    """
    dc, oc = d.by_id.get(cid), o.by_id.get(cid)
    if dc is None or oc is None:
        return []
    out: list[dict] = []
    lab = _label(o, cid)
    if dc.text != oc.text:
        out.append({
            "type": "self_body", "clause_id": cid, "label": lab,
            "message": "当前草稿已修改该条款正文（相对旧版），该条款的任一变更均禁止覆盖",
        })
    d_alive, o_alive = cid in d.alive, cid in o.alive
    if d_alive != o_alive:
        what = "删除" if o_alive and not d_alive else "恢复"
        out.append({
            "type": "self_lifecycle", "clause_id": cid, "label": lab,
            "message": f"当前草稿已对该条款做{what}（相对旧版），该条款的任一变更均禁止覆盖",
        })
    elif d_alive and r_d.get(cid) != r_o.get(cid):
        out.append({
            "type": "self_order", "clause_id": cid, "label": lab,
            "message": "当前草稿已调整该条款的兄弟顺序（相对旧版），该条款的任一变更均禁止覆盖",
        })
    return out


def _clause_conflicts(d: _Side, o: _Side, cid: str,
                      r_d: dict[str, int], r_o: dict[str, int]) -> list[dict]:
    """同一条款自身或任一父级在草稿中相对旧版的改动（条款级冲突原因全集）。"""
    return (_self_conflict_reasons(d, o, cid, r_d, r_o)
            + _parent_reasons(d, o, cid, r_d, r_o))


# ---------------------------------------------------------------- 可勾选项

def _build_items(d: _Side, o: _Side, n: _Side) -> list[dict]:
    """生成逐条可勾选项（正文 / 顺序 / 删除恢复 / 新增）。"""
    peers_do = {cid for cid in (set(d.by_id) & set(o.by_id))
                if cid in d.alive and cid in o.alive}
    r_do_d = _sibling_ranks(d, peers_do)
    r_do_o = _sibling_ranks(o, peers_do)

    all_ids = set(d.by_id) | set(o.by_id) | set(n.by_id)
    items: list[dict] = []

    for cid in all_ids:
        in_d, in_o, in_n = (cid in d.by_id, cid in o.by_id, cid in n.by_id)
        d_alive = in_d and cid in d.alive
        o_alive = in_o and cid in o.alive
        n_alive = in_n and cid in n.alive

        dims: dict[str, dict] = {k: _dim_state([], selectable=False) for k in _DIMS}
        lifecycle_action: str | None = None
        heading = ""
        for s, present in ((d, in_d), (o, in_o), (n, in_n)):
            if present:
                heading = s.by_id[cid].heading or heading

        # ---- 仅新版存在：新增（或草稿已有同路径编号条款 → 冲突）----
        if not in_o and in_n:
            if in_d:
                reasons = [{
                    "type": "add_clash", "clause_id": cid,
                    "label": _label(n, cid),
                    "message": "当前草稿已存在相同原路径编号的条款，无法新增带入",
                }]
                dims[ADD] = _dim_state(reasons, selectable=False)
            else:
                reasons = []
                for pid in _ancestors(n, cid):
                    if pid in d.by_id:
                        if pid not in o.by_id:
                            reasons.append({
                                "type": "parent_add_clash",
                                "clause_id": pid, "label": _label(n, pid),
                                "message": f"父级位置 {_label(n, pid)} 在草稿中为另一条款，无法新增带入",
                            })
                        else:
                            reasons.extend(
                                _parent_reasons(d, o, pid, r_do_d, r_do_o))
                dims[ADD] = _dim_state(reasons, selectable=True)
            items.append(_item_row(cid, heading, dims, d, o, n,
                                   lifecycle_action))
            continue

        # ---- 仅旧版存在：新版删除（整条编号消失）----
        if in_o and not in_n:
            if in_d and d_alive:
                lifecycle_action = "delete"
                dims[LIFECYCLE] = _dim_state(
                    _clause_conflicts(d, o, cid, r_do_d, r_do_o),
                    selectable=True)
            elif in_d:
                # 草稿中已删除：与新版结果一致
                dims[LIFECYCLE] = _dim_state([], selectable=False, already=True)
            items.append(_item_row(cid, heading, dims, d, o, n,
                                   lifecycle_action))
            continue

        if not (in_o and in_n):
            continue

        # ---- 两侧共有 ----
        oc, nc = o.by_id[cid], n.by_id[cid]
        body_diff = o_alive and n_alive and oc.text != nc.text
        rank_change = (o_alive and n_alive
                       and _rank_changed(o, n, cid))
        # 条款级冲突原因：自身或父级相对旧版被草稿改动 → 该条款所有维度统一禁用
        conflicts = (_clause_conflicts(d, o, cid, r_do_d, r_do_o)
                     if in_d else [])

        if in_d and d_alive:
            # 正文维度
            if body_diff:
                dims[BODY] = _dim_state(conflicts, selectable=True)
            # 顺序维度
            if rank_change:
                dims[ORDER] = _dim_state(conflicts, selectable=True)
            # 生命周期：旧版存活、新版删除
            if o_alive and not n_alive:
                lifecycle_action = "delete"
                dims[LIFECYCLE] = _dim_state(conflicts, selectable=True)
            # 旧版删除、新版恢复
            if not o_alive and n_alive:
                # 草稿存活则与新版一致；草稿删除的分支见下方
                dims[LIFECYCLE] = _dim_state([], selectable=False, already=True)
        elif in_d:
            # 草稿中该条款已删除
            if not o_alive and n_alive:
                # 恢复变更：草稿仍删除 → 可恢复（自身 / 父级被改则冲突）
                lifecycle_action = "restore"
                dims[LIFECYCLE] = _dim_state(
                    _self_conflict_reasons(d, o, cid, r_do_d, r_do_o)
                    + _parent_reasons(d, o, cid, r_do_d, r_do_o),
                    selectable=True)
            elif o_alive and n_alive:
                # 草稿自行删除了旧版 / 新版都存活的条款：正文 / 顺序无法带入
                for dim in (BODY, ORDER):
                    if (body_diff if dim == BODY else rank_change):
                        dims[dim] = _dim_state([{
                            "type": "self_lifecycle", "clause_id": cid,
                            "label": _label(d, cid),
                            "message": "当前草稿已删除该条款，无法只带入正文 / 顺序（新版该条款仍存活）",
                        }], selectable=False)
            # o_alive 且 not n_alive：草稿与新版都删除 → 已一致
            if o_alive and not n_alive:
                dims[LIFECYCLE] = _dim_state([], selectable=False, already=True)

        if any(v["status"] != "none" for v in dims.values()):
            items.append(_item_row(cid, heading, dims, d, o, n,
                                   lifecycle_action))

    items.sort(key=lambda it: (it["sort_pos"], it["clause_id"]))
    for it in items:
        it.pop("sort_pos", None)
    return items


def _rank_changed(o: _Side, n: _Side, cid: str) -> bool:
    common = {x for x in (set(o.by_id) & set(n.by_id))
              if x in o.alive and x in n.alive}
    ro = _sibling_ranks(o, common)
    rn = _sibling_ranks(n, common)
    return ro.get(cid) != rn.get(cid)


def _item_row(cid: str, heading: str, dims: dict, d: _Side, o: _Side,
              n: _Side, lifecycle_action: str | None) -> dict:
    pos = min(p for p in (n.position(cid), d.position(cid), o.position(cid))
              if p is not None)
    return {
        "clause_id": cid,
        "heading": heading,
        "dims": dims,
        "lifecycle_action": lifecycle_action,
        "sort_pos": pos,
        "draft_loc": d.clause_loc(cid),
        "old_loc": o.clause_loc(cid),
        "new_loc": n.clause_loc(cid),
    }


# ---------------------------------------------------------------- 选择项规整

def normalize_selections(items: list[dict], selections: list[dict]
                         ) -> tuple[dict[str, set[str]], list[dict]]:
    """把前端选择规整为 {cid: {dim,...}}；非法（冲突 / 不可选 / 未知）逐项返回。"""
    by_id = {it["clause_id"]: it for it in items}
    chosen: dict[str, set[str]] = {}
    invalid: list[dict] = []
    for raw in selections or []:
        cid = raw.get("clause_id")
        it = by_id.get(cid)
        if it is None:
            invalid.append({"clause_id": cid, "dim": None,
                            "message": "该条款不在合并范围内"})
            continue
        for dim in _DIMS:
            if not raw.get(dim):
                continue
            st = it["dims"].get(dim, {})
            if st.get("status") != "selectable":
                invalid.append({
                    "clause_id": cid, "dim": dim,
                    "message": f"该条款的{_DIM_CN[dim]}变更不可带入（冲突或已一致）",
                    "reasons": st.get("reasons", []),
                })
            else:
                chosen.setdefault(cid, set()).add(dim)
    return chosen, invalid


_DIM_CN = {
    BODY: "正文", ORDER: "顺序", LIFECYCLE: "删除 / 恢复", ADD: "新增",
}


# ---------------------------------------------------------------- 合并状态

def _merge_content(d: _Side, n: _Side, chosen: dict[str, set[str]]
                   ) -> tuple[str, list[str]]:
    """拼装合并后正文，返回 (content, 块在拼装序列中的 cid 顺序)。

    以草稿原文块为底（保留草稿未勾选部分）；勾选「正文」用新版整块原文替换；
    勾选「新增」的新版块按新版文档序插入到（已在底流中的）新版相邻条款之间。
    前置说明始终保留当前草稿。
    """
    override = set()
    added: list[str] = []
    for cid, dims in chosen.items():
        if BODY in dims and cid in d.by_id and cid in n.by_id:
            override.add(cid)
        if ADD in dims and cid not in d.by_id and cid in n.by_id:
            added.append(cid)

    # 底流：草稿解析序的条款块
    stream: list[str] = [c.id for c in d.clauses]
    raw_of: dict[str, str] = {c.id: c.raw for c in d.clauses}
    for c in n.clauses:
        raw_of.setdefault(c.id, c.raw)
    for cid in override:
        raw_of[cid] = n.by_id[cid].raw

    # 新增块按新版文档序插入（锚点：新版相邻、且已在流中的条款）
    present = set(stream)
    n_order = [c.id for c in n.clauses]
    for cid in sorted(added, key=lambda x: n.doc[x]):
        i = n_order.index(cid)
        prev_id = next((q for q in reversed(n_order[:i]) if q in present), None)
        next_id = next((q for q in n_order[i + 1:] if q in present), None)
        if prev_id is not None:
            stream.insert(stream.index(prev_id) + 1, cid)
        elif next_id is not None:
            stream.insert(stream.index(next_id), cid)
        else:
            stream.append(cid)
        present.add(cid)

    blocks = [raw_of[cid].strip("\n") for cid in stream if cid in raw_of]
    content = d.preamble.strip()
    if blocks:
        content = (content + "\n\n" if content else "") + "\n\n".join(blocks)
    return content + "\n", stream


def _merge_deleted(o: _Side, d: _Side, merged_clauses,
                   chosen: dict[str, set[str]]
                   ) -> tuple[set[str], list[str]]:
    """合并显式删除集，返回 (级联后删除集, 显式删除列表)。"""
    import json

    explicit = set(cid for cid in json.loads(d.row_deleted_json)
                   if cid in {c.id for c in merged_clauses})
    by_id = {c.id: c for c in merged_clauses}
    for cid, dims in chosen.items():
        if LIFECYCLE not in dims:
            continue
        if cid not in by_id:
            continue
        pref = by_id[cid].path
        if _side_alive_path(o, cid):  # 旧版存活 → 新版删除：删除
            explicit.add(cid)
        else:                  # 旧版删除 → 新版恢复：连同后代一起恢复
            for x in merged_clauses:
                if x.path[:len(pref)] == pref:
                    explicit.discard(x.id)
    dead = cascade_deleted(merged_clauses, sorted(explicit))
    return dead, sorted(explicit)


def _merge_order(d: _Side, o: _Side, n: _Side, merged_clauses,
                 dead: set[str], chosen: dict[str, set[str]]) -> list[str]:
    """节点级三方顺序合并 → 存活条款的合法树先序（再交 analyze 规整）。

    规则（每个父节点的直接子节点组内）：

    - 勾选了「顺序 / 恢复 / 新增」的子节点为**驱动节点**，按新版相对次序输出；
    - 未勾选节点保留当前草稿：以其在草稿序列中相邻的前一个驱动节点为锚
      （草稿中无前置驱动的放最前），同锚的未勾选节点按草稿次序排列；
    - 草稿侧不存在的节点（新增带来的虚父节点）退而以新版序列找锚。
    """
    tree = _build_tree(merged_clauses)

    # 驱动节点：勾选了顺序 / 恢复 / 新增的存活条款自身所在节点
    driven_paths: set[tuple[int, ...]] = set()
    merged_by_id = {c.id: c for c in merged_clauses}
    for cid, dims in chosen.items():
        c = merged_by_id.get(cid)
        if c and cid not in dead and (ORDER in dims or LIFECYCLE in dims
                                      or ADD in dims):
            driven_paths.add(c.path)

    def subtree_alive(node) -> list[str]:
        ids = [cid for cid in node.clause_ids if cid not in dead]
        for ch in node.children.values():
            ids.extend(subtree_alive(ch))
        return ids

    def side_rank(side: _Side, kids, members_of) -> list[tuple[int, ...]]:
        out = []
        for ch in kids:
            mem = [cid for cid in members_of[ch.path] if cid in side.alive]
            if mem:
                out.append(ch.path)
        out.sort(key=lambda p: min(side.position(c) for c in members_of[p]
                                   if c in side.alive))
        return out

    merged_child_order: dict[tuple, list[tuple[int, ...]]] = {}

    def merge_group(node) -> None:
        kids = [ch for ch in node.children.values() if subtree_alive(ch)]
        members_of = {ch.path: subtree_alive(ch) for ch in kids}

        d_rank = side_rank(d, kids, members_of)
        n_rank = side_rank(n, kids, members_of)
        d_pos = {p: i for i, p in enumerate(d_rank)}
        n_pos = {p: i for i, p in enumerate(n_rank)}

        # 仅驱动直接子节点自身（含虚节点时虚节点永远不可直接驱动）
        spine = [p for p in n_rank if p in driven_paths]
        spine_set = set(spine)

        buckets: dict[tuple[int, ...] | None, list[tuple[int, ...]]] = {}
        for ch in kids:
            p = ch.path
            if p in spine_set:
                continue
            anchor = None
            if p in d_pos:
                # 草稿锚：草稿序列中前一个驱动节点
                for q in reversed(d_rank[:d_pos[p]]):
                    if q in spine_set:
                        anchor = q
                        break
            else:
                # 新增带来的虚节点：退用新版锚
                for q in reversed(n_rank[:n_pos.get(p, 0)]):
                    if q in spine_set:
                        anchor = q
                        break
            buckets.setdefault(anchor, []).append(p)
        for v in buckets.values():
            v.sort(key=lambda p: (d_pos.get(p, 10 ** 9),
                                  n_pos.get(p, 10 ** 9)))

        ordered: list[tuple[int, ...]] = list(buckets.get(None, []))
        for s in spine:
            ordered.append(s)
            ordered.extend(buckets.get(s, []))
        for ch in kids:  # 兜底
            if ch.path not in ordered:
                ordered.append(ch.path)
        merged_child_order[node.path] = ordered

        for ch in kids:
            merge_group(ch)

    merge_group(tree)

    flat: list[str] = []
    doc_index = {c.id: i for i, c in enumerate(merged_clauses)}

    def emit(node) -> None:
        own = sorted((cid for cid in node.clause_ids if cid not in dead),
                     key=lambda x: doc_index[x])
        flat.extend(own)
        for p in merged_child_order.get(node.path, []):
            ch = node.children.get(p)
            if ch is not None:
                emit(ch)

    emit(tree)
    return flat


# ---------------------------------------------------------------- 主流程

def _recompute(content: str, order: list[str], explicit_deleted: list[str],
               ignores: list[str]) -> dict:
    preamble, clauses = parse_contract(content)
    known = {c.id for c in clauses}
    order = [cid for cid in order if cid in known]
    deleted = [cid for cid in explicit_deleted if cid in known]
    ref_result = analyze(clauses, order, deleted)
    ref_result["preamble"] = preamble
    dead = cascade_deleted(clauses, deleted)
    term_result = review_terms(preamble, clauses, dead)
    ignore_set = set(ignores)
    blockers = {
        "reference_issues": list(ref_result["issues"]),
        "term_issues": [it for it in term_result["issues"]
                        if it["occ_id"] not in ignore_set],
    }
    doc_index = {c.id: i for i, c in enumerate(clauses)}
    deleted_out = sorted(dead, key=lambda x: doc_index.get(x, 10 ** 9))
    return {
        "content": content,
        "order": list(ref_result["order"]),
        "deleted": deleted_out,
        "term_ignores": list(ignores),
        "ref_result": ref_result,
        "term_result": term_result,
        "blockers": blockers,
    }


def build_merge_plan(draft_row: dict, old_row: dict, new_row: dict,
                     selections: list[dict] | None = None) -> dict[str, Any]:
    """生成合并计划（selections 非空时附带合并预览与阻塞项）。不写库。"""
    d = _draft_side(draft_row)
    o = _load_side(old_row)
    n = _load_side(new_row)

    items = _build_items(d, o, n)
    chosen, invalid = normalize_selections(items, selections or [])

    conflicts = []
    for it in items:
        for dim, st in it["dims"].items():
            if st["status"] == "conflict":
                conflicts.append({
                    "clause_id": it["clause_id"], "dim": dim,
                    "reasons": st["reasons"],
                })

    selected_out = [
        dict({"clause_id": cid}, **{dim: (dim in dims) for dim in _DIMS})
        for cid, dims in sorted(chosen.items())
    ]

    counts = {
        "items": len(items),
        "selectable": sum(
            1 for it in items
            for st in it["dims"].values() if st["status"] == "selectable"),
        "conflicts": len(conflicts),
        "selected": len(chosen),
        "invalid_selections": len(invalid),
    }

    plan: dict[str, Any] = {
        "draft": {
            "id": draft_row["id"], "title": draft_row["title"],
            "revision": draft_row["revision"],
            "fingerprint": snapshot_fingerprint(
                draft_row["content"],
                _json(draft_row["order_json"]),
                _json(draft_row["deleted_json"]),
                _json(draft_row["term_ignores_json"] or "[]"),
            ),
        },
        "old": _version_meta(old_row),
        "new": _version_meta(new_row),
        "items": items,
        "conflicts": conflicts,
        "invalid_selections": invalid,
        "selection": selected_out,
        "counts": counts,
        "sides": {
            "draft": _draft_view(d),
            "old": _side_snapshot(o),
            "new": _side_snapshot(n),
        },
        "merged": None,
    }

    if chosen:
        content, _ = _merge_content(d, n, chosen)
        merged_clauses = parse_contract(content)[1]
        dead, explicit_deleted = _merge_deleted(
            o, d, merged_clauses, chosen)
        order = _merge_order(d, o, n, merged_clauses, dead, chosen)
        merged = _recompute(
            content, order, explicit_deleted,
            _json(draft_row["term_ignores_json"] or "[]"))
        merged["has_blockers"] = bool(
            merged["blockers"]["reference_issues"]
            or merged["blockers"]["term_issues"])
        plan["merged"] = merged
    return plan


def _json(raw: str):
    import json
    return json.loads(raw)


def _version_meta(row: dict) -> dict:
    return {
        "id": row["id"], "version_no": row["version_no"],
        "revision": row["revision"], "fingerprint": row["fingerprint"],
    }


def _draft_view(s: _Side) -> dict:
    view = _side_snapshot(s)
    view["version_no"] = None
    return view


def has_plan_blockers(plan: dict) -> bool:
    merged = plan.get("merged")
    return bool(merged and merged.get("has_blockers"))
