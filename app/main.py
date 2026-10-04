"""FastAPI 入口：解析/重排分析 + 草稿持久化（SQLite）。"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .parser import parse_contract
from .references import analyze, cascade_deleted
from .terms import review_terms
from .preview import preview_deletion
from .issuance import (
    build_summary,
    has_blockers,
    snapshot_fingerprint,
    validate_snapshot,
)
from .audit import audit_versions
from .merge import build_merge_plan

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "..", "data"))
DB_PATH = os.path.join(DATA_DIR, "drafts.db")


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                order_json TEXT NOT NULL DEFAULT '[]',
                deleted_json TEXT NOT NULL DEFAULT '[]',
                term_ignores_json TEXT NOT NULL DEFAULT '[]',
                revision INTEGER NOT NULL DEFAULT 1,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        # 兼容旧库：补列（SQLite 不支持 IF NOT EXISTS 加列，先查再补）
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(drafts)")}
        if "term_ignores_json" not in cols:
            conn.execute(
                "ALTER TABLE drafts ADD COLUMN term_ignores_json "
                "TEXT NOT NULL DEFAULT '[]'"
            )
        if "revision" not in cols:
            conn.execute(
                "ALTER TABLE drafts ADD COLUMN revision "
                "INTEGER NOT NULL DEFAULT 1"
            )

        # 生效版本：快照四要素（正文/顺序/删除/忽略）+ 签发时的编号、改写与绑定；
        # 一经写入不可修改，后续编辑只产生新版本号，不覆盖旧版本。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                draft_id INTEGER NOT NULL REFERENCES drafts(id),
                version_no INTEGER NOT NULL,
                revision INTEGER NOT NULL,
                fingerprint TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                order_json TEXT NOT NULL DEFAULT '[]',
                deleted_json TEXT NOT NULL DEFAULT '[]',
                term_ignores_json TEXT NOT NULL DEFAULT '[]',
                ref_result_json TEXT NOT NULL,
                term_result_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                issued_at INTEGER NOT NULL,
                UNIQUE(draft_id, fingerprint),
                UNIQUE(draft_id, version_no)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_versions_draft "
            "ON versions(draft_id, version_no)"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="条款引用重排台", lifespan=lifespan)


# ---------------------------------------------------------------- 请求模型

class AnalyzeRequest(BaseModel):
    text: str = ""
    order: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)


class DraftIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    content: str = ""
    order: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    term_ignores: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- API

@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/analyze")
def analyze_text(req: AnalyzeRequest) -> dict:
    preamble, clauses = parse_contract(req.text)
    known = {c.id for c in clauses}
    # order 仅作「同父兄弟换序」的意图输入：未列出的条款、跨父级拖动等
    # 不合法排列都交由 analyze 按父子树规整（未列出者回到解析顺序）。
    order = [cid for cid in req.order if cid in known]
    # 删除父条款会在 analyze 中级联删除整棵子树
    deleted = [cid for cid in req.deleted if cid in known]
    result = analyze(clauses, order, deleted)
    result["preamble"] = preamble
    return result


class TermsRequest(BaseModel):
    text: str = ""
    deleted: list[str] = Field(default_factory=list)


class DeletePreviewRequest(BaseModel):
    text: str = ""
    order: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)


@app.post("/api/delete-preview")
def delete_preview(req: DeletePreviewRequest) -> dict:
    """删除影响预演（只读）：勾选存活条款 → 级联集 + 删除前后绑定差异 + 原因链。"""
    preamble, clauses = parse_contract(req.text)
    known = {c.id for c in clauses}
    order = [cid for cid in req.order if cid in known]
    deleted = [cid for cid in req.deleted if cid in known]
    candidates = [cid for cid in req.candidates if cid in known]
    try:
        return preview_deletion(preamble, clauses, order, deleted, candidates)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/terms")
def review(req: TermsRequest) -> dict:
    """定义术语审校：以当前删除状态（含级联删除）为准。"""
    preamble, clauses = parse_contract(req.text)
    known = {c.id for c in clauses}
    deleted = [cid for cid in req.deleted if cid in known]
    deleted_set = cascade_deleted(clauses, deleted)
    result = review_terms(preamble, clauses, deleted_set)
    result["deleted_clauses"] = [
        {"id": c.id, "label": c.label}
        for c in clauses if c.id in deleted_set
    ]
    return result


@app.get("/api/drafts")
def list_drafts() -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT d.id, d.title, d.revision, d.created_at, d.updated_at, "
            "COUNT(v.id) AS version_count "
            "FROM drafts d LEFT JOIN versions v ON v.draft_id = d.id "
            "GROUP BY d.id ORDER BY d.updated_at DESC"
        ).fetchall()
    out = []
    for r in rows:
        row = dict(r)
        row["version_count"] = row.get("version_count") or 0
        out.append(row)
    return out


@app.post("/api/drafts")
def create_draft(req: DraftIn) -> dict:
    import json
    now = int(time.time())
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO drafts (title, content, order_json, deleted_json, "
            "term_ignores_json, revision, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (req.title, req.content, json.dumps(req.order, ensure_ascii=False),
             json.dumps(req.deleted, ensure_ascii=False),
             json.dumps(req.term_ignores, ensure_ascii=False), now, now),
        )
        return {"id": cur.lastrowid, "revision": 1}


def _draft_payload(row: sqlite3.Row) -> dict:
    import json
    return {
        "id": row["id"],
        "title": row["title"],
        "content": row["content"],
        "order": json.loads(row["order_json"]),
        "deleted": json.loads(row["deleted_json"]),
        "term_ignores": json.loads(row["term_ignores_json"] or "[]"),
        "revision": row["revision"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@app.get("/api/drafts/{draft_id}")
def get_draft(draft_id: int) -> dict:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM drafts WHERE id = ?", (draft_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="草稿不存在")
    return _draft_payload(row)


@app.put("/api/drafts/{draft_id}")
def update_draft(draft_id: int, req: DraftIn) -> dict:
    import json
    now = int(time.time())
    with db() as conn:
        cur = conn.execute(
            "UPDATE drafts SET title=?, content=?, order_json=?, "
            "deleted_json=?, term_ignores_json=?, "
            "revision=revision+1, updated_at=? WHERE id=?",
            (req.title, req.content, json.dumps(req.order, ensure_ascii=False),
             json.dumps(req.deleted, ensure_ascii=False),
             json.dumps(req.term_ignores, ensure_ascii=False), now, draft_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="草稿不存在")
        rev = conn.execute(
            "SELECT revision FROM drafts WHERE id=?", (draft_id,)
        ).fetchone()[0]
    return {"ok": True, "revision": rev}


@app.delete("/api/drafts/{draft_id}")
def delete_draft(draft_id: int) -> dict:
    with db() as conn:
        cur = conn.execute("DELETE FROM drafts WHERE id=?", (draft_id,))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="草稿不存在")
    # 已签发版本保留（不可变凭证，不随草稿删除而销毁）
    return {"ok": True}


# ---------------------------------------------------------------- 生效版本签发

class IssueRequest(BaseModel):
    # 乐观并发：前端持有的草稿修订号；与库内不一致则拒绝（不产生版本）
    revision: int


def _version_row_to_dict(row: sqlite3.Row, include_snapshot: bool) -> dict:
    import json
    out = {
        "id": row["id"],
        "draft_id": row["draft_id"],
        "version_no": row["version_no"],
        "revision": row["revision"],
        "title": row["title"],
        "issued_at": row["issued_at"],
        "summary": json.loads(row["summary_json"]),
    }
    if include_snapshot:
        out.update({
            "content": row["content"],
            "order": json.loads(row["order_json"]),
            "deleted": json.loads(row["deleted_json"]),
            "term_ignores": json.loads(row["term_ignores_json"] or "[]"),
            "ref_result": json.loads(row["ref_result_json"]),
            "term_result": json.loads(row["term_result_json"]),
        })
    return out


@app.get("/api/drafts/{draft_id}/versions")
def list_versions(draft_id: int) -> list[dict]:
    with db() as conn:
        # 草稿删除后其不可变版本仍保留、可读：这里不再要求草稿行存在。
        rows = conn.execute(
            "SELECT * FROM versions WHERE draft_id=? "
            "ORDER BY version_no DESC", (draft_id,)
        ).fetchall()
    return [_version_row_to_dict(r, include_snapshot=False) for r in rows]


@app.post("/api/drafts/{draft_id}/issue", status_code=201)
def issue_version(draft_id: int, req: IssueRequest) -> dict:
    """从当前草稿签发生效版本。

    一次操作内固定 正文 / 顺序 / 删除状态 / 术语忽略项，重跑引用与术语终审：
    - 存在无法解析 / 目标已删除 / 循环引用 / 未忽略术语问题 → 422 拒绝，草稿不变；
    - 校验期间草稿修订号已变化（乐观并发）→ 409 拒绝，不产生版本；
    - 相同快照重复提交 → 返回原版本（幂等，200 语义由 issued=false 表达）。
    """
    import json

    with db() as conn:
        row = conn.execute(
            "SELECT * FROM drafts WHERE id=?", (draft_id,)
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="草稿不存在")
        draft = _draft_payload(row)

        if not draft["content"].strip():
            raise HTTPException(status_code=400, detail="草稿为空，无法签发")

        # 1) 固定快照并重跑引用 + 术语终审（此处尚未写库，任何拒绝都不改动草稿）
        preamble, ref_result, term_result, blockers = validate_snapshot(
            draft["content"], draft["order"], draft["deleted"],
            draft["term_ignores"],
        )
        ref_result["preamble"] = preamble
        if has_blockers(blockers):
            raise HTTPException(status_code=422, detail={
                "message": "存在未通过的引用或术语问题，已拒绝签发（草稿未修改）",
                "blockers": blockers,
            })

        fingerprint = snapshot_fingerprint(
            draft["content"], draft["order"], draft["deleted"],
            draft["term_ignores"],
        )

        # 2) 加事务锁后复查：并发期间草稿被改 / 已被同一快照抢先签发
        conn.execute("BEGIN IMMEDIATE")
        cur_row = conn.execute(
            "SELECT revision FROM drafts WHERE id=?", (draft_id,)
        ).fetchone()
        if cur_row is None:
            conn.rollback()
            raise HTTPException(status_code=404, detail="草稿不存在")
        current_rev = cur_row[0]
        if current_rev != req.revision or current_rev != draft["revision"]:
            conn.rollback()
            raise HTTPException(status_code=409, detail={
                "message": "草稿在校验期间已被修改（修订号变化），已拒绝签发",
                "current_revision": current_rev,
            })

        existing = conn.execute(
            "SELECT * FROM versions WHERE draft_id=? AND fingerprint=?",
            (draft_id, fingerprint),
        ).fetchone()
        if existing is not None:
            conn.rollback()
            payload = _version_row_to_dict(existing, include_snapshot=False)
            payload["issued"] = False
            payload["reused"] = True
            return payload

        summary = build_summary(ref_result, term_result, draft["term_ignores"])
        max_no = conn.execute(
            "SELECT COALESCE(MAX(version_no), 0) FROM versions WHERE draft_id=?",
            (draft_id,),
        ).fetchone()[0]
        version_no = max_no + 1
        now = int(time.time())

        cur = conn.execute(
            "INSERT INTO versions (draft_id, version_no, revision, fingerprint, "
            "title, content, order_json, deleted_json, term_ignores_json, "
            "ref_result_json, term_result_json, summary_json, issued_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (draft_id, version_no, current_rev, fingerprint, draft["title"],
             draft["content"], json.dumps(draft["order"], ensure_ascii=False),
             json.dumps(draft["deleted"], ensure_ascii=False),
             json.dumps(draft["term_ignores"], ensure_ascii=False),
             json.dumps(ref_result, ensure_ascii=False),
             json.dumps(term_result, ensure_ascii=False),
             json.dumps(summary, ensure_ascii=False), now),
        )
        new_id = cur.lastrowid
        conn.commit()
        saved = conn.execute(
            "SELECT * FROM versions WHERE id=?", (new_id,)
        ).fetchone()

    payload = _version_row_to_dict(saved, include_snapshot=False)
    payload["issued"] = True
    return payload


@app.get("/api/versions/{version_id}")
def get_version(version_id: int) -> dict:
    """查看单个不可变版本的完整内容（含编号、引用改写与术语绑定快照）。"""
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM versions WHERE id=?", (version_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="版本不存在")
    payload = _version_row_to_dict(row, include_snapshot=True)
    payload["issued"] = False
    return payload


class AuditRequest(BaseModel):
    # 审计方向固定为 old_version → new_version（均须为同一草稿的已签发版本）
    old_version_id: int
    new_version_id: int


@app.post("/api/versions/audit")
def audit(req: AuditRequest) -> dict:
    """版本差异审计：对齐同一草稿的两份不可变快照。

    只读取 versions 行：草稿后续编辑 / 删除均不影响审计结果。
    两版非同草稿或指向同一版本 → 400；版本不存在 → 404。
    """
    with db() as conn:
        rows = {
            r["id"]: r
            for r in conn.execute(
                "SELECT * FROM versions WHERE id IN (?, ?)",
                (req.old_version_id, req.new_version_id),
            ).fetchall()
        }
    old_row = rows.get(req.old_version_id)
    new_row = rows.get(req.new_version_id)
    if old_row is None or new_row is None:
        missing = req.old_version_id if old_row is None else req.new_version_id
        raise HTTPException(status_code=404, detail=f"版本不存在：{missing}")
    try:
        return audit_versions(old_row, new_row)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------- 审计结果应用到当前草稿

class MergeSelection(BaseModel):
    clause_id: str
    body: bool = False
    order: bool = False
    lifecycle: bool = False
    add: bool = False


class MergePlanRequest(BaseModel):
    draft_id: int
    old_version_id: int
    new_version_id: int
    revision: int
    selections: list[MergeSelection] = Field(default_factory=list)


class MergeCommitRequest(BaseModel):
    draft_id: int
    old_version_id: int
    new_version_id: int
    # 乐观锁：草稿修订号；与库内不一致 → 409，不写任何结果
    revision: int
    # 两版快照指纹：版本不可变但要校验调用方预览的就是库内这两版
    old_fingerprint: str
    new_fingerprint: str
    selections: list[MergeSelection] = Field(default_factory=list)


def _load_merge_rows(conn, draft_id: int, old_id: int, new_id: int):
    draft = conn.execute(
        "SELECT * FROM drafts WHERE id=?", (draft_id,)
    ).fetchone()
    if draft is None:
        raise HTTPException(status_code=404, detail="草稿不存在")
    vrows = {
        r["id"]: r
        for r in conn.execute(
            "SELECT * FROM versions WHERE id IN (?, ?)", (old_id, new_id)
        ).fetchall()
    }
    old_row, new_row = vrows.get(old_id), vrows.get(new_id)
    if old_row is None or new_row is None:
        missing = old_id if old_row is None else new_id
        raise HTTPException(status_code=404, detail=f"版本不存在：{missing}")
    return draft, old_row, new_row


def _selections_dicts(items: list[MergeSelection]) -> list[dict]:
    return [it.model_dump() for it in items]


def _validate_merge_pair(draft_id: int, old_id: int, new_id: int,
                         old_row, new_row) -> None:
    if old_id == new_id:
        raise HTTPException(status_code=400, detail="请选择两个不同的版本")
    if old_row["draft_id"] != draft_id or new_row["draft_id"] != draft_id:
        raise HTTPException(status_code=400,
                            detail="只能把同一草稿签发的版本合并回该草稿")


@app.post("/api/versions/merge-plan")
def merge_plan(req: MergePlanRequest) -> dict:
    """审计结果「应用到当前草稿」的只读预览：三方差异可选项、冲突与合并后分析。

    以旧版为基线、新版为候选，读取当前草稿，返回逐条可勾选维度
    （正文 / 顺序 / 删除恢复 / 新增）；草稿相对旧版已改动同一条款或其父级的
    维度标为冲突。携带 selections 时在合并结果上重跑引用与术语审校并给出
    阻塞项（引用改写与术语绑定一律重新计算，不拼接）。
    """
    with db() as conn:
        draft, old_row, new_row = _load_merge_rows(
            conn, req.draft_id, req.old_version_id, req.new_version_id)
    _validate_merge_pair(req.draft_id, req.old_version_id,
                         req.new_version_id, old_row, new_row)
    if draft["revision"] != req.revision:
        raise HTTPException(status_code=409, detail={
            "message": "草稿修订号已变化，请载入最新草稿后重新生成合并预览",
            "current_revision": draft["revision"],
        })
    try:
        return build_merge_plan(
            draft, old_row, new_row, _selections_dicts(req.selections))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/drafts/{draft_id}/merge", status_code=200)
def merge_into_draft(draft_id: int, req: MergeCommitRequest) -> dict:
    """把审计中勾选的变更一次性合并进当前草稿（单事务，新修订）。

    - 修订号过期（409）、两版快照指纹不符（409）、草稿被删（404）；
    - 选择项落在冲突 / 不可选维度，或合并后存在引用 / 未忽略术语阻塞项
      → 422 整体拒绝，不写部分结果；
    - 通过后在 BEGIN IMMEDIATE 事务内复查修订号与快照，一次性保存新修订。
    """
    import json

    if req.draft_id != draft_id:
        raise HTTPException(status_code=400, detail="草稿 id 不一致")
    with db() as conn:
        draft, old_row, new_row = _load_merge_rows(
            conn, draft_id, req.old_version_id, req.new_version_id)
    _validate_merge_pair(draft_id, req.old_version_id, req.new_version_id,
                         old_row, new_row)
    if draft["revision"] != req.revision:
        raise HTTPException(status_code=409, detail={
            "message": "草稿修订号已变化，合并已拒绝（未写入）",
            "current_revision": draft["revision"],
        })
    if (old_row["fingerprint"] != req.old_fingerprint
            or new_row["fingerprint"] != req.new_fingerprint):
        raise HTTPException(status_code=409, detail={
            "message": "旧版 / 新版快照与预览时不一致，合并已拒绝（未写入）",
        })

    plan = build_merge_plan(
        draft, old_row, new_row, _selections_dicts(req.selections))

    if plan["invalid_selections"]:
        raise HTTPException(status_code=422, detail={
            "message": "存在冲突或不可带入的勾选项，合并已拒绝（未写入）",
            "invalid_selections": plan["invalid_selections"],
            "conflicts": plan["conflicts"],
        })
    merged = plan["merged"]
    if merged is None:
        raise HTTPException(status_code=400,
                            detail="未勾选任何变更，无需合并")
    if merged["has_blockers"]:
        raise HTTPException(status_code=422, detail={
            "message": "合并后的草稿存在未通过的引用或术语问题，已整体拒绝（未写入）",
            "blockers": merged["blockers"],
        })

    now = int(time.time())
    with db() as conn:
        conn.execute("BEGIN IMMEDIATE")
        cur_row = conn.execute(
            "SELECT revision FROM drafts WHERE id=?", (draft_id,)
        ).fetchone()
        if cur_row is None:
            conn.rollback()
            raise HTTPException(status_code=404, detail="草稿不存在")
        current_rev = cur_row[0]
        if current_rev != req.revision:
            conn.rollback()
            raise HTTPException(status_code=409, detail={
                "message": "草稿在合并校验期间已被修改（修订号变化），已拒绝（未写入）",
                "current_revision": current_rev,
            })
        conn.execute(
            "UPDATE drafts SET title=?, content=?, order_json=?, "
            "deleted_json=?, term_ignores_json=?, "
            "revision=revision+1, updated_at=? WHERE id=?",
            (draft["title"], merged["content"],
             json.dumps(merged["order"], ensure_ascii=False),
             json.dumps(merged["deleted"], ensure_ascii=False),
             json.dumps(merged["term_ignores"], ensure_ascii=False),
             now, draft_id),
        )
        new_revision = conn.execute(
            "SELECT revision FROM drafts WHERE id=?", (draft_id,)
        ).fetchone()[0]
        conn.commit()

    return {
        "ok": True,
        "revision": new_revision,
        "applied": plan["selection"],
        "content": merged["content"],
        "order": merged["order"],
        "deleted": merged["deleted"],
        "term_ignores": merged["term_ignores"],
    }


# ---------------------------------------------------------------- 静态页面

STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
