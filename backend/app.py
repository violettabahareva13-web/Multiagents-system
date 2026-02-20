import logging
import uuid
import time
import math
from datetime import datetime
from contextlib import asynccontextmanager
from html import escape

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langgraph.types import Command

import database
from config import CORS_ORIGINS
from database import (
    init_database_pool,
    shutdown_database_pool,
    load_schema,
    DB_SCHEMA,
    reconfigure_database,
    get_db_status,
    get_schema_structured,
)
import qdrant
from qdrant import init_vectorstore_async, delete_cache_entry
from agent.graph import build_graph, graph
import agent.graph as agent_graph_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

ERROR_KEYWORDS = [
    'ошибка', 'error', 'не могу', 'невозможно',
    'некорректн', 'не найден', 'failed', '{', 'success'
]

_last_schema_structured: dict | None = None
_last_schema_source: str = "live"
_last_schema_fallback_reason: str | None = None


def _extract_interrupt_value(result: dict):
    """Извлекает payload из __interrupt__ (Interrupt namedtuple или dict)."""
    interrupts = result.get("__interrupt__", [])
    first = interrupts[0] if interrupts else None
    if first is None:
        return None
    if hasattr(first, "value"):
        return first.value
    if isinstance(first, dict) and "value" in first:
        return first["value"]
    return first


def _try_cache_response(query: str, response_text: str, data_rows, session_id: str):
    """Пытается закэшировать ответ в Qdrant, если он не ошибочный."""
    store = qdrant.get_active_vectorstore()
    if not store or not qdrant.embeddings:
        return
    if not query:
        return

    response = response_text.strip()
    is_error = any(kw in response.lower() for kw in ERROR_KEYWORDS)

    if is_error:
        logger.info("⚠️ Ошибочный ответ не кэшируется")
        return

    try:
        doc = Document(
            page_content=query,
            metadata={
                'response': response,
                'data': data_rows or [],
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
            }
        )
        point_id = str(uuid.uuid4())
        store.add_documents(documents=[doc], ids=[point_id])
        logger.info(f"📥 Добавлено в Qdrant кэш | ID: {point_id[:8]}...")
    except Exception as e:
        logger.error(f"❌ Ошибка при добавлении в кэш: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Startup начинается...")

    global _last_schema_structured
    try:
        init_database_pool()
        load_schema()
        agent_graph_module.DB_SCHEMA = database.DB_SCHEMA
        try:
            _last_schema_structured = get_schema_structured()
        except Exception:
            pass
    except Exception as e:
        logger.error(f"❌ Не удалось подключиться к БД при старте: {e}")
        logger.warning("⚠️ Приложение запущено без активного подключения к БД")

    init_vectorstore_async()
    if qdrant.get_active_vectorstore() is None:
        logger.warning("⚠️ Qdrant недоступен — приложение работает БЕЗ кэширования")

    global graph
    graph = build_graph()
    logger.info("✅ Граф построен и сохранён глобально")

    if qdrant.get_active_vectorstore() is None:
        qdrant.start_reconnect_task(interval=30)
        logger.info("✅ Сервер готов (кэш отключён, переподключение в фоне)")
    else:
        logger.info("✅ Сервер полностью готов")

    yield

    shutdown_database_pool()


app = FastAPI(title='SQL Agent', lifespan=lifespan)

_default_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + _default_origins if CORS_ORIGINS else _default_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    session_id: str
    message: str | None = None
    question: str | None = None
    mode: str | None = None


class ResumeRequest(BaseModel):
    session_id: str
    data: dict


class AuthLoginRequest(BaseModel):
    session_id: str
    username: str
    password: str


class DbConnectProfile(BaseModel):
    name: str | None = None
    dsn: str | None = None
    host: str | None = None
    port: int | None = None
    database: str | None = None
    user: str | None = None
    password: str | None = None


class DbConnectRequest(BaseModel):
    session_id: str | None = None
    profile: DbConnectProfile


@app.get("/health")
async def health():
    status = get_db_status()
    return {"ok": True, **status}


@app.get("/db/status")
async def db_status():
    return get_db_status()


@app.post("/db/connect")
async def db_connect(request: DbConnectRequest):
    global graph
    try:
        profile = request.profile
        reconfigure_database(
            {
                "dsn": profile.dsn,
                "host": profile.host,
                "port": profile.port,
                "database": profile.database,
                "user": profile.user,
                "password": profile.password or "",
            }
        )
        qdrant.get_active_vectorstore()
        agent_graph_module.DB_SCHEMA = database.DB_SCHEMA
        graph = build_graph()
        status = get_db_status()
        if not status.get("connected"):
            raise HTTPException(status_code=400, detail=status.get("error", "Не удалось подключиться"))
        return {"ok": True, "cache_collection": qdrant.get_active_collection_name(), **status}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка подключения к БД: {e}")


@app.post("/db/disconnect")
async def db_disconnect():
    shutdown_database_pool()
    return {"ok": True, "connected": False}


@app.post("/auth/login")
async def auth_login(request: AuthLoginRequest):
    if not request.username:
        raise HTTPException(status_code=400, detail="username обязателен")
    return {"ok": True, "connected": bool(get_db_status().get("connected")), "user": request.username}


@app.get("/auth/me")
async def auth_me():
    status = get_db_status()
    return {
        "ok": True,
        "connected": bool(status.get("connected")),
        "user": status.get("user"),
    }


@app.get("/api/db/schema")
async def api_db_schema(refresh: bool = Query(default=False)):
    global _last_schema_structured, _last_schema_source, _last_schema_fallback_reason
    try:
        if refresh:
            load_schema()
        schema = get_schema_structured()

        meta = dict(schema.get("metadata", {}) or {})
        meta["source"] = "live"
        meta["is_fallback"] = False
        if _last_schema_fallback_reason:
            meta["last_fallback_error"] = _last_schema_fallback_reason
        schema["metadata"] = meta

        _last_schema_structured = schema
        _last_schema_source = "live"
        _last_schema_fallback_reason = None
        return schema
    except Exception as e:
        if _last_schema_structured is not None:
            logger.warning(f"⚠️ Возвращаем кэш схемы из-за ошибки live-запроса: {e}")
            fallback = dict(_last_schema_structured)
            fallback_meta = dict(fallback.get("metadata", {}) or {})
            fallback_meta["source"] = "cache"
            fallback_meta["is_fallback"] = True
            fallback_meta["fallback_reason"] = str(e)
            fallback_meta["fallback_at"] = int(datetime.now().timestamp())
            fallback["metadata"] = fallback_meta
            _last_schema_source = "cache"
            _last_schema_fallback_reason = str(e)
            return fallback

        raise HTTPException(status_code=500, detail=f"Ошибка получения схемы: {e}")


def _port_id(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    return safe or "col"


def _dom_id(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(value))
    return safe or "id"


def _build_erd_dot(schema: dict) -> str:
    tables = sorted(schema.get("tables", []))
    columns_map = schema.get("columns", {}) or {}
    pk_map = schema.get("primary_keys", {}) or {}
    foreign_keys = schema.get("foreign_keys", []) or []

    fk_cols_by_table: dict[str, set[str]] = {}
    for fk in foreign_keys:
        ft = str(fk.get("from_table", ""))
        fc = str(fk.get("from_column", ""))
        if ft and fc:
            fk_cols_by_table.setdefault(ft, set()).add(fc)

    lines = [
        "digraph ERD {",
        "  rankdir=LR;",
        "  graph [fontname=\"Arial\", bgcolor=\"white\", splines=true, pad=\"0.3\", nodesep=\"0.35\", ranksep=\"0.7\"];",
        "  node [shape=plain, fontname=\"Arial\"];",
        "  edge [fontname=\"Arial\", color=\"#64748b\", arrowsize=0.7, penwidth=1.2];",
    ]

    for table in tables:
        pk_set = set(pk_map.get(table, []))
        fk_set = fk_cols_by_table.get(table, set())
        cols = columns_map.get(table, []) or []

        rows = [
            f'    <TR><TD COLSPAN="2" BGCOLOR="#0f172a"><FONT COLOR="white"><B>{escape(str(table))}</B></FONT></TD></TR>'
        ]

        if not cols:
            rows.append('    <TR><TD ALIGN="LEFT" COLSPAN="2"><FONT COLOR="#64748b">нет колонок</FONT></TD></TR>')
        else:
            for col in cols:
                name = str(col.get("name", ""))
                typ = str(col.get("type", ""))
                mark = []
                if name in pk_set:
                    mark.append("PK")
                if name in fk_set:
                    mark.append("FK")
                mark_text = f"[{','.join(mark)}] " if mark else ""
                port = _port_id(name)
                rows.append(
                    f'    <TR><TD PORT="{port}" ALIGN="LEFT">{escape(mark_text + name)}</TD><TD ALIGN="LEFT"><FONT COLOR="#475569">{escape(typ)}</FONT></TD></TR>'
                )

        label = "<\n  <TABLE BORDER=\"1\" CELLBORDER=\"0\" CELLSPACING=\"0\" CELLPADDING=\"4\" COLOR=\"#334155\">\n"
        label += "\n".join(rows)
        label += "\n  </TABLE>\n>"
        lines.append(f'  "{table}" [label={label}];')

    for fk in foreign_keys:
        ft = str(fk.get("from_table", ""))
        fc = str(fk.get("from_column", ""))
        tt = str(fk.get("to_table", ""))
        tc = str(fk.get("to_column", ""))
        if not (ft and tt):
            continue

        from_port = _port_id(fc) if fc else ""
        to_port = _port_id(tc) if tc else ""

        from_ref = f"\"{ft}\":{from_port}" if from_port else f"\"{ft}\""
        to_ref = f"\"{tt}\":{to_port}" if to_port else f"\"{tt}\""
        label = escape(f"{fc} → {tc}") if (fc and tc) else ""
        edge_label = f' [label="{label}", fontsize=10]' if label else ""
        lines.append(f"  {from_ref} -> {to_ref}{edge_label};")

    lines.append("}")
    return "\n".join(lines)


def _build_erd_svg(schema: dict) -> str:
    tables = sorted(schema.get("tables", []))
    columns_map = schema.get("columns", {}) or {}
    pk_map = schema.get("primary_keys", {}) or {}
    foreign_keys = schema.get("foreign_keys", []) or []

    fk_cols_by_table: dict[str, set[str]] = {}
    for fk in foreign_keys:
        ft = str(fk.get("from_table", ""))
        fc = str(fk.get("from_column", ""))
        if ft and fc:
            fk_cols_by_table.setdefault(ft, set()).add(fc)

    col_defs: dict[str, list[tuple[str, str, bool, bool]]] = {}
    table_heights: dict[str, int] = {}

    header_h = 32
    row_h = 20
    node_w = 390
    margin = 28
    gap_x = 46
    gap_y = 34

    for table in tables:
        pk_set = set(pk_map.get(table, []))
        fk_set = fk_cols_by_table.get(table, set())
        cols_raw = columns_map.get(table, []) or []
        defs: list[tuple[str, str, bool, bool]] = []
        for col in cols_raw:
            name = str(col.get("name", ""))
            typ = str(col.get("type", ""))
            defs.append((name, typ, name in pk_set, name in fk_set))
        col_defs[table] = defs
        visible_rows = max(1, len(defs))
        table_heights[table] = header_h + visible_rows * row_h + 10

    if not tables:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="180">'
            '<rect width="100%" height="100%" fill="#ffffff"/>'
            '<text x="24" y="44" font-family="Arial" font-size="18" fill="#0f172a">ERD: таблицы не найдены</text>'
            "</svg>"
        )

    ncols = min(4, max(1, math.ceil(math.sqrt(len(tables)))))
    rows: list[list[str]] = [tables[i:i + ncols] for i in range(0, len(tables), ncols)]
    row_heights = [max(table_heights[t] for t in row_tables) for row_tables in rows]

    y_starts = []
    y = margin
    for rh in row_heights:
        y_starts.append(y)
        y += rh + gap_y

    positions: dict[str, tuple[int, int]] = {}
    for r_idx, row_tables in enumerate(rows):
        for c_idx, table in enumerate(row_tables):
            x = margin + c_idx * (node_w + gap_x)
            positions[table] = (x, y_starts[r_idx])

    width = margin * 2 + ncols * node_w + (ncols - 1) * gap_x
    height = y + margin

    col_anchor: dict[tuple[str, str], tuple[float, float]] = {}
    for table in tables:
        x, y0 = positions[table]
        defs = col_defs.get(table, [])
        for idx, (name, _typ, _is_pk, _is_fk) in enumerate(defs):
            yy = y0 + header_h + 5 + idx * row_h + row_h / 2
            col_anchor[(table, name)] = (x, yy)

    edge_groups: list[str] = []
    for fk in foreign_keys:
        ft = str(fk.get("from_table", ""))
        fc = str(fk.get("from_column", ""))
        tt = str(fk.get("to_table", ""))
        tc = str(fk.get("to_column", ""))
        if ft not in positions or tt not in positions:
            continue

        fx, fy = positions[ft]
        tx, ty = positions[tt]

        from_anchor = col_anchor.get((ft, fc))
        to_anchor = col_anchor.get((tt, tc))

        if from_anchor is None:
            from_anchor = (fx + node_w, fy + header_h / 2)
        if to_anchor is None:
            to_anchor = (tx, ty + header_h / 2)

        from_left_to_right = fx <= tx
        start_x = fx + node_w if from_left_to_right else fx
        end_x = tx if from_left_to_right else tx + node_w
        start_y = from_anchor[1]
        end_y = to_anchor[1]
        dx = abs(end_x - start_x)
        c = max(40, dx * 0.35)
        c1x = start_x + c if from_left_to_right else start_x - c
        c2x = end_x - c if from_left_to_right else end_x + c

        edge_id = f"{_dom_id(ft)}__{_dom_id(fc)}__to__{_dom_id(tt)}__{_dom_id(tc)}"
        edge_items = [
            f'<path class="erd-edge-path" d="M {start_x:.1f} {start_y:.1f} C {c1x:.1f} {start_y:.1f}, {c2x:.1f} {end_y:.1f}, {end_x:.1f} {end_y:.1f}" '
            f'stroke="#475569" stroke-width="1.4" fill="none" marker-end="url(#arrow)"/>'
        ]
        if fc and tc:
            lx = (start_x + end_x) / 2
            ly = (start_y + end_y) / 2 - 4
            edge_items.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" font-family="Arial" font-size="10" fill="#334155">{escape(fc)}→{escape(tc)}</text>'
            )
        edge_groups.append(
            f'<g class="erd-edge" id="edge_{edge_id}" data-from-table="{escape(ft)}" data-to-table="{escape(tt)}" '
            f'data-from-column="{escape(fc)}" data-to-column="{escape(tc)}">'
            f'{"".join(edge_items)}'
            "</g>"
        )

    table_groups: list[str] = []
    for table in tables:
        x, y0 = positions[table]
        h = table_heights[table]
        defs = col_defs.get(table, [])
        table_id = _dom_id(table)
        node_items = [
            f'<rect class="erd-table-body" x="{x}" y="{y0}" width="{node_w}" height="{h}" rx="8" ry="8" fill="#ffffff" stroke="#334155" stroke-width="1.2"/>',
            f'<rect class="erd-table-header" x="{x}" y="{y0}" width="{node_w}" height="{header_h}" rx="8" ry="8" fill="#0f172a"/>',
            f'<text class="erd-table-title" x="{x + 10}" y="{y0 + 21}" font-family="Arial" font-size="13" font-weight="700" fill="#ffffff">{escape(table)}</text>',
        ]

        if not defs:
            node_items.append(
                f'<text x="{x + 10}" y="{y0 + header_h + 18}" font-family="Arial" font-size="12" fill="#64748b">нет колонок</text>'
            )
        else:
            for idx, (name, typ, is_pk, is_fk) in enumerate(defs):
                yy = y0 + header_h + 5 + idx * row_h
                if idx > 0:
                    node_items.append(
                        f'<line x1="{x + 1}" y1="{yy}" x2="{x + node_w - 1}" y2="{yy}" stroke="#e2e8f0" stroke-width="1"/>'
                    )
                flags = []
                if is_pk:
                    flags.append("PK")
                if is_fk:
                    flags.append("FK")
                flag_text = f"[{','.join(flags)}] " if flags else ""
                node_items.append(
                    f'<text class="erd-col-name" data-column="{escape(name)}" x="{x + 10}" y="{yy + 14}" font-family="Arial" font-size="11" fill="#0f172a">{escape(flag_text + name)}</text>'
                )
                node_items.append(
                    f'<text class="erd-col-type" x="{x + node_w - 10}" y="{yy + 14}" text-anchor="end" font-family="Arial" font-size="10" fill="#64748b">{escape(typ)}</text>'
                )
        table_groups.append(
            f'<g class="erd-table" id="table_{table_id}" data-table="{escape(table)}">{"".join(node_items)}</g>'
        )

    legend_x = margin
    legend_y = height - margin - 16
    legend = (
        f'<text x="{legend_x}" y="{legend_y}" font-family="Arial" font-size="11" fill="#475569">'
        "Обозначения: [PK] первичный ключ, [FK] внешний ключ, стрелка = связь FK → PK"
        "</text>"
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L10,4 L0,8 z" fill="#475569"/></marker></defs>'
        '<rect width="100%" height="100%" fill="#f8fafc"/>'
        f'{"".join(edge_groups)}{"".join(table_groups)}{legend}'
        "</svg>"
    )


@app.get("/api/db/schema/erd")
async def api_db_schema_erd(format: str = Query(default="svg"), refresh: bool = Query(default=False)):
    try:
        schema = await api_db_schema(refresh=refresh)
        dot = _build_erd_dot(schema)

        if format == "dot":
            return Response(content=dot, media_type="text/vnd.graphviz; charset=utf-8")
        if format != "svg":
            raise HTTPException(status_code=400, detail="Поддерживаются только format=svg|dot")

        svg = _build_erd_svg(schema)
        return Response(content=svg, media_type="image/svg+xml; charset=utf-8")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка построения ERD: {e}")


@app.post("/chat")
async def chat(request: MessageRequest):

    if graph is None:
        raise HTTPException(status_code=503, detail="Сервер еще загружается, попробуйте позже")

    user_message = (request.message or request.question or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message (или question) обязателен")

    logger.info(f"💬 Запрос [{request.session_id}]: {user_message}")
    start_time = time.perf_counter()
    config = {"configurable": {"thread_id": request.session_id}}
    input_data = {"messages": [HumanMessage(content=user_message)]}

    try:
        result = await graph.ainvoke(input_data, config)

        if isinstance(result, dict) and "__interrupt__" in result:
            value = _extract_interrupt_value(result)
            execution_time = time.perf_counter() - start_time
            itype = value.get('type') if isinstance(value, dict) else '?'
            logger.info(f"⏸ Граф приостановлен, interrupt type={itype}")

            return {
                "status": "needs_human_input",
                "session_id": request.session_id,
                "interrupt": value,
                "execution_time": round(execution_time, 3),
            }

        if not isinstance(result, dict) or "messages" not in result:
            logger.error(f"Некорректный формат результата: {type(result)}")
            raise ValueError("Ошибка логики графа: некорректный ответ")

        final_content = result["messages"][-1].content
        data_rows = result.get("query_result", [])
        execution_time = time.perf_counter() - start_time

        if result.get("from_cache") is not True:
            _try_cache_response(user_message, final_content, data_rows, request.session_id)

        logger.info(f"✅ Успешно за {execution_time:.2f}s")

        return {
            "status": "ok",
            "response": final_content,
            "data": data_rows,
            "session_id": request.session_id,
            "execution_time": round(execution_time, 3),
            "from_cache": bool(result.get("from_cache")),
        }

    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e)
        logger.error(f"❌ Ошибка в /chat: {error_str[:500]}")
        if "tool_use_failed" in error_str or "failed_generation" in error_str:
            raise HTTPException(
                status_code=502,
                detail="Модель сгенерировала некорректный запрос. Пожалуйста, переформулируйте ваш вопрос."
            )
        if "rate_limit" in error_str.lower():
            raise HTTPException(
                status_code=429,
                detail="Превышен лимит запросов к LLM. Подождите несколько секунд и попробуйте снова."
            )
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера. Попробуйте ещё раз.")


@app.post("/chat/resume")
async def chat_resume(request: ResumeRequest):

    if graph is None:
        raise HTTPException(status_code=503, detail="Сервер еще загружается, попробуйте позже")

    logger.info(f"🔁 Резюмирование сессии [{request.session_id}] с человеческим вводом")
    start_time = time.perf_counter()
    config = {"configurable": {"thread_id": request.session_id}}

    try:
        result = await graph.ainvoke(Command(resume=request.data), config)

        if isinstance(result, dict) and "__interrupt__" in result:
            value = _extract_interrupt_value(result)
            execution_time = time.perf_counter() - start_time
            itype = value.get('type') if isinstance(value, dict) else '?'
            logger.info(f"⏸ Граф снова приостановлен, interrupt type={itype}")

            return {
                "status": "needs_human_input",
                "session_id": request.session_id,
                "interrupt": value,
                "execution_time": round(execution_time, 3),
            }

        if not isinstance(result, dict) or "messages" not in result:
            logger.error(f"Некорректный формат результата при resume: {type(result)}")
            raise ValueError("Ошибка логики графа при resume: некорректный ответ")

        final_content = result["messages"][-1].content
        data_rows = result.get("query_result", [])
        execution_time = time.perf_counter() - start_time

        if result.get("from_cache") is not True:
            original_query = (result.get("original_query") or "").strip()

            reject_query = (result.get("cache_reject_query") or "").strip()
            if reject_query:
                delete_cache_entry(reject_query)

            _try_cache_response(original_query, final_content, data_rows, request.session_id)

        logger.info(f"✅ (resume) Успешно за {execution_time:.2f}s")

        return {
            "status": "ok",
            "response": final_content,
            "data": data_rows,
            "session_id": request.session_id,
            "execution_time": round(execution_time, 3),
            "from_cache": bool(result.get("from_cache")),
        }

    except HTTPException:
        raise
    except Exception as e:
        error_str = str(e)
        logger.error(f"❌ Ошибка в /chat/resume: {error_str[:500]}")
        if "tool_use_failed" in error_str or "failed_generation" in error_str:
            raise HTTPException(
                status_code=502,
                detail="Модель сгенерировала некорректный запрос. Пожалуйста, переформулируйте ваш вопрос."
            )
        if "rate_limit" in error_str.lower():
            raise HTTPException(
                status_code=429,
                detail="Превышен лимит запросов к LLM. Подождите несколько секунд и попробуйте снова."
            )
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера. Попробуйте ещё раз.")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
