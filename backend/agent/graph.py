'''
Основной граф
'''
import json
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq

from agent.state import AgentState
from execute_tools import execute_tool_node, should_continue
from qdrant import checked_cache, cache_should_continue
from database import DB_SCHEMA, run_sql, get_postgres_schema
from config import GROQ_API_KEY
from visual import graph_vis, safe_exec, review_visualization

import logging
logger = logging.getLogger(__name__)

critic_llm = ChatGroq(model="llama-3.1-8b-instant", 
               temperature=0, 
               api_key=GROQ_API_KEY,
               max_tokens=1024,
               )

llm = ChatGroq(model="llama-3.3-70b-versatile", 
               temperature=0, 
               api_key=GROQ_API_KEY,
               max_tokens=4096,
               model_kwargs={
                "top_p": 0.1,  
                "frequency_penalty": 0.5,
            })


JSON_HINTS = """
ВАЖНО — работа с JSON/JSONB колонками:
- Колонки типа JSONB (например city, airport_name) содержат объекты вида {"ru": "Москва", "en": "Moscow"}
- Для фильтрации: WHERE city->>'ru' = 'Москва' (НЕ city = 'Москва')
- Для вывода: SELECT city->>'ru' AS city_name
- Для поиска: WHERE city->>'ru' ILIKE '%Моск%'
- НИКОГДА не сравнивай JSONB-колонку напрямую со строкой!"""

VIS_KEYWORDS = [
    "график", "графики", "диаграмм", "chart", "plot", "нарисуй",
    "визуализ", "построй", "покажи график", "bar", "pie",
    "линейный", "столбчат", "круговая", "heatmap", "визуализируй",
    "построй график", "покажи на графике"
]


def _is_vis_request(text: str) -> bool:
    """Проверяет, является ли текст запросом на визуализацию."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in VIS_KEYWORDS)


def _schema_for_prompt(max_chars: int = 12000) -> str:
    """Ограничивает размер схемы в промпте, чтобы не раздувать контекст."""
    schema = DB_SCHEMA or "Схема БД недоступна"
    if len(schema) <= max_chars:
        return schema
    return f"{schema[:max_chars]}\n\n...[schema truncated]..."


def _final_no_data_message(original_query: str) -> str:
    query = (original_query or "").strip()
    if not query:
        return (
            "По текущему запросу не найдено данных. "
            "Уточните период, фильтры или названия сущностей и попробуйте снова."
        )
    return (
        f"По запросу «{query}» данные не найдены. "
        "Проверьте условия фильтрации, диапазон дат или формулировку запроса."
    )


def assistant(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1] if messages else None
    critic_attempts = state.get("critic_attempts", 0)

    if isinstance(last_msg, ToolMessage) and getattr(last_msg, "name", "") == "run_sql":
        try:
            parsed = json.loads(last_msg.content)
            if parsed.get("success") and int(parsed.get("row_count", 0)) == 0:
                return {"messages": [AIMessage(content=_final_no_data_message(state.get("original_query", "")))]}
        except Exception:
            pass

    if (
        critic_attempts >= 3
        and isinstance(last_msg, ToolMessage)
        and getattr(last_msg, "name", "") == "run_sql"
    ):
        try:
            parsed = json.loads(last_msg.content)
            if parsed.get("success") and int(parsed.get("row_count", 0)) == 0:
                return {"messages": [AIMessage(content=_final_no_data_message(state.get("original_query", "")))]}
            if not parsed.get("success"):
                err = str(parsed.get("error", "")).strip()
                if err:
                    return {
                        "messages": [
                            AIMessage(
                                content=(
                                    "Не удалось получить данные из БД после нескольких попыток. "
                                    f"Последняя ошибка: {err[:220]}"
                                )
                            )
                        ]
                    }
        except Exception:
            pass

    MAX_HISTORY = 12
    if len(messages) > MAX_HISTORY:
        messages_for_llm = list(messages[-MAX_HISTORY:])
    else:
        messages_for_llm = list(messages)

    came_from_critic = (
        isinstance(last_msg, AIMessage) and
        getattr(last_msg, "name", None) == "sql_critic"
    )

    has_data_to_present = False
    query_result = state.get("query_result", [])
    if query_result and isinstance(query_result, list) and len(query_result) > 0:
        has_data_to_present = True

    original_query = state.get("original_query", "")
    is_vis = _is_vis_request(original_query)
    schema_prompt = _schema_for_prompt()

    if came_from_critic:
        llm_with_tools = llm.bind_tools([run_sql], tool_choice="required")
        system_content = f"""Критик указал на ошибку. Схема БД:
{schema_prompt}
{JSON_HINTS}

Прочитай критику выше и СЕЙЧАС ЖЕ:
1. Напиши исправленный SQL
2. Вызови run_sql
НЕ ПИШИ НИКАКОГО ТЕКСТА — ТОЛЬКО инструмент!"""

    elif is_vis and not has_data_to_present:
        llm_with_tools = llm.bind_tools([run_sql], tool_choice="required")
        system_content = f"""Ты — SQL-агент с возможностью визуализации. Схема БД:
{schema_prompt}
{JSON_HINTS}

Пользователь просит визуализировать данные.
Данных в текущем состоянии НЕТ — нужно их получить.

Твои действия:
1. Посмотри историю переписки и найди, о каких данных идёт речь.
2. Составь SQL-запрос, который вернёт нужные данные.
3. ОБЯЗАТЕЛЬНО вызови run_sql с этим запросом.

НЕ ПИШИ ТЕКСТ. НЕ ГОВОРИ, ЧТО НЕ МОЖЕШЬ. Просто вызови run_sql!"""

    elif has_data_to_present:
        llm_with_tools = llm.bind_tools([run_sql])
        system_content = f"""Ты — SQL-аналитик. Данные УЖЕ получены из базы (последний ToolMessage).

Твоя задача:
1. **Кратко ответь на вопрос пользователя**, опираясь только на полученные данные.
2. **Основной формат ответа — таблица в Markdown**:
   - Столбцы должны соответствовать самым важным полям данных (например, города, количество рейсов, суммарные значения и т.п.).
   - Не добавляй длинных текстовых описаний; максимум 1–2 короткие строки выше или ниже таблицы.
3. **НЕ показывай SQL-код** пользователю.
4. **НЕ вызывай инструменты** — данные уже есть.

Если данные представляют собой список городов/типов/объектов — выведи ИХ ЧЁТКИЙ СПИСОК В ТАБЛИЦЕ (одна строка на элемент, с понятными заголовками столбцов).

Пиши ответ строго в одном сообщении, таблица должна быть валидной Markdown-таблицей."""

    else:
        llm_with_tools = llm.bind_tools([run_sql, get_postgres_schema])
        system_content = f"""Ты — SQL-агент. Схема БД:
{schema_prompt}
{JSON_HINTS}

Правила:
- Для любых вопросов о данных → сразу вызывай run_sql
- Если пользователь просит визуализацию/график — найди в истории нужный запрос и вызови run_sql
- Никогда не показывай SQL в ответе пользователю
- После получения данных — сразу пиши красивый ответ
- Если вопрос не про данные — скажи, что ты работаешь только с базой"""

    system_msg = SystemMessage(content=system_content)
    
    try:
        response = llm_with_tools.invoke([system_msg] + messages_for_llm)
    except Exception as e:
        error_str = str(e)
        if "tool_use_failed" in error_str or "failed_generation" in error_str:
            logger.warning(f"⚠️ LLM сгенерировал невалидный tool_call, повтор с упрощённым промптом: {error_str[:200]}")
            try:
                retry_system = SystemMessage(content=f"""Ты — SQL-агент. Схема БД:
{schema_prompt}
{JSON_HINTS}

Напиши ОДИН короткий SQL SELECT запрос для ответа на вопрос пользователя.
Используй ТОЛЬКО таблицы и колонки из схемы. Запрос должен быть максимально простым.
Вызови run_sql с этим запросом.""")
                retry_messages = [m for m in messages_for_llm if isinstance(m, HumanMessage)][-1:]
                retry_llm = llm.bind_tools([run_sql], tool_choice="required")
                response = retry_llm.invoke([retry_system] + retry_messages)
            except Exception as retry_e:
                logger.error(f"❌ Повторная попытка тоже не удалась: {retry_e}")
                response = AIMessage(content="Произошла ошибка при формировании запроса. Пожалуйста, попробуйте переформулировать ваш вопрос проще.")
        else:
            logger.error(f"❌ Ошибка LLM: {error_str[:300]}")
            response = AIMessage(content="Произошла временная ошибка. Пожалуйста, попробуйте ещё раз.")

    return {
        "messages": [response]
    }

CRITIC_PROMPT = """Ты — строгий SQL-ревьюер. Анализируй ошибку и давай КОНКРЕТНОЕ решение.

Исходный вопрос: {original_query}
Неправильный SQL: {last_sql}
Ошибка: {tool_result}

ПОЛНАЯ СХЕМА БД:
{schema_preview}

КРИТИЧЕСКИ ВАЖНО — JSON/JSONB колонки:
- Если колонка имеет тип JSONB (например city, airport_name), она содержит объект вида {{"ru": "Москва", "en": "Moscow"}}
- НЕЛЬЗЯ писать: WHERE city = 'Москва' (это вызовет ошибку или вернёт 0 строк!)
- ПРАВИЛЬНО: WHERE city->>'ru' = 'Москва'
- Для вывода: SELECT city->>'ru' AS city_name
- Если ошибка содержит "invalid input syntax for type json" — значит ты сравниваешь JSONB колонку со строкой!
- Если запрос вернул 0 строк — проверь, правильно ли обращаешься к JSONB полям через ->>

ТВОЯ ЗАДАЧА:
1. Найди таблицу из схемы, которая подходит для вопроса
2. Найди правильные названия колонок для JOIN
3. Проверь, есть ли JSONB-колонки, и используй ->> для доступа к их значениям
4. Напиши ТОЧНЫЙ ПОЛНЫЙ SQL с правильными названиями

ФОРМАТ ОТВЕТА:

ОШИБКА: [что не так]
ПРАВИЛЬНЫЕ ТАБЛИЦЫ: [список таблиц из схемы, которые нужно использовать]
ПРАВИЛЬНЫЕ КОЛОНКИ: [список колонок для JOIN]
ИСПРАВЛЕННЫЙ SQL: [конкретный рабочий SQL запрос — ПОЛНЫЙ, от SELECT до конца]

Пример для JSONB:
ОШИБКА: Сравнение JSONB-колонки city со строкой напрямую
ПРАВИЛЬНЫЕ ТАБЛИЦЫ: flights, airports_data
ПРАВИЛЬНЫЕ КОЛОНКИ: flights.departure_airport, airports_data.airport_code
ИСПРАВЛЕННЫЙ SQL: SELECT ad2.city->>'ru' AS city, COUNT(*) AS cnt FROM flights f JOIN airports_data ad1 ON f.departure_airport = ad1.airport_code JOIN airports_data ad2 ON f.arrival_airport = ad2.airport_code WHERE ad1.city->>'ru' = 'Москва' GROUP BY ad2.city->>'ru' ORDER BY cnt DESC LIMIT 50

Будь максимально конкретным! Используй ТОЛЬКО таблицы и колонки из схемы выше. Пиши SQL ПОЛНОСТЬЮ!"""

def critic_node(state: AgentState):
    global DB_SCHEMA
    logger.info("🧐 Запущен критик")
    
    critic_attempts = state.get("critic_attempts", 0) + 1
    logger.info(f"🔢 Попытка критика: {critic_attempts}")
    
    last_tool_msg = None
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage) and m.name == "run_sql":
            last_tool_msg = m
            break
    
    if not last_tool_msg:
        logger.warning("Критик вызван, но нет последнего run_sql")
        return {**state, "critic_attempts": critic_attempts}
    
    if not DB_SCHEMA:
        DB_SCHEMA = get_postgres_schema.invoke({})
    schema_preview = _schema_for_prompt()

    prompt = CRITIC_PROMPT.format(
        original_query=state.get("original_query", "—"),
        last_sql=state.get("last_sql", "—"),
        tool_result=last_tool_msg.content[:800],
        schema_preview=schema_preview
    )
    
    try:
        response = critic_llm.invoke(prompt)
        critic_text = response.content.strip()
        
        logger.info(f"🧐 Критика (попытка {critic_attempts}): {critic_text[:200]}")
        
        critic_message = AIMessage(
            content=f"[Критик SQL - Попытка {critic_attempts}]\n{critic_text}",
            name="sql_critic"
        )
        
        return {
            **state,
            "messages": state["messages"] + [critic_message],
            "critic_ran_last": True,
            "critic_attempts": critic_attempts,
        }
    except Exception as e:
        logger.error(f"❌ Ошибка критика: {e}")
        return {
            **state,
            "messages": state["messages"] + [AIMessage(
                content="[Критик] Ошибка анализа. Продолжаем без критики.",
                name="sql_critic"
            )],
            "critic_ran_last": True,
            "critic_attempts": critic_attempts,
        }

def after_tools_decision(state: AgentState) -> str:
    logger.info("🔀 === РОУТЕР: after_tools_decision ===")
    
    critic_attempts = state.get("critic_attempts", 0)
    
    last_tool_msg = state["messages"][-1]
    tool_name = getattr(last_tool_msg, "name", "")

    if tool_name == "get_postgres_schema":
        schema_calls = sum(
            1 for m in state["messages"]
            if isinstance(m, ToolMessage) and m.name == "get_postgres_schema"
        )
        if schema_calls >= 2:
            logger.warning(f"⚠️ Схема вызвана {schema_calls} раз → END")
            return END
        return "assistant"

    if tool_name != "run_sql":
        return END 

    try:
        result = json.loads(last_tool_msg.content)
        success = result.get("success", False)
        row_count = result.get("row_count", 0)
        error_text = result.get("error", "")
        is_connection_error = result.get("is_connection_error", False)
    except (json.JSONDecodeError, ValueError, TypeError):
        success = False
        row_count = 0
        error_text = last_tool_msg.content[:200]
        is_connection_error = "server closed" in error_text.lower()

    if is_connection_error:
        logger.critical("🚨 Инфраструктурная ошибка БД — пропускаем критика")
        return END

    has_error = not success or "error" in error_text.lower()
    has_no_data = row_count == 0
    
    critic_messages = [
        m for m in state["messages"] 
        if isinstance(m, AIMessage) and getattr(m, "name", "") == "sql_critic"
    ][-3:]
    
    if len(critic_messages) >= 2:
        last_two_texts = [m.content.lower() for m in critic_messages[-2:]]
        if all("clients" in t for t in last_two_texts):
            logger.warning("⚠️ Критик зациклился на таблице 'clients' → assistant")
            return "assistant"

        zero_rows_in_current_request = 0
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                break
            if isinstance(m, ToolMessage) and m.name == "run_sql":
                try:
                    r = json.loads(m.content)
                    if r.get("success") and int(r.get("row_count", -1)) == 0:
                        zero_rows_in_current_request += 1
                except Exception:
                    pass
        if zero_rows_in_current_request >= 3:
            logger.warning("⚠️ 3+ подряд запросов с 0 строк — критик зациклился → assistant")
            return "assistant"

    max_critic_attempts = 3
    
    logger.info(f"📊 has_error={has_error}, has_no_data={has_no_data}, "
                f"critic_attempts={critic_attempts}/{max_critic_attempts}")


    if has_error and critic_attempts < max_critic_attempts:
        logger.info(f"❌ SQL неуспешен → критик (будет попытка {critic_attempts + 1})")
        return "critic"

    if has_no_data:
        logger.info("📭 SQL успешен, но данных нет → assistant (человеческий fallback)")
        return "assistant"
    
    if critic_attempts >= max_critic_attempts:
        logger.warning(f"⚠️ Достигнут лимит попыток критика ({max_critic_attempts})")
        return "assistant"

    if not success or row_count == 0:
        logger.info("📚 Нет данных → END")
        return END

    query_result = result.get("data", [])
    has_data = bool(query_result and len(query_result) > 0)

    original_query = state.get("original_query", "")
    has_vis_request = _is_vis_request(original_query)

    if has_vis_request and has_data:
        return "graph_vis"
    
    if has_data:
        return "assistant"
    
    return END

def build_graph():
    memory = MemorySaver()
    builder = StateGraph(AgentState)
    builder.add_node('checked_cache', checked_cache)
    builder.add_node('critic', critic_node)
    builder.add_node('assistant', assistant)
    builder.add_node('tools', execute_tool_node)
    builder.add_node('graph_vis', graph_vis)
    builder.add_node('review_visualization', review_visualization)
    builder.add_node('safe_exec', safe_exec)
 
    builder.add_edge(START, 'checked_cache')
    builder.add_conditional_edges(
        "checked_cache",
        cache_should_continue,
        {
            "assistant": "assistant",
            END: END,
        }
    )
    builder.add_edge('critic', 'assistant')
    
    builder.add_conditional_edges(
        "assistant",
        should_continue,
        {
            "tools": "tools",
            END: END,
        }
    )

    builder.add_conditional_edges(
        "tools",
        after_tools_decision,
        {
            "assistant": "assistant",
            "graph_vis": "graph_vis",
            "critic": "critic",
            END: END
        }
    )
    
    builder.add_edge('graph_vis', 'review_visualization')
    builder.add_edge('review_visualization', 'safe_exec')
    builder.add_edge('safe_exec', END)
    
    return builder.compile(checkpointer=memory)

graph = build_graph
