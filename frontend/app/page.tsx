"use client";

import Image from "next/image";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import {
  chat,
  connectDb,
  disconnectDb,
  getApiBaseUrl,
  ping,
  type ChatResponse,
  type InterruptPayload,
} from "@/lib/api";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Toaster } from "@/components/ui/sonner";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";

type ConnState = "connecting" | "connected" | "disconnected";

type DbProfile = {
  id: string;
  name: string;
  host: string;
  port: string;
  db: string;
  user: string;
  password: string;
};

type HistoryItem = {
  q: string;
  mode: string;
  ts: number;
};

type CacheReviewState = {
  query: string;
  cachedResponse: string;
  score: number | null;
};

type VisualizationReviewState = {
  rowCount: number;
  columns: string[];
  chartBase64: string | null;
  previewError: string | null;
};

const AUTH_TOKEN_KEY = "t2sql_auth_token_v1";

const PROFILES_KEY = "t2sql_db_profiles_v1";
const ACTIVE_PROFILE_KEY = "t2sql_db_active_profile_v1";
const HISTORY_PREFIX = "t2sql_history_v1";
const HISTORY_COLLAPSED_KEY = "t2sql_history_collapsed_v1";
const SESSION_ID_KEY = "t2sql_session_id";
const STATUS_POLL_MS = 60000;
const COLUMN_WINDOW_SIZE = 8;

const historyKey = (profileId: string) => `${HISTORY_PREFIX}:${profileId}`;

function loadProfiles(): { profiles: DbProfile[]; activeId: string | null } {
  try {
    const raw = localStorage.getItem(PROFILES_KEY);
    const activeId = localStorage.getItem(ACTIVE_PROFILE_KEY);
    if (!raw) return { profiles: [], activeId };
    const parsed = JSON.parse(raw) as DbProfile[];
    if (!Array.isArray(parsed)) return { profiles: [], activeId };
    return { profiles: parsed, activeId };
  } catch {
    return { profiles: [], activeId: null };
  }
}

function saveProfiles(list: DbProfile[], activeId?: string) {
  localStorage.setItem(PROFILES_KEY, JSON.stringify(list));
  if (activeId) {
    localStorage.setItem(ACTIVE_PROFILE_KEY, activeId);
  } else {
    localStorage.removeItem(ACTIVE_PROFILE_KEY);
  }
}

function loadHistory(profileId: string): HistoryItem[] {
  try {
    const raw = localStorage.getItem(historyKey(profileId));
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HistoryItem[];
    if (!Array.isArray(parsed)) return [];
    return parsed.sort((a, b) => b.ts - a.ts).slice(0, 40);
  } catch {
    return [];
  }
}

function saveHistory(profileId: string, list: HistoryItem[]) {
  localStorage.setItem(historyKey(profileId), JSON.stringify(list.slice(0, 40)));
}

function getSessionId(): string {
  const existing = localStorage.getItem(SESSION_ID_KEY);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(SESSION_ID_KEY, id);
  return id;
}

function rotateSessionId(): string {
  const id = crypto.randomUUID();
  localStorage.setItem(SESSION_ID_KEY, id);
  return id;
}

export default function Page() {
  const [question, setQuestion] = useState("Покажи 5 строк из любой таблицы");
  const [sessionId, setSessionId] = useState<string>("");

  const [apiBaseUrl, setApiBaseUrlState] = useState<string>(process.env.NEXT_PUBLIC_API_BASE_URL ?? "/backend");

  // auth + status
  const [connState, setConnState] = useState<ConnState>("disconnected");

  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [out, setOut] = useState<ChatResponse | null>(null);
  const [cacheReview, setCacheReview] = useState<CacheReviewState | null>(null);
  const [visualizationReview, setVisualizationReview] = useState<VisualizationReviewState | null>(null);
  const [visualizationCodeDraft, setVisualizationCodeDraft] = useState("");
  const interruptResolverRef = useRef<((data: Record<string, unknown>) => void) | null>(null);

  // profiles
  const [profiles, setProfiles] = useState<DbProfile[]>([]);
  const [activeProfileId, setActiveProfileId] = useState<string | null>(null);

  // connect modal (db profile)
  const [connOpen, setConnOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [pName, setPName] = useState("");
  const [host, setHost] = useState("");
  const [port, setPort] = useState("");
  const [db, setDb] = useState("");
  const [user, setUser] = useState("postgres");
  const [pwd, setPwd] = useState("");

  const refreshRef = useRef<() => Promise<void>>(() => Promise.resolve());

  // history
  const [historyItems, setHistoryItems] = useState<HistoryItem[]>([]);
  const [historyCollapsed, setHistoryCollapsed] = useState(false);
  const activeProfile = useMemo(() => profiles.find((p) => p.id === activeProfileId) ?? null, [profiles, activeProfileId]);

  // table extraction for search/export
  const rows = useMemo(() => {
    const d: any = out?.data;
    if (!d) return [];
    if (Array.isArray(d)) return d;
    if (Array.isArray(d?.rows)) return d.rows;
    if (Array.isArray(d?.data)) return d.data;
    return [];
  }, [out]);

  const serverColumns = useMemo(() => {
    const d: any = out?.data;
    if (Array.isArray(d?.columns)) return d.columns.map(String);
    return null;
  }, [out]);

  const columns = useMemo(() => {
    if (serverColumns?.length) return serverColumns;
    if (!rows.length) return [];
    const first = rows[0];
    if (first && typeof first === "object" && !Array.isArray(first)) return Object.keys(first);
    if (Array.isArray(first)) return first.map((_, i) => String(i));
    return [];
  }, [rows, serverColumns]);

  const [search, setSearch] = useState("");
  const [columnFilters, setColumnFilters] = useState<Record<string, string>>({});
  const [columnWindowStart, setColumnWindowStart] = useState(0);
  const pingFailStreakRef = useRef(0);
  const responseText = useMemo(() => {
    if (!out) return "";

    const raw = String(out.response ?? "").trim();
    const hasChart = Boolean(out.chart_base64) || /data:image\/png;base64,[A-Za-z0-9+/=]+/.test(raw);
    const rowCount = rows.length;
    const colCount = columns.length;

    const plural = (n: number, one: string, few: string, many: string) => {
      const n10 = n % 10;
      const n100 = n % 100;
      if (n10 === 1 && n100 !== 11) return one;
      if (n10 >= 2 && n10 <= 4 && (n100 < 12 || n100 > 14)) return few;
      return many;
    };

    const cleanedRaw = raw
      .replace(/!\[.*?\]\(data:image\/png;base64,[A-Za-z0-9+/=]+\)/g, "")
      .replace(/data:image\/png;base64,[A-Za-z0-9+/=]+/g, "")
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => {
        if (!line) return false;
        if (/^\|.*\|$/.test(line)) return false;
        if (/^[-:|\s]+$/.test(line)) return false;
        return true;
      })
      .join(" ");

    const lines: string[] = [];
    if (hasChart) lines.push("Построен график по результатам запроса.");
    if (rowCount > 0) {
      lines.push(`Найдено ${rowCount} ${plural(rowCount, "строка", "строки", "строк")} в результате.`);
    }
    if (colCount > 0) {
      const cols = columns.slice(0, 8).join(", ");
      const suffix = columns.length > 8 ? ` и ещё ${columns.length - 8}` : "";
      lines.push(`Колонки: ${cols}${suffix}.`);
    }
    if (out.from_cache) lines.push("Ответ получен из кэша.");
    if (typeof out.execution_time === "number") lines.push(`Время выполнения: ${out.execution_time.toFixed(2)} сек.`);

    if (cleanedRaw) {
      const shortText = cleanedRaw.length > 320 ? `${cleanedRaw.slice(0, 320).trim()}...` : cleanedRaw;
      lines.push(`Описание: ${shortText}`);
    }

    return lines.join("\n");
  }, [out, rows.length, columns]);
  const cachePreview = useMemo(() => {
    const raw = String(cacheReview?.cachedResponse ?? "").trim();
    if (!raw) return { chartBase64: null as string | null, text: "" };

    const markdownMatch = raw.match(/!\[.*?\]\(data:image\/png;base64,([A-Za-z0-9+/=]+)\)/);
    const plainMatch = raw.match(/data:image\/png;base64,([A-Za-z0-9+/=]+)/);
    const chartBase64 = markdownMatch?.[1] ?? plainMatch?.[1] ?? null;

    const text = raw
      .replace(/!\[.*?\]\(data:image\/png;base64,[A-Za-z0-9+/=]+\)/g, "[chart image]")
      .replace(/data:image\/png;base64,[A-Za-z0-9+/=]+/g, "[chart image]")
      .trim();

    return { chartBase64, text };
  }, [cacheReview]);
  const columnWindowMaxStart = Math.max(0, columns.length - COLUMN_WINDOW_SIZE);
  const visibleColumns = useMemo(
    () => columns.slice(columnWindowStart, columnWindowStart + COLUMN_WINDOW_SIZE),
    [columns, columnWindowStart]
  );

  useEffect(() => {
    setColumnWindowStart((prev) => Math.min(prev, columnWindowMaxStart));
  }, [columnWindowMaxStart]);

  function getCellValue(row: any, column: string, columnIndex: number): unknown {
    if (Array.isArray(row)) return row[columnIndex];
    if (row && typeof row === "object") return row[column];
    return row;
  }

  function toDisplayText(value: unknown): string {
    if (value === null || value === undefined) return "";
    if (typeof value === "string") return value;
    if (typeof value === "number" || typeof value === "boolean") return String(value);
    if (Array.isArray(value)) {
      if (value.length === 0) return "";
      const rendered = value.map((item) => {
        if (item === null || item === undefined) return "";
        if (typeof item === "string" || typeof item === "number" || typeof item === "boolean") return String(item);
        if (typeof item === "object") {
          const obj = item as Record<string, unknown>;
          if (typeof obj.ru === "string" && obj.ru.trim()) return obj.ru;
          if (typeof obj.en === "string" && obj.en.trim()) return obj.en;
          try {
            return JSON.stringify(obj);
          } catch {
            return String(item);
          }
        }
        return String(item);
      });
      return rendered.filter(Boolean).join(", ");
    }
    if (typeof value === "object") {
      const obj = value as Record<string, unknown>;
      // Common JSONB pattern from DB: {"ru":"...", "en":"..."}.
      if (typeof obj.ru === "string" && obj.ru.trim()) return obj.ru;
      if (typeof obj.en === "string" && obj.en.trim()) return obj.en;
      if (typeof obj.name === "string" && obj.name.trim()) return obj.name;
      const scalarEntries = Object.entries(obj).filter(([, v]) => typeof v !== "object");
      if (scalarEntries.length && scalarEntries.length <= 3) {
        return scalarEntries
          .map(([k, v]) => `${k}: ${String(v ?? "")}`)
          .join(", ");
      }
      try {
        return JSON.stringify(obj);
      } catch {
        return String(value);
      }
    }
    return String(value);
  }

  function valueMatchesFilter(value: unknown, rawExpr: string): boolean {
    const expr = rawExpr.trim();
    if (!expr) return true;

    const text = toDisplayText(value);
    const lower = text.toLowerCase();
    const exprLower = expr.toLowerCase();

    const numeric = Number(text);
    const isNumeric = Number.isFinite(numeric);
    const time = Date.parse(text);
    const isDate = !Number.isNaN(time);

    const cmpValue = (s: string) => {
      const n = Number(s);
      if (!Number.isNaN(n)) return { kind: "number" as const, value: n };
      const d = Date.parse(s);
      if (!Number.isNaN(d)) return { kind: "date" as const, value: d };
      return { kind: "text" as const, value: s.toLowerCase() };
    };

    if (exprLower.includes("..")) {
      const [fromRaw, toRaw] = expr.split("..").map((x) => x.trim());
      if (!fromRaw || !toRaw) return true;
      const from = cmpValue(fromRaw);
      const to = cmpValue(toRaw);
      if (from.kind === "number" && to.kind === "number" && isNumeric) return numeric >= from.value && numeric <= to.value;
      if (from.kind === "date" && to.kind === "date" && isDate) return time >= from.value && time <= to.value;
      return lower >= String(from.value) && lower <= String(to.value);
    }

    const opMatch = expr.match(/^(>=|<=|>|<|=)(.+)$/);
    if (opMatch) {
      const op = opMatch[1];
      const rhs = cmpValue(opMatch[2].trim());
      if (rhs.kind === "number" && isNumeric) {
        if (op === ">=") return numeric >= rhs.value;
        if (op === "<=") return numeric <= rhs.value;
        if (op === ">") return numeric > rhs.value;
        if (op === "<") return numeric < rhs.value;
        return numeric === rhs.value;
      }
      if (rhs.kind === "date" && isDate) {
        if (op === ">=") return time >= rhs.value;
        if (op === "<=") return time <= rhs.value;
        if (op === ">") return time > rhs.value;
        if (op === "<") return time < rhs.value;
        return time === rhs.value;
      }
      return op === "=" ? lower === String(rhs.value) : lower.includes(String(rhs.value));
    }

    return lower.includes(exprLower);
  }

  const filteredRows = useMemo(() => {
    const s = search.trim().toLowerCase();
    return rows.filter((r: any) => {
      const globalOk = !s || JSON.stringify(r).toLowerCase().includes(s);
      if (!globalOk) return false;
      return columns.every((c, idx) => valueMatchesFilter(getCellValue(r, c, idx), columnFilters[c] ?? ""));
    });
  }, [rows, search, columns, columnFilters]);

  useEffect(() => {
    // api url init
    const cur = getApiBaseUrl();
    setApiBaseUrlState(cur);
    setSessionId("");

    // db profiles init
    const { profiles: loaded } = loadProfiles();
    if (loaded.length === 0) {
      const first: DbProfile = {
        id: crypto.randomUUID(),
        name: "Local",
        host: "127.0.0.1",
        port: "5432",
        db: "bookings",
        user: "postgres",
        password: "",
      };
      saveProfiles([first]);
      setProfiles([first]);
      setActiveProfileId(null);
    } else {
      setProfiles(loaded);
      setActiveProfileId(null);
    }

    // backward-compat cleanup for old auth state
    const t = localStorage.getItem(AUTH_TOKEN_KEY);
    if (t) localStorage.removeItem(AUTH_TOKEN_KEY);

    setHistoryCollapsed(localStorage.getItem(HISTORY_COLLAPSED_KEY) === "1");
  }, []);

  useEffect(() => {
    if (!activeProfileId) {
      setHistoryItems([]);
      return;
    }
    setHistoryItems(loadHistory(activeProfileId));
  }, [activeProfileId]);

  // health/connection state machine
  useEffect(() => {
    if (!activeProfileId) {
      setConnState("disconnected");
      pingFailStreakRef.current = 0;
      refreshRef.current = async () => {
        setConnState("disconnected");
      };
      return;
    }

    let alive = true;

    const tick = async (silent = false) => {
      if (!silent) setConnState("connecting");
      const ok = await ping();
      if (!alive) return;
      if (ok) {
        pingFailStreakRef.current = 0;
        setConnState("connected");
      } else {
        pingFailStreakRef.current += 1;
        // Avoid false "disconnect" on single transient status failure.
        if (pingFailStreakRef.current >= 2) {
          setConnState("disconnected");
        } else if (!silent) {
          setConnState("connected");
        }
      }

    };

    refreshRef.current = () => tick(false);

    void tick(false);
    const id = window.setInterval(() => {
      if (typeof document !== "undefined" && document.hidden) return;
      void tick(true);
    }, STATUS_POLL_MS);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [activeProfileId, apiBaseUrl]);

  function toggleHistoryCollapsed() {
    setHistoryCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem(HISTORY_COLLAPSED_KEY, next ? "1" : "0");
      return next;
    });
  }

  function clearHistory() {
    if (!activeProfileId) return;
    if (!window.confirm("Очистить историю запросов?")) return;
    setHistoryItems([]);
    saveHistory(activeProfileId, []);
    toast.success("История очищена");
  }

  function addToHistory(q: string, mode: string) {
    if (!activeProfileId) return;
    const trimmed = q.trim();
    if (!trimmed) return;

    setHistoryItems((prev) => {
      const next = [{ q: trimmed, mode, ts: Date.now() }, ...prev.filter((x) => x.q !== trimmed || x.mode !== mode)];
      saveHistory(activeProfileId, next);
      return next;
    });
  }

  async function setActive(id: string) {
    setActiveProfileId(id);
    saveProfiles(profiles, id);

    const p = profiles.find((x) => x.id === id) ?? null;
    toast.message("Активный профиль изменён", { description: p?.name ?? id });

    if (!p) return;

    try {
      setConnState("connecting");
      const sid = rotateSessionId();
      setSessionId(sid);
      await connectDb(sid, p);
      toast.success("Подключено", { description: `${p.user}@${p.host}:${p.port}/${p.db}` });
    } catch (e: any) {
      setSessionId("");
      setConnState("disconnected");
      toast.error("Ошибка подключения", { description: e?.message ?? "Ошибка подключения" });
    } finally {
      refreshRef.current?.();
    }
  }

  function newProfile() {
    setEditingId(null);
    setPName("Новый профиль");
    setHost("127.0.0.1");
    setPort("5432");
    setDb("bookings");
    setUser("postgres");
    setPwd("");
    setConnOpen(true);
  }

  async function saveCurrentProfile() {
    const name = pName.trim() || "Профиль";
    const next: DbProfile = {
      id: editingId ?? crypto.randomUUID(),
      name,
      host: host.trim(),
      port: port.trim(),
      db: db.trim(),
      user: user.trim(),
      password: pwd ?? "",
    };

    const nextProfiles = editingId ? profiles.map((p) => (p.id === editingId ? next : p)) : [next, ...profiles];
    setProfiles(nextProfiles);
    setActiveProfileId(next.id);
    saveProfiles(nextProfiles, next.id);
    setEditingId(next.id);

    toast.success("Профиль сохранён", { description: `${next.name} — ${next.user}@${next.host}:${next.port}/${next.db}` });

    try {
      setConnState("connecting");
      const sid = rotateSessionId();
      setSessionId(sid);
      await connectDb(sid, next);
      toast.success("Подключено", { description: `${next.user}@${next.host}:${next.port}/${next.db}` });
      setConnOpen(false);
    } catch (e: any) {
      setSessionId("");
      setConnState("disconnected");
      toast.error("Ошибка подключения", { description: e?.message ?? "Ошибка подключения" });
    } finally {
      refreshRef.current?.();
    }
  }

  function deleteCurrentProfile() {
    if (!editingId) return;
    const filtered = profiles.filter((p) => p.id !== editingId);
    if (filtered.length === 0) return toast.error("Нельзя удалить последний профиль");
    const nextActive = activeProfileId === editingId ? filtered[0].id : activeProfileId;

    setProfiles(filtered);
    setActiveProfileId(nextActive);
    saveProfiles(filtered, nextActive ?? undefined);
    setEditingId(null);
    toast.success("Профиль удалён");
  }

  function openEditProfile(id: string) {
    const p = profiles.find((x) => x.id === id);
    if (!p) return;
    setEditingId(id);
    setPName(p.name);
    setHost(p.host);
    setPort(p.port);
    setDb(p.db);
    setUser(p.user);
    setPwd(p.password);
    setConnOpen(true);
  }

  async function logoutProfile() {
    const activeName = activeProfile?.name ?? null;
    try {
      await disconnectDb();
    } catch {
      // Local logout should still proceed even if backend disconnect failed.
    }
    setActiveProfileId(null);
    setSessionId("");
    setConnState("disconnected");
    saveProfiles(profiles);
    clearPendingInterrupt();
    setErr(null);
    setOut(null);
    toast.success("Вы вышли из профиля", { description: activeName ? `Профиль «${activeName}» отключён.` : "Подключение к БД закрыто." });
  }

  function resetConversationContext() {
    const sid = rotateSessionId();
    setSessionId(sid);
    setOut(null);
    setErr(null);
    clearPendingInterrupt();
    toast.success("Контекст сброшен", { description: "Новая сессия для модели запущена." });
  }

  function clearPendingInterrupt() {
    setCacheReview(null);
    setVisualizationReview(null);
    setVisualizationCodeDraft("");
    interruptResolverRef.current = null;
  }

  // Cleanup pending interrupt on unmount to avoid hanging promises.
  useEffect(() => {
    return () => {
      if (interruptResolverRef.current) {
        interruptResolverRef.current({});
        interruptResolverRef.current = null;
      }
    };
  }, []);

  function resolveInterrupt(data: Record<string, unknown>) {
    const resolve = interruptResolverRef.current;
    clearPendingInterrupt();
    resolve?.(data);
  }

  async function handleInterrupt(interrupt: InterruptPayload): Promise<Record<string, unknown>> {
    const interruptType = String(interrupt?.type ?? "");
    if (interruptType === "cache_review") {
      return await new Promise((resolve) => {
        interruptResolverRef.current = resolve;
        setVisualizationReview(null);
        setVisualizationCodeDraft("");
        setCacheReview({
          query: String(interrupt?.query ?? ""),
          cachedResponse: String(interrupt?.cached_response ?? ""),
          score: typeof interrupt?.score === "number" ? interrupt.score : null,
        });
      });
    }

    if (interruptType === "visualization_review") {
      const code = String(interrupt?.code ?? "");
      const columnsRaw = Array.isArray(interrupt?.columns) ? interrupt.columns : [];
      return await new Promise((resolve) => {
        interruptResolverRef.current = resolve;
        setCacheReview(null);
        setVisualizationCodeDraft(code);
        setVisualizationReview({
          rowCount: Number(interrupt?.row_count ?? 0),
          columns: columnsRaw.map((x) => String(x)),
          chartBase64: typeof interrupt?.chart_base64 === "string" ? interrupt.chart_base64 : null,
          previewError: typeof interrupt?.preview_error === "string" ? interrupt.preview_error : null,
        });
      });
    }

    return {};
  }

  async function run() {
    setErr(null);
    clearPendingInterrupt();
    setLoading(true);

    try {
      const base = question.trim();
      if (!base) return;

      if (connState !== "connected") {
        setErr("Нет подключения к БД. Выберите профиль в меню БД и подключитесь.");
        toast.error("Нет подключения", { description: "Выберите профиль в меню БД и запустите подключение." });
        return;
      }

      const res = await chat(sessionId || getSessionId(), base, "table", undefined, {
        onInterrupt: handleInterrupt,
      });
      setOut(res);
      addToHistory(base, "table");
      toast.success("Готово");
    } catch (e: any) {
      setOut(null);
      setErr(e?.message ?? "Ошибка");
      toast.error("Ошибка", { description: e?.message ?? "Ошибка" });
    } finally {
      setLoading(false);
    }
  }

  function escapeCsv(v: any) {
    const s = toDisplayText(v);
    if (/[",\n]/.test(s)) return `"${s.replaceAll('"', '""')}"`;
    return s;
  }

  function exportCsv() {
    if (!rows.length) return;
    const header = columns.map(escapeCsv).join(",");
    const body = filteredRows
      .map((r: any) => {
        if (r && typeof r === "object" && !Array.isArray(r)) return columns.map((c) => escapeCsv(r[c])).join(",");
        if (Array.isArray(r)) return columns.map((c, idx) => escapeCsv(getCellValue(r, c, idx))).join(",");
        return escapeCsv(r);
      })
      .join("\n");
    const csv = `${header}\n${body}`;
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "result.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast.success("CSV скачан");
  }

  function exportChartPng() {
    const b64 = out?.chart_base64;
    if (!b64) return toast.error("Нет графика для скачивания");
    const a = document.createElement("a");
    a.href = `data:image/png;base64,${b64}`;
    a.download = "chart.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
    toast.success("PNG скачан");
  }

  async function rebuildChartWithFilters() {
    if (!filteredRows.length) {
      toast.error("Нет данных после фильтрации");
      return;
    }
    setLoading(true);
    setErr(null);
    clearPendingInterrupt();
    try {
      const base = question.trim() || "Построй график по данным";
      const subset = filteredRows.slice(0, 500);
      const msg =
        `${base}\n\n` +
        "Построй график только по отфильтрованным данным ниже. " +
        "Игнорируй иные таблицы и источники.\n" +
        `FILTERED_DATA_JSON:\n${JSON.stringify(subset)}`;
      const res = await chat(sessionId || getSessionId(), msg, "chart", undefined, {
        onInterrupt: handleInterrupt,
      });
      setOut(res);
      toast.success("График обновлён по фильтрам");
    } catch (e: any) {
      toast.error("Ошибка", { description: e?.message ?? "Ошибка обновления графика" });
    } finally {
      setLoading(false);
    }
  }

  const shortSessionId = connState === "connected" && sessionId ? sessionId.slice(0, 8) : "—";
  const profileLabel = activeProfile?.name ?? "не выбран";
  const dbUserLabel = connState === "connected" ? activeProfile?.user ?? "—" : "—";

  const statusPill =
    connState === "connecting"
      ? { dot: "bg-amber-300", text: "Подключение" }
    : connState === "connected"
      ? { dot: "bg-emerald-400", text: "Подключено" }
      : { dot: "bg-rose-400", text: "Не подключено" };

  return (
    <main className="relative min-h-screen overflow-hidden text-white">
      <Toaster richColors />
      {/* Background layers */}
      <div className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute inset-0 bg-[radial-gradient(1200px_circle_at_15%_0%,rgba(56,189,248,0.16),transparent_55%),radial-gradient(1000px_circle_at_80%_20%,rgba(168,85,247,0.16),transparent_55%),radial-gradient(900px_circle_at_30%_90%,rgba(34,197,94,0.10),transparent_60%),linear-gradient(to_bottom,rgba(2,6,23,0.96),rgba(2,6,23,0.86),rgba(2,6,23,0.98))]" />
        <svg
          className="absolute inset-0 h-full w-full opacity-[0.28]"
          viewBox="0 0 1200 700"
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2.3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {Array.from({ length: 120 }).map((_, i) => {
            const x = (i * 73) % 1200;
            const y = (i * 149) % 700;
            const r = (i % 4 === 0 ? 1.6 : i % 3 === 0 ? 1.2 : 0.9) + (i % 7 === 0 ? 0.6 : 0);
            const o = i % 5 === 0 ? 0.9 : i % 3 === 0 ? 0.65 : 0.45;
            return <circle key={i} cx={x} cy={y} r={r} fill="white" opacity={o} filter="url(#glow)" />;
          })}

          <g stroke="rgba(255,255,255,0.18)" strokeWidth="1">
            <path d="M120 140 L220 90 L310 160 L420 120" />
            <path d="M780 210 L860 160 L940 240 L1020 190" />
            <path d="M260 520 L340 470 L420 540 L520 500" />
            <path d="M640 560 L720 520 L800 600 L920 560" />
          </g>
        </svg>

        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(255,255,255,0.05),transparent_60%)] opacity-70" />
      </div>

      <div className="relative mx-auto w-full max-w-[1920px] px-6 py-10">
        {/* Header */}
        <div className="flex flex-col gap-5 md:flex-row md:items-center md:justify-between">
          <div className="space-y-3">
            <div className="flex items-center">
               <Link href="/" className="inline-flex items-center" aria-label="Home">
                <Image
                  src={encodeURI("/brand/Привет.webp")}
                  alt="Multiagents"
                  width={1024}
                  height={300}
                  priority
                  sizes="(max-width: 768px) 92vw, (max-width: 1280px) 680px, 780px"
                  className="h-28 w-auto max-w-[92vw] object-contain md:h-36 md:max-w-[680px] lg:h-44 lg:max-w-[780px]"
                />
              </Link>
            </div>
          </div>

          <div className="flex flex-col items-end gap-2">
            <div className="flex flex-wrap items-center justify-end gap-2">
              {connState === "connected" ? (
                <Link href="/db-schema">
                  <Button variant="secondary" className="border-white/10 bg-white/5 backdrop-blur-xl transition hover:-translate-y-0.5 hover:bg-white/10">
                    Структура БД
                  </Button>
                </Link>
              ) : null}

              <Link href="/authors">
                <Button variant="secondary" className="border-white/10 bg-white/5 backdrop-blur-xl transition hover:-translate-y-0.5 hover:bg-white/10">
                  Авторы
                </Button>
              </Link>

              {/* DB */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="secondary"
                    className="border-white/10 bg-white/5 backdrop-blur-xl transition hover:-translate-y-0.5 hover:bg-white/10"
                  >
                    БД: {activeProfile?.name ?? "не выбрана"}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="min-w-[220px]">
                  {profiles.map((p) => (
                    <DropdownMenuItem key={p.id} onClick={() => setActive(p.id)}>
                      <div className="flex w-full items-center justify-between gap-2">
                        <span>{p.name}</span>
                        <span className="text-xs text-muted-foreground">{p.host}</span>
                      </div>
                    </DropdownMenuItem>
                  ))}
                  <Separator className="my-1 bg-white/10" />
                  <DropdownMenuItem onClick={newProfile}>Новый профиль</DropdownMenuItem>
                  {activeProfile ? <DropdownMenuItem onClick={() => openEditProfile(activeProfile.id)}>Редактировать активный</DropdownMenuItem> : null}
                  {activeProfile ? <DropdownMenuItem onClick={() => void logoutProfile()}>Выйти из профиля</DropdownMenuItem> : null}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-xs text-white/75">
              <div className="flex flex-wrap items-center gap-2">
                <span className={`h-2.5 w-2.5 rounded-full ${statusPill.dot}`} />
                <span>{statusPill.text}</span>
                <Separator orientation="vertical" className="mx-1 h-4 bg-white/15" />
                <span>Профиль: {profileLabel}</span>
                <Separator orientation="vertical" className="mx-1 h-4 bg-white/15" />
                <span>Пользователь БД: {dbUserLabel}</span>
                <Separator orientation="vertical" className="mx-1 h-4 bg-white/15" />
                <span>Сессия: {shortSessionId}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="mt-8 grid gap-6 lg:grid-cols-[420px_1fr]">
          {/* Left panel */}
          <Card className="border-white/10 bg-white/5 backdrop-blur-xl">
            <CardHeader>
              <CardTitle className="text-xl">Запрос</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                className="min-h-[140px] border-white/10 bg-white/5 text-white placeholder:text-white/40"
                placeholder="Введите запрос..."
              />

              {/* CTA button */}
              <div className="grid grid-cols-2 gap-2">
                <Button
                  variant="secondary"
                  disabled={loading}
                  className="border-white/10 bg-white/5 transition hover:-translate-y-0.5 hover:bg-white/10"
                  onClick={resetConversationContext}
                >
                  Сбросить
                </Button>
                <Button
                  disabled={loading || !question.trim() || connState !== "connected"}
                  className="cta-accent text-black shadow-[0_18px_60px_rgba(0,0,0,0.45)] transition hover:-translate-y-0.5 hover:opacity-95"
                  onClick={() => run()}
                >
                  {loading ? (
                    <span className="inline-flex items-center gap-2">
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-black/20 border-t-black/70" />
                      Выполняю…
                    </span>
                  ) : (
                    "Выполнить"
                  )}
                </Button>
              </div>

              {err ? <div className="rounded-xl border border-rose-500/25 bg-rose-500/10 p-3 text-sm text-rose-200">{err}</div> : null}

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="text-sm font-medium text-white/80">История</div>
                  <div className="flex items-center gap-2">
                    <Button variant="secondary" className="h-8 border-white/10 bg-white/5 px-2 text-xs" onClick={toggleHistoryCollapsed}>
                      {historyCollapsed ? "Развернуть" : "Свернуть"}
                    </Button>
                    <Button variant="secondary" className="h-8 border-white/10 bg-white/5 px-2 text-xs" onClick={clearHistory}>
                      Очистить историю
                    </Button>
                  </div>
                </div>
                {!historyCollapsed ? <div className="grid gap-2">
                  {historyItems.slice(0, 8).map((h) => (
                    <button
                      key={`${h.ts}-${h.mode}-${h.q}`}
                      className="group rounded-xl border border-white/10 bg-white/5 p-3 text-left text-sm transition hover:-translate-y-0.5 hover:bg-white/10"
                      onClick={() => {
                        setQuestion(h.q);
                        toast.message("Подставлено", { description: h.q });
                      }}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="line-clamp-2 text-white/85">{h.q}</span>
                        <Badge variant="secondary" className="border-white/10 bg-white/10 text-white/75">
                          {h.mode === "chart" ? "график" : ">>"}
                        </Badge>
                      </div>
                      <div className="mt-1 text-xs text-white/45 group-hover:text-white/55">
                        {new Date(h.ts).toLocaleString()}
                      </div>
                    </button>
                  ))}
                  {historyItems.length === 0 ? <div className="text-xs text-white/45">Пока пусто</div> : null}
                </div> : null}
              </div>
            </CardContent>
          </Card>

          {/* Right panel */}
          <Card className="border-white/10 bg-white/5 backdrop-blur-xl">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-xl">Результат</CardTitle>
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  className="border-white/10 bg-white/5 transition hover:bg-white/10"
                  onClick={exportCsv}
                  disabled={!rows.length}
                >
                  Скачать CSV
                </Button>

                <Button
                  variant="secondary"
                  className="border-white/10 bg-white/5 transition hover:bg-white/10"
                  onClick={exportChartPng}
                  disabled={!out?.chart_base64}
                >
                  Скачать PNG
                </Button>
              </div>
            </CardHeader>

            <CardContent className="space-y-4">
              {cacheReview ? (
                <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-4">
                  <div className="mb-2 text-sm font-medium text-amber-100">Найден кэшированный ответ</div>
                  <div className="text-xs text-amber-100/80">
                    Запрос: <span className="text-amber-50">{cacheReview.query || "—"}</span>
                    {cacheReview.score !== null ? ` · Сходство ${(cacheReview.score * 100).toFixed(1)}%` : ""}
                  </div>
                  {cachePreview.chartBase64 ? (
                    <div className="mt-3 rounded-lg border border-white/10 bg-black/20 p-2">
                      <img
                        src={`data:image/png;base64,${cachePreview.chartBase64}`}
                        alt="Предпросмотр графика из кэша"
                        className="max-h-72 w-full rounded-md object-contain"
                      />
                    </div>
                  ) : null}
                  <div className="scrollbar-custom mt-3 max-h-40 overflow-auto whitespace-pre-wrap rounded-lg border border-white/10 bg-black/20 p-3 text-sm text-white/85">
                    {cachePreview.text || "Пустой кэшированный ответ"}
                  </div>
                  <div className="mt-3 flex items-center justify-end gap-2">
                    <Button
                      variant="secondary"
                      className="border-white/10 bg-white/5 transition hover:bg-white/10"
                      onClick={() => resolveInterrupt({ use_cache: false })}
                    >
                      Сгенерировать заново
                    </Button>
                    <Button onClick={() => resolveInterrupt({ use_cache: true })}>Использовать кэш</Button>
                  </div>
                </div>
              ) : null}

              {visualizationReview ? (
                <div className="rounded-xl border border-sky-400/30 bg-sky-500/10 p-4">
                  <div className="mb-2 text-sm font-medium text-sky-100">Подтверждение визуализации</div>
                  <div className="text-xs text-sky-100/80">
                    Строк: {visualizationReview.rowCount} · Колонки:{" "}
                    {visualizationReview.columns.length ? visualizationReview.columns.join(", ") : "—"}
                  </div>
                  {visualizationReview.chartBase64 ? (
                    <div className="mt-3 overflow-hidden rounded-lg border border-white/10 bg-black/20 p-2">
                      <img
                        alt="Предпросмотр графика"
                        src={`data:image/png;base64,${visualizationReview.chartBase64}`}
                        className="max-h-72 w-full rounded-md object-contain"
                      />
                    </div>
                  ) : null}
                  {!visualizationReview.chartBase64 && visualizationReview.previewError ? (
                    <div className="mt-3 rounded-lg border border-amber-300/30 bg-amber-400/10 p-2 text-xs text-amber-100/90">
                      Не удалось сгенерировать предпросмотр графика: {visualizationReview.previewError}
                    </div>
                  ) : null}
                  <Textarea
                    value={visualizationCodeDraft}
                    onChange={(e) => setVisualizationCodeDraft(e.target.value)}
                    className="mt-3 min-h-[220px] border-white/10 bg-black/20 font-mono text-xs text-white"
                  />
                  <div className="mt-3 flex items-center justify-end gap-2">
                    <Button
                      variant="secondary"
                      className="border-white/10 bg-white/5 transition hover:bg-white/10"
                      onClick={() => resolveInterrupt({ approved: false })}
                    >
                      Отклонить
                    </Button>
                    <Button onClick={() => resolveInterrupt({ approved: true, code: visualizationCodeDraft })}>
                      Подтвердить и выполнить
                    </Button>
                  </div>
                </div>
              ) : null}

              {out?.chart_base64 ? (
                <div className="overflow-hidden rounded-xl border border-white/10 bg-black/20 p-3">
                  <img
                    alt="график"
                    src={`data:image/png;base64,${out.chart_base64}`}
                    className="h-auto w-full rounded-lg"
                  />
                </div>
              ) : null}

              {responseText ? (
                <div className="rounded-xl border border-white/10 bg-white/5 p-3">
                  <div className="mb-2 text-xs uppercase tracking-wide text-white/50">Ответ ассистента</div>
                  <div className="scrollbar-custom max-h-32 overflow-auto rounded-md border border-white/10 bg-black/20 p-3">
                    <pre className="whitespace-pre-wrap break-words text-xs leading-relaxed text-white/85">{responseText}</pre>
                  </div>
                </div>
              ) : null}

              {rows.length ? (
                <>
                  <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                    <div className="text-sm text-white/70">
                      Строки: <span className="text-white/85">{filteredRows.length}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Label className="text-xs text-white/55">Поиск</Label>
                      <Input
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        className="h-9 w-[240px] border-white/10 bg-white/5 text-white placeholder:text-white/40"
                        placeholder="поиск по таблице…"
                      />
                    </div>
                  </div>
                  {columns.length ? (
                    <div className="grid gap-2 md:grid-cols-2">
                      {visibleColumns.map((c) => (
                        <div key={c} className="flex items-center gap-2">
                          <Label className="min-w-24 text-xs text-white/55">{c}</Label>
                          <Input
                            value={columnFilters[c] ?? ""}
                            onChange={(e) => setColumnFilters((prev) => ({ ...prev, [c]: e.target.value }))}
                            className="h-8 border-white/10 bg-white/5 text-white placeholder:text-white/40"
                            placeholder="содержит, =x, >x, <x, a..b"
                          />
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {columns.length > COLUMN_WINDOW_SIZE ? (
                    <div className="rounded-lg border border-white/10 bg-white/5 px-3 py-2">
                      <div className="mb-1 flex items-center justify-between text-xs text-white/60">
                        <span>
                          Колонки: {columnWindowStart + 1}-
                          {Math.min(columnWindowStart + COLUMN_WINDOW_SIZE, columns.length)} из {columns.length}
                        </span>
                        <span>Сдвиг окна</span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={columnWindowMaxStart}
                        step={1}
                        value={columnWindowStart}
                        onChange={(e) => setColumnWindowStart(Number(e.target.value))}
                        className="w-full accent-cyan-400"
                      />
                    </div>
                  ) : null}
                  {out?.chart_base64 ? (
                    <div className="flex justify-end">
                      <Button
                        variant="secondary"
                        className="border-white/10 bg-white/5 transition hover:bg-white/10"
                        disabled={loading}
                        onClick={rebuildChartWithFilters}
                      >
                        Обновить график по фильтрам
                      </Button>
                    </div>
                  ) : null}

                  <div 
                    className="scrollbar-custom max-h-[600px] overflow-auto rounded-xl border border-white/10"
                  >
                    <table className="min-w-full text-left text-sm">
                      <thead className="sticky top-0 z-10 bg-white/5 text-white/70 backdrop-blur-sm">
                        <tr>
                          {visibleColumns.map((c) => (
                            <th key={c} className="whitespace-nowrap px-3 py-2 font-medium first:pl-4 last:pr-4">
                              {c}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {filteredRows.slice(0, 200).map((r: any, i: number) => (
                          <tr key={i} className="border-t border-white/10">
                            {visibleColumns.map((c, idx) => {
                              const value = getCellValue(r, c, columnWindowStart + idx);
                              return (
                                <td key={c} className="whitespace-nowrap px-3 py-2 text-white/80 first:pl-4 last:pr-4">
                                  {toDisplayText(value)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {filteredRows.length > 200 ? (
                    <div className="text-xs text-white/45">Показаны первые 200 строк (для скорости)</div>
                  ) : null}
                </>
              ) : (
                <div className="rounded-xl border border-white/10 bg-white/5 p-4 text-sm text-white/70">
                  {out ? "Данных таблицы не найдено в ответе." : "Пока нет результата."}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Connect modal */}
      <Dialog open={connOpen} onOpenChange={setConnOpen}>
        <DialogContent className="border-white/10 bg-[#0b1220]/95 text-white backdrop-blur-xl">
          <DialogHeader>
            <DialogTitle>{editingId ? "Редактировать профиль БД" : "Новый профиль БД"}</DialogTitle>
          </DialogHeader>

          <div className="grid gap-3">
            <div className="grid gap-1.5">
              <Label className="text-white/70">Название</Label>
              <Input value={pName} onChange={(e) => setPName(e.target.value)} className="border-white/10 bg-white/5" />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="grid gap-1.5">
                <Label className="text-white/70">Хост</Label>
                <Input value={host} onChange={(e) => setHost(e.target.value)} className="border-white/10 bg-white/5" />
              </div>
              <div className="grid gap-1.5">
                <Label className="text-white/70">Порт</Label>
                <Input value={port} onChange={(e) => setPort(e.target.value)} className="border-white/10 bg-white/5" />
              </div>
            </div>

            <div className="grid gap-1.5">
              <Label className="text-white/70">База данных</Label>
              <Input value={db} onChange={(e) => setDb(e.target.value)} className="border-white/10 bg-white/5" />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div className="grid gap-1.5">
                <Label className="text-white/70">Пользователь</Label>
                <Input value={user} onChange={(e) => setUser(e.target.value)} className="border-white/10 bg-white/5" />
              </div>
              <div className="grid gap-1.5">
                <Label className="text-white/70">Пароль</Label>
                <Input
                  type="password"
                  value={pwd}
                  onChange={(e) => setPwd(e.target.value)}
                  className="border-white/10 bg-white/5"
                />
              </div>
            </div>

            <div className="flex items-center justify-between gap-2 pt-2">
              <div className="flex items-center gap-2">
                {editingId ? (
                  <Button
                    variant="secondary"
                    className="border-white/10 bg-white/5 transition hover:bg-white/10"
                    onClick={deleteCurrentProfile}
                  >
                    Удалить
                  </Button>
                ) : null}
              </div>

              <div className="flex items-center gap-2">
                <Button
                  variant="secondary"
                  className="border-white/10 bg-white/5 transition hover:bg-white/10"
                  onClick={() => toast.success("Проверка", { description: `${user}@${host}:${port}/${db}` })}
                >
                  Проверить
                </Button>
                <Button onClick={saveCurrentProfile}>Сохранить</Button>
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>

    </main>
  );
}
