/** Frontend API client for FastAPI backend. */

export type ChatMode = "table" | "chart";

export type ChatResponse = {
  response: string;
  data?: unknown;
  chart_base64?: string | null;
  from_cache?: boolean;
  execution_time?: number;
};

export type DbSchemaInfo = {
  tables: string[];
  columns: Record<string, Array<{ name: string; type: string; nullable: string; position: number }>>;
  primary_keys: Record<string, string[]>;
  foreign_keys: Array<{
    from_table: string;
    from_column: string;
    to_table: string;
    to_column: string;
  }>;
  indexes: Record<string, Array<{ name: string; definition: string }>>;
  metadata?: {
    generated_at?: number;
    query_timeout_ms?: number;
  };
};

type ChatApiResponse =
  | {
      status: "ok";
      response: string;
      data?: unknown;
      from_cache?: boolean;
      execution_time?: number;
    }
  | {
      status: "needs_human_input";
      session_id: string;
      interrupt: InterruptPayload;
      execution_time?: number;
    };

export type InterruptPayload = {
  type?: string;
  code?: string;
  [key: string]: unknown;
};

export type InterruptDecision = Record<string, unknown>;
export type InterruptHandler = (interrupt: InterruptPayload) => InterruptDecision | Promise<InterruptDecision>;

const API_BASE_URL_KEY = "t2sql_api_base_url_v2";
const LEGACY_API_BASE_URL_KEY = "t2sql_api_base_url_v1";
const SESSION_ID_KEY = "t2sql_session_id";

function isLocalHost(hostname: string) {
  return hostname === "localhost" || hostname === "127.0.0.1";
}

function normalizeApiBaseUrl(value?: string | null) {
  let raw = String(value ?? "").trim();
  if (!raw) raw = String(process.env.NEXT_PUBLIC_API_BASE_URL ?? "/backend").trim();

  // Fix the common typo that broke production for many users.
  raw = raw.replace(/\/beckend(\/|$)/gi, "/backend$1");

  if (raw.startsWith("/")) {
    const path = raw.replace(/\/$/, "");
    return path || "/backend";
  }

  if (typeof window !== "undefined") {
    const currentHost = window.location.hostname;
    const currentIsLocal = isLocalHost(currentHost);

    try {
      const parsed = new URL(raw);
      const parsedPath = parsed.pathname.replace(/\/$/, "") || "/backend";

      if (!currentIsLocal) {
        if (isLocalHost(parsed.hostname)) return "/backend";
        if (parsed.hostname !== currentHost) return "/backend";
      }

      return `${parsed.origin}${parsedPath}`;
    } catch {
      if (!currentIsLocal) return "/backend";
    }
  }

  const withNoTrailingSlash = raw.replace(/\/$/, "");
  return withNoTrailingSlash || "/backend";
}

let apiBaseUrl = normalizeApiBaseUrl(
  (typeof window !== "undefined" &&
    (localStorage.getItem(API_BASE_URL_KEY) ?? localStorage.getItem(LEGACY_API_BASE_URL_KEY))) ||
    process.env.NEXT_PUBLIC_API_BASE_URL ||
    "/backend"
);

export function setApiBaseUrl(url: string) {
  apiBaseUrl = normalizeApiBaseUrl(url);
  if (typeof window !== "undefined") {
    localStorage.setItem(API_BASE_URL_KEY, apiBaseUrl);
    localStorage.removeItem(LEGACY_API_BASE_URL_KEY);
  }
}

export function getApiBaseUrl() {
  if (typeof window !== "undefined") {
    const v = localStorage.getItem(API_BASE_URL_KEY) ?? localStorage.getItem(LEGACY_API_BASE_URL_KEY);
    apiBaseUrl = normalizeApiBaseUrl(v);
    localStorage.setItem(API_BASE_URL_KEY, apiBaseUrl);
    localStorage.removeItem(LEGACY_API_BASE_URL_KEY);
  }
  return apiBaseUrl;
}

function getOrCreateSessionId() {
  if (typeof window === "undefined") return "server";
  const existing = localStorage.getItem(SESSION_ID_KEY);
  if (existing) return existing;
  const id = crypto.randomUUID();
  localStorage.setItem(SESSION_ID_KEY, id);
  return id;
}

async function jsonFetch<T>(path: string, init?: RequestInit, token?: string | null): Promise<T> {
  const base = getApiBaseUrl();
  const res = await fetch(`${base}${path}`, {
    cache: "no-store",
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(init?.headers ?? {}),
    },
  });

  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const detail = (data && (data.detail ?? data.error)) || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data as T;
}

/** True if backend is up and DB is connected. */
export async function ping(): Promise<boolean> {
  try {
    const st = await jsonFetch<{ connected: boolean }>("/db/status", { method: "GET" });
    return !!st.connected;
  } catch {
    return false;
  }
}

/** Returns current DB user. */
export async function whoami(_token?: string | null): Promise<string> {
  void _token;
  const me = await jsonFetch<{ ok: boolean; connected: boolean; user?: string }>("/auth/me", { method: "GET" });
  return me.user ?? "";
}

/** Login is UI-level for compatibility with existing page. */
export async function login(username: string, password: string): Promise<{ token: string; user: string }> {
  const session_id = getOrCreateSessionId();
  await jsonFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ session_id, username, password }),
  });
  return { token: `dev:${username}:${Date.now()}`, user: username };
}

export async function connectDb(
  session_id: string,
  profile: { name: string; host: string; port: string; db: string; user: string; password: string }
) {
  return jsonFetch("/db/connect", {
    method: "POST",
    body: JSON.stringify({
      session_id,
      profile: {
        name: profile.name,
        host: profile.host,
        port: Number(profile.port),
        database: profile.db,
        user: profile.user,
        password: profile.password ?? "",
      },
    }),
  });
}

export async function disconnectDb() {
  return jsonFetch("/db/disconnect", { method: "POST", body: JSON.stringify({}) });
}

export async function getDbSchema(refresh = false): Promise<DbSchemaInfo> {
  const suffix = refresh ? "?refresh=true" : "";
  return jsonFetch<DbSchemaInfo>(`/api/db/schema${suffix}`, { method: "GET" });
}

export async function getDbSchemaErd(format: "svg" | "dot" = "svg", refresh = false): Promise<string> {
  const params = new URLSearchParams({ format });
  if (refresh) params.set("refresh", "true");
  const base = getApiBaseUrl();
  const res = await fetch(`${base}/api/db/schema/erd?${params.toString()}`, { cache: "no-store" });
  const text = await res.text();
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = text ? JSON.parse(text) : null;
      detail = data?.detail ?? data?.error ?? detail;
    } catch {
      detail = text || detail;
    }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return text;
}

function makeInterruptDecision(interrupt: InterruptPayload | null | undefined) {
  const interruptType = interrupt?.type;
  if (interruptType === "cache_review") {
    const query = String(interrupt?.query ?? "");
    const cachedResponse = String(interrupt?.cached_response ?? "");
    const score = typeof interrupt?.score === "number" ? interrupt.score : null;

    // If called outside browser (SSR/tests), fallback to fresh generation.
    if (typeof window === "undefined") {
      return { use_cache: false };
    }

    const preview = cachedResponse.length > 220 ? `${cachedResponse.slice(0, 220)}...` : cachedResponse;
    const scoreText = score !== null ? `\nСходство: ${(score * 100).toFixed(1)}%` : "";
    const text =
      `Найден похожий ответ в кэше.\n\n` +
      `Запрос: ${query || "—"}${scoreText}\n\n` +
      `Предпросмотр:\n${preview || "—"}\n\n` +
      `Использовать кэшированный ответ?`;

    const useCache = window.confirm(text);
    return { use_cache: useCache };
  }
  if (interruptType === "visualization_review") {
    return { approved: true, code: interrupt?.code };
  }
  return {};
}

/** Chat with auto-resume for FastAPI interrupts. */
export async function chat(
  session_id: string,
  message: string,
  mode: ChatMode = "table",
  token?: string | null,
  options?: { onInterrupt?: InterruptHandler }
): Promise<ChatResponse> {
  void mode;
  let current = await jsonFetch<ChatApiResponse>(
    "/chat",
    { method: "POST", body: JSON.stringify({ session_id, message }) },
    token ?? null
  );

  let guard = 0;
  while (current.status === "needs_human_input" && guard < 5) {
    const data = options?.onInterrupt
      ? await options.onInterrupt(current.interrupt)
      : makeInterruptDecision(current.interrupt);
    current = await jsonFetch<ChatApiResponse>(
      "/chat/resume",
      { method: "POST", body: JSON.stringify({ session_id, data }) },
      token ?? null
    );
    guard += 1;
  }

  if (current.status !== "ok") {
    throw new Error("Сервер ожидает дополнительный ввод пользователя");
  }

  const responseText = current.response ?? "";
  const chartMatch = responseText.match(/data:image\/png;base64,([A-Za-z0-9+/=]+)/);
  const chartBase64 = chartMatch?.[1] ?? null;

  return {
    response: responseText,
    data: current.data,
    chart_base64: chartBase64,
    execution_time: current.execution_time,
    from_cache: current.from_cache,
  };
}
