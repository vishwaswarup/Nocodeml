import { supabase } from "./supabase";

export class ApiError extends Error {
  constructor(public status: number, message: string, public issues: string[] = []) {
    super(message);
  }
}

const RETRY_STATUS = new Set([502, 503, 504]);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Call the FastAPI backend as the signed-in user. Read-only (GET) requests are retried twice on a network
 * error or a temporary server/sign-in outage, so a brief blip never shows up as an error.
 */
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const { data } = await supabase().auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new ApiError(401, "Not signed in.");
  const safe = !init.method || init.method.toUpperCase() === "GET";
  const send = () => fetch(`${process.env.NEXT_PUBLIC_API_URL}${path}`, {
    ...init,
    headers: {
      ...(init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...init.headers,
      Authorization: `Bearer ${token}`,
    },
  });
  let res: Response | null = null;
  for (let attempt = 0; attempt < (safe ? 3 : 1); attempt++) {
    if (attempt) await sleep(400 * attempt);
    try {
      res = await send();
    } catch (e) {
      if (!safe || attempt === 2) throw e;
      continue;
    }
    if (!safe || !RETRY_STATUS.has(res.status)) break;
  }
  if (!res) throw new TypeError("Network error");
  if (!res.ok) {
    let body: { detail?: unknown; issues?: string[] } = {};
    try { body = await res.json(); } catch { /* non-JSON error */ }
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${res.status}).`;
    throw new ApiError(res.status, detail, body.issues ?? []);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}
