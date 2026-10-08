import { supabase } from "./supabase";

export class ApiError extends Error {
  constructor(public status: number, message: string, public issues: string[] = []) {
    super(message);
  }
}

const RETRY_STATUS = new Set([502, 503, 504]);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
// Waits between attempts of a read request (about 1 minute in all). A free host puts an idle server to sleep and takes
// up to a minute to wake it on the first request; waiting it out beats showing an error.
const RETRY_DELAYS_MS = [1000, 2000, 4000, 8000, 12000, 15000, 15000];

/** Send a request to the FastAPI backend as the signed-in user and return the successful Response (errors throw). */
async function request(path: string, init: RequestInit = {}): Promise<Response> {
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
  const attempts = safe ? RETRY_DELAYS_MS.length + 1 : 1;
  for (let attempt = 0; attempt < attempts; attempt++) {
    if (attempt) await sleep(RETRY_DELAYS_MS[attempt - 1]);
    try {
      res = await send();
    } catch (e) {
      if (!safe || attempt === attempts - 1) throw e;
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
  return res;
}

/**
 * Call the FastAPI backend as the signed-in user. Read-only (GET) requests are retried for up to a minute on a network
 * error or a temporary server/sign-in outage, so a brief blip (or a sleeping free server waking up) never shows up as an error.
 */
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await request(path, init);
  return res.status === 204 ? (undefined as T) : res.json();
}

/** Like `api`, for endpoints that return a file. */
export async function apiFile(path: string, init: RequestInit = {}): Promise<{ blob: Blob; filename: string }> {
  const res = await request(path, init);
  const name = /filename="?([^";]+)"?/.exec(res.headers.get("content-disposition") ?? "")?.[1];
  return { blob: await res.blob(), filename: name ?? "download" };
}
