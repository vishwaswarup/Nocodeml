import { supabase } from "./supabase";

export class ApiError extends Error {
  constructor(public status: number, message: string, public issues: string[] = []) {
    super(message);
  }
}

/** Call the FastAPI backend as the signed-in user. */
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const { data } = await supabase().auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new ApiError(401, "Not signed in.");
  const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}${path}`, {
    ...init,
    headers: {
      ...(init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...init.headers,
      Authorization: `Bearer ${token}`,
    },
  });
  if (!res.ok) {
    let body: { detail?: unknown; issues?: string[] } = {};
    try { body = await res.json(); } catch { /* non-JSON error */ }
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${res.status}).`;
    throw new ApiError(res.status, detail, body.issues ?? []);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}
