"""Supabase client construction.

Security rule: user data is only ever accessed with the *user's own* access token, so
Postgres row-level security applies. The service-role key (which bypasses RLS) is never
read or used by this package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SupabaseSettings:
    url: str
    anon_key: str

    @classmethod
    def from_env(cls) -> "SupabaseSettings":
        try:
            return cls(url=os.environ["SUPABASE_URL"], anon_key=os.environ["SUPABASE_ANON_KEY"])
        except KeyError as e:
            raise RuntimeError(f"Missing environment variable {e.args[0]}") from None


def build_client(settings: SupabaseSettings, access_token: str):
    """A Supabase client that sends `access_token` on every request (no network call here).

    Build one per request: the client's HTTP/2 connections are not safe to share between threads.
    """
    from supabase import ClientOptions, create_client

    return create_client(settings.url, settings.anon_key, options=ClientOptions(
        headers={"Authorization": f"Bearer {access_token}"}))


def authenticated_client(settings: SupabaseSettings, access_token: str):
    """Return (client, user_id) acting as the owner of `access_token`.

    The token is validated with Supabase Auth; every request then carries it, so RLS sees
    auth.uid() == the user.
    """
    client = build_client(settings, access_token)
    user = client.auth.get_user(access_token)
    if user is None or user.user is None:
        raise PermissionError("Invalid or expired access token")
    return client, user.user.id


def sign_in_with_password(settings: SupabaseSettings, email: str, password: str) -> str:
    """Convenience for scripts/tests; returns an access token. Web UIs use Supabase Auth directly."""
    from supabase import create_client

    res = create_client(settings.url, settings.anon_key).auth.sign_in_with_password(
        {"email": email, "password": password})
    return res.session.access_token
