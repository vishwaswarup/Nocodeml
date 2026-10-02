"""Print a login token for a test user (for pasting into the /docs "Authorize" button).

    .venv/bin/python get_token.py        # user A;   add B for the second user
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
from nocodeml_engine.persistence import SupabaseSettings, sign_in_with_password  # noqa: E402

w = (sys.argv[1] if len(sys.argv) > 1 else "A").upper()
print(sign_in_with_password(SupabaseSettings.from_env(), os.environ[f"SUPABASE_TEST_EMAIL_{w}"],
                            os.environ[f"SUPABASE_TEST_PASSWORD_{w}"]))
