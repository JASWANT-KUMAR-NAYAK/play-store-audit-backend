from supabase import Client, create_client

from config.settings import SUPABASE_KEY, SUPABASE_URL


def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase configuration is missing.")

    return create_client(SUPABASE_URL, SUPABASE_KEY)