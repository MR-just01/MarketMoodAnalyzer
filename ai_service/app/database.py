import os
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("https://fdzznuelgacwahsjzjji.supabase.co/rest/v1/", "")
SUPABASE_KEY = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZkenpudWVsZ2Fjd2Foc2p6amppIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODIyMTczNCwiZXhwIjoyMDkzNzk3NzM0fQ.yQJPwXWDoE52dNI4PoTmnbzzscamWROgW8wjdMVAXuU", "")

# Initialize the client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)