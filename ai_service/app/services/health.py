def get_health_status() -> dict:
    return {
        "status": "healthy",
        "database": "connected" # You can expand this to actually ping Supabase later
    }