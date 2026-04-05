"""Monitoring and statistics API router for LLM calls."""

from fastapi import APIRouter

from app.llm_monitor import monitor

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/stats")
async def get_llm_stats():
    """Get aggregated LLM call statistics.

    Returns:
        Dictionary with total calls, success rate, and per-operation stats.
    """
    stats = monitor.get_stats()
    return {
        "statistics": stats,
        "active_calls": len(monitor._active_calls),
        "total_completed": len(monitor._completed_calls),
    }


@router.get("/errors")
async def get_llm_errors(limit: int = 100):
    """Get recent LLM errors.

    Args:
        limit: Maximum number of errors to return (default: 100)

    Returns:
        List of recent error records with details.
    """
    errors = monitor.get_errors(limit)
    return {
        "errors": errors,
        "total_errors": len(errors),
    }


@router.get("/active-calls")
async def get_active_calls():
    """Get currently active LLM calls.

    Returns:
        List of active call records with elapsed time.
    """
    active_calls = monitor.get_active_calls()
    return {
        "active_calls": active_calls,
        "count": len(active_calls),
    }


@router.get("/recent-calls")
async def get_recent_calls(limit: int = 50):
    """Get recent completed LLM calls.

    Args:
        limit: Maximum number of calls to return (default: 50)

    Returns:
        List of recent completed call records.
    """
    recent_calls = monitor.get_recent_calls(limit)
    return {
        "recent_calls": recent_calls,
        "count": len(recent_calls),
    }
