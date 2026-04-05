"""LLM call monitoring and statistics module.

Tracks LLM API calls for performance analysis and error diagnosis.
"""

import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


logger = logging.getLogger(__name__)


class CallStatus(Enum):
    """Status of an LLM call."""
    STARTED = "started"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ErrorType(Enum):
    """Classification of LLM call errors."""
    SOCKET_HANG_UP = "socket_hang_up"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT = "rate_limit"
    SERVER_ERROR = "server_error"
    CONTENT_ERROR = "content_error"
    AUTH_ERROR = "auth_error"
    NOT_FOUND = "not_found"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class LLMCallMetrics:
    """Metrics for a single LLM call."""
    request_id: str
    operation: str  # health_check, completion, complete_json
    provider: str
    model: str
    start_time: datetime
    end_time: datetime | None = None
    status: CallStatus = CallStatus.STARTED
    error_type: ErrorType | None = None
    error_message: str | None = None
    retry_count: int = 0
    max_tokens: int | None = None
    timeout: int | None = None
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging/API responses."""
        return {
            "request_id": self.request_id,
            "operation": self.operation,
            "provider": self.provider,
            "model": self.model,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "error_type": self.error_type.value if self.error_type else None,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "latency_ms": self.latency_ms,
        }


class LLMMonitor:
    """Thread-safe singleton for monitoring LLM calls."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if not hasattr(self, "_initialized"):
            self._local_lock = threading.Lock()
            self._active_calls: dict[str, LLMCallMetrics] = {}
            self._completed_calls: list[LLMCallMetrics] = []
            self._max_history = 10000
            self._stats = defaultdict(lambda: defaultdict(int))
            self._errors: list[dict[str, Any]] = []
            self._initialized = True

    def start_call(
        self,
        operation: str,
        provider: str,
        model: str,
        timeout: int,
        max_tokens: int | None = None,
    ) -> str:
        """Start tracking a new LLM call.

        Args:
            operation: Type of operation (health_check, completion, complete_json)
            provider: LLM provider name
            model: Model name
            timeout: Timeout in seconds
            max_tokens: Maximum tokens for the request

        Returns:
            Unique request ID for this call
        """
        request_id = str(uuid.uuid4())[:8]  # Short ID for logs

        metrics = LLMCallMetrics(
            request_id=request_id,
            operation=operation,
            provider=provider,
            model=model,
            start_time=datetime.now(),
            timeout=timeout,
            max_tokens=max_tokens,
        )

        with self._local_lock:
            self._active_calls[request_id] = metrics

        # Log call start
        logger.info(
            f"[LLM] [INFO] [{request_id}] Starting {operation} | "
            f"provider={provider} model={model} timeout={timeout}s"
        )

        return request_id

    def update_call(
        self,
        request_id: str,
        status: CallStatus,
        error_type: ErrorType | None = None,
        error_message: str | None = None,
        retry_count: int = 0,
    ) -> float | None:
        """Update call status and metrics.

        Args:
            request_id: The request ID returned by start_call
            status: Final status of the call
            error_type: Type of error if failed
            error_message: Error message if failed
            retry_count: Number of retries attempted

        Returns:
            Latency in milliseconds, or None if request not found
        """
        with self._local_lock:
            if request_id not in self._active_calls:
                logger.warning(f"[LLM] Request ID {request_id} not found in active calls")
                return None

            metrics = self._active_calls[request_id]
            metrics.status = status
            metrics.error_type = error_type
            metrics.error_message = error_message
            metrics.retry_count = retry_count
            metrics.end_time = datetime.now()

            # Calculate latency
            latency = (metrics.end_time - metrics.start_time).total_seconds() * 1000
            metrics.latency_ms = round(latency, 2)

            # Move to completed calls
            self._completed_calls.append(metrics)
            del self._active_calls[request_id]

            # Update statistics
            self._stats[metrics.operation][status.value] += 1

            # Log completion
            if status == CallStatus.SUCCESS:
                logger.info(
                    f"[LLM] [INFO] [{request_id}] Completed {metrics.operation} | "
                    f"latency={metrics.latency_ms}ms status=success retry_count={retry_count}"
                )
            else:
                error_label = error_type.value if error_type else "unknown"
                logger.error(
                    f"[LLM] [ERROR] [{request_id}] Failed {metrics.operation} | "
                    f"latency={metrics.latency_ms}ms status={status.value} "
                    f"error_type={error_label} retry_count={retry_count} "
                    f"error={error_message[:100] if error_message else ''}"
                )

                # Record error
                self._errors.append({
                    "request_id": request_id,
                    "operation": metrics.operation,
                    "provider": metrics.provider,
                    "model": metrics.model,
                    "error_type": error_label,
                    "error_message": error_message,
                    "retry_count": retry_count,
                    "latency_ms": metrics.latency_ms,
                    "timestamp": metrics.end_time.isoformat(),
                })

            # Trim history if needed
            if len(self._completed_calls) > self._max_history:
                remove_count = len(self._completed_calls) - self._max_history
                self._completed_calls = self._completed_calls[remove_count:]

            return metrics.latency_ms

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated statistics.

        Returns:
            Dictionary with aggregated statistics
        """
        with self._local_lock:
            total_calls = sum(len(calls) for calls in self._stats.values())
            successful_calls = sum(
                calls.get(CallStatus.SUCCESS.value, 0)
                for calls in self._stats.values()
            )
            failed_calls = sum(
                calls.get(CallStatus.FAILED.value, 0)
                for calls in self._stats.values()
            )
            timeout_calls = sum(
                calls.get(CallStatus.TIMEOUT.value, 0)
                for calls in self._stats.values()
            )

            # Calculate average latency by operation
            operation_stats: dict[str, Any] = {}
            for op, calls in self._stats.items():
                op_completed = [
                    c for c in self._completed_calls
                    if c.operation == op and c.latency_ms is not None
                ]
                if op_completed:
                    avg_latency = sum(c.latency_ms for c in op_completed) / len(op_completed)
                    max_latency = max(c.latency_ms for c in op_completed)
                    min_latency = min(c.latency_ms for c in op_completed)
                else:
                    avg_latency = max_latency = min_latency = 0

                operation_stats[op] = {
                    "total": calls.get("started", 0) + calls.get("success", 0) + calls.get("failed", 0),
                    "success": calls.get(CallStatus.SUCCESS.value, 0),
                    "failed": calls.get(CallStatus.FAILED.value, 0),
                    "timeout": calls.get(CallStatus.TIMEOUT.value, 0),
                    "avg_latency_ms": round(avg_latency, 2),
                    "max_latency_ms": round(max_latency, 2),
                    "min_latency_ms": round(min_latency, 2),
                }

            return {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "failed_calls": failed_calls,
                "timeout_calls": timeout_calls,
                "success_rate": round(successful_calls / total_calls, 4) if total_calls > 0 else 0,
                "by_operation": operation_stats,
                "active_calls": len(self._active_calls),
                "total_errors": len(self._errors),
            }

    def get_errors(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent errors.

        Args:
            limit: Maximum number of errors to return

        Returns:
            List of error records
        """
        with self._local_lock:
            return self._errors[-limit:] if self._errors else []

    def get_active_calls(self) -> list[dict[str, Any]]:
        """Get currently active calls.

        Returns:
            List of active call records
        """
        with self._local_lock:
            now = datetime.now()
            active = []
            for call in self._active_calls.values():
                elapsed = (now - call.start_time).total_seconds() * 1000
                active.append({
                    "request_id": call.request_id,
                    "operation": call.operation,
                    "provider": call.provider,
                    "model": call.model,
                    "start_time": call.start_time.isoformat(),
                    "elapsed_ms": round(elapsed, 2),
                    "timeout": call.timeout,
                })
            return active

    def get_recent_calls(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent completed calls.

        Args:
            limit: Maximum number of calls to return

        Returns:
            List of recent call records
        """
        with self._local_lock:
            recent = self._completed_calls[-limit:]
            return [call.to_dict() for call in recent]


def classify_error(error: Exception) -> ErrorType:
    """Classify an exception into an ErrorType.

    Args:
        error: The exception to classify

    Returns:
        The classified ErrorType
    """
    error_str = str(error).lower()

    # Check for socket hang up
    if "socket hang up" in error_str or "sockettimeout" in error_str:
        return ErrorType.SOCKET_HANG_UP

    # Check for timeout
    if "timeout" in error_str or "timed out" in error_str:
        return ErrorType.TIMEOUT_ERROR

    # Check for rate limit
    if "rate limit" in error_str or "429" in error_str:
        return ErrorType.RATE_LIMIT

    # Check for auth errors
    if "auth" in error_str or "api key" in error_str or "401" in error_str or "403" in error_str:
        return ErrorType.AUTH_ERROR

    # Check for not found
    if "404" in error_str:
        return ErrorType.NOT_FOUND

    # Check for server errors (5xx)
    if any(code in error_str for code in ["500", "502", "503", "504"]):
        return ErrorType.SERVER_ERROR

    # Check for content errors (JSON parsing, empty response, etc.)
    if "json" in error_str and any(word in error_str for word in ["decode", "parse", "extract"]):
        return ErrorType.CONTENT_ERROR
    if "empty" in error_str and "response" in error_str:
        return ErrorType.CONTENT_ERROR
    if "truncat" in error_str:
        return ErrorType.CONTENT_ERROR

    return ErrorType.UNKNOWN_ERROR


# Global monitor instance
monitor = LLMMonitor()
