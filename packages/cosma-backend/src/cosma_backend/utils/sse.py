"""
ServerSentEvent helper class for Quart SSE endpoints
"""
import json
from typing import Optional, Any


def sse_comment(text: str = "keepalive") -> str:
    """
    Create an SSE comment for keep-alive or debugging purposes.
    
    According to the SSE spec, lines starting with ':' are comments
    and are ignored by clients. These are useful for:
    - Keep-alive messages to prevent connection timeouts
    - Debugging/logging without affecting the client
    
    Args:
        text: Optional comment text (default: "keepalive")
    
    Returns:
        Properly formatted SSE comment string with trailing double newline
    
    Example:
        >>> sse_comment()
        ': keepalive\\n\\n'
        >>> sse_comment("heartbeat")
        ': heartbeat\\n\\n'
    """
    return f": {text}\n\n"


class ServerSentEvent:
    """
    Helper class to format Server-Sent Events according to the SSE specification.
    
    SSE format:
        event: event_name
        id: event_id
        retry: retry_time
        data: message_data
        
    """
    
    def __init__(
        self,
        data: Any,
        event: Optional[str] = None,
        id: Optional[str] = None,
        retry: Optional[int] = None,
    ):
        """
        Args:
            data: The message data (will be JSON-encoded if not a string)
            event: Optional event name for named events
            id: Optional event ID (used for reconnection)
            retry: Optional reconnection time in milliseconds
        """
        self.data = data
        self.event = event
        self.id = id
        self.retry = retry
    
    def encode(self) -> str:
        """
        Encode the event in SSE format.
        
        Returns:
            Properly formatted SSE string with trailing double newline
        """
        lines = []
        
        if self.event:
            lines.append(f"event: {self.event}")
        
        if self.id:
            lines.append(f"id: {self.id}")
        
        if self.retry:
            lines.append(f"retry: {self.retry}")
        
        # Handle data - convert to JSON if not a string
        if isinstance(self.data, str):
            data_str = self.data
        else:
            data_str = json.dumps(self.data)
        
        # Support multi-line data
        for line in data_str.splitlines():
            lines.append(f"data: {line}")
        
        # SSE spec requires double newline at the end
        return "\n".join(lines) + "\n\n"
