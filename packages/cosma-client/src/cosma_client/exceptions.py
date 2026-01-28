"""
Exceptions for the Cosma client library.
"""


class CosmaClientError(Exception):
    """Base exception for all Cosma client errors."""
    pass


class ConnectionError(CosmaClientError):
    """Raised when unable to connect to the Cosma server."""
    pass


class ServerNotRunningError(ConnectionError):
    """Raised when the Cosma server is not running."""

    def __init__(self, base_url: str = "http://127.0.0.1:60534"):
        self.base_url = base_url
        super().__init__(
            f"Cannot connect to Cosma server at {base_url}. "
            "Is the server running? Start it with: cosma serve"
        )


class APIError(CosmaClientError):
    """Raised when the API returns an error response."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"API error ({status_code}): {message}")
