class DatabaseClosingError(Exception):
    """Raised when a DB operation is attempted after the pool has started closing."""

    def __init__(self) -> None:
        super().__init__("Database pool is closed")
