from dataclasses import dataclass, field
from typing import Any, Dict, Self
import enum

from cosma_backend.utils.sse import ServerSentEvent


class UpdateOpcode(enum.Enum):
    """
    Opcodes for different types of backend updates sent to the frontend via SSE.
    """
    # File processing updates
    FILE_PARSING = "file_parsing"
    FILE_PARSED = "file_parsed"
    FILE_SUMMARIZING = "file_summarizing"
    FILE_SUMMARIZED = "file_summarized"
    FILE_EMBEDDING = "file_embedding"
    FILE_EMBEDDED = "file_embedded"
    FILE_COMPLETE = "file_complete"
    FILE_FAILED = "file_failed"
    FILE_SKIPPED = "file_skipped"
    
    # File system events (from watcher)
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    
    # Watch directory updates
    WATCH_ADDED = "watch_added"
    WATCH_REMOVED = "watch_removed"
    WATCH_STARTED = "watch_started"
    
    # Directory processing updates
    DIRECTORY_PROCESSING_STARTED = "directory_processing_started"
    DIRECTORY_PROCESSING_COMPLETED = "directory_processing_completed"
    
    # General updates
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    INFO = "info"
    
    SHUTTING_DOWN = "shutting_down"


@dataclass
class Update:
    """
    A model representing a backend update to be sent to the frontend via SSE.
    
    Each update has an opcode (message type) and optional data payload.
    """
    opcode: UpdateOpcode
    data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, opcode: UpdateOpcode, **kwargs) -> Self:
        """
        Create an Update instance with the given opcode and data.
        
        Args:
            opcode: The type of update (UpdateOpcode enum value)
            **kwargs: Arbitrary keyword arguments that will be stored in the data dict
        
        Returns:
            An Update instance
        
        Example:
            >>> update = Update.create(UpdateOpcode.FILE_DISCOVERED, path="/docs/file.pdf", size=1024)
            >>> update.opcode
            <UpdateOpcode.FILE_DISCOVERED: 'file_discovered'>
            >>> update.data
            {'path': '/docs/file.pdf', 'size': 1024}
        """
        return cls(opcode=opcode, data=kwargs)
    
    @classmethod
    def file_parsing(cls, path: str, filename: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_PARSING update."""
        return cls.create(UpdateOpcode.FILE_PARSING, path=path, filename=filename, **kwargs)
    
    @classmethod
    def file_parsed(cls, path: str, filename: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_PARSED update."""
        return cls.create(UpdateOpcode.FILE_PARSED, path=path, filename=filename, **kwargs)
    
    @classmethod
    def file_summarizing(cls, path: str, filename: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_SUMMARIZING update."""
        return cls.create(UpdateOpcode.FILE_SUMMARIZING, path=path, filename=filename, **kwargs)
    
    @classmethod
    def file_summarized(cls, path: str, filename: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_SUMMARIZED update."""
        return cls.create(UpdateOpcode.FILE_SUMMARIZED, path=path, filename=filename, **kwargs)
    
    @classmethod
    def file_embedding(cls, path: str, filename: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_EMBEDDING update."""
        return cls.create(UpdateOpcode.FILE_EMBEDDING, path=path, filename=filename, **kwargs)
    
    @classmethod
    def file_embedded(cls, path: str, filename: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_EMBEDDED update."""
        return cls.create(UpdateOpcode.FILE_EMBEDDED, path=path, filename=filename, **kwargs)
    
    @classmethod
    def file_complete(cls, path: str, filename: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_COMPLETE update."""
        return cls.create(UpdateOpcode.FILE_COMPLETE, path=path, filename=filename, **kwargs)
    
    @classmethod
    def file_skipped(cls, path: str, filename: str, reason: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_SKIPPED update."""
        return cls.create(UpdateOpcode.FILE_SKIPPED, path=path, filename=filename, reason=reason, **kwargs)
    
    @classmethod
    def file_failed(cls, path: str, filename: str, error: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_FAILED update."""
        return cls.create(UpdateOpcode.FILE_FAILED, path=path, filename=filename, error=error, **kwargs)
    
    @classmethod
    def file_created(cls, path: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_CREATED update."""
        return cls.create(UpdateOpcode.FILE_CREATED, path=path, **kwargs)
    
    @classmethod
    def file_modified(cls, path: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_MODIFIED update."""
        return cls.create(UpdateOpcode.FILE_MODIFIED, path=path, **kwargs)
    
    @classmethod
    def file_deleted(cls, path: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_DELETED update."""
        return cls.create(UpdateOpcode.FILE_DELETED, path=path, **kwargs)
    
    @classmethod
    def file_moved(cls, src_path: str, dest_path: str, **kwargs) -> Self:
        """Convenience method for creating a FILE_MOVED update."""
        return cls.create(UpdateOpcode.FILE_MOVED, src_path=src_path, dest_path=dest_path, **kwargs)
    
    @classmethod
    def directory_processing_started(cls, path: str, **kwargs) -> Self:
        """Convenience method for creating a DIRECTORY_PROCESSING_STARTED update."""
        return cls.create(UpdateOpcode.DIRECTORY_PROCESSING_STARTED, path=path, **kwargs)
    
    @classmethod
    def directory_processing_completed(cls, path: str, **kwargs) -> Self:
        """Convenience method for creating a DIRECTORY_PROCESSING_COMPLETED update."""
        return cls.create(UpdateOpcode.DIRECTORY_PROCESSING_COMPLETED, path=path, **kwargs)
    
    @classmethod
    def error(cls, message: str, **kwargs) -> Self:
        """Convenience method for creating an ERROR update."""
        return cls.create(UpdateOpcode.ERROR, message=message, **kwargs)
    
    @classmethod
    def info(cls, message: str, **kwargs) -> Self:
        """Convenience method for creating an INFO update."""
        return cls.create(UpdateOpcode.INFO, message=message, **kwargs)
        
    @classmethod
    def shutting_down(cls, **kwargs) -> Self:
        """Convenience method for creating a SHUTTING_DOWN update."""
        return cls.create(UpdateOpcode.SHUTTING_DOWN, **kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Update to a dictionary for serialization.
        
        Returns:
            A dictionary with 'opcode' and 'data' keys
        """
        return {
            "opcode": self.opcode.value,
            "data": self.data
        }
    
    def to_sse(self, event_id: str | None = None) -> ServerSentEvent:
        """
        Convert the Update to a ServerSentEvent for SSE transmission.
        
        Args:
            event_id: Optional event ID for SSE reconnection support
        
        Returns:
            A ServerSentEvent instance ready to be encoded and sent
        
        Example:
            >>> update = Update.file_parsed("/docs/file.pdf", "file.pdf")
            >>> sse = update.to_sse()
            >>> message = sse.encode()  # Ready to send via SSE endpoint
        """
        return ServerSentEvent(
            data=self.to_dict(),
            event="update",  # All updates use the same event type
            id=event_id,
        )
    
    def __str__(self) -> str:
        """Return a string representation of the update."""
        return f"Update(opcode={self.opcode.value}, data={self.data})"
