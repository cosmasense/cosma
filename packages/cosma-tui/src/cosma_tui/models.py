from dataclasses import dataclass, field
from typing import Any, Dict, Self
import enum
import json


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
    A model representing a backend update received from the server via SSE.
    
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
        """
        return cls(opcode=opcode, data=kwargs)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        """
        Create an Update instance from a dictionary (parsed from SSE).
        
        Args:
            data: Dictionary with 'opcode' and 'data' keys
        
        Returns:
            An Update instance
        """
        opcode_str = data.get('opcode', '')
        try:
            opcode = UpdateOpcode(opcode_str)
        except ValueError:
            # Fallback to INFO if unknown opcode
            opcode = UpdateOpcode.INFO
        
        update_data = data.get('data', {})
        return cls(opcode=opcode, data=update_data)
    
    @classmethod
    def from_sse_data(cls, sse_data: str) -> Self:
        """
        Create an Update instance from SSE data string.
        
        Args:
            sse_data: Raw data string from SSE (JSON)
        
        Returns:
            An Update instance
        """
        try:
            data = json.loads(sse_data)
            return cls.from_dict(data)
        except json.JSONDecodeError:
            # Fallback: treat as INFO message with raw data
            return cls.create(UpdateOpcode.INFO, message=sse_data)
    
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
    
    def get_display_message(self) -> str:
        """
        Get a human-readable display message for this update.
        
        Returns:
            A formatted string suitable for display in the TUI
        """
        # File processing messages
        if self.opcode == UpdateOpcode.FILE_PARSING:
            filename = self.data.get('filename', 'Unknown file')
            return f"Parsing {filename}..."
        
        elif self.opcode == UpdateOpcode.FILE_PARSED:
            filename = self.data.get('filename', 'Unknown file')
            return f"Parsed {filename}"
        
        elif self.opcode == UpdateOpcode.FILE_SUMMARIZING:
            filename = self.data.get('filename', 'Unknown file')
            return f"Summarizing {filename}..."
        
        elif self.opcode == UpdateOpcode.FILE_SUMMARIZED:
            filename = self.data.get('filename', 'Unknown file')
            return f"Summarized {filename}"
        
        elif self.opcode == UpdateOpcode.FILE_EMBEDDING:
            filename = self.data.get('filename', 'Unknown file')
            return f"Embedding {filename}..."
        
        elif self.opcode == UpdateOpcode.FILE_EMBEDDED:
            filename = self.data.get('filename', 'Unknown file')
            return f"Embedded {filename}"
        
        elif self.opcode == UpdateOpcode.FILE_COMPLETE:
            filename = self.data.get('filename', 'Unknown file')
            return f"Completed {filename}"
        
        elif self.opcode == UpdateOpcode.FILE_FAILED:
            filename = self.data.get('filename', 'Unknown file')
            error = self.data.get('error', 'Unknown error')
            return f"Failed {filename}: {error}"
        
        elif self.opcode == UpdateOpcode.FILE_SKIPPED:
            filename = self.data.get('filename', 'Unknown file')
            reason = self.data.get('reason', 'Unknown reason')
            return f"Skipped {filename}: {reason}"
        
        # File system events
        elif self.opcode == UpdateOpcode.FILE_CREATED:
            path = self.data.get('path', 'Unknown path')
            return f"Created {path}"
        
        elif self.opcode == UpdateOpcode.FILE_MODIFIED:
            path = self.data.get('path', 'Unknown path')
            return f"Modified {path}"
        
        elif self.opcode == UpdateOpcode.FILE_DELETED:
            path = self.data.get('path', 'Unknown path')
            return f"Deleted {path}"
        
        elif self.opcode == UpdateOpcode.FILE_MOVED:
            src = self.data.get('src_path', 'Unknown source')
            dst = self.data.get('dest_path', 'Unknown destination')
            return f"Moved {src} -> {dst}"
        
        # Directory processing
        elif self.opcode == UpdateOpcode.DIRECTORY_PROCESSING_STARTED:
            path = self.data.get('path', 'Unknown path')
            return f"Processing directory: {path}"
        
        elif self.opcode == UpdateOpcode.DIRECTORY_PROCESSING_COMPLETED:
            path = self.data.get('path', 'Unknown path')
            return f"Completed directory: {path}"
        
        # Watch directory updates
        elif self.opcode == UpdateOpcode.WATCH_ADDED:
            return f"Added watch directory"
        
        elif self.opcode == UpdateOpcode.WATCH_REMOVED:
            return f"Removed watch directory"
        
        elif self.opcode == UpdateOpcode.WATCH_STARTED:
            return "Started watching for changes"
        
        # General updates
        elif self.opcode == UpdateOpcode.STATUS_UPDATE:
            message = self.data.get('message', 'Status update')
            return f"Status: {message}"
        
        elif self.opcode == UpdateOpcode.ERROR:
            message = self.data.get('message', 'Error occurred')
            return f"Error: {message}"
        
        elif self.opcode == UpdateOpcode.INFO:
            message = self.data.get('message', 'Info')
            return f"Info: {message}"
        
        elif self.opcode == UpdateOpcode.SHUTTING_DOWN:
            return "Server shutting down"
        
        # Fallback for unknown opcodes
        else:
            return f"Unknown update: {self.opcode.value}"
    
    def __str__(self) -> str:
        """Return a string representation of the update."""
        return self.get_display_message()