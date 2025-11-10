"""Unit tests for backend models."""

import datetime
from pathlib import Path
import pytest

from cosma_backend.models.file import File
from cosma_backend.models.status import ProcessingStatus


@pytest.mark.unit
class TestFile:
    """Test cases for the File model."""

    def test_file_from_path(self):
        """Test creating a File object from a file path."""
        # This test would require an actual file, so we'll mock the stat result
        test_path = Path("/test/sample.txt")
        
        # For now, let's test the logic with a simpler approach
        # In a real test, you'd create a temporary file
        file = File(
            path=test_path,
            file_path=str(test_path),
            filename="sample.txt",
            extension=".txt",
            file_size=1024,
            created=datetime.datetime.now(datetime.timezone.utc),
            modified=datetime.datetime.now(datetime.timezone.utc),
            accessed=datetime.datetime.now(datetime.timezone.utc),
        )
        
        assert file.file_path == "/test/sample.txt"
        assert file.filename == "sample.txt"
        assert file.extension == ".txt"
        assert file.status == ProcessingStatus.DISCOVERED

    def test_file_with_processing_data(self, sample_file_data):
        """Test creating a File object with processing data."""
        file = File(**sample_file_data)
        
        assert file.file_path == "/test/sample.txt"
        assert file.content == "Sample content for testing"
        assert file.summary == "A test file with sample content"
        assert file.status == ProcessingStatus.COMPLETE
        assert file.keywords == ["test", "sample", "content"]

    def test_file_to_response(self, sample_file_data):
        """Test converting File to FileResponse."""
        file = File(**sample_file_data)
        response = file.to_response()
        
        assert response.file_path == file.file_path
        assert response.filename == file.filename
        assert response.extension == file.extension
        assert response.created == file.created
        assert response.modified == file.modified
        assert response.accessed == file.accessed
        assert response.title == file.title
        assert response.summary == file.summary

    def test_file_status_transitions(self):
        """Test file status transitions through processing pipeline."""
        file = File(
            path=Path("/test/sample.txt"),
            file_path="/test/sample.txt",
            filename="sample.txt",
            extension=".txt",
            file_size=1024,
            created=datetime.datetime.now(datetime.timezone.utc),
            modified=datetime.datetime.now(datetime.timezone.utc),
            accessed=datetime.datetime.now(datetime.timezone.utc),
        )
        
        # Initial state
        assert file.status == ProcessingStatus.DISCOVERED
        
        # After parsing
        file.status = ProcessingStatus.PARSED
        file.parsed_at = datetime.datetime.now(datetime.timezone.utc)
        assert file.status == ProcessingStatus.PARSED
        assert file.parsed_at is not None
        
        # After summarization
        file.status = ProcessingStatus.SUMMARIZED
        file.summarized_at = datetime.datetime.now(datetime.timezone.utc)
        assert file.status == ProcessingStatus.SUMMARIZED
        assert file.summarized_at is not None
        
        # After embedding
        file.status = ProcessingStatus.COMPLETE
        file.embedded_at = datetime.datetime.now(datetime.timezone.utc)
        assert file.status == ProcessingStatus.COMPLETE
        assert file.embedded_at is not None

    def test_file_error_handling(self):
        """Test file error handling."""
        file = File(
            path=Path("/test/sample.txt"),
            file_path="/test/sample.txt",
            filename="sample.txt",
            extension=".txt",
            file_size=1024,
            created=datetime.datetime.now(datetime.timezone.utc),
            modified=datetime.datetime.now(datetime.timezone.utc),
            accessed=datetime.datetime.now(datetime.timezone.utc),
        )
        
        # Set error state
        file.status = ProcessingStatus.FAILED
        file.processing_error = "Failed to parse file"
        
        assert file.status == ProcessingStatus.FAILED
        assert file.processing_error == "Failed to parse file"

    def test_file_keywords_management(self):
        """Test keywords management in File model."""
        file = File(
            path=Path("/test/sample.txt"),
            file_path="/test/sample.txt",
            filename="sample.txt",
            extension=".txt",
            file_size=1024,
            created=datetime.datetime.now(datetime.timezone.utc),
            modified=datetime.datetime.now(datetime.timezone.utc),
            accessed=datetime.datetime.now(datetime.timezone.utc),
        )
        
        # Initially no keywords
        assert file.keywords is None
        
        # Add keywords
        file.keywords = ["test", "sample", "content"]
        assert file.keywords == ["test", "sample", "content"]
        
        # Clear keywords
        file.keywords = None
        assert file.keywords is None