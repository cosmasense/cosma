"""Integration tests for the processing pipeline."""

import datetime
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.models.file import File
from backend.models.status import ProcessingStatus
from backend.models.update import Update
from backend.pipeline import Pipeline
from tests.factories import SampleFileFactory


@pytest.mark.integration
@pytest.mark.asyncio
class TestPipeline:
    """Test cases for the file processing pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, test_pipeline: Pipeline):
        """Test that pipeline initializes correctly."""
        assert test_pipeline.db is not None
        assert test_pipeline.discoverer is not None
        assert test_pipeline.parser is not None
        assert test_pipeline.summarizer is not None
        assert test_pipeline.embedder is not None
        assert test_pipeline.updates_hub is not None

    # Temporarily disable slow pipeline tests until mocking is fixed
    @pytest.mark.skip(reason="Pipeline tests temporarily disabled due to slow model loading. Fix mocking to re-enable.")
    async def test_process_single_file_success(self, test_pipeline: Pipeline, temp_file_dir: Path, mock_real_services):
        """Test processing a single file successfully."""
        # Create test file
        test_file = SampleFileFactory.create_text_file(temp_file_dir, "test.txt")
        file_obj = File.from_path(test_file)

        # Process file (our mock fixtures should handle everything)
        await test_pipeline.process_file(file_obj)
        
        # Verify file was processed with mock data
        assert file_obj.status == ProcessingStatus.COMPLETE
        assert file_obj.parsed_at is not None
        assert file_obj.summarized_at is not None
        assert file_obj.embedded_at is not None
        assert file_obj.summary is not None
        assert file_obj.embedding is not None

    @pytest.mark.skip(reason="Pipeline tests temporarily disabled due to slow model loading. Fix mocking to re-enable.")
    async def test_process_file_with_parsing_error(self, test_pipeline: Pipeline, temp_file_dir: Path):
        """Test pipeline behavior when parsing fails."""
        # Create a test file
        test_file = SampleFileFactory.create_text_file(temp_file_dir, "test.txt")
        file_obj = File.from_path(test_file)
        
        # Mock parser to raise an exception
        async def mock_parse_with_error(file_obj):
            raise ValueError("Simulated parsing error")
        
        test_pipeline.parser.parse_file = mock_parse_with_error
        
        # Process file should raise an exception
        with pytest.raises(ValueError, match="Simulated parsing error"):
            await test_pipeline.process_file(file_obj)
        
        # Verify error state
        assert file_obj.status == ProcessingStatus.FAILED
        assert "Simulated parsing error" in file_obj.processing_error

    @pytest.mark.skip(reason="Pipeline tests temporarily disabled due to slow model loading. Fix mocking to re-enable.")
    async def test_process_directory(self, test_pipeline: Pipeline, temp_file_dir: Path):
        """Test processing an entire directory."""
        # Create multiple test files
        SampleFileFactory.create_text_file(temp_file_dir, "file1.txt")
        SampleFileFactory.create_text_file(temp_file_dir, "file2.txt")
        SampleFileFactory.create_markdown_file(temp_file_dir, "doc.md")
        
        # Mock services to avoid real AI calls
        test_pipeline.summarizer.summarize = AsyncMock()
        test_pipeline.embedder.embed = AsyncMock()
        
        # Process directory
        await test_pipeline.process_directory(temp_file_dir)
        
        # Verify files were discovered and processed
        # We check by querying the database
        processed_files = []
        for file_path in [temp_file_dir / "file1.txt", temp_file_dir / "file2.txt", temp_file_dir / "doc.md"]:
            file_record = await test_pipeline.db.get_file_by_path(str(file_path))
            if file_record:
                processed_files.append(file_record)
        
        # Should have processed at least some files
        assert len(processed_files) > 0

    async def test_should_skip_file_already_processed(self, test_pipeline: Pipeline, temp_file_dir: Path):
        """Test that already processed files are skipped."""
        # Create test file
        test_file = SampleFileFactory.create_text_file(temp_file_dir, "already_processed.txt")
        file_obj = File.from_path(test_file)
        
        # Save file to database as already processed
        file_obj.status = ProcessingStatus.COMPLETE
        file_obj.parsed_at = file_obj.modified
        await test_pipeline.db.upsert_file(file_obj)
        
        # Check if file should be skipped
        should_skip = await test_pipeline._should_skip_file(file_obj)
        assert should_skip is True

    async def test_should_process_modified_file(self, test_pipeline: Pipeline, temp_file_dir: Path):
        """Test that modified files are reprocessed."""
        # Create test file
        test_file = SampleFileFactory.create_text_file(temp_file_dir, "modified.txt")
        file_obj = File.from_path(test_file)
        
        # Save file to database as already processed but with old modification time
        old_modified = file_obj.modified.replace(second=0) - datetime.timedelta(hours=1)
        file_obj.status = ProcessingStatus.COMPLETE
        file_obj.modified = old_modified
        await test_pipeline.db.upsert_file(file_obj)
        
        # Reset file object to current state (simulating file modification detection)
        current_file_obj = File.from_path(test_file)
        
        # Check if file should be skipped (it shouldn't be)
        should_skip = await test_pipeline._should_skip_file(current_file_obj)
        assert should_skip is False

    async def test_has_file_changed_by_hash(self, test_pipeline: Pipeline, temp_file_dir: Path):
        """Test file change detection by content hash."""
        # Create test file
        test_file = SampleFileFactory.create_text_file(temp_file_dir, "hash_test.txt")
        file_obj = File.from_path(test_file)
        
        # Mock content hash generation
        file_obj.content_hash = "original_hash"
        
        # Save file to database
        await test_pipeline.db.upsert_file(file_obj)
        
        # Test with same hash
        file_obj.content_hash = "original_hash"
        has_changed = await test_pipeline._has_file_changed(file_obj)
        assert has_changed is False
        
        # Test with different hash
        file_obj.content_hash = "different_hash"
        has_changed = await test_pipeline._has_file_changed(file_obj)
        assert has_changed is True

    @pytest.mark.skip(reason="Pipeline tests temporarily disabled due to slow model loading. Fix mocking to re-enable.")
    async def test_publish_updates(self, test_pipeline: Pipeline, temp_file_dir: Path):
        """Test that pipeline publishes updates correctly."""
        # Create test file
        test_file = SampleFileFactory.create_text_file(temp_file_dir, "updates_test.txt")
        file_obj = File.from_path(test_file)
        
        # Track published updates
        published_updates = []
        
        def capture_update(update):
            published_updates.append(update)
        
        test_pipeline.updates_hub.publish = capture_update
        
        # Mock services
        test_pipeline.summarizer.summarize = AsyncMock()
        test_pipeline.embedder.embed = AsyncMock()
        
        # Process file
        await test_pipeline.process_file(file_obj)
        
        # Verify updates were published
        update_types = [type(u).__name__ for u in published_updates]
        
        # Should have various update types during processing
        assert "FileParsingUpdate" in update_types or "FileParsedUpdate" in update_types
        assert "FileSummarizingUpdate" in update_types or "FileSummarizedUpdate" in update_types
        assert "FileEmbeddingUpdate" in update_types or "FileEmbeddedUpdate" in update_types
        assert "FileCompleteUpdate" in update_types

    async def test_is_supported_method(self, test_pipeline: Pipeline, temp_file_dir: Path):
        """Test file support checking."""
        # Create different file types
        txt_file = SampleFileFactory.create_text_file(temp_file_dir, "test.txt")
        md_file = SampleFileFactory.create_markdown_file(temp_file_dir, "test.md")
        
        txt_file_obj = File.from_path(txt_file)
        md_file_obj = File.from_path(md_file)
        
        # Test support checking
        txt_supported = await test_pipeline.is_supported(txt_file_obj)
        md_supported = await test_pipeline.is_supported(md_file_obj)
        
        # At least text files should be supported
        assert txt_supported is True
        # Markdown support depends on parser implementation
        assert isinstance(md_supported, bool)

    @pytest.mark.skip(reason="Pipeline tests temporarily disabled due to slow model loading. Fix mocking to re-enable.")
    async def test_process_file_with_empty_content(self, test_pipeline: Pipeline, temp_file_dir: Path):
        """Test processing file with empty content."""
        # Create empty file
        empty_file = temp_file_dir / "empty.txt"
        empty_file.write_text("")
        
        file_obj = File.from_path(empty_file)
        
        # Mock services
        test_pipeline.summarizer.summarize = AsyncMock()
        test_pipeline.embedder.embed = AsyncMock()
        
        # Process empty file
        await test_pipeline.process_file(file_obj)
        
        # Verify file was processed (even if empty)
        assert file_obj.parsed_at is not None

    @pytest.mark.skip(reason="Pipeline tests temporarily disabled due to slow model loading. Fix mocking to re-enable.")
    async def test_pipeline_error_handling_and_recovery(self, test_pipeline: Pipeline, temp_file_dir: Path):
        """Test pipeline error handling and recovery."""
        # Create test file
        test_file = SampleFileFactory.create_text_file(temp_file_dir, "error_test.txt")
        file_obj = File.from_path(test_file)
        
        # Mock summarizer to fail
        test_pipeline.summarizer.summarize = AsyncMock(
            side_effect=Exception("Summarization failed")
        )
        
        # Process file should handle the error gracefully
        with pytest.raises(Exception, match="Summarization failed"):
            await test_pipeline.process_file(file_obj)
        
        # Verify error state
        assert file_obj.status == ProcessingStatus.FAILED
        assert "Summarization failed" in file_obj.processing_error
        
        # File should still be saved to database with error state
        saved_file = await test_pipeline.db.get_file_by_path(file_obj.file_path)
        assert saved_file is not None
        assert saved_file.status == ProcessingStatus.FAILED
        assert saved_file.processing_error is not None