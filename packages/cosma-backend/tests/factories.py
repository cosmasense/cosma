"""Factories for creating test data."""

import datetime
from pathlib import Path
from typing import Optional
import factory
import factory.fuzzy

from cosma_backend.models.file import File
from cosma_backend.models.status import ProcessingStatus


class FileFactory(factory.Factory):
    """Factory for creating File objects for testing."""
    
    class Meta:
        model = File
    
    # Basic file metadata
    path = factory.Faker("file_path")
    file_path = factory.LazyAttribute(lambda obj: str(obj.path))
    filename = factory.LazyAttribute(lambda obj: obj.path.name)
    extension = factory.LazyAttribute(lambda obj: obj.path.suffix)
    file_size = factory.fuzzy.FuzzyInteger(100, 10000)
    
    # Timestamps
    created = factory.Faker("date_time_this_year", tzinfo=datetime.timezone.utc)
    modified = factory.Faker("date_time_this_month", tzinfo=datetime.timezone.utc)
    accessed = factory.Faker("date_time_this_week", tzinfo=datetime.timezone.utc)
    
    # Content metadata
    content_type = factory.Iterator([
        "text/plain",
        "text/markdown",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ])
    content = factory.Faker("paragraph", nb_sentences=3)
    content_hash = factory.Faker("sha256")
    
    # Processing metadata
    summary = factory.Faker("sentence", nb_words=10)
    title = factory.Faker("sentence", nb_words=5)
    keywords = factory.List([factory.Faker("word") for _ in range(3)])
    
    # Status
    status = factory.Iterator(list(ProcessingStatus))
    processing_error = factory.Maybe(
        factory.Iterator([ProcessingStatus.FAILED]),
        factory.Faker("sentence")
    )
    
    class Params:
        """Parameters for creating specific types of files."""
        
        complete = factory.Trait(
            status=ProcessingStatus.COMPLETE,
            parsed_at=factory.Faker("date_time_this_day", tzinfo=datetime.timezone.utc),
            summarized_at=factory.Faker("date_time_this_day", tzinfo=datetime.timezone.utc),
            embedded_at=factory.Faker("date_time_this_day", tzinfo=datetime.timezone.utc),
        )
        
        failed = factory.Trait(
            status=ProcessingStatus.FAILED,
            processing_error=factory.Faker("sentence"),
        )
        
        discovered = factory.Trait(
            status=ProcessingStatus.DISCOVERED,
            content=None,
            summary=None,
            title=None,
            keywords=None,
        )
        
        text_file = factory.Trait(
            extension=".txt",
            content_type="text/plain",
            content=factory.Faker("text", max_nb_chars=500),
        )
        
        markdown_file = factory.Trait(
            extension=".md",
            content_type="text/markdown",
            content=factory.LazyAttribute(
                lambda obj: f"# {obj.title}\n\n{obj.content}"
            ),
        )
        
        pdf_file = factory.Trait(
            extension=".pdf",
            content_type="application/pdf",
            file_size=factory.fuzzy.FuzzyInteger(50000, 5000000),  # 50KB - 5MB
        )


class SampleFileFactory:
    """Factory for creating actual files on disk for integration tests."""
    
    @staticmethod
    def create_text_file(directory: Path, filename: str = None, content: str = None) -> Path:
        """Create a text file with sample content."""
        if filename is None:
            fake = factory.Faker('word')
            filename = f"sample_{fake.generate({})}.txt" if hasattr(fake, 'generate') else f"sample_{fake}.txt"
        
        if content is None:
            fake = factory.Faker("paragraph", nb_sentences=5)
            content = fake.generate({}) if hasattr(fake, 'generate') else str(fake)
        
        file_path = directory / filename
        file_path.write_text(content, encoding='utf-8')
        return file_path
    
    @staticmethod
    def create_markdown_file(directory: Path, filename: str = None, content: str = None) -> Path:
        """Create a markdown file with sample content."""
        if filename is None:
            fake = factory.Faker('word')
            filename = f"doc_{fake.generate({}) if hasattr(fake, 'generate') else fake}.md"
        
        if content is None:
            title_fake = factory.Faker("sentence", nb_words=4)
            body_fake = factory.Faker("paragraph", nb_sentences=3)
            
            title = title_fake.generate({}) if hasattr(title_fake, 'generate') else str(title_fake)
            body = body_fake.generate({}) if hasattr(body_fake, 'generate') else str(body_fake)
            content = f"# {title}\n\n{body}"
        
        file_path = directory / filename
        file_path.write_text(content, encoding='utf-8')
        return file_path
    
    @staticmethod
    def create_nested_files(directory: Path, depth: int = 2, files_per_dir: int = 3) -> list[Path]:
        """Create a nested directory structure with files."""
        created_files = []
        
        def create_level(current_dir: Path, current_depth: int):
            if current_depth <= 0:
                return
            
            # Create files in current directory
            for i in range(files_per_dir):
                if i % 2 == 0:
                    file_path = SampleFileFactory.create_text_file(
                        current_dir, f"file_{i}.txt"
                    )
                else:
                    file_path = SampleFileFactory.create_markdown_file(
                        current_dir, f"doc_{i}.md"
                    )
                created_files.append(file_path)
            
            # Create subdirectories
            if current_depth > 1:
                for i in range(2):
                    sub_dir = current_dir / f"subdir_{i}"
                    sub_dir.mkdir(exist_ok=True)
                    create_level(sub_dir, current_depth - 1)
        
        create_level(directory, depth)
        return created_files