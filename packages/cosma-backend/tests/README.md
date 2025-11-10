# Backend Test Suite

This directory contains the test suite for the backend application.

## Test Structure

```
tests/
├── __init__.py          # Test package initialization
├── conftest.py          # Common pytest fixtures and configuration
├── factories.py         # Factory Boy factories for test data
├── unit/                # Unit tests (fast, isolated)
│   └── test_models.py   # Model unit tests
├── integration/         # Integration tests (database, file system)
│   ├── test_database.py # Database integration tests
│   ├── test_pipeline.py # Pipeline integration tests
│   └── test_api.py      # API integration tests
└── README.md           # This file
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=backend --cov-report=html
```

### Run only unit tests
```bash
pytest tests/unit/ -m unit
```

### Run only integration tests
```bash
pytest tests/integration/ -m integration
```

### Run specific test file
```bash
pytest tests/unit/test_models.py
```

### Run with verbose output
```bash
pytest -v
```

## Test Categories

### Unit Tests (`-m unit`)
- Fast tests that don't require external dependencies
- Test individual functions and classes in isolation
- Use mocks to avoid database, file system, or network calls

### Integration Tests (`-m integration`)
- Slower tests that test component interactions
- Use real database connections and file system operations
- Test complete workflows and API endpoints

### Slow Tests (`-m slow`)
- Tests that make network calls or use AI services
- Should be run sparingly during development
- Not typically run in CI/CD pipelines

## Key Fixtures

- `temp_db`: In-memory database for each test
- `temp_file_dir`: Temporary directory for test files
- `test_pipeline`: Pipeline instance with mocked services
- `sample_file_data`: Sample file data for creating test objects
- `test_client`: Quart test client for API testing

## Writing New Tests

### Unit Test Example
```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_some_function():
    # Arrange
    # Act
    # Assert
    pass
```

### Integration Test Example
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_some_workflow(temp_db, temp_file_dir):
    # Use real database and file system
    pass
```

## Test Data

Use the factories in `factories.py` to create test data:

```python
from tests.factories import FileFactory, SampleFileFactory

# Create File object
file = FileFactory()

# Create actual file on disk
file_path = SampleFileFactory.create_text_file(temp_dir)
```

## Best Practices

1. **Keep tests independent**: Each test should set up its own data
2. **Use descriptive names**: Test names should clearly indicate what they test
3. **Arrange-Act-Assert**: Structure tests in this pattern
4. **Mock external services**: Avoid real API calls in tests
5. **Clean up resources**: Use fixtures and context managers for cleanup
6. **Test edge cases**: Test error conditions and boundary cases
7. **Use type hints**: Help catch bugs during development

## Coverage

The test suite aims for 80% code coverage. Coverage reports are generated in:
- Terminal output (basic report)
- `htmlcov/` directory (detailed HTML report)
- `coverage.xml` (CI/CD integration)

## Debugging Tests

### Run with pdb
```bash
pytest --pdb
```

### Stop at first failure
```bash
pytest -x
```

### Show local variables on failure
```bash
pytest -l
```

### Run with logging
```bash
pytest -s --log-cli-level=DEBUG
```