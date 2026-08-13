# Contributing to LLMTivo

## Getting Started

### Prerequisites

- Python 3.12 or higher
- uv
- Git

### Development Setup

1. **Fork the Repository**

   ```bash
   # Fork via GitHub UI, then:
   git clone https://github.com/integrallis/llmtivo-python.git
   cd llmtivo
   ```

2. **Create a Virtual Environment**

   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   uv pip install -e ".[dev]"
   ```

3. **Set Up Pre-commit Hooks**

   ```bash
   pre-commit install
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Your Changes

Write tested, meaningful contributions to the project.

### 4. Ensure Quality Standards

Before submitting:

```bash
# Run all tests
uv run pytest

# Check coverage
uv run pytest --cov=llmtivo --cov-report=term-missing

# Type checking
uv run mypy --strict src/

# Linting
uv run ruff check src/

# Format code
uv run ruff format src/

# IMPORTANT: Check LLMTivo score (must maintain or improve)
uv run llmtivo analyze src/llmtivo --skip-llm
# Current score: 0.753 - Your changes should maintain or improve this
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints for all functions
- Maximum line length: 120 characters
- Use descriptive variable names
- Document all public APIs

### Documentation

- All public functions must have docstrings
- Use Google-style docstrings:

  ```python
  def calculate_metric(code: str, threshold: float = 0.5) -> float:
      """Calculate a quality metric from source code.

      Args:
          code: The source code to analyze.
          threshold: Quality threshold for normalization.

      Returns:
          A normalized metric value between 0.0 and 1.0.

      Raises:
          SyntaxError: If the code cannot be parsed.
      """
  ```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add new complexity metric
fix: correct normalization in pattern detection
docs: update API documentation
test: add integration tests for CLI
refactor: simplify metric calculation logic
perf: optimize AST traversal
chore: update dependencies
```

## Testing Requirements

### Test Coverage

- **Core modules**: 100% coverage required
- **New features**: Must include comprehensive tests
- **Bug fixes**: Must include regression tests

### Test Organization

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests
├── property/       # Property-based tests (Hypothesis)
└── benchmarks/     # Performance benchmarks
```

## Submitting Changes

### Pull Request Process

1. **Ensure all tests pass**
2. **Update documentation** if needed
3. **Update CHANGELOG.md** with your changes
4. **Create Pull Request** with clear description
5. **Address review feedback** promptly

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added
- [ ] Coverage maintained at 100%

## Checklist
- [ ] Tests written for changes
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] LLMTivo score maintained or improved (≥0.753)
```

## Reporting Issues

### Bug Reports

Include:

- Python version
- LLMTivo version
- Minimal reproducible example
- Expected vs actual behavior
- Error messages/stack traces

### Feature Requests

Include:

- Use case description
- Proposed API/interface
- Example usage
- Why existing features don't suffice

## Community

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive criticism
- Assume good intentions

### Getting Help

- Check existing issues first
- Use discussions for questions
- Join our Discord server (coming soon)
- Read the documentation thoroughly

## Advanced Topics

### Adding New Metrics

1. Research the metric thoroughly
2. Submit a proposal in Markdown format
3. Write comprehensive tests
4. Implement extraction logic
5. Add normalization function
6. Integrate with LLMTivo formula
7. Update documentation
8. Add benchmark tests

### LLM Provider Integration

1. Study existing provider implementations
2. Implement provider interface
3. Add configuration support
4. Create integration tests
5. Update CLI commands
6. Document usage and costs

## Questions?

Feel free to:

- Open an issue for discussion
- Reach out to maintainers
- Check FAQ in documentation

---

Remember: **Quality over Quantity**. Better to contribute one well-tested, well-documented feature than multiple
incomplete ones.

Thank you for contributing to LLMTivo.
