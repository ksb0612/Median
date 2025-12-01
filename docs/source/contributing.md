# Contributing to Ridge MMM

Thank you for your interest in contributing to Ridge MMM!

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/ridge-mmm-app.git
   cd ridge-mmm-app
   ```
3. **Install dependencies**:
   ```bash
   poetry install
   poetry install --with dev,docs
   ```

## Development Workflow

### Creating a Branch

```bash
git checkout -b feature/your-feature-name
```

### Making Changes

1. Write your code
2. Add tests for new functionality
3. Update documentation as needed
4. Ensure all tests pass

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **Type hints** for better code clarity

Format your code:
```bash
poetry run black src/ streamlit_app/
poetry run flake8 src/ streamlit_app/
```

### Testing

Run tests:
```bash
poetry run pytest
```

Run tests with coverage:
```bash
poetry run pytest --cov=src tests/
```

### Documentation

Update documentation when adding features:

1. **Docstrings**: Use Google-style docstrings
   ```python
   def my_function(param1: int, param2: str) -> bool:
       """Brief description.

       Detailed description here.

       Args:
           param1: Description of param1
           param2: Description of param2

       Returns:
           Description of return value

       Example:
           >>> my_function(42, "test")
           True
       """
       pass
   ```

2. **API docs**: Sphinx auto-generates from docstrings
3. **User docs**: Update USER_GUIDE.md if needed

Build docs to verify:
```bash
cd docs
make html
```

## Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:
```
Add Hill saturation transformer

- Implement Hill function for diminishing returns
- Add tests for edge cases
- Update documentation with examples
```

### Pull Requests

1. Push your branch to your fork
2. Open a Pull Request on GitHub
3. Describe your changes clearly
4. Reference any related issues

PR template:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Code Review

- Be open to feedback
- Respond to review comments
- Make requested changes
- Keep discussion professional and constructive

## Areas for Contribution

### High Priority

- [ ] Additional transformation functions (custom adstock, saturation)
- [ ] More optimization algorithms
- [ ] Time-varying coefficients
- [ ] Cross-validation utilities
- [ ] More comprehensive tests

### Medium Priority

- [ ] Additional visualizations
- [ ] Export to PowerPoint/PDF
- [ ] Integration with external data sources
- [ ] Improved error messages
- [ ] Performance optimizations

### Documentation

- [ ] More tutorials and examples
- [ ] Video walkthroughs
- [ ] Case studies
- [ ] Best practices guide
- [ ] Troubleshooting expansion

## Questions?

- Open an issue for discussion
- Tag maintainers for guidance
- Join our community discussions

Thank you for contributing! 🎉
