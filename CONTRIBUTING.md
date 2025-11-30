# Contributing to Primate Vocalization Detection Pipeline

We welcome contributions to improve the pipeline. This document provides guidelines for contributing to the project.

## Types of Contributions

### Bug Reports

If you encounter a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- System environment (Python version, OS, GPU availability)
- Relevant error messages or logs

### Feature Requests

For new features, please open an issue describing:
- The proposed functionality
- Use case and motivation
- Potential implementation approach
- Impact on existing functionality

### Code Contributions

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature-name`
3. Make your changes
4. Write or update tests if applicable
5. Update documentation to reflect changes
6. Submit a pull request

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Keep functions focused and modular

Example function structure:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Brief description of function purpose.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    
    Returns:
        Description of return value
    """
    # Implementation
    return result
```

### Documentation

- Use clear, concise academic English
- Avoid decorative symbols or excessive formatting
- Provide context and rationale for design decisions
- Include code examples where helpful
- Update relevant documentation files when changing functionality

### Comments

- Use comments to explain why, not what (code should be self-explanatory)
- Document non-obvious algorithmic choices
- Cite relevant papers or resources when implementing published methods

## Testing

While formal unit tests are not currently implemented, contributions should:
- Be tested on sample data before submission
- Not break existing functionality
- Include instructions for testing new features

Future contributors are encouraged to add unit tests.

## Adding New Species

When adding support for new species:

1. Document the species in configuration
2. Ensure consistent naming conventions
3. Update visualization color schemes if needed
4. Test with actual data before submitting

## Documentation Updates

When modifying code that affects usage:

1. Update relevant sections in README.md
2. Update docstrings in affected modules
3. Add examples to SETUP_TUTORIAL.md if workflow changes
4. Update FILE_MANIFEST.md if file structure changes

## Commit Messages

Use clear, descriptive commit messages:

- Start with a verb in present tense: "Add", "Fix", "Update", "Remove"
- Keep first line under 72 characters
- Provide additional context in message body if needed

Examples:
```
Add support for Cercopithecus nictitans detection

Implement hard negative mining script for domain gap reduction

Fix memory leak in batch preprocessing
```

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Update documentation as needed
3. Describe changes clearly in pull request description
4. Reference any related issues
5. Wait for review and address feedback

## Research Ethics

This project involves audio recordings that may contain sensitive ecological data:

- Do not commit actual audio files to the repository
- Respect data sharing agreements with collaborators
- Maintain appropriate citations for methods and data sources
- Acknowledge collaborators appropriately

## Questions

If you have questions about contributing, please open an issue with the "question" label.

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.
