# Contributing to foxHippoRAG

Thank you for your interest in contributing to foxHippoRAG!

## How to Contribute

### 1. Setting Up Your Development Environment

```bash
# Clone the repository
git clone https://github.com/shunianssy/foxHippoRAG.git
cd foxHippoRAG

# Create and activate virtual environment
conda create -n foxhipporag-dev python=3.10
conda activate foxhipporag-dev

# Install development dependencies
pip install -e .
pip install -r requirements.txt
```

### 2. Making Changes

- **Code Style**: Follow the existing code style and conventions.
- **Testing**: Add tests for any new functionality.
- **Documentation**: Update documentation for any changes.

### 3. Running Tests

```bash
# Run OpenAI tests (requires API key)
export OPENAI_API_KEY=<your-api-key>
python tests_openai.py

# Run local tests (requires vLLM server)
python tests_local.py

# Run Azure tests (requires Azure endpoints)
python tests_azure.py
```

### 4. Submitting a Pull Request

1. Create a new branch for your changes
2. Commit your changes with clear commit messages
3. Push your branch to GitHub
4. Create a pull request with a description of your changes

## Code of Conduct

Please be respectful and constructive in your contributions. All contributors are expected to follow our code of conduct.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on GitHub.

## License

By contributing to foxHippoRAG, you agree that your contributions will be licensed under the MIT License.

We appreciate your contributions to foxHippoRAG!
