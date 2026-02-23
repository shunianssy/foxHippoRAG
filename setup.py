import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "torch>=2.0.0",
    "transformers>=4.40.0",
    "openai>=1.91.0",
    "litellm>=1.73.0",
    "networkx>=3.0",
    "python_igraph>=0.11.0",
    "tiktoken>=0.7.0",
    "pydantic>=2.0.0",
    "tenacity>=8.0.0",
    "einops",
    "tqdm",
    "boto3",
]

extras_require = {
    "gritlm": ["gritlm>=1.0.0"],
    "vllm": ["vllm>=0.6.0; sys_platform != 'win32'"],
    "dev": ["pytest", "ruff", "flake8"],
    "all": [
        "gritlm>=1.0.0",
        "vllm>=0.6.0; sys_platform != 'win32'",
    ],
}

setuptools.setup(
    name="foxHippoRAG",
    version="1.0.0",
    author="shunianssy",
    author_email="shunianssy@example.com",
    description="A powerful graph-based RAG framework (foxHippoRAG distribution) that enables LLMs to identify and leverage connections within new knowledge for improved retrieval.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shunianssy/foxHippoRAG",
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
)