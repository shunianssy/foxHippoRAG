import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

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
    install_requires=[
        "torch==2.5.1",
        "transformers==4.45.2",
        "vllm==0.6.6.post1",
        "openai>=1.91.0",
        "litellm>=1.73.0",
        "gritlm==1.0.2",
        "networkx==3.4.2",
        "python_igraph==0.11.8",
        "tiktoken==0.7.0",
        "pydantic==2.10.4",
        "tenacity==8.5.0",
        "einops", # No version specified
        "tqdm", # No version specified
        "boto3", # No version specified
    ]
)