"""
BiG-RAG - Bipartite Graph Retrieval-Augmented Generation
Setup configuration for testing and development
"""

from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements(filename):
    """Read requirements from file, excluding comments and empty lines"""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    requirements = []
    for line in lines:
        line = line.strip()
        # Skip comments, empty lines, and section headers
        if line and not line.startswith('#') and not line.startswith('INSTALLATION'):
            # Skip inline comments
            if '#' in line:
                line = line.split('#')[0].strip()
            if line:
                requirements.append(line)

    return requirements

# Core dependencies for BiGRAG
install_requires = [
    'numpy>=1.26.0',
    'pandas>=2.0.0',
    'pyarrow>=15.0.0',
    'datasets',
    'networkx>=3.0',
    'pyvis',
    'faiss-cpu',
    'nano-vectordb',
    'openai>=1.0.0',
    'tiktoken',
    'transformers>=4.30.0',
    'nltk',
    'spacy',
    'PyPDF2',
    'jsonlines',
    'xxhash',
    'aiohttp',
    'tqdm',
    'tenacity',
    'fastapi',
    'uvicorn[standard]',
    'pydantic>=2.0.0',
    'python-multipart',
    'python-dotenv',
    'requests',
    'colorama',
]

# Test dependencies
extras_require = {
    'test': [
        'pytest>=8.0.0',
        'pytest-asyncio>=0.23.0',
        'pytest-cov>=4.1.0',
        'pytest-xdist>=3.5.0',
        'pytest-timeout>=2.2.0',
        'pytest-html>=4.1.0',
        'pytest-mock>=3.12.0',
        'pytest-benchmark>=4.0.0',
        'faker>=22.0.0',
        'httpx>=0.26.0',
        'coverage>=7.4.0',
        'memory-profiler>=0.61.0',
    ],
    'reranking': [
        'sentence-transformers>=2.0.0',
    ],
    'dev': [
        'ipython',
        'jupyter',
        'black',
        'flake8',
        'mypy',
    ],
}

# All extras combined
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name='bigrag',
    version='1.0.0',
    description='Bipartite Graph Retrieval-Augmented Generation',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='BiG-RAG Team',
    author_email='',
    url='https://github.com/yourusername/BiG-RAG',
    packages=find_packages(include=['bigrag', 'bigrag.*']),
    python_requires='>=3.11',
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='knowledge-graph rag retrieval bipartite-graph nlp',
    project_urls={
        'Documentation': 'https://github.com/yourusername/BiG-RAG/docs',
        'Source': 'https://github.com/yourusername/BiG-RAG',
        'Bug Reports': 'https://github.com/yourusername/BiG-RAG/issues',
    },
)
