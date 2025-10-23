"""
OpenAI Embedding Wrapper for BiG-RAG
Provides OpenAI text-embedding-3-large with FlagEmbedding-compatible interface
"""
import os
from typing import List, Union
import numpy as np
from openai import OpenAI

class OpenAIEmbedding:
    """
    Wrapper to make OpenAI embeddings compatible with FlagEmbedding interface
    """
    def __init__(
        self,
        model_name: str = "text-embedding-3-large",
        api_key: str = None,
        dimensions: int = 3072,  # text-embedding-3-large default
    ):
        self.model_name = model_name
        self.dimensions = dimensions

        # Load API key
        if api_key is None:
            if os.path.exists('openai_api_key.txt'):
                with open('openai_api_key.txt', 'r') as f:
                    api_key = f.read().strip()
            else:
                api_key = os.environ.get('OPENAI_API_KEY')

        if not api_key:
            raise ValueError("OpenAI API key not found. Create openai_api_key.txt or set OPENAI_API_KEY env var")

        self.client = OpenAI(api_key=api_key)
        print(f"[OpenAI Embedding] Initialized with model: {model_name}")

    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode texts to embeddings
        Compatible with FlagEmbedding.encode() interface
        """
        if isinstance(texts, str):
            texts = [texts]

        # Call OpenAI API
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            dimensions=self.dimensions
        )

        # Extract embeddings
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype=np.float32)

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        """
        Encode queries - same as encode() for OpenAI
        Compatible with FlagAutoModel.encode_queries() interface
        """
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Encode corpus in batches
        Compatible with FlagAutoModel.encode_corpus() interface
        """
        all_embeddings = []

        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i+batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)

            if (i // batch_size + 1) % 10 == 0:
                print(f"[OpenAI Embedding] Encoded {i+len(batch)}/{len(corpus)} texts")

        return np.vstack(all_embeddings)

# Alias for compatibility
class FlagAutoModel:
    @staticmethod
    def from_finetuned(model_name: str, **kwargs):
        """
        Compatibility layer with FlagEmbedding
        """
        if 'openai' in model_name or 'text-embedding' in model_name:
            return OpenAIEmbedding(model_name=model_name)
        else:
            # Fallback to FlagEmbedding if user wants to use BGE
            from FlagEmbedding import FlagAutoModel as OriginalFlagAutoModel
            return OriginalFlagAutoModel.from_finetuned(model_name, **kwargs)