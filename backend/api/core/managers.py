"""
LLM and Embedding Manager Classes

Manages multiple LLM providers and embedding strategies for BiG-RAG API.
"""

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from bigrag.utils import logger


class LLMProviderManager:
    """Manages multiple LLM providers with automatic fallback"""

    def __init__(self, default_provider: str = "openai"):
        self.default_provider = default_provider
        self.available_providers = {}
        self._initialize_providers()

    def _load_api_key(self, key_name: str, file_name: str = None) -> Optional[str]:
        """
        Load API key from configuration.
        Config automatically handles: .env file → *_api_key.txt → environment variable

        Args:
            key_name: Environment variable name (e.g., "OPENAI_API_KEY")
            file_name: Deprecated - kept for backward compatibility

        Returns:
            API key if found, None otherwise
        """
        from bigrag.config import config

        # Map environment variable names to config attributes
        key_map = {
            "OPENAI_API_KEY": config.openai_api_key,
            "ANTHROPIC_API_KEY": config.anthropic_api_key,
            "GOOGLE_API_KEY": config.google_api_key,
            "XAI_API_KEY": config.xai_api_key,
        }

        return key_map.get(key_name)

    def _initialize_providers(self):
        """Initialize all available LLM providers"""
        # OpenAI
        openai_key = self._load_api_key("OPENAI_API_KEY", "openai_api_key.txt")
        if openai_key:
            try:
                from bigrag.llm import gpt_4o_mini_complete, gpt_4o_complete
                self.available_providers["openai"] = {
                    "gpt-4o-mini": gpt_4o_mini_complete,
                    "gpt-4o": gpt_4o_complete,
                    "gpt-4": gpt_4o_complete,  # Alias
                }
                logger.info("OpenAI provider initialized")
            except Exception as e:
                logger.warning(f"OpenAI provider failed: {e}")

        # Anthropic (Claude)
        anthropic_key = self._load_api_key("ANTHROPIC_API_KEY", "anthropic_api_key.txt")
        if anthropic_key:
            try:
                self.available_providers["anthropic"] = self._get_anthropic_funcs()
                logger.info("Anthropic (Claude) provider initialized")
            except Exception as e:
                logger.warning(f"Anthropic provider failed: {e}")

        # Google (Gemini)
        google_key = self._load_api_key("GOOGLE_API_KEY", "google_api_key.txt")
        if google_key:
            try:
                self.available_providers["google"] = self._get_google_funcs()
                logger.info("Google (Gemini) provider initialized")
            except Exception as e:
                logger.warning(f"Google provider failed: {e}")

        # xAI (Grok)
        xai_key = self._load_api_key("XAI_API_KEY", "grok_api_key.txt")
        if xai_key:
            try:
                self.available_providers["grok"] = self._get_grok_funcs()
                logger.info("xAI (Grok) provider initialized")
            except Exception as e:
                logger.warning(f"xAI provider failed: {e}")

    def _get_anthropic_funcs(self):
        """Create Anthropic LLM functions"""
        async def claude_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
            try:
                from anthropic import AsyncAnthropic
                client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

                messages = []
                if history_messages:
                    messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})

                response = await client.messages.create(
                    model=kwargs.get("model", "claude-3-5-sonnet-20241022"),
                    max_tokens=kwargs.get("max_tokens", 1024),
                    temperature=kwargs.get("temperature", 1.0),
                    system=system_prompt or "",
                    messages=messages
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Claude API error: {e}")
                raise

        return {
            "claude-3-5-sonnet": claude_complete,
            "claude-3-opus": claude_complete,
            "claude-3-sonnet": claude_complete,
        }

    def _get_google_funcs(self):
        """Create Google Gemini functions"""
        async def gemini_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

                model = genai.GenerativeModel(
                    model_name=kwargs.get("model", "gemini-pro"),
                    system_instruction=system_prompt
                )

                # Format history
                chat_history = []
                for msg in history_messages:
                    role = "user" if msg["role"] == "user" else "model"
                    chat_history.append({"role": role, "parts": [msg["content"]]})

                chat = model.start_chat(history=chat_history)
                response = await chat.send_message_async(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=kwargs.get("temperature", 1.0),
                        max_output_tokens=kwargs.get("max_tokens", 1024),
                    )
                )
                return response.text
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                raise

        return {
            "gemini-pro": gemini_complete,
            "gemini-1.5-pro": gemini_complete,
        }

    def _get_grok_funcs(self):
        """Create xAI Grok functions"""
        async def grok_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(
                    api_key=os.getenv("XAI_API_KEY"),
                    base_url="https://api.x.ai/v1"
                )

                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.extend(history_messages)
                messages.append({"role": "user", "content": prompt})

                response = await client.chat.completions.create(
                    model="grok-beta",
                    messages=messages,
                    temperature=kwargs.get("temperature", 1.0),
                    max_tokens=kwargs.get("max_tokens", 1024),
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"Grok API error: {e}")
                raise

        return {
            "grok-beta": grok_complete,
            "grok": grok_complete,
        }

    async def complete(self, prompt: str, provider: Optional[str] = None,
                      model: Optional[str] = None, **kwargs) -> str:
        """
        Complete using specified provider or fallback to default

        Args:
            prompt: User prompt
            provider: LLM provider (openai, anthropic, google, grok)
            model: Specific model name
            **kwargs: Additional parameters (system_prompt, temperature, etc.)

        Returns:
            Generated text
        """
        # Use default provider if not specified
        provider = provider or self.default_provider

        # Fallback chain: requested → default → any available
        providers_to_try = [provider]
        if provider != self.default_provider and self.default_provider in self.available_providers:
            providers_to_try.append(self.default_provider)

        # Add any other available provider as last resort
        for p in self.available_providers.keys():
            if p not in providers_to_try:
                providers_to_try.append(p)

        last_error = None
        for prov in providers_to_try:
            if prov not in self.available_providers:
                continue

            try:
                provider_models = self.available_providers[prov]

                # Select model
                if model and model in provider_models:
                    func = provider_models[model]
                else:
                    # Use first available model for this provider
                    func = list(provider_models.values())[0]

                logger.info(f"Using provider: {prov}")
                return await func(prompt, **kwargs)

            except Exception as e:
                last_error = e
                logger.warning(f"Provider {prov} failed: {e}")
                continue

        # All providers failed
        raise Exception(f"All LLM providers failed. Last error: {last_error}")

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.available_providers.keys())

    def get_model_func(self, provider: str, model: str):
        """
        Get LLM function for specified provider and model

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet")

        Returns:
            LLM completion function

        Raises:
            ValueError: If provider or model not available
        """
        if provider not in self.available_providers:
            available = ", ".join(self.available_providers.keys())
            raise ValueError(
                f"Provider '{provider}' not available. Available providers: {available}"
            )

        if model not in self.available_providers[provider]:
            available_models = ", ".join(self.available_providers[provider].keys())
            raise ValueError(
                f"Model '{model}' not available for provider '{provider}'. "
                f"Available models: {available_models}"
            )

        return self.available_providers[provider][model]


class EmbeddingManager:
    """Auto-detects and manages embedding strategy"""

    def __init__(self, working_dir: str):
        self.working_dir = Path(working_dir)
        self.mode = None
        self.model = None
        self.faiss_indices = {}
        self._detect_mode()

    def _detect_mode(self):
        """Detect whether to use OpenAI or FlagEmbedding"""
        # Check for OpenAI-style files (NanoVectorDB)
        if (self.working_dir / "vdb_entities.json").exists():
            self.mode = "openai"
            logger.info("Detected OpenAI embeddings (NanoVectorDB format)")
            self._init_openai()

        # Check for FlagEmbedding-style files (FAISS)
        elif (self.working_dir / "index_entity.bin").exists():
            self.mode = "flagembedding"
            logger.info("Detected FlagEmbedding (FAISS format)")
            self._init_flagembedding()

        else:
            # No existing embeddings - default to OpenAI for new documents
            logger.warning("No embedding files detected! Defaulting to OpenAI embeddings.")
            self.mode = "openai"
            self._init_openai()

    def _init_openai(self):
        """Initialize OpenAI embedding mode"""
        try:
            from bigrag.llm import openai_embedding
            self.embedding_func = openai_embedding
            logger.info("OpenAI embedding function loaded")
        except Exception as e:
            logger.error(f"Failed to load OpenAI embeddings: {e}")
            raise

    def _init_flagembedding(self):
        """
        Initialize FlagEmbedding mode (LEGACY - for backwards compatibility)

        NOTE: This is legacy code for graphs built with FAISS indices.
        New graphs use NanoVectorDB (OpenAI mode). If you're seeing errors here,
        your graph may be from an older version. Consider rebuilding with script_build.py.
        """
        try:
            import faiss
            from FlagEmbedding import FlagAutoModel

            # Load FlagEmbedding model
            self.model = FlagAutoModel.from_finetuned(
                'BAAI/bge-large-en-v1.5',
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                devices="cpu",
            )
            logger.info("FlagEmbedding model loaded")

            # Load FAISS indices
            self.faiss_indices["entity"] = faiss.read_index(str(self.working_dir / "index_entity.bin"))
            self.faiss_indices["relation"] = faiss.read_index(str(self.working_dir / "index_relation.bin"))
            logger.info("FAISS indices loaded")

            # Load corpus mappings from GraphML (new architecture) or JSON (legacy)
            graph_file = self.working_dir / "graph_chunk_entity_relation.graphml"
            legacy_entities_file = self.working_dir / "kv_store_entities.json"
            legacy_edges_file = self.working_dir / "kv_store_relations.json"

            if graph_file.exists():
                # New architecture: Read from GraphML
                logger.info("Reading entity/relation names from GraphML file")
                import networkx as nx
                G = nx.read_graphml(graph_file)

                self.corpus_entity = []
                self.corpus_relation = []

                for node, attrs in G.nodes(data=True):
                    role = attrs.get("role", "")
                    if role == "entity":
                        self.corpus_entity.append(attrs.get("name", node))
                    elif role == "relation":
                        self.corpus_relation.append(attrs.get("name", node))

            elif legacy_entities_file.exists() and legacy_edges_file.exists():
                # Legacy architecture: Read from JSON files
                logger.warning("Using legacy JSON metadata files (consider rebuilding graph)")
                with open(legacy_entities_file) as f:
                    entities = json.load(f)
                    self.corpus_entity = [entities[item]['entity_name'] for item in entities]

                with open(legacy_edges_file) as f:
                    edges = json.load(f)
                    self.corpus_relation = [edges[item]['content'] for item in edges]
            else:
                raise FileNotFoundError(
                    "No entity/relation metadata found! Expected either:\n"
                    f"  - {graph_file} (new architecture)\n"
                    f"  - {legacy_entities_file} + {legacy_edges_file} (legacy)\n"
                    "Please rebuild your graph with script_build.py"
                )

            logger.info(f"Loaded {len(self.corpus_entity)} entities, {len(self.corpus_relation)} relations")

        except ImportError as e:
            logger.error(f"FlagEmbedding dependencies not installed: {e}")
            logger.error("Install with: pip install FlagEmbedding faiss-cpu")
            raise
        except FileNotFoundError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            logger.error(f"Failed to initialize FlagEmbedding: {e}")
            raise

    def get_embedding_func(self):
        """Get embedding function for BiGRAG"""
        if self.mode == "openai":
            return self.embedding_func
        elif self.mode == "flagembedding":
            # For FlagEmbedding, we handle embeddings manually in query
            return None
        else:
            return None

    async def search_entities(self, query: str, top_k: int = 5):
        """Search for entities matching query"""
        if self.mode == "flagembedding":
            embeddings = self.model.encode_queries([query])
            _, ids = self.faiss_indices["entity"].search(embeddings, top_k)
            return [self.corpus_entity[i] for i in ids[0]]
        return None

    async def search_relations(self, query: str, top_k: int = 5):
        """Search for relations matching query"""
        if self.mode == "flagembedding":
            embeddings = self.model.encode_queries([query])
            _, ids = self.faiss_indices["relation"].search(embeddings, top_k)
            return [self.corpus_relation[i] for i in ids[0]]
        return None
