"""
Plugin Registry System for BiG-RAG Modular Indexing

Allows third-party developers to register custom strategies without modifying core code.

Features:
- Dynamic strategy registration for all 6 strategy types
- Validation of custom strategies against interfaces
- Plugin discovery and loading
- Thread-safe registry with locking

Usage:
    from bigrag.registry import StrategyRegistry
    from bigrag.interfaces import ChunkerInterface

    # Define custom strategy
    class MyCustomChunker(ChunkerInterface):
        async def chunk(self, text, metadata):
            # Custom implementation
            pass

    # Register it
    StrategyRegistry.register_chunker("my_chunker", MyCustomChunker)

    # Use it in config
    config = IndexingConfig(chunker="my_chunker", ...)
"""

from typing import Type, Dict, Any, Callable
import threading
from bigrag.interfaces import (
    ChunkerInterface,
    ExtractorInterface,
    ValidatorInterface,
    MergerInterface,
    HITLInterface,
    OrphanLinkerInterface
)


class StrategyRegistry:
    """
    Central registry for all strategy plugins.

    Thread-safe singleton for registering and retrieving custom strategies.
    """

    _instance = None
    _lock = threading.Lock()

    # Registry dictionaries (strategy_name → strategy_class)
    _chunkers: Dict[str, Type[ChunkerInterface]] = {}
    _extractors: Dict[str, Type[ExtractorInterface]] = {}
    _validators: Dict[str, Type[ValidatorInterface]] = {}
    _mergers: Dict[str, Type[MergerInterface]] = {}
    _hitl_strategies: Dict[str, Type[HITLInterface]] = {}
    _orphan_linkers: Dict[str, Type[OrphanLinkerInterface]] = {}

    def __new__(cls):
        """Singleton pattern - only one registry instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    # ========== CHUNKER REGISTRATION ==========

    @classmethod
    def register_chunker(
        cls,
        name: str,
        strategy_class: Type[ChunkerInterface],
        override: bool = False
    ) -> None:
        """
        Register a custom chunker strategy.

        Args:
            name: Unique name for this strategy (e.g., "my_custom_chunker")
            strategy_class: Class implementing ChunkerInterface
            override: Allow overriding built-in strategies (default: False)

        Raises:
            ValueError: If name already registered and override=False
            TypeError: If strategy_class doesn't implement ChunkerInterface

        Example:
            >>> class PDFChunker(ChunkerInterface):
            ...     async def chunk(self, text, metadata):
            ...         # Custom PDF chunking logic
            ...         pass
            >>> StrategyRegistry.register_chunker("pdf", PDFChunker)
        """
        cls._register_strategy(
            registry=cls._chunkers,
            interface=ChunkerInterface,
            name=name,
            strategy_class=strategy_class,
            override=override,
            strategy_type="chunker"
        )

    @classmethod
    def get_chunker(cls, name: str) -> Type[ChunkerInterface]:
        """Get registered chunker strategy by name."""
        return cls._get_strategy(cls._chunkers, name, "chunker")

    @classmethod
    def list_chunkers(cls) -> list:
        """List all registered chunker strategies."""
        return list(cls._chunkers.keys())

    # ========== EXTRACTOR REGISTRATION ==========

    @classmethod
    def register_extractor(
        cls,
        name: str,
        strategy_class: Type[ExtractorInterface],
        override: bool = False
    ) -> None:
        """Register a custom extractor strategy."""
        cls._register_strategy(
            registry=cls._extractors,
            interface=ExtractorInterface,
            name=name,
            strategy_class=strategy_class,
            override=override,
            strategy_type="extractor"
        )

    @classmethod
    def get_extractor(cls, name: str) -> Type[ExtractorInterface]:
        """Get registered extractor strategy by name."""
        return cls._get_strategy(cls._extractors, name, "extractor")

    @classmethod
    def list_extractors(cls) -> list:
        """List all registered extractor strategies."""
        return list(cls._extractors.keys())

    # ========== VALIDATOR REGISTRATION ==========

    @classmethod
    def register_validator(
        cls,
        name: str,
        strategy_class: Type[ValidatorInterface],
        override: bool = False
    ) -> None:
        """Register a custom validator strategy."""
        cls._register_strategy(
            registry=cls._validators,
            interface=ValidatorInterface,
            name=name,
            strategy_class=strategy_class,
            override=override,
            strategy_type="validator"
        )

    @classmethod
    def get_validator(cls, name: str) -> Type[ValidatorInterface]:
        """Get registered validator strategy by name."""
        return cls._get_strategy(cls._validators, name, "validator")

    @classmethod
    def list_validators(cls) -> list:
        """List all registered validator strategies."""
        return list(cls._validators.keys())

    # ========== MERGER REGISTRATION ==========

    @classmethod
    def register_merger(
        cls,
        name: str,
        strategy_class: Type[MergerInterface],
        override: bool = False
    ) -> None:
        """Register a custom merger strategy."""
        cls._register_strategy(
            registry=cls._mergers,
            interface=MergerInterface,
            name=name,
            strategy_class=strategy_class,
            override=override,
            strategy_type="merger"
        )

    @classmethod
    def get_merger(cls, name: str) -> Type[MergerInterface]:
        """Get registered merger strategy by name."""
        return cls._get_strategy(cls._mergers, name, "merger")

    @classmethod
    def list_mergers(cls) -> list:
        """List all registered merger strategies."""
        return list(cls._mergers.keys())

    # ========== HITL REGISTRATION ==========

    @classmethod
    def register_hitl(
        cls,
        name: str,
        strategy_class: Type[HITLInterface],
        override: bool = False
    ) -> None:
        """Register a custom HITL strategy."""
        cls._register_strategy(
            registry=cls._hitl_strategies,
            interface=HITLInterface,
            name=name,
            strategy_class=strategy_class,
            override=override,
            strategy_type="hitl"
        )

    @classmethod
    def get_hitl(cls, name: str) -> Type[HITLInterface]:
        """Get registered HITL strategy by name."""
        return cls._get_strategy(cls._hitl_strategies, name, "hitl")

    @classmethod
    def list_hitl(cls) -> list:
        """List all registered HITL strategies."""
        return list(cls._hitl_strategies.keys())

    # ========== ORPHAN LINKER REGISTRATION ==========

    @classmethod
    def register_orphan_linker(
        cls,
        name: str,
        strategy_class: Type[OrphanLinkerInterface],
        override: bool = False
    ) -> None:
        """Register a custom orphan linker strategy."""
        cls._register_strategy(
            registry=cls._orphan_linkers,
            interface=OrphanLinkerInterface,
            name=name,
            strategy_class=strategy_class,
            override=override,
            strategy_type="orphan_linker"
        )

    @classmethod
    def get_orphan_linker(cls, name: str) -> Type[OrphanLinkerInterface]:
        """Get registered orphan linker strategy by name."""
        return cls._get_strategy(cls._orphan_linkers, name, "orphan_linker")

    @classmethod
    def list_orphan_linkers(cls) -> list:
        """List all registered orphan linker strategies."""
        return list(cls._orphan_linkers.keys())

    # ========== HELPER METHODS ==========

    @classmethod
    def _register_strategy(
        cls,
        registry: Dict[str, Type],
        interface: Type,
        name: str,
        strategy_class: Type,
        override: bool,
        strategy_type: str
    ) -> None:
        """
        Internal method to register a strategy with validation.

        Args:
            registry: Target registry dict
            interface: Expected interface type
            name: Strategy name
            strategy_class: Strategy class to register
            override: Allow overriding existing entries
            strategy_type: Human-readable type name for errors
        """
        with cls._lock:
            # Check if name already exists
            if name in registry and not override:
                raise ValueError(
                    f"Strategy '{name}' already registered for {strategy_type}. "
                    f"Use override=True to replace it."
                )

            # Validate that strategy_class implements the interface
            if not issubclass(strategy_class, interface):
                raise TypeError(
                    f"Strategy class {strategy_class.__name__} must implement {interface.__name__} "
                    f"for {strategy_type} registration."
                )

            # Register the strategy
            registry[name] = strategy_class
            print(f"[StrategyRegistry] Registered {strategy_type}: '{name}' → {strategy_class.__name__}")

    @classmethod
    def _get_strategy(cls, registry: Dict[str, Type], name: str, strategy_type: str) -> Type:
        """
        Internal method to retrieve a strategy.

        Args:
            registry: Source registry dict
            name: Strategy name
            strategy_type: Human-readable type name for errors

        Returns:
            Strategy class

        Raises:
            KeyError: If strategy not found
        """
        if name not in registry:
            available = list(registry.keys())
            raise KeyError(
                f"Strategy '{name}' not found in {strategy_type} registry. "
                f"Available strategies: {available}"
            )
        return registry[name]

    @classmethod
    def list_all_strategies(cls) -> Dict[str, list]:
        """
        List all registered strategies across all types.

        Returns:
            Dict mapping strategy type to list of registered names
        """
        return {
            "chunkers": cls.list_chunkers(),
            "extractors": cls.list_extractors(),
            "validators": cls.list_validators(),
            "mergers": cls.list_mergers(),
            "hitl": cls.list_hitl(),
            "orphan_linkers": cls.list_orphan_linkers()
        }

    @classmethod
    def unregister_strategy(cls, strategy_type: str, name: str) -> None:
        """
        Unregister a custom strategy.

        Args:
            strategy_type: One of: chunker, extractor, validator, merger, hitl, orphan_linker
            name: Strategy name to unregister

        Raises:
            ValueError: If strategy_type invalid
            KeyError: If strategy not found
        """
        registry_map = {
            "chunker": cls._chunkers,
            "extractor": cls._extractors,
            "validator": cls._validators,
            "merger": cls._mergers,
            "hitl": cls._hitl_strategies,
            "orphan_linker": cls._orphan_linkers
        }

        if strategy_type not in registry_map:
            raise ValueError(f"Invalid strategy type: {strategy_type}. Must be one of: {list(registry_map.keys())}")

        registry = registry_map[strategy_type]

        with cls._lock:
            if name not in registry:
                raise KeyError(f"Strategy '{name}' not found in {strategy_type} registry.")
            del registry[name]
            print(f"[StrategyRegistry] Unregistered {strategy_type}: '{name}'")

    @classmethod
    def clear_all(cls) -> None:
        """
        Clear all registered strategies (useful for testing).

        WARNING: This will remove ALL custom strategies. Use with caution.
        """
        with cls._lock:
            cls._chunkers.clear()
            cls._extractors.clear()
            cls._validators.clear()
            cls._mergers.clear()
            cls._hitl_strategies.clear()
            cls._orphan_linkers.clear()
            print("[StrategyRegistry] Cleared all registries")


# ========== CONVENIENCE DECORATORS ==========

def register_chunker(name: str, override: bool = False):
    """
    Decorator to register a chunker strategy.

    Example:
        >>> @register_chunker("pdf")
        ... class PDFChunker(ChunkerInterface):
        ...     async def chunk(self, text, metadata):
        ...         pass
    """
    def decorator(cls):
        StrategyRegistry.register_chunker(name, cls, override=override)
        return cls
    return decorator


def register_extractor(name: str, override: bool = False):
    """Decorator to register an extractor strategy."""
    def decorator(cls):
        StrategyRegistry.register_extractor(name, cls, override=override)
        return cls
    return decorator


def register_validator(name: str, override: bool = False):
    """Decorator to register a validator strategy."""
    def decorator(cls):
        StrategyRegistry.register_validator(name, cls, override=override)
        return cls
    return decorator


def register_merger(name: str, override: bool = False):
    """Decorator to register a merger strategy."""
    def decorator(cls):
        StrategyRegistry.register_merger(name, cls, override=override)
        return cls
    return decorator


def register_hitl(name: str, override: bool = False):
    """Decorator to register a HITL strategy."""
    def decorator(cls):
        StrategyRegistry.register_hitl(name, cls, override=override)
        return cls
    return decorator


def register_orphan_linker(name: str, override: bool = False):
    """Decorator to register an orphan linker strategy."""
    def decorator(cls):
        StrategyRegistry.register_orphan_linker(name, cls, override=override)
        return cls
    return decorator
