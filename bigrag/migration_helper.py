"""
Migration helper for BiG-RAG pipeline version compatibility.

Provides utilities to check graph compatibility and handle version migrations
between standard, production, and enhanced pipelines.
"""

import networkx as nx
from pathlib import Path
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

# Current pipeline versions
CURRENT_VERSIONS = {
    'standard': 'standard-v1.0',
    'production': 'production-v1.0',
    'enhanced': 'enhanced-v1.0'
}

# Compatibility matrix
COMPATIBILITY_MATRIX = {
    'enhanced-v1.0': ['standard-v1.0', 'production-v1.0', 'enhanced-v1.0'],
    'production-v1.0': ['standard-v1.0', 'production-v1.0'],
    'standard-v1.0': ['standard-v1.0']
}


def check_graph_compatibility(
    graph_path: str,
    current_version: str = 'enhanced-v1.0',
    strict: bool = False
) -> bool:
    """
    Check if existing graph is compatible with current pipeline version.

    Args:
        graph_path: Path to GraphML file
        current_version: Current pipeline version
        strict: If True, raise error on incompatibility; if False, only warn

    Returns:
        True if compatible, False otherwise

    Raises:
        ValueError: If strict=True and graph is incompatible
    """
    graph_path = Path(graph_path)

    if not graph_path.exists():
        logger.warning(f"Graph file not found: {graph_path}")
        return False

    try:
        # Load graph metadata
        graph = nx.read_graphml(str(graph_path))
        graph_version = graph.graph.get('pipeline_version', 'unknown')
        graph_compatible = graph.graph.get('backward_compatible', [])

        # Check if current version is in graph's compatibility list
        if isinstance(graph_compatible, str):
            graph_compatible = [graph_compatible]

        if current_version in graph_compatible:
            logger.info(f"✓ Graph {graph_path.name} is compatible (version: {graph_version})")
            return True

        # Check if graph version is in current version's compatibility matrix
        compatible_versions = COMPATIBILITY_MATRIX.get(current_version, [])
        if graph_version in compatible_versions:
            logger.info(f"✓ Graph {graph_path.name} is compatible via matrix (version: {graph_version})")
            return True

        # Incompatible
        message = (
            f"Graph version {graph_version} may not be compatible with {current_version}. "
            f"Compatible versions: {compatible_versions}. "
            f"Consider rebuilding the graph."
        )

        if strict:
            raise ValueError(message)
        else:
            logger.warning(f"⚠ {message}")
            return False

    except Exception as e:
        logger.error(f"Error checking graph compatibility: {e}")
        if strict:
            raise
        return False


def get_graph_version(graph_path: str) -> Optional[str]:
    """
    Get the pipeline version from a graph file.

    Args:
        graph_path: Path to GraphML file

    Returns:
        Pipeline version string, or None if not found
    """
    graph_path = Path(graph_path)

    if not graph_path.exists():
        return None

    try:
        graph = nx.read_graphml(str(graph_path))
        return graph.graph.get('pipeline_version', None)
    except Exception as e:
        logger.error(f"Error reading graph version: {e}")
        return None


def check_dataset_compatibility(
    dataset_path: str,
    current_version: str = 'enhanced-v1.0'
) -> Dict[str, any]:
    """
    Check compatibility of all graphs in a dataset directory.

    Args:
        dataset_path: Path to dataset directory (e.g., expr/kuet_test)
        current_version: Current pipeline version

    Returns:
        {
            'compatible': bool,
            'graph_version': str,
            'files_checked': List[str],
            'warnings': List[str]
        }
    """
    dataset_path = Path(dataset_path)
    graphml_file = dataset_path / "graph_chunk_entity_relation.graphml"

    result = {
        'compatible': False,
        'graph_version': None,
        'files_checked': [],
        'warnings': []
    }

    if not graphml_file.exists():
        result['warnings'].append(f"GraphML file not found: {graphml_file}")
        return result

    result['files_checked'].append(str(graphml_file))

    # Check main graph file
    version = get_graph_version(str(graphml_file))
    result['graph_version'] = version

    if version is None:
        result['warnings'].append(
            f"Graph has no version metadata. Likely from old pipeline (pre-v1.0). "
            f"Recommend rebuilding with enhanced pipeline."
        )
        return result

    # Check compatibility
    compatible = check_graph_compatibility(str(graphml_file), current_version, strict=False)
    result['compatible'] = compatible

    if not compatible:
        result['warnings'].append(
            f"Graph version {version} may not be fully compatible with {current_version}. "
            f"Some features may not work correctly. Consider rebuilding."
        )

    return result


def migrate_config_keys(config: Dict) -> Dict:
    """
    Migrate old config keys to new enhanced pipeline format.

    Args:
        config: Configuration dictionary

    Returns:
        Updated configuration dictionary with deprecation warnings
    """
    migrated = config.copy()
    warnings = []

    # Handle old production pipeline key
    if 'use_production_pipeline' in migrated:
        warnings.append(
            "'use_production_pipeline' is deprecated. Use 'use_enhanced_pipeline' instead."
        )
        if 'use_enhanced_pipeline' not in migrated:
            migrated['use_enhanced_pipeline'] = migrated['use_production_pipeline']
        del migrated['use_production_pipeline']

    # Handle old production_pipeline_config key
    if 'production_pipeline_config' in migrated:
        warnings.append(
            "'production_pipeline_config' is deprecated. Use 'enhanced_pipeline_config' instead."
        )
        if 'enhanced_pipeline_config' not in migrated:
            migrated['enhanced_pipeline_config'] = migrated['production_pipeline_config']
        del migrated['production_pipeline_config']

    # Add extraction_strategy to enhanced_pipeline_config if missing
    if 'enhanced_pipeline_config' in migrated:
        if 'extraction_strategy' not in migrated['enhanced_pipeline_config']:
            migrated['enhanced_pipeline_config']['extraction_strategy'] = 'hybrid'
            warnings.append(
                "Added default 'extraction_strategy': 'hybrid' to enhanced_pipeline_config."
            )

    # Log warnings
    for warning in warnings:
        logger.warning(warning)

    return migrated


def is_graph_from_enhanced_pipeline(graph_path: str) -> bool:
    """
    Check if graph was built with enhanced pipeline.

    Args:
        graph_path: Path to GraphML file

    Returns:
        True if from enhanced pipeline, False otherwise
    """
    version = get_graph_version(graph_path)
    return version == 'enhanced-v1.0' if version else False


def get_compatible_pipelines(graph_path: str) -> List[str]:
    """
    Get list of pipelines compatible with this graph.

    Args:
        graph_path: Path to GraphML file

    Returns:
        List of compatible pipeline names
    """
    version = get_graph_version(graph_path)
    if not version:
        return ['standard']  # Assume old standard pipeline

    compatible = []
    if version in ['standard-v1.0', 'production-v1.0', 'enhanced-v1.0']:
        compatible.append('enhanced')
    if version in ['standard-v1.0', 'production-v1.0']:
        compatible.append('production')
    if version == 'standard-v1.0':
        compatible.append('standard')

    return compatible if compatible else ['standard']
