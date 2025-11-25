"""
Pipeline Selector Helper Module

Analyzes documents and recommends optimal pipeline configuration based on:
- Document characteristics (length, structure, complexity)
- Content type (educational, technical, general)
- Performance requirements (speed vs accuracy)
- Resource constraints (cost, time)

Part of Phase 1 Step 5: Pipeline Selector Helper
"""

from typing import Dict, List, Optional, Tuple, Any
import re
from dataclasses import dataclass
from enum import Enum


class PipelineType(Enum):
    """Available pipeline types."""
    STANDARD = "standard"
    ENHANCED = "enhanced"


class ContentComplexity(Enum):
    """Content complexity levels."""
    SIMPLE = "simple"          # Simple structure, short documents
    MODERATE = "moderate"      # Mixed structure, medium length
    COMPLEX = "complex"        # Complex structure, tables, long documents


class PerformanceProfile(Enum):
    """Performance optimization profiles."""
    SPEED = "speed"            # Fast processing, lower accuracy
    BALANCED = "balanced"      # Balance between speed and accuracy
    ACCURACY = "accuracy"      # Maximum accuracy, slower processing


@dataclass
class DocumentCharacteristics:
    """Analyzed characteristics of a document."""
    avg_length: float              # Average document length (chars)
    has_tables: bool               # Contains table structures
    has_code: bool                 # Contains code blocks
    has_equations: bool            # Contains mathematical equations
    has_lists: bool                # Contains structured lists
    structure_complexity: float    # 0-1 score of structural complexity
    content_type: str              # 'educational', 'technical', 'general'
    estimated_entity_density: float  # Estimated entities per 1000 chars


@dataclass
class PipelineRecommendation:
    """Pipeline configuration recommendation."""
    pipeline_type: PipelineType
    config: Dict[str, Any]
    reasoning: List[str]
    estimated_cost: str            # 'low', 'medium', 'high'
    estimated_time: str            # 'fast', 'medium', 'slow'
    expected_quality: str          # 'good', 'very_good', 'excellent'
    confidence: float              # 0-1 confidence in recommendation


# Configuration Presets (Part 2)
CONFIGURATION_PRESETS = {
    "fast_general": {
        "pipeline_type": PipelineType.STANDARD,
        "config": {
            "entity_merge_strategy": "basic",
            "chunk_size": 1200,
            "chunk_overlap": 100,
        },
        "use_case": "General documents, speed priority",
        "estimated_cost": "low",
        "estimated_time": "fast",
        "expected_quality": "good"
    },

    "balanced_general": {
        "pipeline_type": PipelineType.STANDARD,
        "config": {
            "entity_merge_strategy": "hybrid",
            "chunk_size": 1200,
            "chunk_overlap": 150,
        },
        "use_case": "General documents, balanced speed/accuracy",
        "estimated_cost": "low",
        "estimated_time": "medium",
        "expected_quality": "very_good"
    },

    "accurate_general": {
        "pipeline_type": PipelineType.STANDARD,
        "config": {
            "entity_merge_strategy": "fuzzy",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        "use_case": "General documents, accuracy priority",
        "estimated_cost": "medium",
        "estimated_time": "medium",
        "expected_quality": "very_good"
    },

    "educational_standard": {
        "pipeline_type": PipelineType.ENHANCED,
        "config": {
            "extraction_strategy": "hybrid",
            "extraction_mode": "semi_structured",
            "validation_level": "MODERATE",
            "enable_entity_linking": True,
            "entity_merge_strategy": "fuzzy",
            "chunking_strategy": "semantic",
            "enable_gleaning": False,
        },
        "use_case": "Educational content without heavy tables",
        "estimated_cost": "medium",
        "estimated_time": "medium",
        "expected_quality": "very_good"
    },

    "educational_tables": {
        "pipeline_type": PipelineType.ENHANCED,
        "config": {
            "extraction_strategy": "comprehensive",
            "extraction_mode": "semi_structured",
            "validation_level": "STRICT",
            "enable_entity_linking": True,
            "entity_merge_strategy": "fuzzy",
            "chunking_strategy": "semantic",
            "enable_gleaning": True,
        },
        "use_case": "Educational content with tables, maximum accuracy",
        "estimated_cost": "high",
        "estimated_time": "slow",
        "expected_quality": "excellent"
    },

    "technical_documentation": {
        "pipeline_type": PipelineType.ENHANCED,
        "config": {
            "extraction_strategy": "comprehensive",
            "extraction_mode": "semi_structured",
            "validation_level": "STRICT",
            "enable_entity_linking": True,
            "entity_merge_strategy": "fuzzy",
            "chunking_strategy": "semantic",
            "enable_gleaning": True,
        },
        "use_case": "Technical documentation with code/tables",
        "estimated_cost": "high",
        "estimated_time": "slow",
        "expected_quality": "excellent"
    },

    "large_corpus_fast": {
        "pipeline_type": PipelineType.STANDARD,
        "config": {
            "entity_merge_strategy": "basic",
            "chunk_size": 1500,
            "chunk_overlap": 100,
        },
        "use_case": "Large corpus (>10K docs), speed priority",
        "estimated_cost": "low",
        "estimated_time": "fast",
        "expected_quality": "good"
    },

    "small_corpus_accurate": {
        "pipeline_type": PipelineType.ENHANCED,
        "config": {
            "extraction_strategy": "comprehensive",
            "extraction_mode": "semi_structured",
            "validation_level": "STRICT",
            "enable_entity_linking": True,
            "entity_merge_strategy": "fuzzy",
            "chunking_strategy": "semantic",
            "enable_gleaning": True,
        },
        "use_case": "Small corpus (<1K docs), accuracy priority",
        "estimated_cost": "high",
        "estimated_time": "slow",
        "expected_quality": "excellent"
    }
}


class PipelineSelector:
    """
    Analyzes documents and recommends optimal pipeline configuration.

    Usage:
        selector = PipelineSelector()

        # Analyze sample documents
        chars = selector.analyze_documents(documents)

        # Get recommendation
        recommendation = selector.recommend_pipeline(
            characteristics=chars,
            corpus_size=1000,
            performance_profile=PerformanceProfile.BALANCED
        )

        # Use recommendation
        if recommendation.pipeline_type == PipelineType.ENHANCED:
            pipeline = EnhancedKGPipeline(**recommendation.config)
        else:
            rag = BiGRAG(**recommendation.config)
    """

    def __init__(self):
        """Initialize pipeline selector."""
        self.presets = CONFIGURATION_PRESETS

    def analyze_documents(
        self,
        documents: List[str],
        sample_size: Optional[int] = None
    ) -> DocumentCharacteristics:
        """
        Analyze a sample of documents to determine characteristics.

        Args:
            documents: List of document texts
            sample_size: Number of documents to analyze (None = all)

        Returns:
            DocumentCharacteristics object
        """
        if not documents:
            raise ValueError("documents list cannot be empty")

        # Sample documents if needed
        if sample_size and len(documents) > sample_size:
            import random
            sample = random.sample(documents, sample_size)
        else:
            sample = documents

        # Calculate average length
        avg_length = sum(len(doc) for doc in sample) / len(sample)

        # Detect structural features
        has_tables = self._detect_tables(sample)
        has_code = self._detect_code(sample)
        has_equations = self._detect_equations(sample)
        has_lists = self._detect_lists(sample)

        # Calculate structure complexity (0-1)
        structure_complexity = self._calculate_structure_complexity(
            has_tables, has_code, has_equations, has_lists, avg_length
        )

        # Determine content type
        content_type = self._determine_content_type(
            has_tables, has_code, has_equations, sample
        )

        # Estimate entity density (entities per 1000 chars)
        estimated_entity_density = self._estimate_entity_density(
            content_type, structure_complexity
        )

        return DocumentCharacteristics(
            avg_length=avg_length,
            has_tables=has_tables,
            has_code=has_code,
            has_equations=has_equations,
            has_lists=has_lists,
            structure_complexity=structure_complexity,
            content_type=content_type,
            estimated_entity_density=estimated_entity_density
        )

    def recommend_pipeline(
        self,
        characteristics: DocumentCharacteristics,
        corpus_size: int,
        performance_profile: PerformanceProfile = PerformanceProfile.BALANCED,
        budget_constraint: Optional[str] = None  # 'low', 'medium', 'high'
    ) -> PipelineRecommendation:
        """
        Recommend optimal pipeline configuration.

        Args:
            characteristics: Document characteristics from analyze_documents()
            corpus_size: Total number of documents in corpus
            performance_profile: Speed vs accuracy preference
            budget_constraint: Optional cost constraint

        Returns:
            PipelineRecommendation with config and reasoning
        """
        reasoning = []

        # Decision logic
        use_enhanced = False
        preset_key = None

        # Rule 1: Large corpus (>10K) -> prefer standard pipeline
        if corpus_size > 10000:
            reasoning.append(f"Large corpus ({corpus_size} docs) - standard pipeline recommended for speed")
            if performance_profile == PerformanceProfile.SPEED:
                preset_key = "large_corpus_fast"
            else:
                preset_key = "balanced_general"

        # Rule 2: Educational/technical content with tables -> enhanced pipeline
        elif characteristics.content_type in ['educational', 'technical'] and characteristics.has_tables:
            reasoning.append(f"Educational/technical content with tables detected - enhanced pipeline recommended")
            use_enhanced = True
            if performance_profile == PerformanceProfile.ACCURACY:
                preset_key = "educational_tables"
            else:
                preset_key = "educational_standard"

        # Rule 3: High structure complexity -> enhanced pipeline
        elif characteristics.structure_complexity > 0.6:
            reasoning.append(f"High structure complexity ({characteristics.structure_complexity:.2f}) - enhanced pipeline recommended")
            use_enhanced = True
            preset_key = "technical_documentation"

        # Rule 4: Small corpus with accuracy priority -> enhanced pipeline
        elif corpus_size < 1000 and performance_profile == PerformanceProfile.ACCURACY:
            reasoning.append(f"Small corpus ({corpus_size} docs) with accuracy priority - enhanced pipeline recommended")
            use_enhanced = True
            preset_key = "small_corpus_accurate"

        # Rule 5: Speed priority -> standard pipeline
        elif performance_profile == PerformanceProfile.SPEED:
            reasoning.append("Speed priority - standard pipeline recommended")
            preset_key = "fast_general"

        # Rule 6: Default to balanced
        else:
            reasoning.append("Standard use case - balanced configuration recommended")
            if characteristics.content_type == 'educational':
                use_enhanced = True
                preset_key = "educational_standard"
            else:
                preset_key = "balanced_general"

        # Budget constraint override
        if budget_constraint == 'low':
            if use_enhanced:
                reasoning.append("Budget constraint (low) - switching to standard pipeline")
                use_enhanced = False
                preset_key = "fast_general"

        # Get preset configuration
        preset = self.presets[preset_key]

        # Build recommendation
        recommendation = PipelineRecommendation(
            pipeline_type=preset["pipeline_type"],
            config=preset["config"].copy(),
            reasoning=reasoning,
            estimated_cost=preset["estimated_cost"],
            estimated_time=preset["estimated_time"],
            expected_quality=preset["expected_quality"],
            confidence=self._calculate_confidence(characteristics, preset)
        )

        return recommendation

    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """
        Get a specific configuration preset by name.

        Args:
            preset_name: Name of preset (e.g., 'educational_tables')

        Returns:
            Preset configuration dictionary

        Raises:
            KeyError: If preset_name not found
        """
        if preset_name not in self.presets:
            available = list(self.presets.keys())
            raise KeyError(f"Preset '{preset_name}' not found. Available: {available}")

        return self.presets[preset_name].copy()

    def list_presets(self) -> Dict[str, str]:
        """
        List all available presets with their use cases.

        Returns:
            Dictionary mapping preset names to use case descriptions
        """
        return {name: preset["use_case"] for name, preset in self.presets.items()}

    def compare_presets(self, preset_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple presets side-by-side.

        Args:
            preset_names: List of preset names to compare

        Returns:
            Dictionary with comparison data
        """
        comparison = {}
        for name in preset_names:
            if name in self.presets:
                preset = self.presets[name]
                comparison[name] = {
                    "pipeline_type": preset["pipeline_type"].value,
                    "use_case": preset["use_case"],
                    "cost": preset["estimated_cost"],
                    "time": preset["estimated_time"],
                    "quality": preset["expected_quality"],
                    "config": preset["config"]
                }
        return comparison

    # Helper methods for analysis

    def _detect_tables(self, documents: List[str]) -> bool:
        """Detect if documents contain table structures."""
        table_patterns = [
            r'\|.*\|.*\|',  # Markdown tables
            r'<table>',     # HTML tables
            r'\t.*\t',      # Tab-separated
        ]

        for doc in documents:
            for pattern in table_patterns:
                if re.search(pattern, doc):
                    return True
        return False

    def _detect_code(self, documents: List[str]) -> bool:
        """Detect if documents contain code blocks."""
        code_patterns = [
            r'```',                    # Markdown code blocks
            r'<code>',                 # HTML code tags
            r'^\s{4,}\w+',             # Indented code
            r'def\s+\w+\s*\(',         # Python functions
            r'function\s+\w+\s*\(',    # JavaScript functions
            r'class\s+\w+\s*[\{:]',    # Class definitions
        ]

        for doc in documents:
            for pattern in code_patterns:
                if re.search(pattern, doc, re.MULTILINE):
                    return True
        return False

    def _detect_equations(self, documents: List[str]) -> bool:
        """Detect if documents contain mathematical equations."""
        equation_patterns = [
            r'\$\$',           # LaTeX display math
            r'\\\[',           # LaTeX display math
            r'\\begin{equation}',  # LaTeX equation environment
        ]

        for doc in documents:
            for pattern in equation_patterns:
                if re.search(pattern, doc):
                    return True
        return False

    def _detect_lists(self, documents: List[str]) -> bool:
        """Detect if documents contain structured lists."""
        list_patterns = [
            r'^\s*[-*+]\s',     # Markdown lists
            r'^\s*\d+\.\s',     # Numbered lists
            r'<ul>|<ol>',       # HTML lists
        ]

        for doc in documents:
            for pattern in list_patterns:
                if re.search(pattern, doc, re.MULTILINE):
                    return True
        return False

    def _calculate_structure_complexity(
        self,
        has_tables: bool,
        has_code: bool,
        has_equations: bool,
        has_lists: bool,
        avg_length: float
    ) -> float:
        """
        Calculate structural complexity score (0-1).

        Higher score = more complex structure requiring enhanced pipeline.
        """
        score = 0.0

        # Base complexity from structural elements
        if has_tables:
            score += 0.3
        if has_code:
            score += 0.2
        if has_equations:
            score += 0.2
        if has_lists:
            score += 0.1

        # Length factor (longer = more complex)
        if avg_length > 5000:
            score += 0.2
        elif avg_length > 2000:
            score += 0.1

        return min(score, 1.0)

    def _determine_content_type(
        self,
        has_tables: bool,
        has_code: bool,
        has_equations: bool,
        documents: List[str]
    ) -> str:
        """
        Determine content type: 'educational', 'technical', or 'general'.
        """
        # Technical indicators
        if has_code or (has_equations and has_tables):
            return 'technical'

        # Educational indicators
        educational_keywords = [
            'chapter', 'section', 'exercise', 'question', 'answer',
            'course', 'lecture', 'tutorial', 'example', 'definition'
        ]

        sample_text = ' '.join(documents[:3]).lower()
        educational_count = sum(1 for kw in educational_keywords if kw in sample_text)

        if educational_count >= 3 or (has_tables and educational_count >= 1):
            return 'educational'

        return 'general'

    def _estimate_entity_density(
        self,
        content_type: str,
        structure_complexity: float
    ) -> float:
        """
        Estimate entities per 1000 characters.

        Used to predict graph size and processing time.
        """
        # Base density by content type
        base_density = {
            'educational': 8.0,   # Higher entity density
            'technical': 10.0,    # Highest entity density
            'general': 5.0        # Lower entity density
        }

        density = base_density.get(content_type, 5.0)

        # Adjust for structure complexity
        density *= (1.0 + structure_complexity * 0.5)

        return density

    def _calculate_confidence(
        self,
        characteristics: DocumentCharacteristics,
        preset: Dict[str, Any]
    ) -> float:
        """
        Calculate confidence score (0-1) for recommendation.

        Higher confidence when characteristics clearly match preset use case.
        """
        confidence = 0.5  # Base confidence

        # Increase confidence for clear matches
        if characteristics.content_type == 'educational':
            if 'educational' in preset['use_case'].lower():
                confidence += 0.2

        if characteristics.has_tables:
            if 'table' in preset['use_case'].lower():
                confidence += 0.15

        if characteristics.structure_complexity > 0.6:
            if preset['pipeline_type'] == PipelineType.ENHANCED:
                confidence += 0.15

        return min(confidence, 1.0)


# Convenience functions

def quick_recommend(
    documents: List[str],
    corpus_size: int,
    performance_profile: str = "balanced",
    sample_size: int = 10
) -> PipelineRecommendation:
    """
    Quick recommendation with minimal setup.

    Args:
        documents: Sample documents (at least 3-5 recommended)
        corpus_size: Total corpus size
        performance_profile: 'speed', 'balanced', or 'accuracy'
        sample_size: Number of documents to analyze

    Returns:
        PipelineRecommendation

    Example:
        rec = quick_recommend(docs, corpus_size=500, performance_profile='accuracy')
        print(f"Use {rec.pipeline_type.value} pipeline")
        print(f"Config: {rec.config}")
    """
    selector = PipelineSelector()

    # Map string to enum
    profile_map = {
        'speed': PerformanceProfile.SPEED,
        'balanced': PerformanceProfile.BALANCED,
        'accuracy': PerformanceProfile.ACCURACY
    }
    profile = profile_map.get(performance_profile.lower(), PerformanceProfile.BALANCED)

    # Analyze and recommend
    chars = selector.analyze_documents(documents, sample_size=sample_size)
    recommendation = selector.recommend_pipeline(chars, corpus_size, profile)

    return recommendation


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """
    Get a preset configuration by name.

    Args:
        preset_name: Name of preset

    Returns:
        Configuration dictionary

    Example:
        config = get_preset_config('educational_tables')
        pipeline = EnhancedKGPipeline(**config['config'])
    """
    selector = PipelineSelector()
    return selector.get_preset(preset_name)


def list_all_presets() -> Dict[str, str]:
    """
    List all available presets.

    Returns:
        Dictionary mapping preset names to descriptions

    Example:
        presets = list_all_presets()
        for name, desc in presets.items():
            print(f"{name}: {desc}")
    """
    selector = PipelineSelector()
    return selector.list_presets()
