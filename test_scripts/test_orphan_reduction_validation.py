"""
Comprehensive Unit Tests for Orphan Node Reduction
Tests all validation, sanitization, and parsing logic
"""
import pytest
import asyncio
from bigrag.utils import (
    sanitize_extracted_text,
    fix_delimiter_corruption,
    description_quality_score,
    split_string_by_multi_markers
)
from bigrag.operate import (
    _handle_single_entity_extraction,
    _handle_single_hyperrelation_extraction
)


class TestDelimiterFix:
    """Test fix_delimiter_corruption function"""

    def test_double_brackets(self):
        """Test fixing double angle brackets <<|>>"""
        input_text = '"relation"<<|>>"content"<<|>>8'
        expected = '"relation"<|>"content"<|>8'
        result = fix_delimiter_corruption(input_text, "<|>")
        assert result == expected, f"Expected {expected}, got {result}"

    def test_double_pipes(self):
        """Test fixing double pipes <||>"""
        input_text = '"relation"<||>"content"<||>8'
        expected = '"relation"<|>"content"<|>8'
        result = fix_delimiter_corruption(input_text, "<|>")
        assert result == expected

    def test_empty_brackets(self):
        """Test fixing empty brackets <>"""
        input_text = '"relation"<>"content"<>8'
        expected = '"relation"<|>"content"<|>8'
        result = fix_delimiter_corruption(input_text, "<|>")
        assert result == expected

    def test_no_corruption(self):
        """Test that valid delimiter is not changed"""
        input_text = '"relation"<|>"content"<|>8'
        result = fix_delimiter_corruption(input_text, "<|>")
        assert result == input_text, "Valid delimiter should not be changed"


class TestSanitization:
    """Test sanitize_extracted_text function"""

    def test_entity_name_with_quotes(self):
        """Test entity name with quoted content"""
        input_text = '"Lionel Messi"'
        result = sanitize_extracted_text(input_text, "entity_name")
        assert result == "Lionel Messi"

    def test_entity_name_with_inner_quotes(self):
        """Test entity name with quotes in content"""
        input_text = 'Lionel "Leo" Messi'
        result = sanitize_extracted_text(input_text, "entity_name")
        assert result == "Lionel Leo Messi"  # Inner quotes removed

    def test_entity_type_lowercase(self):
        """Test entity type is lowercased"""
        input_text = '"PERSON"'
        result = sanitize_extracted_text(input_text, "entity_type")
        assert result == "person"

    def test_entity_type_with_spaces(self):
        """Test entity type removes spaces"""
        input_text = '"person player"'
        result = sanitize_extracted_text(input_text, "entity_type")
        assert result == "personplayer"

    def test_entity_type_invalid_chars(self):
        """Test entity type rejects invalid characters"""
        input_text = '"person<|>player"'
        result = sanitize_extracted_text(input_text, "entity_type")
        assert result == "", "Should reject type with delimiter"

    def test_description_truncation(self):
        """Test description is truncated if too long"""
        long_desc = "A" * 2000  # 2000 chars
        result = sanitize_extracted_text(long_desc, "description")
        assert len(result) <= 1500, "Description should be truncated to 1500 chars"

    def test_relation_content_preserves_detail(self):
        """Test relation content preserves important detail"""
        content = '"Lionel Messi scored 672 goals for Barcelona between 2004-2021."'
        result = sanitize_extracted_text(content, "relation")
        assert "672" in result
        assert "Barcelona" in result
        assert "2004-2021" in result


class TestQualityScoring:
    """Test description_quality_score function"""

    def test_empty_description(self):
        """Test empty description scores 0"""
        assert description_quality_score("") == 0.0

    def test_short_description_penalty(self):
        """Test short descriptions get penalty"""
        short = "A player"  # 8 chars
        long = "A player who plays for Barcelona" * 3  # >100 chars
        assert description_quality_score(short) < description_quality_score(long)

    def test_complete_sentence_bonus(self):
        """Test complete sentence (ends with period) gets bonus"""
        incomplete = "A football player"
        complete = "A football player."
        assert description_quality_score(complete) > description_quality_score(incomplete)

    def test_keyword_bonus(self):
        """Test quality keywords increase score"""
        generic = "A person in sports"
        specific = "A professional player who is known for winning championships"
        assert description_quality_score(specific) > description_quality_score(generic)

    def test_numbers_bonus(self):
        """Test numbers/dates increase score"""
        no_numbers = "A football player for Barcelona"
        with_numbers = "A football player for Barcelona since 2004"
        assert description_quality_score(with_numbers) > description_quality_score(no_numbers)


class TestParsing:
    """Test split_string_by_multi_markers function"""

    def test_split_by_delimiter(self):
        """Test basic splitting by delimiter"""
        text = '"relation"<|>"content"<|>8'
        result = split_string_by_multi_markers(text, ["<|>"])
        assert len(result) == 3
        assert result[0] == '"relation"'
        assert result[1] == '"content"'
        assert result[2] == '8'

    def test_split_by_multiple_delimiters(self):
        """Test splitting by multiple delimiters"""
        text = 'part1<|>part2##part3<COMPLETE>'
        result = split_string_by_multi_markers(text, ["<|>", "##", "<COMPLETE>"])
        assert len(result) == 3
        assert "part1" in result
        assert "part2" in result
        assert "part3" in result

    def test_split_removes_empty_parts(self):
        """Test that empty parts are removed"""
        text = 'part1<|><|>part2'  # Double delimiter
        result = split_string_by_multi_markers(text, ["<|>"])
        assert len(result) == 2  # Empty part removed


class TestRelationValidation:
    """Test _handle_single_hyperrelation_extraction function"""

    @pytest.mark.asyncio
    async def test_valid_relation(self):
        """Test valid relation is accepted"""
        record_attributes = [
            '"relation"',
            '"Lionel Messi plays for Inter Miami."',
            '9'
        ]
        result = await _handle_single_hyperrelation_extraction(record_attributes, "chunk-test")
        assert result is not None
        assert result['hyper_relation_content'] == "Lionel Messi plays for Inter Miami."
        assert result['weight'] == 9.0

    @pytest.mark.asyncio
    async def test_relation_wrong_field_count(self):
        """Test relation with wrong field count is rejected"""
        record_attributes = ['"relation"', '"content"']  # Only 2 fields
        result = await _handle_single_hyperrelation_extraction(record_attributes, "chunk-test")
        assert result is None

    @pytest.mark.asyncio
    async def test_relation_wrong_type(self):
        """Test record with wrong type is rejected"""
        record_attributes = ['"entity"', '"content"', '8']  # Says entity, not relation
        result = await _handle_single_hyperrelation_extraction(record_attributes, "chunk-test")
        assert result is None

    @pytest.mark.asyncio
    async def test_relation_score_clamping(self):
        """Test score out of range is clamped"""
        record_attributes = ['"relation"', '"content"', '15']  # Score > 10
        result = await _handle_single_hyperrelation_extraction(record_attributes, "chunk-test")
        assert result is not None
        assert result['weight'] == 10.0  # Clamped to max

    @pytest.mark.asyncio
    async def test_relation_empty_content(self):
        """Test relation with empty content is rejected"""
        record_attributes = ['"relation"', '""', '8']  # Empty content
        result = await _handle_single_hyperrelation_extraction(record_attributes, "chunk-test")
        assert result is None


class TestEntityValidation:
    """Test _handle_single_entity_extraction function"""

    @pytest.mark.asyncio
    async def test_valid_entity(self):
        """Test valid entity is accepted"""
        record_attributes = [
            '"entity"',
            '"Lionel Messi"',
            '"person"',
            '"Widely regarded as one of the greatest football players."',
            '90'
        ]
        result = await _handle_single_entity_extraction(
            record_attributes,
            "chunk-test",
            "rel-abc123"  # Valid relation context
        )
        assert result is not None
        assert result['entity_name'] == "LIONEL MESSI"  # Uppercase
        assert result['entity_type'] == "person"
        assert result['weight'] == 90.0

    @pytest.mark.asyncio
    async def test_entity_wrong_field_count(self):
        """Test entity with wrong field count is rejected"""
        record_attributes = ['"entity"', '"MESSI"', '"person"']  # Only 3 fields
        result = await _handle_single_entity_extraction(record_attributes, "chunk-test", "rel-123")
        assert result is None

    @pytest.mark.asyncio
    async def test_entity_no_relation_context(self):
        """Test entity without relation context is rejected"""
        record_attributes = [
            '"entity"', '"MESSI"', '"person"', '"A player"', '90'
        ]
        result = await _handle_single_entity_extraction(
            record_attributes,
            "chunk-test",
            ""  # No relation context!
        )
        assert result is None  # Should be rejected

    @pytest.mark.asyncio
    async def test_entity_name_uppercase_conversion(self):
        """Test entity name is converted to uppercase"""
        record_attributes = [
            '"entity"', '"lionel messi"', '"person"', '"A player"', '90'
        ]
        result = await _handle_single_entity_extraction(record_attributes, "chunk-test", "rel-123")
        assert result is not None
        assert result['entity_name'] == "LIONEL MESSI"

    @pytest.mark.asyncio
    async def test_entity_weight_clamping(self):
        """Test weight out of range is clamped"""
        record_attributes = [
            '"entity"', '"MESSI"', '"person"', '"A player"', '150'  # > 100
        ]
        result = await _handle_single_entity_extraction(record_attributes, "chunk-test", "rel-123")
        assert result is not None
        assert result['weight'] == 100.0  # Clamped


class TestEndToEnd:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_full_llm_output_parsing(self):
        """Test parsing complete LLM output with multiple records"""
        # Simulate actual LLM output with double brackets
        llm_output = (
            '("relation"<<|>>"Lionel Messi plays for Inter Miami."<<|>>9)##'
            '("entity"<<|>>"Lionel Messi"<<|>>"person"<<|>>"Football player."<<|>>90)##'
            '("entity"<<|>>"Inter Miami"<<|>>"organization"<<|>>"MLS club."<<|>>85)##'
        )

        # Parse into records
        records = split_string_by_multi_markers(llm_output, ["##"])
        assert len(records) == 3

        # Process each record
        import re
        relation_context = ""
        entities = []

        for record in records:
            # Extract from parentheses
            match = re.search(r"\((.*)\)", record)
            if not match:
                continue
            record_content = match.group(1)

            # Fix delimiter corruption
            record_content = fix_delimiter_corruption(record_content, "<|>")

            # Split by delimiter
            parts = split_string_by_multi_markers(record_content, ["<|>"])

            if parts[0] == '"relation"':
                result = await _handle_single_hyperrelation_extraction(parts, "chunk-test")
                assert result is not None
                relation_context = result['hyper_relation']

            elif parts[0] == '"entity"':
                result = await _handle_single_entity_extraction(parts, "chunk-test", relation_context)
                assert result is not None
                entities.append(result)

        # Verify results
        assert len(entities) == 2
        assert entities[0]['entity_name'] == "LIONEL MESSI"
        assert entities[1]['entity_name'] == "INTER MIAMI"


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
