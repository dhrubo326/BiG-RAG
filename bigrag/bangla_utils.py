"""
Bangla Numeral Normalization Utilities

This module provides utilities for normalizing Bangla numerals to English numerals
for accurate comparison and validation in bilingual content.

CRITICAL for educational domain: Ensures "১২০" (Bangla) == "120" (English) in validation.
"""

import re
from typing import List


class BanglaNumeralNormalizer:
    """
    Normalize Bangla numerals for accurate comparison and validation.

    CRITICAL for educational domain: "১২০" vs "120" must be treated as same.
    """

    # Mapping table
    BANGLA_TO_ENGLISH = {
        '০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4',
        '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'
    }

    ENGLISH_TO_BANGLA = {v: k for k, v in BANGLA_TO_ENGLISH.items()}

    @staticmethod
    def bangla_to_english(text: str) -> str:
        """
        Convert Bangla numerals to English.

        Examples:
            >>> BanglaNumeralNormalizer.bangla_to_english("১২০")
            "120"
            >>> BanglaNumeralNormalizer.bangla_to_english("৪.০০")
            "4.00"
            >>> BanglaNumeralNormalizer.bangla_to_english("CSE: ১২০ seats")
            "CSE: 120 seats"

        Args:
            text: Input text with Bangla numerals

        Returns:
            Text with Bangla numerals converted to English
        """
        result = text
        for bn, en in BanglaNumeralNormalizer.BANGLA_TO_ENGLISH.items():
            result = result.replace(bn, en)
        return result

    @staticmethod
    def english_to_bangla(text: str) -> str:
        """
        Convert English numerals to Bangla.

        Examples:
            >>> BanglaNumeralNormalizer.english_to_bangla("120")
            "১২০"
            >>> BanglaNumeralNormalizer.english_to_bangla("4.00")
            "৪.০০"

        Args:
            text: Input text with English numerals

        Returns:
            Text with English numerals converted to Bangla
        """
        result = text
        for en, bn in BanglaNumeralNormalizer.ENGLISH_TO_BANGLA.items():
            result = result.replace(en, bn)
        return result

    @staticmethod
    def normalize_for_comparison(text: str) -> str:
        """
        Normalize text for comparison (always convert to English).

        Use in validation to ensure "১২০" == "120".

        Examples:
            >>> BanglaNumeralNormalizer.normalize_for_comparison("১২০")
            "120"
            >>> BanglaNumeralNormalizer.normalize_for_comparison("120")
            "120"

        Args:
            text: Input text with either Bangla or English numerals

        Returns:
            Text with all numerals in English
        """
        return BanglaNumeralNormalizer.bangla_to_english(text)

    @staticmethod
    def extract_numbers(text: str, normalize: bool = True) -> List[str]:
        """
        Extract all numbers from text (Bangla + English).

        Examples:
            >>> BanglaNumeralNormalizer.extract_numbers("CSE: ১২০, EEE: 120")
            ["120", "120"]
            >>> BanglaNumeralNormalizer.extract_numbers("GPA: ৪.০০ or 4.00")
            ["4.00", "4.00"]

        Args:
            text: Input text
            normalize: If True, convert all to English numerals

        Returns:
            List of numbers as strings
        """
        # First normalize if requested
        if normalize:
            text = BanglaNumeralNormalizer.bangla_to_english(text)

        # Extract all numbers (including decimals)
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        return numbers

    @staticmethod
    def is_bangla_numeral(text: str) -> bool:
        """
        Check if text contains any Bangla numerals.

        Examples:
            >>> BanglaNumeralNormalizer.is_bangla_numeral("১২০")
            True
            >>> BanglaNumeralNormalizer.is_bangla_numeral("120")
            False
            >>> BanglaNumeralNormalizer.is_bangla_numeral("CSE: ১২০ seats")
            True

        Args:
            text: Input text

        Returns:
            True if text contains Bangla numerals, False otherwise
        """
        return any(char in text for char in BanglaNumeralNormalizer.BANGLA_TO_ENGLISH.keys())

    @staticmethod
    def is_english_numeral(text: str) -> bool:
        """
        Check if text contains any English numerals.

        Examples:
            >>> BanglaNumeralNormalizer.is_english_numeral("120")
            True
            >>> BanglaNumeralNormalizer.is_english_numeral("১২০")
            False

        Args:
            text: Input text

        Returns:
            True if text contains English numerals, False otherwise
        """
        return bool(re.search(r'[0-9]', text))

    @staticmethod
    def normalize_number_string(num_str: str) -> str:
        """
        Normalize a number string to English (handles both formats).

        Examples:
            >>> BanglaNumeralNormalizer.normalize_number_string("১২০")
            "120"
            >>> BanglaNumeralNormalizer.normalize_number_string("120")
            "120"
            >>> BanglaNumeralNormalizer.normalize_number_string("৪.০০")
            "4.00"

        Args:
            num_str: Number as string (Bangla or English)

        Returns:
            Normalized number string in English
        """
        return BanglaNumeralNormalizer.bangla_to_english(num_str)


def test_bangla_normalizer():
    """
    Quick test function to verify Bangla normalization works.

    Run with: python -c "from bigrag.bangla_utils import test_bangla_normalizer; test_bangla_normalizer()"
    """
    print("Testing BanglaNumeralNormalizer...")

    # Test 1: Bangla to English conversion
    assert BanglaNumeralNormalizer.bangla_to_english("১২০") == "120", "Test 1 failed"
    print("[OK] Test 1: Bangla to English conversion")

    # Test 2: English to Bangla conversion
    assert BanglaNumeralNormalizer.english_to_bangla("120") == "১২০", "Test 2 failed"
    print("[OK] Test 2: English to Bangla conversion")

    # Test 3: Normalize for comparison
    assert BanglaNumeralNormalizer.normalize_for_comparison("১২০") == "120", "Test 3a failed"
    assert BanglaNumeralNormalizer.normalize_for_comparison("120") == "120", "Test 3b failed"
    print("[OK] Test 3: Normalize for comparison")

    # Test 4: Extract numbers
    numbers = BanglaNumeralNormalizer.extract_numbers("CSE: ১২০, EEE: 120")
    assert numbers == ["120", "120"], f"Test 4 failed: got {numbers}"
    print("[OK] Test 4: Extract numbers")

    # Test 5: Decimal numbers
    assert BanglaNumeralNormalizer.bangla_to_english("৪.০০") == "4.00", "Test 5a failed"
    numbers = BanglaNumeralNormalizer.extract_numbers("GPA: ৪.০০")
    assert "4.00" in numbers, f"Test 5b failed: got {numbers}"
    print("[OK] Test 5: Decimal numbers")

    # Test 6: Mixed text
    text = "কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগের কোড CSE এবং আসন সংখ্যা ১২০।"
    normalized = BanglaNumeralNormalizer.normalize_for_comparison(text)
    assert "120" in normalized, "Test 6 failed"
    print("[OK] Test 6: Mixed Bangla-English text")

    # Test 7: Detection
    assert BanglaNumeralNormalizer.is_bangla_numeral("১২০") == True, "Test 7a failed"
    assert BanglaNumeralNormalizer.is_bangla_numeral("120") == False, "Test 7b failed"
    assert BanglaNumeralNormalizer.is_english_numeral("120") == True, "Test 7c failed"
    assert BanglaNumeralNormalizer.is_english_numeral("১২০") == False, "Test 7d failed"
    print("[OK] Test 7: Numeral detection")

    print("\n[SUCCESS] All tests passed!")
    print("\nBanglaNumeralNormalizer is ready for production use.")


if __name__ == "__main__":
    test_bangla_normalizer()
