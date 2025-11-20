"""
Entity Canonicalization Map for Production Knowledge Graph

Domain-specific entity name normalization for educational institutions.
Maps variations to canonical forms to prevent duplicate entities.

Examples:
  - "CSE" → "COMPUTER SCIENCE AND ENGINEERING"
  - "কম্পিউটার সায়েন্স" → "COMPUTER SCIENCE AND ENGINEERING"
  - "KUET" → "KHULNA UNIVERSITY OF ENGINEERING AND TECHNOLOGY"
"""

from typing import Dict, List


class EntityCanonicalizationMap:
    """
    Domain-specific entity name canonicalization.

    Maps variations to canonical forms (manually curated for high accuracy).
    Critical for educational domain to prevent duplicates.
    """

    def __init__(self):
        """Initialize with pre-defined educational mappings."""
        self.canonical_map = {}  # variant -> canonical
        self.aliases = {}  # canonical -> [variants]

        # Initialize with KUET/BUET departments
        self._initialize_educational_mappings()

    def _initialize_educational_mappings(self):
        """
        Pre-defined mappings for educational domain.

        MUST BE MAINTAINED as new universities are added.
        All KUET and BUET departments included.
        """

        # ===================================================================
        # KUET DEPARTMENTS (16 departments)
        # ===================================================================

        # Department 1: Computer Science and Engineering
        self.add_mapping(
            canonical="COMPUTER SCIENCE AND ENGINEERING",
            variants=[
                "CSE",
                "কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং",
                "কম্পিউটার সায়েন্স",
                "Computer Science and Engineering",
                "COMPUTER SCIENCE AND ENGINEERING",
                "Comp Sci & Eng",
                "Computer Science",
                "CS"
            ]
        )

        # Department 2: Electrical and Electronic Engineering
        self.add_mapping(
            canonical="ELECTRICAL AND ELECTRONIC ENGINEERING",
            variants=[
                "EEE",
                "ইলেক্ট্রিক্যাল এন্ড ইলেক্ট্রনিক ইঞ্জিনিয়ারিং",
                "ইলেক্ট্রিক্যাল",
                "Electrical and Electronic Engineering",
                "ELECTRICAL AND ELECTRONIC ENGINEERING",
                "Elec & Electronic Eng",
                "Electrical Engineering"
            ]
        )

        # Department 3: Civil Engineering
        self.add_mapping(
            canonical="CIVIL ENGINEERING",
            variants=[
                "CE",
                "সিভিল ইঞ্জিনিয়ারিং",
                "সিভিল",
                "Civil Engineering",
                "CIVIL ENGINEERING"
            ]
        )

        # Department 4: Mechanical Engineering
        self.add_mapping(
            canonical="MECHANICAL ENGINEERING",
            variants=[
                "ME",
                "মেকানিক্যাল ইঞ্জিনিয়ারিং",
                "মেকানিক্যাল",
                "Mechanical Engineering",
                "MECHANICAL ENGINEERING"
            ]
        )

        # Department 5: Electronics and Communication Engineering
        self.add_mapping(
            canonical="ELECTRONICS AND COMMUNICATION ENGINEERING",
            variants=[
                "ECE",
                "ইলেকট্রনিক্স এন্ড কমিউনিকেশন ইঞ্জিনিয়ারিং",
                "ইলেকট্রনিক্স",
                "Electronics and Communication Engineering",
                "ELECTRONICS AND COMMUNICATION ENGINEERING"
            ]
        )

        # Department 6: Industrial Engineering and Management
        self.add_mapping(
            canonical="INDUSTRIAL ENGINEERING AND MANAGEMENT",
            variants=[
                "IEM",
                "ইন্ডাস্ট্রিয়াল ইঞ্জিনিয়ারিং এন্ড ম্যানেজমেন্ট",
                "Industrial Engineering and Management",
                "INDUSTRIAL ENGINEERING AND MANAGEMENT"
            ]
        )

        # Department 7: Urban and Regional Planning
        self.add_mapping(
            canonical="URBAN AND REGIONAL PLANNING",
            variants=[
                "URP",
                "আরবান এন্ড রিজিওনাল প্ল্যানিং",
                "Urban and Regional Planning",
                "URBAN AND REGIONAL PLANNING"
            ]
        )

        # Department 8: Building Engineering and Construction Management
        self.add_mapping(
            canonical="BUILDING ENGINEERING AND CONSTRUCTION MANAGEMENT",
            variants=[
                "BECM",
                "বিল্ডিং ইঞ্জিনিয়ারিং এন্ড কনস্ট্রাকশন ম্যানেজমেন্ট",
                "Building Engineering and Construction Management",
                "BUILDING ENGINEERING AND CONSTRUCTION MANAGEMENT"
            ]
        )

        # Department 9: Mathematics
        self.add_mapping(
            canonical="MATHEMATICS",
            variants=[
                "MATH",
                "গণিত",
                "Mathematics",
                "MATHEMATICS"
            ]
        )

        # Department 10: Chemistry
        self.add_mapping(
            canonical="CHEMISTRY",
            variants=[
                "CHEM",
                "রসায়ন",
                "Chemistry",
                "CHEMISTRY"
            ]
        )

        # Department 11: Physics
        self.add_mapping(
            canonical="PHYSICS",
            variants=[
                "PHY",
                "পদার্থবিজ্ঞান",
                "Physics",
                "PHYSICS"
            ]
        )

        # Department 12: Humanities
        self.add_mapping(
            canonical="HUMANITIES",
            variants=[
                "HUM",
                "মানবিক",
                "Humanities",
                "HUMANITIES"
            ]
        )

        # Department 13: Textile Engineering
        self.add_mapping(
            canonical="TEXTILE ENGINEERING",
            variants=[
                "TE",
                "টেক্সটাইল ইঞ্জিনিয়ারিং",
                "Textile Engineering",
                "TEXTILE ENGINEERING"
            ]
        )

        # Department 14: Leather Engineering
        self.add_mapping(
            canonical="LEATHER ENGINEERING",
            variants=[
                "LE",
                "লেদার ইঞ্জিনিয়ারিং",
                "Leather Engineering",
                "LEATHER ENGINEERING"
            ]
        )

        # Department 15: Materials Science and Engineering
        self.add_mapping(
            canonical="MATERIALS SCIENCE AND ENGINEERING",
            variants=[
                "MSE",
                "ম্যাটেরিয়ালস সায়েন্স এন্ড ইঞ্জিনিয়ারিং",
                "Materials Science and Engineering",
                "MATERIALS SCIENCE AND ENGINEERING"
            ]
        )

        # Department 16: Biomedical Engineering
        self.add_mapping(
            canonical="BIOMEDICAL ENGINEERING",
            variants=[
                "BME",
                "বায়োমেডিকেল ইঞ্জিনিয়ারিং",
                "Biomedical Engineering",
                "BIOMEDICAL ENGINEERING"
            ]
        )

        # ===================================================================
        # BUET DEPARTMENTS (15 departments)
        # ===================================================================

        # BUET shares some departments with KUET, so we add BUET-specific variants

        # Department 1: Architecture
        self.add_mapping(
            canonical="ARCHITECTURE",
            variants=[
                "Arch",
                "স্থাপত্য",
                "Architecture",
                "ARCHITECTURE"
            ]
        )

        # Department 2: Chemical Engineering
        self.add_mapping(
            canonical="CHEMICAL ENGINEERING",
            variants=[
                "ChE",
                "কেমিক্যাল ইঞ্জিনিয়ারিং",
                "Chemical Engineering",
                "CHEMICAL ENGINEERING"
            ]
        )

        # Department 3: Petroleum and Mineral Resources Engineering
        self.add_mapping(
            canonical="PETROLEUM AND MINERAL RESOURCES ENGINEERING",
            variants=[
                "PMRE",
                "পেট্রোলিয়াম এন্ড মিনারেল রিসোর্সেস ইঞ্জিনিয়ারিং",
                "Petroleum and Mineral Resources Engineering",
                "PETROLEUM AND MINERAL RESOURCES ENGINEERING"
            ]
        )

        # Department 4: Naval Architecture and Marine Engineering
        self.add_mapping(
            canonical="NAVAL ARCHITECTURE AND MARINE ENGINEERING",
            variants=[
                "NAME",
                "নেভাল আর্কিটেকচার এন্ড মেরিন ইঞ্জিনিয়ারিং",
                "Naval Architecture and Marine Engineering",
                "NAVAL ARCHITECTURE AND MARINE ENGINEERING"
            ]
        )

        # Department 5: Industrial and Production Engineering
        self.add_mapping(
            canonical="INDUSTRIAL AND PRODUCTION ENGINEERING",
            variants=[
                "IPE",
                "ইন্ডাস্ট্রিয়াল এন্ড প্রোডাকশন ইঞ্জিনিয়ারিং",
                "Industrial and Production Engineering",
                "INDUSTRIAL AND PRODUCTION ENGINEERING"
            ]
        )

        # Department 6: Glass and Ceramic Engineering
        self.add_mapping(
            canonical="GLASS AND CERAMIC ENGINEERING",
            variants=[
                "GCE",
                "গ্লাস এন্ড সিরামিক ইঞ্জিনিয়ারিং",
                "Glass and Ceramic Engineering",
                "GLASS AND CERAMIC ENGINEERING"
            ]
        )

        # Department 7: Water Resources Engineering
        self.add_mapping(
            canonical="WATER RESOURCES ENGINEERING",
            variants=[
                "WRE",
                "ওয়াটার রিসোর্সেস ইঞ্জিনিয়ারিং",
                "Water Resources Engineering",
                "WATER RESOURCES ENGINEERING"
            ]
        )

        # Department 8: Metallurgical and Materials Engineering
        self.add_mapping(
            canonical="METALLURGICAL AND MATERIALS ENGINEERING",
            variants=[
                "MME",
                "মেটালারজিক্যাল এন্ড ম্যাটেরিয়ালস ইঞ্জিনিয়ারিং",
                "Metallurgical and Materials Engineering",
                "METALLURGICAL AND MATERIALS ENGINEERING"
            ]
        )

        # Department 9: Nuclear Engineering
        self.add_mapping(
            canonical="NUCLEAR ENGINEERING",
            variants=[
                "NE",
                "নিউক্লিয়ার ইঞ্জিনিয়ারিং",
                "Nuclear Engineering",
                "NUCLEAR ENGINEERING"
            ]
        )

        # Department 10: Aerospace Engineering
        self.add_mapping(
            canonical="AEROSPACE ENGINEERING",
            variants=[
                "AE",
                "অ্যারোস্পেস ইঞ্জিনিয়ারিং",
                "Aerospace Engineering",
                "AEROSPACE ENGINEERING"
            ]
        )

        # ===================================================================
        # UNIVERSITIES
        # ===================================================================

        # KUET
        self.add_mapping(
            canonical="KHULNA UNIVERSITY OF ENGINEERING AND TECHNOLOGY",
            variants=[
                "KUET",
                "খুলনা প্রকৌশল ও প্রযুক্তি বিশ্ববিদ্যালয়",
                "Khulna University of Engineering and Technology",
                "KHULNA UNIVERSITY OF ENGINEERING AND TECHNOLOGY",
                "Khulna University"
            ]
        )

        # BUET
        self.add_mapping(
            canonical="BANGLADESH UNIVERSITY OF ENGINEERING AND TECHNOLOGY",
            variants=[
                "BUET",
                "বাংলাদেশ প্রকৌশল বিশ্ববিদ্যালয়",
                "Bangladesh University of Engineering and Technology",
                "BANGLADESH UNIVERSITY OF ENGINEERING AND TECHNOLOGY",
                "Bangladesh University of Engineering"
            ]
        )

        # CUET
        self.add_mapping(
            canonical="CHITTAGONG UNIVERSITY OF ENGINEERING AND TECHNOLOGY",
            variants=[
                "CUET",
                "চট্টগ্রাম প্রকৌশল ও প্রযুক্তি বিশ্ববিদ্যালয়",
                "Chittagong University of Engineering and Technology",
                "CHITTAGONG UNIVERSITY OF ENGINEERING AND TECHNOLOGY"
            ]
        )

        # RUET
        self.add_mapping(
            canonical="RAJSHAHI UNIVERSITY OF ENGINEERING AND TECHNOLOGY",
            variants=[
                "RUET",
                "রাজশাহী প্রকৌশল ও প্রযুক্তি বিশ্ববিদ্যালয়",
                "Rajshahi University of Engineering and Technology",
                "RAJSHAHI UNIVERSITY OF ENGINEERING AND TECHNOLOGY"
            ]
        )

    def add_mapping(self, canonical: str, variants: List[str]):
        """
        Add entity name mapping.

        Args:
            canonical: Canonical form (uppercase, English preferred)
            variants: List of all known variations (Bangla, English, abbreviations)
        """
        self.aliases[canonical] = variants

        # Add all case variations to mapping
        for variant in variants:
            self.canonical_map[variant] = canonical
            self.canonical_map[variant.upper()] = canonical
            self.canonical_map[variant.lower()] = canonical

    def canonicalize(self, entity_name: str) -> str:
        """
        Return canonical form of entity name.

        Args:
            entity_name: Entity name to canonicalize

        Returns:
            Canonical name if mapped, otherwise original name
        """
        # Try exact match
        if entity_name in self.canonical_map:
            return self.canonical_map[entity_name]

        # Try case-insensitive
        for variant, canonical in self.canonical_map.items():
            if variant.lower() == entity_name.lower():
                return canonical

        # No mapping found - return original
        return entity_name

    def get_aliases(self, canonical_name: str) -> List[str]:
        """
        Get all aliases for a canonical entity.

        Args:
            canonical_name: Canonical entity name

        Returns:
            List of all known variants
        """
        return self.aliases.get(canonical_name, [])

    def is_canonical(self, entity_name: str) -> bool:
        """
        Check if entity name is already in canonical form.

        Args:
            entity_name: Entity name to check

        Returns:
            True if entity_name is a canonical form
        """
        return entity_name in self.aliases

    def get_all_canonicals(self) -> List[str]:
        """
        Get list of all canonical entity names.

        Returns:
            List of canonical names
        """
        return list(self.aliases.keys())

    def get_mapping_stats(self) -> Dict:
        """
        Get statistics about the mapping.

        Returns:
            {
                'total_canonicals': int,
                'total_variants': int,
                'avg_variants_per_canonical': float
            }
        """
        total_canonicals = len(self.aliases)
        total_variants = sum(len(variants) for variants in self.aliases.values())
        avg_variants = total_variants / total_canonicals if total_canonicals > 0 else 0

        return {
            'total_canonicals': total_canonicals,
            'total_variants': total_variants,
            'avg_variants_per_canonical': avg_variants
        }
