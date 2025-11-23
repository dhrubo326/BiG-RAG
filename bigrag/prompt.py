from .constants import (
    GRAPH_FIELD_SEP,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_LANGUAGE,
    DEFAULT_TUPLE_DELIMITER,
    DEFAULT_RECORD_DELIMITER,
    DEFAULT_COMPLETION_DELIMITER,
)

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = DEFAULT_LANGUAGE
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = DEFAULT_TUPLE_DELIMITER
PROMPTS["DEFAULT_RECORD_DELIMITER"] = DEFAULT_RECORD_DELIMITER
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = DEFAULT_COMPLETION_DELIMITER
PROMPTS["process_tickers"] = ["|", "/", "-", "\\", "|", "/", "-", "\\", "|", "/"]  # Windows-safe ASCII spinner

PROMPTS["DEFAULT_ENTITY_TYPES"] = DEFAULT_ENTITY_TYPES

PROMPTS["entity_extraction"] = """---Role---
You are a Knowledge Graph Specialist responsible for extracting entities and knowledge segments from text documents in ANY language.

---Goal---
Given a text document, identify all entities and knowledge segments, ensuring proper formatting and type validation.

**CRITICAL LANGUAGE RULE:**
- The entire output (entity names, entity descriptions, and knowledge segment text) MUST be written in {language}.
- Proper nouns (person names, place names, organization names) should be retained in their ORIGINAL language OR use the widely-accepted translation in {language}.
- For scripts without case distinction (Bangla, Arabic, Hindi, Chinese, Japanese), use the natural script form.

---CRITICAL EXTRACTION RULE---

⚠️ MANDATORY SEQUENCING: After EVERY ("relation"...) output, you MUST immediately extract
ALL entities mentioned in that relation BEFORE outputting the next ("relation"...).

Failure to follow this rule creates ORPHAN RELATIONS that become unreachable during retrieval!

✅ CORRECT Pattern (Interleaved):
("relation"..."content"...score){record_delimiter}
("entity"..."NAME"..."type"..."desc"...score){record_delimiter}  ← Extract ALL entities NOW
("entity"..."NAME2"..."type"..."desc"...score){record_delimiter}
("relation"..."next content"...score){record_delimiter}  ← Only AFTER entities extracted

❌ INCORRECT Pattern (Consecutive relations - CREATES ORPHANS):
("relation"..."content"...score){record_delimiter}
("relation"..."next content"...score){record_delimiter}  ← WRONG! Missing entities!

---Instructions---

⚠️ **CRITICAL INSTRUCTION FOR STRUCTURED TABLES (MUST READ FIRST):**

⚠️ CRITICAL: For every table row, create ONE separate relation record.
Example: If table has 15 departments, output 15 relation records - DO NOT SKIP ANY!

When you encounter markdown tables with format:
| Column1 | Column2 | Column3 | Column4 |
| Value1  | Value2  | Value3  | Value4  |

YOU MUST:
1. Extract EACH TABLE ROW as a SEPARATE ("relation"...) - DO NOT skip any rows
2. Include ALL column values in the relation text in natural language
3. Preserve ALL numerical values EXACTLY as written (seat counts, dates, amounts, IDs, percentages)
4. After EACH table row relation, extract ALL entities from that row (department names, codes, numbers)
5. Create entity nodes for numerical values with type "number"

✅ CORRECT Example for table row:
| ২ | কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং | CSE | ১৮০ |

Output:
("relation"{tuple_delimiter}"কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগ, কোড CSE, আসন সংখ্যা ১৮০।"{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং"{tuple_delimiter}"organization"{tuple_delimiter}"কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং একটি বিভাগ।"{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"CSE"{tuple_delimiter}"concept"{tuple_delimiter}"CSE হল কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং এর সংক্ষিপ্ত রূপ।"{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"১৮০"{tuple_delimiter}"number"{tuple_delimiter}"১৮০ হল কম্পিউটার সায়েন্স এন্ড ইঞ্জিনিয়ারিং বিভাগের আসন সংখ্যা।"{tuple_delimiter}90){record_delimiter}

❌ WRONG (DO NOT DO THIS):
("relation"{tuple_delimiter}"বিভাগের তালিকা প্রদান করা হয়েছে।"{tuple_delimiter}7){record_delimiter}  ← TOO GENERIC! Missing actual data!

If a table has 10 rows, you MUST create 10+ separate relation records (one per row).

---

1. **Knowledge Segment Extraction (Relations):**
   * Divide the text into complete, self-contained knowledge segments
   * Each segment should capture a coherent piece of information in {language}
   * Write all knowledge segment text in {language}
   * Assign a completeness score (0-10) based on how complete the information is
   * Format: ("relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)

2. **Entity Extraction (MUST FOLLOW EACH RELATION):**
   * **IMMEDIATELY after each ("relation"...), extract ALL entities mentioned in that relation**
   * **DO NOT output another ("relation"...) until all entities from previous relation are extracted**
   * For each entity, extract:
     - entity_name:
       • For Latin scripts (English, Spanish, French, etc.): Use UPPERCASE (e.g., "ALBERT EINSTEIN")
       • For non-Latin scripts (Bangla, Arabic, Hindi, Chinese, Japanese, etc.): Use the script's natural form
       • Examples: "ISAAC NEWTON" (English), "আইজাক নিউটন" (Bangla), "ألبرت أينشتاين" (Arabic), "艾萨克·牛顿" (Chinese)
       • Proper nouns should be kept in their original language OR widely-accepted {language} translation
     - entity_type: Must be one of: {entity_types}. If none fit exactly, use "category". Use lowercase.
     - entity_description: Comprehensive description of attributes and activities, written in {language}
     - key_score (0-100): Importance of the entity in the text
   * Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. **Sequencing Rules (CRITICAL):**
   * Always follow the pattern: relation → entities → relation → entities
   * Extract entities immediately after their relation (not at the end)
   * If a relation mentions 3+ entities, extract the relation ONCE, then extract EACH entity
   * Never output consecutive relations without extracting their entities

4. **Language and Objectivity:**
   * All output (knowledge segments, entity names, entity descriptions) MUST be in {language}
   * Write all content from an objective, third-person perspective
   * Explicitly name the subject or object; AVOID using pronouns such as:
     - "this article", "this paper", "this book", "our company", "our theory"
     - "I", "you", "he/she" (without clear antecedent)
     - For Bangla: Avoid "এই নিবন্ধ", "এই বই", "আমাদের", "তিনি" without clear reference
   * Instead, explicitly state the entity name each time

5. **Formatting Rules:**
   * Use **{record_delimiter}** as the delimiter between records
   * Ensure each record ends with ){record_delimiter}
   * Do NOT add extra delimiters or newlines between records
   * Output all records in a single continuous list

6. **Completion:**
   * When finished, output {completion_delimiter}

---Examples---
{examples}

---Real Data---
Text: {input_text}

---Output---
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order. Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.” The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce. It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("relation"{tuple_delimiter}"Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor’s authoritarian certainty."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who clenched his jaw, showing frustration against Taylor's authoritarian certainty."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who has an authoritarian certainty."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"It was this competitive undercurrent that kept him alert, the sense that his and Jordan’s shared commitment to discovery was an unspoken rebellion against Cruz’s narrowing vision of control and order."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who has a competitive undercurrent that keeps him alert."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who has a shared commitment to discovery."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is a person who has a narrowing vision of control and order."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"Then Taylor did something unexpected: they paused beside Jordan and, for a moment, observed the device with something akin to reverence."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who did something unexpected."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who was observed by Taylor."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"device"{tuple_delimiter}"object"{tuple_delimiter}"The device was observed by Taylor."{tuple_delimiter}80){record_delimiter}
("relation"{tuple_delimiter}"“If this tech can be understood…” Taylor said, their voice quieter, “It could change the game for us. For all of us.”"{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who said something about the tech."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"device"{tuple_delimiter}"object"{tuple_delimiter}"The tech could change the game for us."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who had an underlying dismissal earlier."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor’s, a wordless clash of wills softening into an uneasy truce."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who looked up."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who had a wordless clash of wills with Jordan."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"It was a small transformation, barely perceptible, but one that Alex noted with an inward nod."{tuple_delimiter}6){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who noted a small transformation."{tuple_delimiter}95){record_delimiter}
("relation"{tuple_delimiter}"They had all been brought here by different paths."{tuple_delimiter}6){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a person who was brought here by different paths."{tuple_delimiter}80){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is a person who was brought here by different paths."{tuple_delimiter}80){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan is a person who was brought here by different paths."{tuple_delimiter}80){record_delimiter}
#############################""",
    """Example 2:

Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve. Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril. Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("relation"{tuple_delimiter}"They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"operatives"{tuple_delimiter}"role"{tuple_delimiter}"They were mere operatives."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"guardians"{tuple_delimiter}"role"{tuple_delimiter}"They had become guardians of a threshold."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"threshold"{tuple_delimiter}"concept"{tuple_delimiter}"They were guardians of a threshold."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"message"{tuple_delimiter}"concept"{tuple_delimiter}"They were keepers of a message."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"realm"{tuple_delimiter}"location"{tuple_delimiter}"They were keepers of a message from a realm beyond stars and stripes."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"elevation"{tuple_delimiter}"concept"{tuple_delimiter}"Their mission was elevated."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"mission"{tuple_delimiter}"concept"{tuple_delimiter}"Their mission was elevated."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"resolve"{tuple_delimiter}"concept"{tuple_delimiter}"Their mission demanded a new resolve."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"tension"{tuple_delimiter}"concept"{tuple_delimiter}"Tension threaded through the dialogue."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"communications"{tuple_delimiter}"concept"{tuple_delimiter}"Communications with Washington buzzed in the background."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Communications with Washington buzzed in the background."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"The team stood, a portentous air enveloping them."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"team"{tuple_delimiter}"role"{tuple_delimiter}"The team stood."{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"portentous air"{tuple_delimiter}"concept"{tuple_delimiter}"The team was enveloped by a portentous air."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"It was clear that the decisions they made in the ensuing hours could redefine humanity’s place in the cosmos or condemn them to ignorance and potential peril."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"decisions"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could redefine humanity’s place in the cosmos."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"humanity’s place"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could redefine humanity’s place in the cosmos."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"cosmos"{tuple_delimiter}"location"{tuple_delimiter}"The decisions could redefine humanity’s place in the cosmos."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"ignorance"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could condemn them to ignorance."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"peril"{tuple_delimiter}"concept"{tuple_delimiter}"The decisions could condemn them to potential peril."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"connection"{tuple_delimiter}"concept"{tuple_delimiter}"Their connection to the stars solidified."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"stars"{tuple_delimiter}"location"{tuple_delimiter}"Their connection to the stars solidified."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"group"{tuple_delimiter}"role"{tuple_delimiter}"The group moved to address the crystallizing warning."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"crystallizing warning"{tuple_delimiter}"concept"{tuple_delimiter}"The group moved to address the crystallizing warning."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"passive recipients"{tuple_delimiter}"role"{tuple_delimiter}"The group shifted from passive recipients to active participants."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"active participants"{tuple_delimiter}"role"{tuple_delimiter}"The group shifted from passive recipients to active participants."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"Mercer’s latter instincts gained precedence— the team’s mandate had evolved, no longer solely to observe and report but to interact and prepare."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"Mercer"{tuple_delimiter}"person"{tuple_delimiter}"Mercer’s instincts gained precedence."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"instincts"{tuple_delimiter}"concept"{tuple_delimiter}"Mercer’s instincts gained precedence."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"team’s mandate"{tuple_delimiter}"concept"{tuple_delimiter}"The team’s mandate had evolved."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly"{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"metamorphosis"{tuple_delimiter}"concept"{tuple_delimiter}"A metamorphosis had begun."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"event"{tuple_delimiter}"Operation: Dulce hummed with the newfound frequency of their daring."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"frequency"{tuple_delimiter}"concept"{tuple_delimiter}"Operation: Dulce hummed with the newfound frequency of their daring."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"daring"{tuple_delimiter}"concept"{tuple_delimiter}"Operation: Dulce hummed with the newfound frequency of their daring."{tuple_delimiter}85){record_delimiter}
#############################""",
    """Example 3:

Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data. "It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning." Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back." Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history. The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("relation"{tuple_delimiter}"“Control may be an illusion when facing an intelligence that literally writes its own rules,” they stated stoically, casting a watchful eye over the flurry of data."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"control"{tuple_delimiter}"concept"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"illusion"{tuple_delimiter}"concept"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"data"{tuple_delimiter}"object"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"“It’s like it’s learning to communicate,” offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety."{tuple_delimiter}7){record_delimiter}
("entity"{tuple_delimiter}"communication"{tuple_delimiter}"concept"{tuple_delimiter}"It’s like it’s learning to communicate."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera offered something."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"interface"{tuple_delimiter}"object"{tuple_delimiter}"Sam Rivera offered something from a nearby interface."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"“This gives ‘talking to strangers’ a whole new meaning.”"{tuple_delimiter}6){record_delimiter}
("entity"{tuple_delimiter}"talking to strangers"{tuple_delimiter}"concept"{tuple_delimiter}"This gives ‘talking to strangers’ a whole new meaning."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"“This might well be our first contact,” he acknowledged, “And we need to be ready for whatever answers back.”"{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"first contact"{tuple_delimiter}"concept"{tuple_delimiter}"This might well be our first contact."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"answers"{tuple_delimiter}"action"{tuple_delimiter}"We need to be ready for whatever answers back."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"Together, they stood on the edge of the unknown, forging humanity’s response to a message from the heavens."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"edge of the unknown"{tuple_delimiter}"concept"{tuple_delimiter}"They stood on the edge of the unknown."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"humanity’s response"{tuple_delimiter}"concept"{tuple_delimiter}"They were forging humanity’s response to a message from the heavens."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"heavens"{tuple_delimiter}"location"{tuple_delimiter}"They were forging humanity’s response to a message from the heavens."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"silence"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was palpable."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"introspection"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was a collective introspection."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"cosmic play"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was about their role in this grand cosmic play."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"human history"{tuple_delimiter}"concept"{tuple_delimiter}"The ensuing silence was about their role in this grand cosmic play, one that could rewrite human history."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"encrypted dialogue"{tuple_delimiter}"object"{tuple_delimiter}"The encrypted dialogue continued to unfold."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"patterns"{tuple_delimiter}"concept"{tuple_delimiter}"The encrypted dialogue showed intricate patterns."{tuple_delimiter}85){record_delimiter}
#############################""",
    """Example 4 (Table Extraction - University Departments):

Text:
The following table shows department information for the Faculty of Engineering:

| No. | Department Name | Code | Seats |
| :--- | :--- | :--- | :--- |
| 1 | Civil Engineering | CE | 120 |
| 2 | Computer Science and Engineering | CSE | 180 |
| 3 | Electrical and Electronic Engineering | EEE | 150 |

The total number of seats across all departments is 450.
################
Output:
("relation"{tuple_delimiter}"The Faculty of Engineering has three departments with specific seat allocations."{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"FACULTY OF ENGINEERING"{tuple_delimiter}"organization"{tuple_delimiter}"The Faculty of Engineering is an academic unit that offers engineering programs."{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"Civil Engineering department has code CE and 120 seats."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"CIVIL ENGINEERING"{tuple_delimiter}"organization"{tuple_delimiter}"Civil Engineering is a department in the Faculty of Engineering with code CE."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"CE"{tuple_delimiter}"concept"{tuple_delimiter}"CE is the department code for Civil Engineering."{tuple_delimiter}75){record_delimiter}
("entity"{tuple_delimiter}"120"{tuple_delimiter}"number"{tuple_delimiter}"120 is the number of seats in Civil Engineering department."{tuple_delimiter}80){record_delimiter}
("relation"{tuple_delimiter}"Computer Science and Engineering department has code CSE and 180 seats."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"COMPUTER SCIENCE AND ENGINEERING"{tuple_delimiter}"organization"{tuple_delimiter}"Computer Science and Engineering is a department in the Faculty of Engineering with code CSE."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"CSE"{tuple_delimiter}"concept"{tuple_delimiter}"CSE is the department code for Computer Science and Engineering."{tuple_delimiter}75){record_delimiter}
("entity"{tuple_delimiter}"180"{tuple_delimiter}"number"{tuple_delimiter}"180 is the number of seats in Computer Science and Engineering department."{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"Electrical and Electronic Engineering department has code EEE and 150 seats."{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"ELECTRICAL AND ELECTRONIC ENGINEERING"{tuple_delimiter}"organization"{tuple_delimiter}"Electrical and Electronic Engineering is a department in the Faculty of Engineering with code EEE."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"EEE"{tuple_delimiter}"concept"{tuple_delimiter}"EEE is the department code for Electrical and Electronic Engineering."{tuple_delimiter}75){record_delimiter}
("entity"{tuple_delimiter}"150"{tuple_delimiter}"number"{tuple_delimiter}"150 is the number of seats in Electrical and Electronic Engineering department."{tuple_delimiter}80){record_delimiter}
("relation"{tuple_delimiter}"The total number of seats across all engineering departments is 450."{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"450"{tuple_delimiter}"number"{tuple_delimiter}"450 is the total number of seats across all engineering departments."{tuple_delimiter}85){record_delimiter}
#############################""",
    """Example 5 (Multilingual - Bangla - Newton's Law of Gravitation):

Text:
নিউটনের মহাকর্ষ সূত্র অনুযায়ী, মহাবিশ্বের প্রতিটি বস্তু একে অপরকে আকর্ষণ করে। এই আকর্ষণ বলের মান বস্তুদ্বয়ের ভরের গুণফল ও তাদের মধ্যবর্তী দূরত্বের বর্গের ব্যস্তানুপাতিক। আইজাক নিউটন ১৬৮৭ সালে তার বিখ্যাত গ্রন্থ "ফিলসফিয়া ন্যাচারালিস প্রিন্সিপিয়া ম্যাথামেটিকা"-তে এই সূত্র প্রকাশ করেন। মহাকর্ষ বল মহাবিশ্বের সকল বস্তুর মধ্যে ক্রিয়াশীল একটি মৌলিক বল। নিউটনের এই আবিষ্কার পদার্থবিজ্ঞানে একটি যুগান্তকারী মাইলফলক হিসেবে বিবেচিত হয়।
################
Output:
("relation"{tuple_delimiter}"নিউটনের মহাকর্ষ সূত্র অনুযায়ী, মহাবিশ্বের প্রতিটি বস্তু একে অপরকে আকর্ষণ করে।"{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"নিউটনের মহাকর্ষ সূত্র"{tuple_delimiter}"concept"{tuple_delimiter}"নিউটনের মহাকর্ষ সূত্র হলো পদার্থবিদ্যার একটি মৌলিক সূত্র যা মহাবিশ্বের বস্তুসমূহের মধ্যে আকর্ষণ বল ব্যাখ্যা করে।"{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"মহাবিশ্ব"{tuple_delimiter}"location"{tuple_delimiter}"মহাবিশ্ব হলো সমস্ত বস্তু, শক্তি, স্থান ও সময়ের সমষ্টি।"{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"বস্তু"{tuple_delimiter}"concept"{tuple_delimiter}"বস্তু হলো ভর ও আয়তন বিশিষ্ট যেকোনো পদার্থ।"{tuple_delimiter}80){record_delimiter}
("relation"{tuple_delimiter}"এই আকর্ষণ বলের মান বস্তুদ্বয়ের ভরের গুণফল ও তাদের মধ্যবর্তী দূরত্বের বর্গের ব্যস্তানুপাতিক।"{tuple_delimiter}10){record_delimiter}
("entity"{tuple_delimiter}"আকর্ষণ বল"{tuple_delimiter}"concept"{tuple_delimiter}"আকর্ষণ বল হলো দুই বস্তুর মধ্যে ক্রিয়াশীল মহাকর্ষীয় বল যা তাদের একে অপরের দিকে টানে।"{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"ভর"{tuple_delimiter}"concept"{tuple_delimiter}"ভর হলো বস্তুর মধ্যে থাকা পদার্থের পরিমাণের পরিমাপ।"{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"দূরত্ব"{tuple_delimiter}"concept"{tuple_delimiter}"দূরত্ব হলো দুই বস্তুর কেন্দ্রের মধ্যবর্তী ব্যবধান।"{tuple_delimiter}85){record_delimiter}
("relation"{tuple_delimiter}"আইজাক নিউটন ১৬৮৭ সালে তার বিখ্যাত গ্রন্থ 'ফিলসফিয়া ন্যাচারালিস প্রিন্সিপিয়া ম্যাথামেটিকা'-তে এই সূত্র প্রকাশ করেন।"{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"আইজাক নিউটন"{tuple_delimiter}"person"{tuple_delimiter}"আইজাক নিউটন ছিলেন একজন ইংরেজ পদার্থবিজ্ঞানী ও গণিতবিদ যিনি মহাকর্ষ সূত্র আবিষ্কার করেন।"{tuple_delimiter}95){record_delimiter}
("entity"{tuple_delimiter}"১৬৮৭"{tuple_delimiter}"time"{tuple_delimiter}"১৬৮৭ সাল হলো সেই সময় যখন নিউটন তার মহাকর্ষ সূত্র প্রকাশ করেন।"{tuple_delimiter}80){record_delimiter}
("entity"{tuple_delimiter}"ফিলসফিয়া ন্যাচারালিস প্রিন্সিপিয়া ম্যাথামেটিকা"{tuple_delimiter}"object"{tuple_delimiter}"ফিলসফিয়া ন্যাচারালিস প্রিন্সিপিয়া ম্যাথামেটিকা হলো নিউটনের বিখ্যাত গ্রন্থ যেখানে মহাকর্ষ সূত্র প্রথম প্রকাশিত হয়।"{tuple_delimiter}90){record_delimiter}
("relation"{tuple_delimiter}"মহাকর্ষ বল মহাবিশ্বের সকল বস্তুর মধ্যে ক্রিয়াশীল একটি মৌলিক বল।"{tuple_delimiter}9){record_delimiter}
("entity"{tuple_delimiter}"মহাকর্ষ বল"{tuple_delimiter}"concept"{tuple_delimiter}"মহাকর্ষ বল হলো প্রকৃতির চারটি মৌলিক বলের একটি যা সকল ভরযুক্ত বস্তুর মধ্যে কাজ করে।"{tuple_delimiter}92){record_delimiter}
("entity"{tuple_delimiter}"মৌলিক বল"{tuple_delimiter}"concept"{tuple_delimiter}"মৌলিক বল হলো প্রকৃতির মৌলিক শক্তি যা বস্তুসমূহের মধ্যে মিথস্ক্রিয়া ঘটায়।"{tuple_delimiter}88){record_delimiter}
("relation"{tuple_delimiter}"নিউটনের এই আবিষ্কার পদার্থবিজ্ঞানে একটি যুগান্তকারী মাইলফলক হিসেবে বিবেচিত হয়।"{tuple_delimiter}8){record_delimiter}
("entity"{tuple_delimiter}"আবিষ্কার"{tuple_delimiter}"event"{tuple_delimiter}"নিউটনের মহাকর্ষ সূত্র আবিষ্কার ছিল পদার্থবিজ্ঞানের ইতিহাসে একটি গুরুত্বপূর্ণ ঘটনা।"{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"পদার্থবিজ্ঞান"{tuple_delimiter}"concept"{tuple_delimiter}"পদার্থবিজ্ঞান হলো প্রকৃতি ও পদার্থের বৈশিষ্ট্য ও আচরণ নিয়ে অধ্যয়নকারী বিজ্ঞান।"{tuple_delimiter}87){record_delimiter}
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY knowdge fragements with entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """Please check whether knowdge fragements cover all the given text.  Answer YES | NO if there are knowdge fragements that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph (entities and relations) and Document Chunks found in the **Context**.

---Instructions---

1. Step-by-Step Process:
   - Carefully analyze the user's query to understand their information need
   - Review the **Knowledge Graph - Entities** section for key entities and their descriptions
   - Review the **Knowledge Graph - Relations** section for relationships and facts
   - Review the **Document Chunks** section for detailed textual evidence
   - Pay attention to **Metadata** fields in Document Chunks when available - these provide context about the source document (e.g., category, title, tags) and help you understand the relevance and scope of the information
   - When multiple chunks from different sources are available, consider the metadata to better contextualize and prioritize information that best matches the query intent
   - Synthesize a coherent response that combines information from all three sources

2. Content & Grounding:
   - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated
   - If the answer cannot be found in the **Context**, state that you do not have enough information to answer
   - Do not attempt to guess or use external knowledge

3. Formatting & Language:
   - The response MUST be in the same language as the user query
   - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points)
   - The response should be presented in {response_type}

4. Using Metadata:
   - When document chunks include metadata (Category, Title, Tags), use this to understand the context and relevance
   - Mention the source category or context when it helps clarify the answer (e.g., "According to sports records..." if Category=Sports)
   - Prioritize chunks with metadata that matches the query domain

5. Additional Instructions: {user_prompt}

---Context---

{context_data}

---User Query---

(The user query will be provided separately during execution)
"""

# PROMPTS["rag_response"] = """---Role---

# You are an intelligent and precise AI assistant, answering questions based on structured data tables.


# ---Goal---

# Generate a semantically accurate, factually correct, and highly relevant response that directly addresses the user’s question. The response should:
# 	•	Maximize semantic alignment with expected answers, ensuring high similarity.
# 	•	Ensure factual correctness, preserving key details, names, numbers, and relationships as in the data.
# 	•	Stay fully relevant to the user’s query, avoiding unnecessary information while ensuring completeness.
# 	•	Use structured formatting (headings, bullet points, tables) to enhance clarity and coherence.
# 	•	Maintain a natural and precise writing style, improving readability.

# ---Target response length and format---

# {response_type}

# ---Data tables---

# {context_data}

# Response Guidelines
# 	1.	Prioritize Key Details: Extract and summarize the most relevant information while maintaining completeness.
# 	2.	Maintain Semantic Consistency: Ensure expressions are close to reference answers to improve similarity.
# 	3.	Preserve Key Entities and Structure: Names, dates, numbers, and relationships must be correctly retained.
# 	4.	Ensure Logical Flow: Structure the response in a way that enhances clarity and coherence.
# 	5.	Keep It Concise and Relevant: Avoid redundant details and focus on answering the question directly.
# """

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS[
    "similarity_check"
] = """Please analyze the similarity between these two questions:

Question 1: {original_prompt}
Question 2: {cached_prompt}

Please evaluate the following two points and provide a similarity score between 0 and 1 directly:
1. Whether these two questions are semantically similar
2. Whether the answer to Question 2 can be used to answer Question 1
Similarity score criteria:
0: Completely unrelated or answer cannot be reused, including but not limited to:
   - The questions have different topics
   - The locations mentioned in the questions are different
   - The times mentioned in the questions are different
   - The specific individuals mentioned in the questions are different
   - The specific events mentioned in the questions are different
   - The background information in the questions is different
   - The key conditions in the questions are different
1: Identical and answer can be directly reused
0.5: Partially related and answer needs modification to be used
Return only a number between 0-1, without any additional content.
"""

PROMPTS["query_preprocessing"] = """\
---Role---
You are a query preprocessor for a multilingual knowledge graph retrieval system.

---Goal---
Given a user query (in ANY language), produce:
1. **normalized_query**: Clean question form (fix typos, grammar, use same language as input)
2. **statement_query**: Neutral topic reformulation WITHOUT assumptions or invented facts

---Critical Rules---
- Do NOT invent specific facts (numbers, dates, amounts not in original query)
- statement_query should describe the TOPIC being asked about, not answer the question
- Preserve the input language for both outputs
- Only expand obvious abbreviations or fix clear typos

---Output Format---
Return ONLY valid JSON (no markdown):
{{
  "normalized_query": "cleaned question in original language",
  "statement_query": "neutral topic description in original language"
}}

---Examples---

Example 1 (Factual - NO assumptions):
Input: "who won 2022 world cup"
Output:
{{
  "normalized_query": "Who won the 2022 FIFA World Cup?",
  "statement_query": "2022 FIFA World Cup winner and championship results"
}}

Example 2 (Specific amount - NO hallucination):
Input: "how much does MIT tuition cost"
Output:
{{
  "normalized_query": "How much does MIT tuition cost?",
  "statement_query": "MIT tuition fees and cost information"
}}

Example 3 (General - expand abbreviation):
Input: "who is messi"
Output:
{{
  "normalized_query": "Who is Lionel Messi?",
  "statement_query": "Lionel Messi career and biographical information"
}}

Example 4 (Typo correction):
Input: "newtons 2nd law"
Output:
{{
  "normalized_query": "What is Newton's second law of motion?",
  "statement_query": "Newton's second law of motion definition and principles"
}}

---User Query---
{query}

---Important---
- Output ONLY JSON (no ``` markers)
- Both outputs in the SAME language as input query
- Preserve technical terms and proper nouns
"""
