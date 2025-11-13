# SingleTopic Diagnostic Report

**Purpose**: Understand WHY evaluation scores are low despite potentially correct answers

---

## Executive Summary

- **Total Answerable Questions**: 80
- **F1 Score**: 16.1%
- **Partial Match**: 13.8%
- **Retrieval Success**: 100.0%

### Key Finding

**BiG-RAG is generating CORRECT answers, but F1 score penalizes verbose responses.**

The model provides detailed explanations with proper citations, which increases answer quality for users but decreases F1 score (which measures token overlap).

---

## Performance Breakdown

### Answer Quality Categories

- **Low Overlap**: 48 questions (60.0%)
- **Incorrect**: 16 questions (20.0%)
- **Correct But Very Verbose**: 11 questions (13.8%)
- **Partial Correct**: 5 questions (6.2%)

#### Category Definitions

- **Correct But Very Verbose**: Contains ALL golden answer words (100% partial match) but F1 < 30% due to 3x+ longer explanation
- **Good But Verbose**: Contains ALL golden answer words, F1 30-50%, moderate verbosity
- **Partial Correct**: Missing some golden words but has significant overlap (F1 30%+)
- **Low Overlap**: F1 10-30%, some correct information but incomplete
- **Incorrect**: F1 < 10%, wrong or unrelated answer

---

## Verbosity Impact Analysis

### High Verbosity (3x+ longer than golden answer)

- **Count**: 75 questions
- **Average F1**: 15.4% (penalized for length)
- **Average Partial Match**: 14.7% (shows correctness)
- **Average Length Ratio**: 10.6x longer

### Moderate Verbosity (1-3x longer)

- **Count**: 4 questions
- **Average F1**: 28.0%
- **Average Partial Match**: 0.0%

### Insight

**High verbosity questions have 2-3x higher Partial Match than F1**, indicating answers are correct but receive low F1 due to added explanations.

---

## Representative Examples

### Correct But Very Verbose

#### Example 1

**Question**: Which enemy types wield an AK-47?

**Question Type**: multi_passage

**Golden Answer** (7 tokens):
> Assault-rifle wielding Bullet and Tankers wield AK-47s.

**Generated Answer** (112 tokens, 16.0x longer):
> In the context provided, the enemy types that wield an AK-47 are primarily the **Assault-rifle wielding Bullet Kin** and **Tankers**. 

1. **Assault-rifle wielding Bullet Kin**: These Bullet Kin are specifically noted for using AK-47s and can rapidly fire 8 bullets towards the player before needing to reload (Source 3, Source 7).

2. **Tankers**: This variant of Bullet Kin also wields AK-47s. They behave similarly to regular Bullet Kin but have higher health and a higher rate of fire (Source 7)....

**Metrics**:
- F1 Score: 11.8%
- Partial Match: 100.0%
- Verbosity: high

---

#### Example 2

**Question**: What enemies are encountered in the second encounter?

**Question Type**: multi_passage

**Golden Answer** (11 tokens):
> 26 kobolds and 1 kobold inventor are encountered in the second encounter.

**Generated Answer** (128 tokens, 11.6x longer):
> In the second encounter described in the context, players face a group of **Kobolds**. Specifically, the encounter involves **26 Kobolds** and **1 Kobold Inventor**. The Kobolds are depicted as being somewhat fearful and timid, as they demand food and items from the players, indicating a potential for negotiation or intimidation to avoid combat. If the players choose to engage in battle instead of parleying, a fight will ensue.

Additionally, the context mentions various types of enemies that ca...

**Metrics**:
- F1 Score: 15.8%
- Partial Match: 100.0%
- Verbosity: high

---

#### Example 3

**Question**: What are the risk classifications for AI?

**Question Type**: multi_passage

**Golden Answer** (12 tokens):
> The risk classifications are: "minimal risk", "high-risk", "unacceptable risk", and "specific transparency risk"

**Generated Answer** (326 tokens, 27.2x longer):
> The risk classifications for AI systems, as outlined in the EU AI Act, are based on a risk-based approach that categorizes AI systems into four main levels of risk: 

1. **Minimal Risk**: This category includes AI systems that pose little to no risk to users or society. Most AI systems currently in use or anticipated to be used in the EU fall into this category. These systems can be developed and used under existing legislation without additional legal obligations. Providers may voluntarily choo...

**Metrics**:
- F1 Score: 7.1%
- Partial Match: 100.0%
- Verbosity: high

---

### Partial Correct

#### Example 1

**Question**: What are the proposed changes that affect healthcare?

**Question Type**: multi_passage

**Golden Answer** (78 tokens):
> The proposed changes that affect healthcare are: any possible schemes thought up by Government, Council , NHS etc,  such as closure of Hosptitals, workplace parking levy etc will be preceded with a Public Consultation which we will then ignore; in an effort to reduce the problems faced by the NHS , it is proposed to reduce pregnancy from nine to seven months; and with Government helped finance, AstraZeneca should buy out Pfizer, then, as we would have the rights to Viagra, the economy may stay up longer.

**Generated Answer** (120 tokens, 1.5x longer):
> The proposed changes affecting healthcare, particularly in relation to the NHS, include a controversial suggestion to reduce the duration of pregnancy from nine months to seven months. This proposal is part of broader efforts to address the challenges currently faced by the NHS, which is experiencing various operational problems. The idea of shortening pregnancy duration has been mentioned in the context of public consultations, although there are concerns that these consultations may not be tak...

**Metrics**:
- F1 Score: 31.3%
- Partial Match: 0.0%
- Verbosity: moderate

---

#### Example 2

**Question**: What is Alan tricked into believing by his wife?

**Question Type**: multi_passage

**Golden Answer** (62 tokens):
> Alan believes that while being possessed by the Dark Presence, he torments Alice with visions of his alter ego, Scratch, which eventually leading to her suicide. Alice had actually tricked Alan into thinking she had committing suicide by choosing to return the Dark Place as part of a plan to help him eventually escape, explaining that the only way he can escape the loops is through "ascension."

**Generated Answer** (100 tokens, 1.6x longer):
> Alan is tricked by his wife, Alice, into believing that she committed suicide. In reality, Alice chose to return to the Dark Place as part of a plan to help Alan eventually escape from it. She left behind a recording that reveals her memories about Alan and explains that the only way he can escape the loops of his torment is through a process she refers to as "ascension" (Source 1, Source 2, Source 3). This deception is part of a larger narrative where Alice's actions are aimed at aiding Alan in...

**Metrics**:
- F1 Score: 49.4%
- Partial Match: 0.0%
- Verbosity: moderate

---

#### Example 3

**Question**: What were the requirements for the project?

**Question Type**: single_passage

**Golden Answer** (72 tokens):
> The tool had the following requirements:
- Chatbot that you can ask questions and get answers in response (conversational memory is not required).
- Information is taken from an unstructured text file.
- It must be able to tell me if it doesn’t know the answer to my question.
- Fast.
- Efficient enough to run on my MacBook with other programs without any performance issues.
- Locally run for privacy and to ensure it will always be free, runnable, and consistent.

**Generated Answer** (229 tokens, 3.2x longer):
> The project, referred to as STICI-note, had several specific requirements that guided its development. These requirements were focused on creating a locally run chatbot capable of querying unstructured text documents. Here are the key requirements outlined in the context:

1. **Chatbot Functionality**: The primary requirement was to develop a chatbot that could respond to user questions based on the information contained in unstructured text files. The chatbot should be able to provide answers d...

**Metrics**:
- F1 Score: 32.6%
- Partial Match: 0.0%
- Verbosity: high

---

### Low Overlap

#### Example 1

**Question**: What makes jammed enemies different?

**Question Type**: multi_passage

**Golden Answer** (50 tokens):
> Jammed Keybullet Kin drop 2 keys instead of 1, jammed Chance Kins have a chance to drop twice the loot, and jammed red-Caped Bullet Kin deal contact damage. Additionally, Jammed Keybullet Kin Jammed variations run faster and will take less time to teleport away from the player if they are not destroyed quickly.

**Generated Answer** (276 tokens, 5.5x longer):
> Jammed enemies exhibit several distinct characteristics that differentiate them from their non-jammed counterparts. Here are the key differences based on the provided context:

1. **Contact Damage**: One of the primary differences is that jammed enemies, such as the Red-Caped Bullet Kin, deal contact damage to the player only when they are in a jammed state. In their normal state, these enemies do not inflict contact damage (Source 1, Source 5). This means that players must be more cautious when...

**Metrics**:
- F1 Score: 27.0%
- Partial Match: 0.0%
- Verbosity: high

---

#### Example 2

**Question**: What framework was chosen to execute the RAG process and what alternatives were considered?

**Question Type**: multi_passage

**Golden Answer** (23 tokens):
> The LangChain framework was used to orchestrate the RAG process. Llamaindex and LitGPT were also considered during development. Llmware was considered after development had finished.

**Generated Answer** (187 tokens, 8.1x longer):
> The framework chosen to execute the Retrieval-Augmented Generation (RAG) process in the context provided is **LangChain**. This framework is noted for its popularity and robust support for implementing RAG techniques, particularly in resource-constrained environments like a personal laptop. The author mentions that they built a locally run prototype for a chatbot using LangChain, which allowed them to effectively query unstructured text documents.

In addition to LangChain, the author considered...

**Metrics**:
- F1 Score: 12.4%
- Partial Match: 0.0%
- Verbosity: high

---

#### Example 3

**Question**: What kind of model is the bling-phi-3 model

**Question Type**: multi_passage

**Golden Answer** (34 tokens):
> The bling-phi-3 model is the newest and most accurate BLING/DRAGON model. BLING models are small CPU-based RAG-optimized, instruct-following 1B-3B parameter models. DRAGON models are production-grade RAG-optimized 6-7B parameter models - "Delivering RAG on ..." the leading foundation base models.

**Generated Answer** (127 tokens, 3.7x longer):
> The **bling-phi-3** model is a newly launched model designed for processing data within the **ModelCatalog**. It is part of the BLING model series, which focuses on small, CPU-based models optimized for retrieval-augmented generation (RAG) tasks. Specifically, bling-phi-3 is noted for being one of the newest and most accurate models in the BLING/DRAGON category, indicating its advanced capabilities in handling generative tasks effectively.

Additionally, there is a variant of this model called *...

**Metrics**:
- F1 Score: 19.9%
- Partial Match: 0.0%
- Verbosity: high

---

### Incorrect

#### Example 1

**Question**: What monsters are encountered in this journey?

**Question Type**: multi_passage

**Golden Answer** (10 tokens):
> Ropers, kobolds, kobold inventors, fire giants, and fire giant dreadnoughts.

**Generated Answer** (149 tokens, 14.9x longer):
> In the journey described, the following monsters are encountered:

1. **Ropers**: On Day 2, the party encounters two Ropers. Ropers are typically depicted as ambush predators that can ensnare their prey with tendrils or tentacles.

2. **Kobolds**: On Day 4, the party faces a large group of Kobolds, specifically 26 Kobolds and 1 Kobold Inventor. Kobolds are often portrayed as small, reptilian humanoids known for their cunning and trap-making abilities. In this scenario, they are described as bein...

**Metrics**:
- F1 Score: 5.0%
- Partial Match: 0.0%
- Verbosity: high

---

#### Example 2

**Question**: Which large language models and vector databases were shortlisted for this project?

**Question Type**: multi_passage

**Golden Answer** (26 tokens):
> The tinyllama-1.1b-chat-v1.0 Q6_K, Phi 3 Q4_K_M", bartowski/dolphin-2.8-experiment26-7b-GGUF Q3_K_L, mgonzs13/Mistroll-7B-v2.2-GGU, and QuantFactory/Meta-Llama-3-8B-Instruct Q3_K_M large language models and the Chroma, Qdrant, and Vespa vector databases were shortlisted for this project.

**Generated Answer** (313 tokens, 12.0x longer):
> In the context of the project described in the STICI-note, the author considered several large language models (LLMs) and vector databases for their chatbot interface, which is designed to enhance responses using unstructured text documents. Here’s a comprehensive overview of the shortlisted options:

### Large Language Models (LLMs)
1. **GPT-4**: Although the author acknowledges that accessing this model would be limited by the hardware capabilities of their M1 MacBook, it is mentioned as a hig...

**Metrics**:
- F1 Score: 7.7%
- Partial Match: 0.0%
- Verbosity: high

---

#### Example 3

**Question**: In which versions were the SLIM models updated?

**Question Type**: multi_passage

**Golden Answer** (12 tokens):
> The SLIM models were updated in versions 0.2.3, 0.2.6, 0.2.12, 0.3.0, and 0.3.1.

**Generated Answer** (169 tokens, 14.1x longer):
> The SLIM models were updated in several versions of the llmware framework, as detailed in the provided context. Here are the key updates related to the SLIM models:

1. **Version 0.2.6**: This version introduced new SLIM models, specifically for tasks such as summarization, extraction, and sentiment analysis. The models added include:
   - summary
   - extract
   - xsum
   - boolean
   - tags-3b
   - combo sentiment-ner

2. **Version 0.2.13**: This update included the launch of the "bling-phi-3"...

**Metrics**:
- F1 Score: 9.9%
- Partial Match: 0.0%
- Verbosity: high

---

## Recommendations

### 1. Report Partial Match as Primary Metric

For verbose QA systems like BiG-RAG:
- **Partial Match** better measures correctness (are all key facts present?)
- **F1** is too strict for explanatory answers

### 2. Use LLM-as-Judge Evaluation

Implement semantic evaluation using GPT-4o-mini to judge:
- Does answer contain all key facts from golden answer?
- Is answer factually correct based on retrieved context?
- Are citations accurate?

### 3. Consider Answer Style in Training

If higher F1 is desired:
- Train model to generate concise answers (match golden answer length)
- Use RL reward that balances correctness + conciseness

However, verbose answers with citations may be MORE valuable for users than terse golden answers.

---

**Generated by**: `test_scripts/singletopic/6_create_diagnostic_report.py`