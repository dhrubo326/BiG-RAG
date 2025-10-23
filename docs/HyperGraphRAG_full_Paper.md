# HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation

## Authors

**Haoran Luo¹'²**, **Haihong E¹*** (Corresponding author), **Guanting Chen¹**, **Yandan Zheng²**, **Xiaobao Wu²**, **Yikai Guo³**, **Qika Lin⁴**, **Yu Feng¹**, **Zemin Kuang⁵**, **Meina Song¹**, **Yifan Zhu¹**, **Luu Anh Tuan²**

¹ Beijing University of Posts and Telecommunications
² Nanyang Technological University
³ Beijing Institute of Computer Technology and Application
⁴ National University of Singapore
⁵ Beijing Anzhen Hospital, Capital Medical University

**Contact:** {luohaoran, ehaihong, yifan_zhu}@bupt.edu.cn, anhtuan.luu@ntu.edu.sg

**arXiv:** 2503.21322v2 [cs.AI] 22 May 2025
**Status:** Preprint. Under review.
**Code & Data:** https://github.com/LHRLAB/HyperGraphRAG

---

## Abstract

Standard Retrieval-Augmented Generation (RAG) relies on chunk-based retrieval, whereas GraphRAG advances this approach by graph-based knowledge representation. However, existing graph-based RAG approaches are constrained by binary relations, as each edge in an ordinary graph connects only two entities, limiting their ability to represent the n-ary relations (n ≥ 2) in real-world knowledge.

In this work, we propose **HyperGraphRAG**, a novel hypergraph-based RAG method that represents n-ary relational facts via hyperedges, and consists of knowledge hypergraph construction, retrieval, and generation. Experiments across medicine, agriculture, computer science, and law demonstrate that HyperGraphRAG outperforms both standard RAG and previous graph-based RAG methods in answer accuracy, retrieval efficiency, and generation quality.

---

## 1. Introduction

Retrieval-Augmented Generation (RAG) has advanced knowledge-intensive tasks by integrating knowledge retrieval with large language models (LLMs), thereby enhancing factual awareness and generation accuracy. Standard RAG typically relies on chunk-based retrieval, segmenting documents into fixed-length text chunks retrieved via dense vector similarity, which overlooks the relationships between entities. Recently, GraphRAG has emerged as a promising direction that structures knowledge as a graph to capture inter-entity relations, with the potential to improve retrieval efficiency and knowledge-driven generation.

### Figure 1: HyperGraphRAG System Overview

**Description:** This figure illustrates the complete HyperGraphRAG pipeline:
- **Input:** Domain knowledge documents and an LLM for processing
- **Step 1 - N-ary Relation Extraction:** The LLM extracts complex multi-entity relationships from domain knowledge
- **Step 2 - Knowledge HyperGraph:** A hypergraph structure is created where:
  - Nodes (v1-v7) represent entities
  - Hyperedges (e1-e4, shown as colored regions) connect multiple entities simultaneously, representing n-ary relations
- **Step 3 - User Question:** A user poses a question
- **Step 4 - HyperGraph Retrieval:** The system retrieves relevant hyperedges and entities based on the question
- **Step 5 - HyperGraph-Guided Generation:** The LLM generates a knowledgeable answer using retrieved n-ary relational facts
- **Output:** A comprehensive, factually grounded response

### The Limitation of Binary Relations

However, since each edge in an ordinary graph connects only two entities, existing graph-based RAG approaches are all restricted to **binary relations**, making them insufficient for modeling the **n-ary relations** among more than two entities that are widespread in real-world domain knowledge.

**Medical Example:** In the medical domain, representing the fact that *"Male hypertensive patients with serum creatinine levels between 115–133 µmol/L are diagnosed with mild serum creatinine elevation"* requires decomposing it into several binary relational triples, such as:
- `Gender:(Hypertensive patient, Male)`
- `Diagnosed_with:(Hypertensive patient, Mild serum creatinine elevation)`

This decomposition leads to **representation loss and sparsity** during the conversion.

### HyperGraphRAG Solution

To address these limitations, we propose **HyperGraphRAG**, a novel graph-based RAG method built upon hypergraph-structured knowledge representation. In contrast to prior graph-based RAG methods constrained to binary relations, HyperGraphRAG leverages **hyperedges** to represent n-ary relational facts, where each hyperedge connects n entities (n ≥ 2).

**Example:**
```
Hyperedge:(Hypertensive patient, Male, Serum creatinine levels between 115–133 µmol/L, Mild serum creatinine elevation)
```

Each hyperedge is expressed through **natural language descriptions**. This design ensures:
- **Knowledge completeness**
- **Structural expressiveness**
- **Inferential capability**

Thereby providing more comprehensive support for knowledge-intensive applications.

### Three Key Components

Our proposed HyperGraphRAG is built upon three key steps:

1. **Knowledge Hypergraph Construction:** We leverage LLM-based n-ary relation extraction to extract and structure multi-entity relationships. The resulting hypergraph is stored in a bipartite graph database, with separate vector databases for entities and hyperedges to facilitate efficient retrieval.

2. **Hypergraph Retrieval Strategy:** We employ vector similarity search to retrieve relevant entities and hyperedges, ensuring that the knowledge retrieved is both precise and contextually relevant.

3. **Hypergraph-Guided Generation:** We combine retrieved n-ary facts with traditional chunk-based RAG passages, thereby improving response quality.

### Experimental Validation

We conduct experiments in multiple knowledge-intensive domains, including medicine, agriculture, computer science, and law. Results demonstrate that HyperGraphRAG outperforms standard RAG and previous graph-based RAG methods in:
- **Answer accuracy**
- **Retrieval efficiency**
- **Generation quality**

Showcasing its strong potential for real-world applications.

---

## 2. Related Work

### 2.1 Graph-based RAG

**GraphRAG** is the first graph-based RAG method that improves LLM generation via graph-based retrieval. Based on GraphRAG, several methods focus on building graph-based RAG for different applications:

- **LightRAG:** Enhances efficiency via graph indexing and updates
- **PathRAG:** Refines retrieval with path pruning
- **HippoRAG2:** Uses Personalized PageRank for retrieval

However, **all rely on binary relations**, limiting knowledge expressiveness. In this work, we propose HyperGraphRAG, the **first graph-based RAG method via hypergraph-structured knowledge representation**.

### 2.2 Hypergraph Representation

Hypergraph-structured knowledge representation aims to overcome ordinary graph's limitations in modeling n-ary relations:

- **Early methods:** Employ various embedding techniques to represent n-ary relational entities
- **Later methods:** Utilize GNN or attention mechanisms to enhance embedding

However, existing methods mainly focus on **link prediction**, while hypergraphs also show potential for enhancing knowledge representation in graph-based RAG.

### Figure 2: Comparison of Knowledge Representations

**Description:** This figure compares three knowledge representation approaches using a medical example:

**Domain Knowledge (Top):**
*"Male hypertensive patients with serum creatinine levels between 115–133 µmol/L are diagnosed with mild serum creatinine elevation."*

**Three Approaches (Left to Right):**

1. **Standard RAG - Chunk-based Knowledge Representation:**
   - Text is divided into fixed-length chunks
   - Each chunk is treated as an independent unit
   - No explicit relationships between entities
   - Multiple text boxes containing fragments of the original statement

2. **GraphRAG - Ordinary Graph Knowledge Representation:**
   - Entities shown as nodes: "Hypertensive patient", "Male", "Mild serum creatinine elevation", "Serum creatinine levels between 115–133 µmol/L"
   - Binary edges connect pairs of entities
   - The complex 4-way relationship is decomposed into multiple binary relationships
   - Loss of semantic integrity - the holistic n-ary relationship is fragmented

3. **HyperGraphRAG - Hypergraph-Structured Knowledge Representation:**
   - Same entities as nodes
   - Multiple hyperedges (Hyperedge_1, Hyperedge_2, Hyperedge_3, Hyperedge_4) shown as colored regions
   - Each hyperedge can connect multiple entities simultaneously
   - The complete 4-way relationship can be captured in a single hyperedge
   - Preserves the semantic integrity of the n-ary relation

**Key Insight:** The figure demonstrates that hypergraph representation can capture complex multi-entity relationships without decomposition, avoiding the representation loss inherent in binary graph structures.

---

## 3. Preliminaries

### Definition 1: RAG (Retrieval-Augmented Generation)

Given a question **q** and domain knowledge **K**, standard RAG first selects relevant document fragments **d** from **K** based on **q**, and then generates an answer **y** based on **q** and **d**. The probability model is formulated as:

```
P(y|q) = Σ P(y|q, d)P(d|q, K)
        d∈K
```

### Definition 2: Graph-based RAG

Graph-based RAG optimizes retrieval by representing knowledge as a graph structure **G = (V, E)**, where:
- **V** is the set of entities
- **E** is the set of relationships between entities

**G** consists of facts represented as **F = (e, Vₑ) ∈ G**, where:
- **e** is the relation
- **Vₑ** is the entity set connected to **e**

Given a question **q**, the retrieval process is defined as:

```
P(y|q) = Σ P(y|q, F)P(F|q, G)
        F∈G
```

### Definition 3: Hypergraph

A hypergraph **G_H = (V, E_H)** is a generalized graph, where:
- **V** is the entity set
- **E_H** is the hyperedge set
- Each hyperedge **e_H ∈ E_H** connects 2 or more entities:

```
V_{e_H} = (v₁, v₂, ..., vₙ), where n ≥ 2
```

**Key Difference:** Unlike ordinary graphs, where relationships are binary **Vₑ = (v_h, v_t)**, hypergraphs model n-ary relational facts **F_n = (e_H, V_{e_H}) ∈ G_H**.

---

## 4. Method: HyperGraphRAG

In this section, we introduce the proposed HyperGraphRAG, including:
1. Knowledge hypergraph construction
2. Hypergraph retrieval strategy
3. Hypergraph-guided generation

### Figure 3: Complete HyperGraphRAG System Architecture

**Description:** This detailed system architecture diagram shows the complete pipeline:

**Left Section - Knowledge HyperGraph Construction:**
- **Input:** Domain knowledge files and knowledge documents
- **N-ary Relation Extraction:** LLM with prompts extracts structured n-ary relational facts
- **Example Output:**
  ```xml
  <hyperedge>(text:"Hypertension is defined as an office systolic blood pressure ≥140 mmHg or diastolic blood pressure ≥90 mmHg.", score:"9")
  - <entity>(name:"Hypertension", type:"Disease", explain:"Hypertension is defined as systolic blood pressure ≥140 mmHg or diastolic blood pressure ≥90 mmHg.", score:"95")
  - <entity>(name:"Systolic blood pressure ≥140 mmHg", type:"Measurement", explain:"...", score:"85")
  - <entity>(name:"Diastolic blood pressure ≥90 mmHg", type:"Measurement", explain:"...", score:"85")
  ```

**Middle Section - Bipartite Hypergraph Storage:**
- **Visual Representation:** A graph showing:
  - Entity nodes (circles): "Hypertension", "Systolic BP ≥140 mmHg", "Diastolic BP ≥90 mmHg", "Elevated BP", "Stroke", "Cardiovascular Disease", "Low-sodium Diet", "Regular Exercise", "Stress Management", "Ejection Fraction", "BNP", "Heart Failure"
  - Hyperedge nodes (larger circles): Hyperedge_1, Hyperedge_2, Hyperedge_3, Hyperedge_4
  - Edges connecting hyperedges to their associated entities
- **Vector Storage:**
  - Entity Vector Base (stores entity embeddings)
  - Hyperedge Vector Base (stores hyperedge embeddings)

**Right Section - HyperGraph Retrieval and Generation:**
- **User Question Input:** "How does hypertension affect cardiovascular health, and what interventions can effectively reduce its risk?"
- **Entity Extract:** Extracts key entities from the question (hypertension, cardiovascular health, ...)
- **Entity Retrieval:** Retrieves relevant entities ("Hypertension", "Cardiovascular Disease", ...)
- **Hyperedge Retrieval:** Retrieves relevant hyperedges (Hyperedge_1, Hyperedge_2, ...)
- **Retrieved N-ary Relational Facts:** Shows the complete multi-entity relationships
- **Traditional Chunks:** Also retrieves relevant text chunks
- **Hybrid Generation:** Combines hypergraph knowledge with chunks
- **LLM Generation:** Produces final knowledgeable response: "Hypertension is a major risk factor for..."

### 4.1 Knowledge Hypergraph Construction

To represent and store knowledge, we propose a knowledge hypergraph construction method that includes:
1. N-ary relational extraction
2. Bipartite hypergraph storage
3. Vector representation storage

#### 4.1.1 N-ary Relation Extraction

To construct the knowledge hypergraph **G_H**, our first step is to extract multiple n-ary relational facts **F_n** from natural language documents **d ∈ K**.

Unlike traditional hyper-relations, events, or other n-ary relation models, in the era of LLMs, to preserve richer and more diverse n-ary relations among entities, we propose a new n-ary relation representation **F_n = (e_H, V_{e_H})**, utilizing **natural language descriptions**, instead of structured relations, to represent hyperedges **e_H** among multiple entities **V_{e_H}** as follows:

**(a) Hyperedge:** Given an input text **d**, it is parsed into several independent knowledge fragments, each treated as a hyperedge: **E^d_H = {e₁, e₂, ..., eₖ}**. Each hyperedge **eᵢ = (e^text_i, e^score_i)** consists of two parts:
- **e^text_i:** A natural language description
- **e^score_i ∈ (0, 10]:** A confidence score indicating the association degree between eᵢ and d

**(b) Entity:** For each hyperedge **eᵢ**, entity recognition is performed to extract all contained entities: **V_{eᵢ} = {v₁, v₂, ..., vₙ}**, where **V_{eᵢ}** is the entity set associated with **eᵢ**. Each entity **vⱼ = (v^name_j, v^type_j, v^explain_j, v^score_j)** consists of four parts:
- **v^name_j ⊆ e^text_i:** Entity name
- **v^type_j:** Entity type
- **v^explain_j:** Explanation
- **v^score_j ∈ (0, 100]:** Confidence score indicating the extraction certainty

Following this hypergraph-structured knowledge representation, we design an n-ary relation extraction prompt **p_ext** to enable the LLM **π** to perform end-to-end knowledge fragment segmentation and entity recognition, thereby forming the n-ary relational fact set **F^d_n**:

```
F^d_n = {f₁, f₂, ..., fₖ} ~ π(F_n|p_ext, d)
```

where each extracted n-ary relational fact **fᵢ = (eᵢ, V_{eᵢ})** contains information about the corresponding hyperedge **eᵢ** and its associated entity set **V_{eᵢ}**. We convert all documents **d ∈ K** into hyperedges and entities using n-ary relation extraction, forming a complete knowledge hypergraph **G_H**.

**Proposition 1.** *Hypergraph-structured knowledge representation is more comprehensive than binary.*

*Proof.* We provide experimental results in Section 5.4 and proofs in Appendix B.1. □

#### 4.1.2 Bipartite Hypergraph Storage

After n-ary relation extraction, we store the constructed knowledge hypergraph **G_H** in a graph database to support efficient query. We adopt an ordinary graph database represented as a bipartite graph structure **G_B = (V_B, E_B) = Φ(G_H)**, to store the knowledge hypergraph **G_H = (V, E_H)**, where **Φ** is a transformation function defined as:

```
Φ : V_B = V ∪ E_H
    E_B = {(e_H, v) | e_H ∈ E_H, v ∈ V_{e_H}}
```

where:
- **V_B** is the set of nodes in **G_B**, formed by merging the entity set **V** and the hyperedge set **E_H** from **G_H**
- **E_B** captures the connections between each hyperedge **e_H ∈ E_H** and its associated entities **v ∈ V_{e_H}**

**Benefits:**
- We can efficiently query all entities associated with a hyperedge **e_H**
- We can query all hyperedges linked to a specific entity **v**
- This benefits from the optimized query efficiency of an ordinary graph database
- While preserving the complete hypergraph-structured knowledge representation

**Incremental Updates:** Moreover, **G_B** allows incremental updates through dynamically expansion:

```
G_B ← G_B ∪ Φ(G'_H)
```

where **G'_H** represents newly added hypergraph information. The transformation of hyperedges and entities into the bipartite graph storage format enables seamless updates to the graph database.

**Proposition 2.** *A bipartite graph can losslessly preserve and query a knowledge hypergraph.*

*Proof.* We provide proofs in Appendix B.2. □

#### 4.1.3 Vector Representation Storage

To support efficient semantic retrieval, we embed hyperedges **e_H ∈ E_H** and entities **v ∈ V** using the same embedding model **f**, ensuring that the vector representation of hyperedges and entities is in the same vector space as questions.

Let **Ψ** be the vector function, then the vector representation storage for the knowledge hypergraph **G_H** is defined as: **Ψ(G_H) = (ℰ_H, ℰ_V)**, where **ℰ_H** is the vector base of hyperedges and **ℰ_V** is the vector base of entities:

```
Ψ : ℰ_H = {h_{e_H} | e_H ∈ E_H}
    ℰ_V = {h_v | v ∈ V}
```

where each hyperedge **e_H** and entity **v** in **G_H** is embedded into their vector representations:
- **h_{e_H} = f(e_H)**
- **h_v = f(v)**

### 4.2 Hypergraph Retrieval Strategy

After constructing and storing the hypergraph **G_H**, we design an efficient retrieval strategy to match user questions with relevant hyperedges and entities.

#### 4.2.1 Entity Retrieval

**Step 1: Extract Entities from Question**

First, we extract key entities from the question **q** to facilitate subsequent matching. We design an entity extraction prompt **p_{q_ext}** along with the LLM **π** to extract the entity set **V_q**:

```
V_q ~ π(V|p_{q_ext}, q)
```

**Step 2: Retrieve Relevant Entities**

After extracting entities, we retrieve the most relevant entities from the entity set **V** of the knowledge hypergraph **G_H**. We define the entity retrieval function **ℛ_V**, which retrieves the most relevant entities from **ℰ_V** using cosine similarity:

```
ℛ_V(q) = argmax^{k_V}_{v∈V} (sim(h_{V_q}, h_v) ⊙ v^{score})_{>τ_V}
```

where:
- **h_{V_q} = f(V_q):** Concatenated text vector representation of the extracted entity set **V_q**
- **h_v ∈ ℰ_V:** Vector representation of entity **v**
- **sim(·, ·):** Similarity function
- **⊙:** Element-wise multiplication between similarity and entity relevance score **v^{score}** determining the final ranking score
- **τ_V:** Threshold for the entity retrieval score
- **k_V:** Limit on the number of retrieved entities

#### 4.2.2 Hyperedge Retrieval

Moreover, to expand the retrieval scope and capture complete n-ary relations within the hyperedge set **E_H** of the knowledge hypergraph **G_H**, we define the hyperedge retrieval function **ℛ_H**, which retrieves a set of hyperedges related to **q**:

```
ℛ_H(q) = argmax^{k_H}_{e_H∈E_B} (sim(h_q, h_{e_H}) ⊙ e^{score}_H)_{>τ_H}
```

where:
- **h_q = f(q):** Text vector representation of **q**
- **h_{e_H} ∈ ℰ_H:** Vector representation of the hyperedge **e_H**
- **⊙:** Element-wise multiplication between similarity and hyperedge relevance score **e^{score}_H** determining the final ranking score
- **τ_H:** Threshold for the hyperedge retrieval score
- **k_H:** Limits the number of retrieved hyperedges

### 4.3 Hypergraph-Guided Generation

To fully utilize the structured knowledge in the hypergraph, we propose a Hypergraph-Guided Generation mechanism, which consists of hypergraph knowledge fusion and generation augmentation.

#### 4.3.1 Hypergraph Knowledge Fusion

The primary goal of hypergraph knowledge fusion is to expand and reorganize the retrieved n-ary relational knowledge to form a comprehensive knowledge input. Since **q** may only match partial entities or hyperedges, we further expand the retrieval scope.

To obtain a complete set of n-ary relational facts, we design a **bidirectional expansion strategy**, that includes:
1. Expanding hyperedges from retrieved entities
2. Expanding entities from retrieved hyperedges

**Step 1: Expand Hyperedges from Retrieved Entities**

Given the entity set retrieved from **q**, denoted as **ℛ_V(q) = {v₁, v₂, ..., v_{k_V}}**, we retrieve all hyperedges in the knowledge hypergraph **G_H** that connect these entities:

```
ℱ*_V = ⋃_{vᵢ∈ℛ_V(q)} {(e_H, V_{e_H}) | vᵢ ∈ V_{e_H}, e_H ∈ E_H}
```

**Step 2: Expand Entities from Retrieved Hyperedges**

Next, we expand the set of entities connected to the retrieved hyperedges **ℛ_H(q) = {e₁, e₂, ..., e_{k_H}}**:

```
ℱ*_H = ⋃_{eᵢ∈ℛ_H(q)} {(eᵢ, V_{eᵢ}) | V_{eᵢ} ⊆ V}
```

**Step 3: Merge Results**

Finally, we merge the expanded hyperedge set **ℱ*_V** with the expanded entity set **ℱ*_H** to form a complete retrieved n-ary relational fact set:

```
K_H = ℱ*_V ∪ ℱ*_H
```

This set contains all necessary n-ary relational knowledge for reasoning and generation, ensuring a comprehensive input for the LLM.

#### 4.3.2 Generation Augmentation

Following hypergraph knowledge fusion, we augment the generation strategy to improve the accuracy and readability of the responses. We adopt a **hybrid RAG fusion mechanism**, combining hypergraph knowledge **K_H** with retrieved chunk-based text fragments **K_{chunk}** to form the final knowledge input:

```
K* = K_H ∪ K_{chunk}
```

where **K_{chunk}** consists of chunk-based text fragments retrieved using traditional RAG.

**Final Generation:**

Finally, we use a retrieval-augmented generation prompt **p_gen** that combines hypergraph knowledge **K*** and the user question **q** as input to LLM **π** to generate final response **y***:

```
y* ~ π(y|p_gen, K*, q)
```

**Proposition 3.** *Retrieving knowledge on a knowledge hypergraph improves retrieval efficiency compared to methods based on ordinary binary graphs, leading to gains in generation quality.*

*Proof.* We provide experimental results in Sections 5.5 and 5.6 and proofs in Appendix B.3. □

---

## 5. Experiments

This section presents the experimental setup, main results, and analysis. We answer the following research questions (RQs):

- **RQ1:** Does HyperGraphRAG outperform other methods?
- **RQ2:** Does the main component of HyperGraphRAG work?
- **RQ3:** How effective is the knowledge hypergraph constructed by HyperGraphRAG across various domains?
- **RQ4:** Could the hypergraph retrieval strategy improve retrieval efficiency?
- **RQ5:** How effective is the generation quality of HyperGraphRAG?
- **RQ6:** How are the time and cost of HyperGraphRAG in construction and generation phases?

### 5.1 Experimental Setup

#### Datasets

To evaluate the performance of HyperGraphRAG across multiple domains, we select four knowledge contexts from UltraDomain, as used in LightRAG:

1. **Agriculture:** Agricultural production, crop management, and pest control
2. **Computer Science (CS):** Computer architecture, algorithms, and machine learning
3. **Legal:** Legal provisions, judicial precedents, and regulatory interpretations
4. **Mix:** A mixed domain combining multiple areas

In addition, we include the latest **international hypertension guidelines** as the foundational knowledge for the **Medicine** domain.

For each of the five domains, we sample knowledge fragments one, two, and three hops away to construct questions with ground-truth answers verified by human annotators. We then categorize the questions into:
- **Binary Source:** Questions based on binary relational facts
- **N-ary Source:** Questions based on n-ary relational facts (n > 2)

More details can be found in Appendix D.

#### Baselines

We compare HyperGraphRAG against six publicly available baseline methods:

1. **NaiveGeneration:** Directly generates responses using LLM
2. **StandardRAG:** Traditional chunk-based RAG approach
3. **GraphRAG:** First graph-based RAG method
4. **LightRAG:** Enhances efficiency via graph indexing
5. **PathRAG:** Uses path pruning for retrieval
6. **HippoRAG2:** Uses Personalized PageRank

To ensure fairness, we use the same generation prompt (see Appendix E).

#### Evaluation Metrics

We evaluate using 3 key metrics:

1. **F1:** Measures word-level similarity between the generated answer and the ground-truth answer
2. **Retrieval Similarity (R-S):** Assesses the semantic similarity between the retrieved knowledge and the ground-truth knowledge
3. **Generation Evaluation (G-E):** Uses LLM-as-a-judge to evaluate generation quality in 7 dimensions:
   - Comprehensiveness
   - Knowledgeability
   - Correctness
   - Relevance
   - Diversity
   - Logical Coherence
   - Factuality

Details are provided in Appendix E.

#### Implementation Details

- **LLM:** OpenAI's GPT-4o-mini for extraction and generation
- **Embedding:** text-embedding-3-small
- **Retrieval Parameters:**
  - Entity retrieval: k_V = 60, τ_V = 50
  - Hyperedge retrieval: k_H = 60, τ_H = 5
  - Chunk retrieval: k_C = 6, τ_C = 0.5
- **Hardware:** 80-core CPU and 512GB RAM

### Table 1: Comparison of RAG Methods

**Description:** This table compares different RAG methods in terms of their knowledge construction and retrieval approaches:

| Method | Knowledge Construction | Knowledge Retrieval |
|--------|------------------------|---------------------|
| **NaiveGeneration** | K = ∅ (no knowledge) | K*_q = ∅ (no retrieval) |
| **StandardRAG** | K = {cᵢ}^N_{i=1}, where cᵢ is a chunk | K*_q = Top_k{c ∈ K \| sim(h_q, h_c)} |
| **GraphRAG** | K = S = {s_g \| g ∈ Community(G)}, where S is the community summary set | K*_q = Detect{s_g ∈ S \| q}, detected community summaries are retrieved |
| **LightRAG** | K = G = (V, E), where V & E are entity & relation sets | K*_q = ℱ{v ∈ V, e ∈ E \| q} ∪ K_{chunk}, entities & relations retrieved with chunks |
| **PathRAG** | K = G = (V, E), same as LightRAG | K*_q = Prune{p ∈ P_q \| q}, relational paths retrieved via pruning |
| **HippoRAG2** | K = G = (V ∪ M, E), where V & M are phrase & passage nodes | K*_q = PageRank{m ∈ M \| q}, passages retrieved via Personalized PageRank |
| **HyperGraphRAG (ours)** | K = G_H = (V, E_H), where G_H is structured as a hypergraph | K*_q = ℱ_n{v ∈ V \| q} ∪ ℱ_n{e ∈ E_H \| q} ∪ K_{chunk}, n-ary relational facts retrieved with chunks |

**Key Insight:** HyperGraphRAG is the only method that constructs and retrieves n-ary relational facts through hypergraph representation, enabling richer knowledge representation compared to chunk-based or binary graph-based methods.

### 5.2 Main Results (RQ1)

### Table 2: Performance Comparison Across Different Domains

**Description:** This comprehensive results table shows F1, R-S (Retrieval Similarity), and G-E (Generation Evaluation) scores across five domains (Medicine, Agriculture, CS, Legal, Mix) for all methods. Results are broken down into three categories: Binary Source (questions based on binary relations), N-ary Source (questions based on n-ary relations), and Overall.

**Key Findings:**

**Medicine Domain:**
- HyperGraphRAG achieves the best overall performance: F1=35.35, R-S=70.19, G-E=59.35
- Outperforms StandardRAG by +7.45 F1, +7.62 R-S, +3.69 G-E
- Graph-based baselines (GraphRAG, LightRAG, PathRAG) often underperform StandardRAG due to knowledge fragmentation

**Agriculture Domain:**
- HyperGraphRAG: F1=33.89, R-S=62.27, G-E=59.79
- Significant improvements in both binary and n-ary scenarios
- Consistent advantages over all baselines

**Computer Science Domain:**
- HyperGraphRAG: F1=31.30, R-S=60.09, G-E=57.94
- Strong performance on structured technical knowledge
- Maintains lead across all evaluation metrics

**Legal Domain:**
- HyperGraphRAG: F1=43.81, R-S=60.47, G-E=63.61
- Largest F1 improvement (+6.47 over StandardRAG)
- Demonstrates effectiveness on complex legal reasoning

**Mix Domain:**
- HyperGraphRAG: F1=48.71, R-S=68.21, G-E=66.90
- Excellent generalization across mixed knowledge types
- Highest overall scores across all domains

**Overall Comparison Across Methods:**

HyperGraphRAG consistently outperforms all baselines across F1, R-S, and G-E metrics. Compared to StandardRAG, it achieves gains of:
- **+7.45 (F1)**
- **+7.62 (R-S)**
- **+3.69 (G-E)**

Interestingly, existing graph-based RAG baselines often underperform StandardRAG, as their reliance on binary relational graphs causes knowledge fragmentation, sparsified retrieval, and incomplete context reconstruction during generation.

**Comparison Across Source Types:**

HyperGraphRAG maintains strong gains under both Binary and N-ary settings:
- **Binary Source:** Improves F1, R-S, and G-E by +8.6, +8.8, and +4.4
- **N-ary Source:** Improvements are +5.3, +6.4, and +2.9

This confirms its robustness across different relation types.

**Comparison Across Domains:**

Performance gains are consistent across domains, with the largest improvements in Medicine and Legal (over +7 F1), and stable advantages in Agriculture and CS. HyperGraphRAG adapts well to both highly structured and more general knowledge tasks.

### 5.3 Ablation Study (RQ2)

### Figure 4: Ablation Study Results

**Description:** This bar chart shows ablation study results in the Medicine domain, comparing:
- **HyperGraphRAG (full model):** F1=35.4, R-S=70.2, G-E=59.4
- **w/o ER (without Entity Retrieval):** F1=29.8, R-S=57.9, G-E=55.3
- **w/o HR (without Hyperedge Retrieval):** F1=26.4, R-S=53.1, G-E=52.2
- **w/o ER & HR:** F1=27.9, R-S=62.6, G-E=55.7
- **w/o CR (without Chunk Retrieval):** F1=29.2, R-S=53.6, G-E=53.4
- **w/o ER & HR & CR (all modules removed):** F1=12.9, R-S=0.0, G-E=43.3

The chart has three groups of bars (F1, R-S, G-E) with different colored bars representing each variant.

**Key Findings:**

**Impact of Entity Retrieval (ER):**
ER is critical for precise retrieval by anchoring key concepts. Without ER, F1 falls from 35.4 to 29.8, underscoring its importance in selecting relevant entities for accurate generation.

**Impact of Hyperedge Retrieval (HR):**
HR captures n-ary, multi-entity facts necessary for complex reasoning. Removing HR drops F1 from 35.4 to 26.4, highlighting its unique role beyond mere entity retrieval.

**Impact of Chunk Retrieval Fusion (CR):**
CR enhances retrieval by integrating unstructured text with hypergraph data. Excluding CR reduces F1 from 35.4 to 29.2, demonstrating that the fusion leads to more complete and fluent generation.

### 5.4 Analysis of Hypergraph-structured Knowledge Representation (RQ3)

### Figure 5: Knowledge Hypergraph Visualizations and Statistics

**Description:** This figure contains six subfigures:

**(a) Medicine HyperGraph:** A network visualization showing medical entities connected by hyperedges. Central nodes include conditions like "Hypertension", "Heart Failure", with surrounding entities representing measurements, treatments, and symptoms. Hyperedges (shown as connecting lines) link multiple related medical concepts.

**(b) Agriculture HyperGraph:** Network showing agricultural concepts with entities like crop types, growing conditions, pest management strategies, and yield factors interconnected through hyperedges representing complex agricultural relationships.

**(c) CS HyperGraph:** Computer science knowledge network with entities representing algorithms, data structures, programming concepts, and system components. Hyperedges capture multi-way relationships between technical concepts.

**(d) Legal HyperGraph:** Legal domain network showing statutes, precedents, legal principles, and case elements connected via hyperedges that represent multi-party legal relationships and dependencies.

**(e) Mix HyperGraph:** A combined network incorporating entities from multiple domains, demonstrating the system's ability to handle interdisciplinary knowledge with diverse n-ary relationships.

**(f) Statistics of Construction:** A comparison table showing:

| Domain | #Knowl. Token | GraphRAG (#Entity, #Community) | LightRAG (#Entity, #Relation) | HyperGraphRAG (#Entity, #Hyperedge) |
|--------|---------------|--------------------------------|-------------------------------|-------------------------------------|
| Med. | 179k | 329, 256 | 3,725, 1,304 | 7,675, 4,818 |
| Agric. | 382k | 699, 523 | 5,032, 3,105 | 16,805, 16,102 |
| CS | 795k | 1449, 930 | 8,967, 5,632 | 19,913, 26,902 |
| Legal | 940k | 1711, 517 | 5,354, 6,002 | 11,098, 18,285 |
| Mix | 122k | 225, 59 | 2,229, 940 | 6,201, 4,356 |

**Key Findings:**

**Visualization of Knowledge Structures:**
As shown in Figure 5(a)-5(e), unlike previous graph-based RAG methods, which only model binary relations, HyperGraphRAG connects multiple entities via hyperedges, forming a more interconnected and expressive network.

**Statistical Analysis:**
HyperGraphRAG surpasses GraphRAG and LightRAG in all domains. For instance:
- In CS domain: HyperGraphRAG constructs **26,902 hyperedges**
- GraphRAG has only **930 communities**
- LightRAG has **5,632 relations**

This shows a **stronger capacity for capturing knowledge** through hypergraph representation.

### 5.5 Analysis of Hypergraph Retrieval Efficiency (RQ4)

### Figure 6: Retrieval Efficiency Analysis

**Description:** This figure contains two subfigures analyzing retrieval efficiency:

**(a) Impact of Top-k on Retrieval Efficiency & Token Length:**
- Line and bar chart showing how performance changes with different top-k values (0, 20, 40, 60, 80, 100)
- Three lines showing F1, R-S, and G-E scores
- Bars showing #R-Token (number of retrieved tokens)
- Key observations:
  - At k=0 (StandardRAG): F1=27.9, R-S=62.6, G-E=55.7, #R-Token=2521
  - At k=60 (HyperGraphRAG): F1=35.4, R-S=70.2, G-E=59.4, #R-Token=7964
  - Performance saturates around k=60
  - Token count increases linearly with k

**(b) F1 Comparison under Limited Lengths:**
- Line chart comparing F1 scores across different retrieved knowledge lengths (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, Full)
- Shows all methods: NaiveGeneration, StandardRAG, GraphRAG, LightRAG, PathRAG, HippoRAG2, HyperGraphRAG
- HyperGraphRAG consistently outperforms all methods at every length limit
- At length 4096: HyperGraphRAG F1=34.3 vs next best HippoRAG2 F1=19.6

**Key Findings:**

**Impact of Retrieved Hyperedge Quantity:**
As shown in Figure 6(a), increasing the top-k hyperedges improves F1, R-S, and G-E, along with the rise in token count. Performance saturates around **k = 60**, indicating that HyperGraphRAG achieves strong retrieval quality with limited input.

**Performance under Constrained Retrieval Length:**
As illustrated in Figure 6(b), HyperGraphRAG outperforms all binary graph-based RAG methods even under retrieval length limits, demonstrating the efficiency of n-ary representations and highlighting the semantic loss inherent in binary structures.

### 5.6 Analysis of Hypergraph-Guided Generation Quality (RQ5)

### Figure 7: Generation Quality Evaluations

**Description:** A radar/spider chart showing seven evaluation dimensions (Comprehensiveness, Knowledgeability, Correctness, Relevance, Diversity, Logical Coherence, Factuality) plus an Overall score. Each method is represented by a different colored line:
- NaiveGeneration (lowest scores across all dimensions)
- StandardRAG
- GraphRAG
- LightRAG
- PathRAG
- HippoRAG2
- HyperGraphRAG (outermost line, highest scores)

**Scores for HyperGraphRAG:**
- Overall: 61.5
- Comprehensiveness: 62.8
- Knowledgeability: 62.0
- Correctness: 64.8
- Relevance: 66.0
- Diversity: 47.0
- Logical Coherence: 64.3
- Factuality: 64.2

**Key Findings:**

**Best Overall Generation Quality:**
HyperGraphRAG achieves the highest Overall score (61.5), significantly outperforming all baseline methods, indicating the comprehensive advantage in hypergraph-guided generation.

**Lead on Key Dimensions:**
HyperGraphRAG achieves notable improvements in:
- **Correctness (64.8)**
- **Relevance (66.0)**
- **Factuality (64.2)**

Outperforming both standard RAG and binary graph-based methods. These gains indicate its strong capacity to produce accurate, context-aware, and knowledge-grounded responses.

**Balanced Performance:**
Although the Diversity score (47.0) is relatively lower than other dimensions, HyperGraphRAG still exceeds all baselines, indicating that it maintains a balanced dimension-wise performance, effectively combining content richness with structural consistency for stable and high-quality generation.

### 5.7 Analysis of Time and Cost in Construction and Generation Phases (RQ6)

### Table 3: Time & Cost Comparisons

**Description:** This table compares construction and generation efficiency:

| Method | Construction | | Generation | |
|--------|-------------|---|------------|---|
| | TP1kT (Time Per 1k Tokens) | CP1kT (Cost Per 1k Tokens) | TPQ (Time Per Query) | CP1kQ (Cost Per 1k Queries) |
| NaiveGeneration | 0 s | 0 $ | 0.131 s | 0.059 $ |
| StandardRAG | 0 s | 0 $ | 0.147 s | 1.016 $ |
| GraphRAG | 9.272 s | 0.0058 $ | 0.221 s | 1.836 $ |
| LightRAG | 5.168 s | 0.0081 $ | 0.359 s | 3.359 $ |
| PathRAG | 5.168 s | 0.0081 $ | 0.436 s | 3.496 $ |
| HippoRAG2 | 2.758 s | 0.0056 $ | 0.240 s | 3.438 $ |
| **HyperGraphRAG** | **3.084 s** | **0.0063 $** | **0.256 s** | **3.184 $** |

**Key Findings:**

**Time & Cost in Construction Phase:**
HyperGraphRAG demonstrates efficient knowledge construction with:
- **3.084 seconds per 1k tokens (TP1kT)**
- **$0.0063 per 1k tokens (CP1kT)**

This places it between the faster HippoRAG2 (2.758s, $0.0056) and slower GraphRAG (9.272s, $0.0058). While its cost is slightly higher than GraphRAG, HyperGraphRAG achieves a better balance between speed, expressiveness, and structure, offering a more compact yet richer representation of n-ary relational knowledge.

**Time & Cost in Generation Phase:**
During the generation phase, HyperGraphRAG requires:
- **0.256 seconds per query (TPQ)**
- **$3.184 per 1k queries (CP1kQ)**

This is moderately higher than StandardRAG (0.147s, $1.016) but significantly lower than PathRAG (0.436s, $3.496) and LightRAG (0.359s, $3.359). Compared to GraphRAG (0.221s, $1.836), HyperGraphRAG slightly increases time and cost but compensates with better retrieval quality and generation outcomes.

The results suggest that HyperGraphRAG achieves a **favorable trade-off** between generation efficiency and output quality, suitable for real-world knowledge-intensive applications.

---

## 6. Conclusion

In this work, we present **HyperGraphRAG**, a retrieval-augmented generation framework that models knowledge as hypergraphs to capture n-ary relational structures. By introducing novel methods for knowledge hypergraph construction, retrieval, and generation, HyperGraphRAG addresses limitations of binary graph-based RAG methods.

Experimental results across diverse domains demonstrate consistent improvements in:
- **Answer accuracy**
- **Retrieval relevance**
- **Generation quality**

Confirming the effectiveness and generalizability of hypergraph-guided retrieval and generation.

---

## References

[1] Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang. Pathrag: Pruning graph-based retrieval augmented generation with relational paths, 2025.

[2] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization, 2024.

[3] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. RAGAs: Automated evaluation of retrieval augmented generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 150–158, St. Julians, Malta, March 2024.

[4] Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao Huang. Minirag: Towards extremely simple retrieval-augmented generation, 2025.

[5] Mikhail Galkin, Priyansh Trivedi, Gaurav Maheshwari, Ricardo Usbeck, and Jens Lehmann. Message passing for hyper-relational knowledge graphs. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7346–7359, Online, November 2020.

[6] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey, 2024.

[7] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented generation, 2024.

[8] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory: Non-parametric continual learning for large language models, 2025.

[9] Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang, and Zhicheng Dou. Flashrag: A modular toolkit for efficient retrieval-augmented generation research. CoRR, abs/2405.13576, 2024.

[10] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020.

[11-29] [Additional references omitted for brevity - full list available in original paper]

---

## Appendix

### A. Prompts Used in HyperGraphRAG

#### A.1 N-ary Relation Extraction Prompt

```
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types,
identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Divide the text into several complete knowledge segments. For each knowledge segment, extract:
   -- knowledge_segment: A sentence that describes the context of the knowledge segment.
   -- completeness_score: A score from 0 to 10 indicating the completeness of the knowledge segment.
   Format: ("hyper-relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)

2. Identify all entities in each knowledge segment. For each identified entity, extract:
   - entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
   - entity_type: Type of the entity.
   - entity_description: Comprehensive description of the entity's attributes and activities.
   - key_score: A score from 0 to 100 indicating the importance of the entity in the text.
   Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2.
   Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
```

#### A.2 Entity Extraction Prompt

```
---Role---
You are a helpful assistant tasked with identifying entities in the user's query.

---Goal---
Given the query, list all entities.

---Instructions---
- Output the keywords in JSON format.

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
```

#### A.3 Retrieval-Augmented Generation Prompt

To ensure a fair comparison across RAG baselines, we adopt a unified Chain-of-Thought (CoT)-based generation prompt for all methods:

```
---Role---
You are a helpful assistant responding to questions based on given knowledge.

---Knowledge---
{d['knowledge']}

---Goal---
Answer the given question. You must first conduct reasoning inside <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>

---Question---
{d['question']}
```

### B. Proofs

#### B.1 Proof of Proposition 1

**Proposition 1.** *Hypergraph-structured knowledge representation is more comprehensive than binary.*

**Proof.** Given a universe of entities **V**, an n-ary fact with n ≥ 3 is denoted as **F = {v₁, ..., vₙ} ⊆ V**.

For hypergraph representation, we represent it with a single hyperedge:
```
e_H = F,  G_H = (V, E_H),  e_H ∈ E_H
```

So the representation function **φ_H : F ↦ e_H** is naturally injective.

For binary graph representation, we connect every pair of entities that co-occur in a fact. For any collection of facts **S ⊆ P(V)**, define the representation function:

```
φ_B(S) = (V_B, E_B)
V_B = ⋃_{F∈S} F
E_B = {(u, w) | u ≠ w, ∃F ∈ S : {u, w} ⊆ F}
```

Let the random variable **X** range over all possible fact sets **S**, with Shannon entropy:
```
H(X) = -Σ_S p(S) log₂ p(S)
```

For hypergraph representation, since **φ_H** is injective:
```
H(X | φ_H(X)) = 0
```

For binary representation, consider three distinct entities **a, b, c ∈ V**, and define:
```
S₁ = {{a, b, c}}
S₂ = {{a, b}, {a, c}, {b, c}}
```

Clearly, **S₁ ≠ S₂**, but:
```
φ_B(S₁) = φ_B(S₂) = ({a, b, c}, {(a,b), (a,c), (b,c)}) = g
```

Thus information is inevitably lost in binary representation. Hence:
```
H(X | φ_B(X)) > 0
I(X; φ_B(X)) = H(X) - H(X | φ_B(X)) < H(X)
```

In contrast, hypergraph representation satisfies **H(X | φ_H(X)) = 0**, so the mutual information reaches its upper bound **H(X)** and all information is preserved.

Therefore, hypergraph representation is more comprehensive than binary in the information-theoretic sense. □

#### B.2 Proof of Proposition 2

**Proposition 2.** *A bipartite graph can losslessly preserve and query a knowledge hypergraph.*

**Proof.** Let the knowledge hypergraph be denoted as **G_H = (V, E_H)**.

Each hyperedge is abstracted as a new node, and combined with the set of entity nodes to form:
```
V_B = V ∪ E_H
E_B = {(e_H, v) | e_H ∈ E_H, v ∈ e_H}
```

Resulting in the incidence bipartite graph **Φ(G_H) = G_B = (V_B, E_B)**.

**G_H** can be represented by the binary incidence matrix:
```
M ∈ {0, 1}^{|V|×|E_H|}
M_{v,e_H} = 1 ⟺ v ∈ e_H
```

The adjacency matrix of **G_B** becomes:
```
A_{G_B} = [  0    M  ]
          [ M^T   0  ]
```

There exists an inverse mapping:
```
Φ^{-1}: G_B → G_H
Φ^{-1}(V_B, E_B) = (V, {N_{G_B}(e_H) | e_H ∈ E_H})
```

where **N_{G_B}(e_H) = {v ∈ V | (e_H, v) ∈ E_B}**.

Clearly, **Φ^{-1} ∘ Φ = id_{G_H}** and **Φ ∘ Φ^{-1} = id_{G_B}**, which means that **Φ** is a bijection and the encoding is lossless.

Therefore, a bipartite graph can losslessly preserve and query a knowledge hypergraph. □

#### B.3 Proof of Proposition 3

**Proposition 3.** *Retrieving knowledge on a knowledge hypergraph improves retrieval efficiency compared to methods based on ordinary binary graphs, leading to gains in generation quality.*

**Proof.** Let the ground-truth knowledge set required for a query **q** be modeled as a discrete random variable **X ⊆ P(V)**. For any n-ary fact **F = {v₁, ..., vₙ}** with n ≥ 3, we define two encoders:

```
φ_H: F ↦ e_H = F
φ_B: F ↦ {(vᵢ, vⱼ) | 1 ≤ i < j ≤ n}
```

Since **φ_H** is injective:
```
H(X | Y_H) = 0,  hence I(X; Y_H) = H(X)
```

However, when **μ({|F| ≥ 3}) > 0**, the encoder **φ_B** becomes non-injective, leading to:
```
H(X | Y_B) > 0
I(X; Y_B) = H(X) - H(X | Y_B) < H(X)
```

Define the information efficiency density (information per bit) as:
```
η* = I(X; Y*) / L*
```

This shows that hypergraph representation transmits more effective information per bit. Let generation quality **E** be a differentiable function **E = g(I(X; Y*), N*)**, where **N*** denotes noise. Under the same bit budget, higher **η_H** implies:

```
E_H(L) - E_B(L) > 0  (unless μ({|F| ≥ 3}) = 0)
```

This formally proves that if there exists at least one fact with arity n ≥ 3 in the knowledge base, then under any fixed retrieval budget **L**, the generation quality under hypergraph encoding strictly exceeds that of the binary encoding. □

### C. HyperGraphRAG Algorithm Detail

#### Algorithm 1: Hypergraph Construction

```
Require: Document collection D
Ensure: Knowledge hypergraph G_H = (V, E_H)

1: Initialize entity set V ← ∅, hyperedge set E_H ← ∅
2: for each document d ∈ D do
3:     Extract n-ary facts: F^d = {(eᵢ, V_{eᵢ})}ᵏᵢ₌₁ ~ π(d)
4:     V ← V ∪ ⋃ᵏᵢ₌₁ V_{eᵢ}
5:     E_H ← E_H ∪ {eᵢ}ᵏᵢ₌₁
6: end for
7: Store (V, E_H) as bipartite graph G_B = Φ(G_H)
8: Compute embeddings: ℰ_V = {f(v) | v ∈ V}, ℰ_{E_H} = {f(e) | e ∈ E_H}
9: return G_H = (V, E_H)
```

**Complexity Analysis:**
Given a corpus of **D** documents, assume each document contains at most **r** relational facts, and each fact involves up to **n** entities. The construction complexity is **O(D·r·n)**. Embedding all nodes and hyperedges requires **O(|V| + |E_H|)** calls to the encoder.

#### Algorithm 2: Hypergraph Retrieval and Generation

```
Require: Query q, knowledge hypergraph G_H = (V, E_H)
Ensure: Final answer y*

1: Extract query entities: V_q ~ π(q)
2: Retrieve top-k entities: V_r ← TopkSim(V_q, ℰ_V)
3: Retrieve top-k hyperedges: E_r ← TopkSim(q, ℰ_{E_H})
4: Expand neighbors: ℱ*_V = ⋃_{v∈V_r} Nbr(v),  ℱ*_E = ⋃_{e∈E_r} Nbr(e)
5: Assemble retrieved knowledge: K_H = ℱ*_V ∪ ℱ*_E
6: Retrieve additional chunks (optional): K_{chunk} = RetrieveChunks(q)
7: Combine all knowledge: K* = K_H ∪ K_{chunk}
8: Generate answer: y* ~ π(q, K*)
9: return y*
```

**Complexity Analysis:**
Given a query **q**, retrieval involves **O(|V| + |E_H|)** embedding comparisons. Neighborhood expansion is bounded by **O(k·d)** where **d** is average node degree. Generation is **O(L)** where **L** is the prompt length.

### D. Dataset Construction

#### D.1 Knowledge Domains

The dataset covers five domains:

1. **Medicine:** Latest international hypertension guidelines covering medical diagnosis, treatment plans, and clinical indicators
2. **Agriculture:** Agricultural production, crop management, and pest control (from UltraDomain)
3. **Computer Science (CS):** Computer architecture, algorithms, and machine learning (from UltraDomain)
4. **Legal:** Legal provisions, judicial precedents, and regulatory interpretations (from UltraDomain)
5. **Mix:** Combination of multiple domains to assess generalization ability

#### D.2 Question Sampling Strategies

For each domain, we sample a total of **512 questions**:

**Binary Source (256 samples):**
- 128 facts via 1-hop traversal
- 64 facts via 2-hop traversal
- 64 facts via 3-hop traversal

**N-ary Source (256 samples):**
- 128 facts via 1-hop traversal
- 64 facts via 2-hop traversal
- 64 facts via 3-hop traversal

For each sampled fact, we prompt GPT to generate a corresponding question and its golden answer. All generated question-answer pairs are manually verified to ensure factual accuracy, relevance, and diversity.

### E. Evaluation Details

#### Evaluation Metrics

**(i) F1 Score:**
Following FlashRAG, we compute the word-level F1 score:

```
F1 = (1/N) Σᵢ₌₁ᴺ (2·Pᵢ·Rᵢ)/(Pᵢ + Rᵢ)

where:
Pᵢ = |Predᵢ ∩ GTᵢ| / |Predᵢ|
Rᵢ = |Predᵢ ∩ GTᵢ| / |GTᵢ|
```

**(ii) Retrieval Similarity (R-S):**
Inspired by RAGAS, R-S quantifies semantic similarity:

```
R-S = (1/N) Σᵢ₌₁ᴺ cos(f(k^(i)_{retr}), f(k^(i)_{gold}))
```

**(iii) Generation Evaluation (G-E):**
Adapted from HelloBench, uses GPT-4o-mini as LLM judge across seven dimensions:

```
G-E = (1/N) Σᵢ₌₁ᴺ mean((1/7)Σ_{d=1}^7 sᵢ,d ; F1ᵢ)
```

**Seven Evaluation Dimensions:**

1. **Comprehensiveness:** Whether the thinking considers all important aspects and is thorough
   - 10: Extremely thorough, covering all relevant angles
   - 8-9: Covers most key aspects clearly
   - 6-7: Covers some important aspects, lacks depth
   - 4-5: Touches on few relevant points
   - 1-3: Sparse or shallow treatment
   - 0: No comprehensiveness at all

2. **Knowledgeability:** Whether the thinking is rich in insightful, domain-relevant knowledge
   - 10: Exceptional depth and insight with strong domain knowledge
   - 8-9: Clear domain knowledge with good insight
   - 6-7: Some understanding, but lacks depth
   - 4-5: Limited knowledge shown
   - 1-3: Poor grasp of relevant knowledge
   - 0: No evidence of meaningful knowledge

3. **Correctness:** Whether the reasoning and answer are logically and factually correct
   - 10: Fully accurate and logically sound
   - 8-9: Mostly correct with minor inaccuracies
   - 6-7: Partially correct with some key flaws
   - 4-5: Noticeable incorrect reasoning
   - 1-3: Largely incorrect or misleading
   - 0: Entirely wrong or nonsensical

4. **Relevance:** Whether the reasoning and answer are highly relevant and helpful to the question
   - 10: Fully focused on the question
   - 8-9: Mostly on point with minor digressions
   - 6-7: Generally relevant with some distractions
   - 4-5: Limited relevance
   - 1-3: Barely related to the question
   - 0: Entirely irrelevant

5. **Diversity:** Whether the reasoning is thought-provoking, offering varied or novel perspectives
   - 10: Exceptionally rich and original
   - 8-9: Contains a few novel angles
   - 6-7: Some variety but generally conventional
   - 4-5: Mostly standard thinking
   - 1-3: Very predictable or monotonous
   - 0: No diversity or originality

6. **Logical Coherence:** Whether the reasoning is internally consistent, clear, and well-structured
   - 10: Highly logical, clear, and easy to follow
   - 8-9: Well-structured with minor lapses
   - 6-7: Some structure but confusing parts
   - 4-5: Often disorganized or unclear
   - 1-3: Poorly structured and incoherent
   - 0: Entirely illogical or unreadable

7. **Factuality:** Whether the reasoning and answer are based on accurate and verifiable facts
   - 10: All facts are accurate and verifiable
   - 8-9: Mostly accurate with minor factual issues
   - 6-7: Contains some factual inaccuracies
   - 4-5: Several significant factual errors
   - 1-3: Mostly false or misleading
   - 0: Completely fabricated or factually wrong

### F. Baseline Details

We compare HyperGraphRAG against six representative baselines:

1. **NaiveGeneration:** Retrieval-free baseline where the LLM directly answers questions without any external knowledge input

2. **StandardRAG:** Follows the original RAG design, retrieving top-k text chunks from a flat corpus using dense vector similarity

3. **GraphRAG:** Constructs a binary relational graph and retrieves community-level summaries linked to query-relevant entities

4. **LightRAG:** Enhances retrieval efficiency by using graph indexing and lightweight entity-relation matching over the binary graph

5. **PathRAG:** Improves graph-based retrieval by selecting paths through the graph that are semantically relevant to the query

6. **HippoRAG2:** Introduces a high-precision multi-hop retrieval mechanism over binary graphs, using Personalized PageRank

### G. Hyperparameter Settings

**Table 4: Hyperparameter settings for all methods**

| Method | Retrieval Type | Top-k Units | Parallel Cores | Generation Model |
|--------|---------------|-------------|----------------|------------------|
| NaiveGeneration | None | – | 16 | GPT-4o-mini |
| StandardRAG | Chunk | 6 chunks | 16 | GPT-4o-mini |
| GraphRAG | Entity → Community | 60 | 16 | GPT-4o-mini |
| LightRAG | Entity + Relation | 60 | 16 | GPT-4o-mini |
| PathRAG | Graph Path | 60 | 16 | GPT-4o-mini |
| HippoRAG2 | PageRank over Graph | 60 | 16 | GPT-4o-mini |
| **HyperGraphRAG (ours)** | **Entity + Hyperedge** | **60** | **16** | **GPT-4o-mini** |

### H. Case Study

**Query:** "What type of renal denervation has been shown to demonstrate BP-lowering efficacy over 24 hours in several randomized, sham-controlled trials, particularly for resistant hypertension?"

**Golden Answer:** "ULTRASOUND RENAL DENERVATION"

**HyperGraphRAG Response:**
```
<think>
The question asks about a type of renal denervation that has demonstrated BP-lowering efficacy
over 24 hours specifically in the context of several randomized, sham-controlled trials related
to resistant hypertension. Based on the provided knowledge, both ultrasound renal denervation
and radiofrequency renal denervation have been mentioned to show BP-lowering efficacy. The
hyperedges also state that several sham-controlled trials have been published demonstrating
this efficacy for these two methods. Therefore, it's essential to identify one type that is
prominently recognized for this efficacy in published studies. Both types could potentially
qualify, but the information about ultrasound renal denervation being mentioned in conjunction
with randomized trials makes it a likely candidate.

Thus, ultrasound renal denervation is a key method that fits the criteria given in the question.
</think>

<answer>
Ultrasound renal denervation
</answer>
```

**Evaluation Scores:**
- F1: 100.00
- R-S: 70.87
- G-E: 92.86
- Comprehensiveness: 90.00
- Knowledgeability: 95.00
- Correctness: 100.00
- Relevance: 100.00
- Diversity: 75.00
- Logical Coherence: 95.00
- Factuality: 95.00

**Key Insight:** HyperGraphRAG provides the most comprehensive and accurate response, clearly identifying ultrasound renal denervation as the answer, and justifying it with domain-relevant reasoning and structured knowledge. In contrast, baseline methods produce factually plausible but imprecise or overly generic answers (e.g., mentioning radiofrequency denervation instead).

### I. Limitations and Future Work

#### I.1 Multimodal HyperGraphRAG

While our current framework focuses on textual knowledge, real-world information often spans multiple modalities, including images, tables, and structured metadata. A promising direction is to extend HyperGraphRAG to the multimodal setting by constructing hypergraphs that integrate both textual and non-textual entities.

#### I.2 HyperGraphRAG with Reinforcement Learning

Another important extension lies in incorporating reinforcement learning (RL) to guide both retrieval and generation. By formulating hypergraph-based retrieval as a sequential decision-making process, we can apply RL to optimize entity and hyperedge selection policies based on long-term generation rewards.

#### I.3 Federated HyperGraphRAG for Privacy-Preserving Retrieval

Many real-world applications involve sensitive or distributed data that cannot be centralized due to privacy constraints. To address this, we propose to integrate HyperGraphRAG with federated learning techniques, allowing hypergraph construction, retrieval, and generation to occur across decentralized data silos.

#### I.4 Toward a Foundation Model for HyperGraph-based Retrieval

As large language models continue to scale and generalize across domains, a natural extension is to explore the development of a foundation model for HyperGraphRAG. Rather than constructing and retrieving from hypergraphs on a per-task or per-domain basis, we envision a pretrained hypergraph reasoning model that jointly learns representations of entities, relations, and higher-order hyperedges across diverse corpora.

#### I.5 Scaling to Harder Tasks and Broader Applications

Finally, we plan to evaluate HyperGraphRAG on more challenging tasks and diverse real-world applications. This includes settings that require deeper compositional reasoning, such as multi-hop question answering, legal argument generation, or complex scientific synthesis.

---

## Implementation Guide

For researchers and practitioners looking to implement HyperGraphRAG:

### Key Components to Implement:

1. **N-ary Relation Extraction Module:**
   - Use LLM with structured prompts to extract hyperedges and entities
   - Store confidence scores for both hyperedges and entities
   - Preserve natural language descriptions

2. **Bipartite Graph Storage:**
   - Implement hypergraph as bipartite graph structure
   - Store entity nodes and hyperedge nodes separately
   - Connect hyperedge nodes to their associated entities

3. **Dual Vector Databases:**
   - Entity vector database for entity embeddings
   - Hyperedge vector database for hyperedge embeddings
   - Use same embedding model for both to ensure semantic alignment

4. **Bidirectional Retrieval:**
   - Entity retrieval: Extract entities from query → retrieve similar entities → expand to connected hyperedges
   - Hyperedge retrieval: Retrieve similar hyperedges → expand to connected entities
   - Merge both retrieval results

5. **Hybrid Generation:**
   - Combine hypergraph knowledge with traditional chunk-based retrieval
   - Format as structured prompt for LLM
   - Use Chain-of-Thought reasoning

### Recommended Parameters:

- Entity retrieval: top-60, threshold=50
- Hyperedge retrieval: top-60, threshold=5
- Chunk retrieval: top-6, threshold=0.5
- Embedding model: text-embedding-3-small or equivalent
- Generation model: GPT-4o-mini or equivalent

---

## J. DETAILED IMPLEMENTATION GUIDE (BASED ON REFERENCE CODE)

This section provides comprehensive implementation guidance based on the official HyperGraphRAG reference implementation in the `hypergraphrag/` folder.

### J.1 CORE ARCHITECTURE & CLASS STRUCTURE

#### HyperGraphRAG Main Class

The `HyperGraphRAG` class (in `hypergraphrag/hypergraphrag.py`) is the main entry point with the following configuration:

**Configuration Parameters:**
```python
@dataclass
class HyperGraphRAG:
    # Working directory
    working_dir: str = f"hypergraphrag_cache_{timestamp}"

    # Storage backends (same as Graph-R1)
    kv_storage: str = "JsonKVStorage"           # Key-value storage
    vector_storage: str = "NanoVectorDBStorage" # Vector database
    graph_storage: str = "NetworkXStorage"      # Bipartite graph storage

    # Text chunking
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # Entity extraction
    entity_extract_max_gleaning: int = 2        # Two-pass extraction
    entity_summary_to_max_tokens: int = 500

    # Embedding configuration
    embedding_func: EmbeddingFunc = openai_embedding
    embedding_batch_num: int = 32
    embedding_func_max_async: int = 16

    # LLM configuration
    llm_model_func: callable = gpt_4o_mini_complete
    llm_model_max_token_size: int = 32768
    llm_model_max_async: int = 16

    # Caching
    enable_llm_cache: bool = True
    embedding_cache_config: dict = {
        "enabled": False,
        "similarity_threshold": 0.95,
        "use_llm_check": False
    }
```

**Storage Components** (Critical Implementation Detail):

The implementation uses THREE separate vector databases:
1. **entities_vdb**: Stores entity embeddings with meta field `{"entity_name"}`
2. **hyperedges_vdb**: Stores hyperedge embeddings with meta field `{"hyperedge_name"}`
3. **chunks_vdb**: Stores traditional text chunk embeddings

Plus a bipartite graph storage:
4. **chunk_entity_relation_graph**: NetworkX bipartite graph connecting entities ↔ hyperedges

### J.2 N-ARY RELATION EXTRACTION IMPLEMENTATION

#### Extraction Format (Missing from Paper)

The paper describes n-ary extraction abstractly, but the implementation uses specific formats:

**Hyperedge Format:**
```python
("hyper-relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)
```

**Example:**
```
("hyper-relation"<|>"Hypertension is defined as systolic BP ≥140 mmHg or diastolic BP ≥90 mmHg."<|>9)
```

**Entity Format:**
```python
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)
```

**Example:**
```
("entity"<|>"Hypertension"<|>"Disease"<|>"Hypertension is defined as systolic BP ≥140 mmHg or diastolic BP ≥90 mmHg."<|>95)
```

**Delimiters (Not in Paper):**
```python
GRAPH_FIELD_SEP = "<SEP>"                    # Used to separate merged descriptions
DEFAULT_TUPLE_DELIMITER = "<|>"              # Separates fields within a record
DEFAULT_RECORD_DELIMITER = "##"              # Separates records
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>" # Signals extraction complete
```

#### Two-Pass Extraction (Gleaning)

The paper doesn't mention gleaning, but the implementation does **two-pass extraction**:

**Pass 1**: Extract obvious hyperedges and entities
```python
# First extraction
results_1 = await llm_func(extraction_prompt, chunk_text)
```

**Pass 2**: Review and extract missed entities (if `entity_extract_max_gleaning >= 2`)
```python
# Second pass: find entities mentioned but not extracted
if gleaning_round < entity_extract_max_gleaning:
    already_extracted = {entity["entity_name"] for entity in entities}
    gleaning_prompt = f"Find entities not yet extracted: {already_extracted}"
    results_2 = await llm_func(gleaning_prompt, chunk_text)
```

This improves recall significantly (not quantified in paper).

### J.3 HYPERGRAPH STORAGE SCHEMA

#### Bipartite Graph Implementation

The paper states hypergraphs are stored as bipartite graphs, but doesn't show the actual structure:

**Entity Node Data:**
```python
entity_node = {
    "entity_type": "ENTITY",                  # Node type marker
    "entity_name": "HYPERTENSION",            # Normalized uppercase
    "description": "Hypertension is...",
    "source_id": "chunk-abc123",
    "weight": 95.0                            # Key score from extraction
}
```

**Hyperedge Node Data:**
```python
hyperedge_node = {
    "entity_type": "HYPEREDGE",               # Node type marker
    "hyperedge_name": "<hyperedge>Hypertension is defined...",
    "description": "Hypertension is defined as...",
    "source_id": "chunk-abc123<SEP>chunk-def456",  # Multiple sources
    "weight": 9.0                             # Completeness score
}
```

**Edge Connections (in NetworkX bipartite graph):**
```python
# Connect hyperedge to its associated entities
graph.add_edge(
    "<hyperedge>Hypertension is defined...",  # Hyperedge node
    "HYPERTENSION",                           # Entity node
    weight=1.0,
    description="HYPERTENSION is part of this hyperedge"
)
```

### J.4 HYPEREDGE NAMING CONVENTION

**Critical Detail (Not in Paper):**

Hyperedge nodes are prefixed with `"<hyperedge>"` to distinguish them from entity nodes:

```python
hyperedge_name = "<hyperedge>" + knowledge_fragment
```

This allows the bipartite graph to differentiate entity nodes from hyperedge nodes without requiring node type attributes in all graph database backends.

### J.5 ENTITY RETRIEVAL IMPLEMENTATION

#### Step-by-Step Process

**Step 1: Extract Query Entities** (Paper: Section 4.2.1)

```python
async def extract_query_entities(query: str, llm_func: callable):
    """
    Extract entities from user question using LLM
    """
    prompt = f"""Extract key entities from this question: {query}
    Return as JSON list of entity names."""

    response = await llm_func(prompt)
    entities = parse_json(response)  # ["Hypertension", "Treatment", ...]
    return entities
```

**Step 2: Retrieve Similar Entities** (with scoring)

The paper shows formula `ℛ_V(q) = argmax^{k_V}_{v∈V} (sim(h_{V_q}, h_v) ⊙ v^{score})_{>τ_V}`, but the implementation is:

```python
async def retrieve_entities(query_entities: List[str], entities_vdb, top_k=60, threshold=50):
    """
    Retrieve entities using cosine similarity × entity weight
    """
    # Concatenate query entities
    query_text = ", ".join(query_entities)
    query_embedding = await embedding_func(query_text)

    # Vector search
    results = await entities_vdb.query(query_text, top_k=top_k * 2)

    # Apply scoring: similarity × entity weight
    scored_results = []
    for entity in results:
        similarity_score = cosine_sim(query_embedding, entity["embedding"])
        entity_weight = entity.get("weight", 50.0)

        # Element-wise multiplication (⊙ in paper)
        final_score = similarity_score * entity_weight

        if final_score > threshold:
            scored_results.append({
                "entity_name": entity["entity_name"],
                "score": final_score,
                "description": entity["description"]
            })

    # Sort and limit
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]
```

### J.6 HYPEREDGE RETRIEVAL IMPLEMENTATION

The paper formula `ℛ_H(q) = argmax^{k_H}_{e_H∈E_B} (sim(h_q, h_{e_H}) ⊙ e^{score}_H)_{>τ_H}` is implemented as:

```python
async def retrieve_hyperedges(query: str, hyperedges_vdb, top_k=60, threshold=5):
    """
    Retrieve hyperedges directly by semantic similarity
    """
    # Embed query
    query_embedding = await embedding_func(query)

    # Vector search in hyperedge vector database
    results = await hyperedges_vdb.query(query, top_k=top_k * 2)

    # Apply scoring: similarity × hyperedge completeness score
    scored_results = []
    for hyperedge in results:
        similarity_score = cosine_sim(query_embedding, hyperedge["embedding"])
        hyperedge_weight = hyperedge.get("weight", 5.0)  # Completeness score

        final_score = similarity_score * hyperedge_weight

        if final_score > threshold:
            scored_results.append({
                "hyperedge_name": hyperedge["hyperedge_name"],
                "score": final_score,
                "description": hyperedge["description"],
                "entities": hyperedge.get("entities", [])
            })

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]
```

### J.7 BIDIRECTIONAL EXPANSION IMPLEMENTATION

The paper describes bidirectional expansion (Section 4.3.1) but doesn't show the graph traversal:

**Expand Hyperedges from Entities:**
```python
async def expand_hyperedges_from_entities(
    retrieved_entities: List[str],
    bipartite_graph: NetworkXStorage
):
    """
    For each retrieved entity, find all connected hyperedges
    Paper formula: ℱ*_V = ⋃_{vᵢ∈ℛ_V(q)} {(e_H, V_{e_H}) | vᵢ ∈ V_{e_H}, e_H ∈ E_H}
    """
    expanded_hyperedges = set()

    for entity_name in retrieved_entities:
        # Get all neighbors of this entity node in bipartite graph
        neighbors = await bipartite_graph.get_node_edges(entity_name)

        for neighbor_node, _ in neighbors:
            # Check if neighbor is a hyperedge (prefixed with "<hyperedge>")
            if neighbor_node.startswith("<hyperedge>"):
                expanded_hyperedges.add(neighbor_node)

    return list(expanded_hyperedges)
```

**Expand Entities from Hyperedges:**
```python
async def expand_entities_from_hyperedges(
    retrieved_hyperedges: List[str],
    bipartite_graph: NetworkXStorage
):
    """
    For each retrieved hyperedge, find all connected entities
    Paper formula: ℱ*_H = ⋃_{eᵢ∈ℛ_H(q)} {(eᵢ, V_{eᵢ}) | V_{eᵢ} ⊆ V}
    """
    expanded_entities = set()

    for hyperedge_name in retrieved_hyperedges:
        # Get all neighbors of this hyperedge node
        neighbors = await bipartite_graph.get_node_edges(hyperedge_name)

        for neighbor_node, _ in neighbors:
            # Check if neighbor is an entity (not prefixed with "<hyperedge>")
            if not neighbor_node.startswith("<hyperedge>"):
                expanded_entities.add(neighbor_node)

    return list(expanded_entities)
```

### J.8 HYBRID RAG FUSION

The paper says `K* = K_H ∪ K_{chunk}` but doesn't detail the combination:

```python
async def hybrid_rag_fusion(
    hypergraph_knowledge: List[Dict],
    chunk_retrieval: List[str],
    chunks_vdb,
    top_k_chunks: int = 6,
    chunk_threshold: float = 0.5
):
    """
    Combine hypergraph knowledge with traditional chunk-based RAG
    """
    # Retrieve traditional chunks
    chunk_results = await chunks_vdb.query(query, top_k=top_k_chunks)
    chunks = [c for c in chunk_results if c["score"] > chunk_threshold]

    # Format hypergraph knowledge
    hypergraph_text = []
    for item in hypergraph_knowledge:
        if item["type"] == "entity":
            hypergraph_text.append(
                f"Entity: {item['entity_name']} ({item['entity_type']})\n"
                f"Description: {item['description']}"
            )
        elif item["type"] == "hyperedge":
            hypergraph_text.append(
                f"N-ary Fact: {item['description']}\n"
                f"Entities: {', '.join(item['entities'])}"
            )

    # Combine both sources
    combined_knowledge = "\n\n".join([
        "=== HYPERGRAPH KNOWLEDGE ===",
        "\n\n".join(hypergraph_text),
        "\n=== ADDITIONAL CONTEXT ===",
        "\n\n".join([c["content"] for c in chunks])
    ])

    return combined_knowledge
```

### J.9 GENERATION PROMPT STRUCTURE

The paper shows a CoT prompt (Appendix A.3) but the actual implementation adds more structure:

```python
def create_generation_prompt(query: str, knowledge: str):
    """
    Create final generation prompt with Chain-of-Thought reasoning
    """
    return f"""---Role---
You are a helpful assistant responding to questions based on given knowledge.

---Knowledge---
{knowledge}

---Goal---
Answer the given question. You must first conduct reasoning inside <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

IMPORTANT: Use the N-ary Facts from the hypergraph knowledge to understand complex multi-entity relationships. These facts capture relationships among multiple entities simultaneously, providing more complete context than individual entities alone.

Output format for answer:
<think>
[Your reasoning process here, considering all relevant entities and n-ary relationships...]
</think>
<answer>
[Your final answer here]
</answer>

---Question---
{query}
"""
```

### J.10 ENTITY MERGING STRATEGY

**Critical Detail (Not in Paper):**

When multiple chunks mention the same entity, descriptions are merged:

```python
async def merge_entity_descriptions(
    entity_name: str,
    new_description: str,
    entities_vdb,
    global_config: dict
):
    """
    Merge descriptions when entity appears in multiple chunks
    """
    # Check if entity already exists
    existing_entity = await entities_vdb.get_by_id(entity_name)

    if existing_entity:
        # Merge descriptions using GRAPH_FIELD_SEP
        merged_description = (
            existing_entity["description"] +
            GRAPH_FIELD_SEP +
            new_description
        )

        # If merged description too long, summarize using LLM
        if len(encode_tokens(merged_description)) > entity_summary_to_max_tokens:
            merged_description = await _handle_entity_relation_summary(
                entity_name,
                merged_description,
                global_config
            )

        # Update entity
        existing_entity["description"] = merged_description
        existing_entity["weight"] = max(existing_entity["weight"], new_weight)

        return existing_entity
    else:
        # Create new entity
        return {
            "entity_name": entity_name,
            "description": new_description,
            "weight": new_weight
        }
```

### J.11 HYPEREDGE WEIGHT AGGREGATION

**Important Implementation Detail:**

When multiple chunks contribute to the same hyperedge, weights are summed:

```python
async def merge_hyperedge_weights(
    hyperedge_name: str,
    new_weight: float,
    new_source_id: str,
    bipartite_graph: NetworkXStorage
):
    """
    Aggregate weights when hyperedge appears in multiple chunks
    """
    existing_hyperedge = await bipartite_graph.get_node(hyperedge_name)

    if existing_hyperedge:
        # Sum weights (not average!)
        total_weight = existing_hyperedge["weight"] + new_weight

        # Merge source IDs
        existing_sources = existing_hyperedge["source_id"].split(GRAPH_FIELD_SEP)
        all_sources = existing_sources + [new_source_id]
        merged_sources = GRAPH_FIELD_SEP.join(set(all_sources))

        return {
            "weight": total_weight,
            "source_id": merged_sources
        }
    else:
        return {
            "weight": new_weight,
            "source_id": new_source_id
        }
```

### J.12 DEFAULT WEIGHT VALUES

**Missing from Paper:**

When extraction doesn't provide weights, defaults are used:

```python
# Entity weight default (key_score)
entity_weight = float(record_attributes[4]) if is_float_regex(record_attributes[4]) else 50.0

# Hyperedge weight default (completeness_score)
hyperedge_weight = float(record_attributes[2]) if is_float_regex(record_attributes[2]) else 1.0
```

Paper says scores are 0-100 for entities and 0-10 for hyperedges, but implementation uses 50.0 and 1.0 as defaults.

### J.13 QUERY MODES

**Not Mentioned in Paper:**

The implementation supports multiple query modes (inherited from GraphRAG):

```python
@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive"] = "hybrid"

    # local: Entity-focused retrieval
    # global: Hyperedge-focused retrieval
    # hybrid: Both entity + hyperedge retrieval (default)
    # naive: Direct LLM without retrieval
```

For HyperGraphRAG, **"hybrid" mode is standard** (retrieves both entities and hyperedges).

### J.14 STORAGE BACKEND OPTIONS

Same as Graph-R1, HyperGraphRAG supports multiple storage backends:

**Key-Value Storage:**
- JsonKVStorage (default)
- MongoKVStorage
- OracleKVStorage
- TiDBKVStorage

**Vector Storage:**
- NanoVectorDBStorage (default)
- MilvusVectorDBStorge
- ChromaVectorDBStorage
- OracleVectorDBStorage
- TiDBVectorDBStorage

**Graph Storage (Bipartite):**
- NetworkXStorage (default) - In-memory
- Neo4JStorage - Graph database
- OracleGraphStorage - Oracle Property Graph

### J.15 PRACTICAL USAGE EXAMPLES

#### Basic Usage

```python
from hypergraphrag import HyperGraphRAG, QueryParam

# Initialize
rag = HyperGraphRAG(
    working_dir="./medical_kb",
    chunk_token_size=1200,
    entity_extract_max_gleaning=2
)

# Insert medical documents
documents = [
    "Hypertension is defined as systolic BP ≥140 mmHg or diastolic BP ≥90 mmHg.",
    "Male hypertensive patients with serum creatinine 115-133 µmol/L are diagnosed with mild elevation.",
]
rag.insert(documents)

# Query
result = rag.query(
    "What defines hypertension?",
    param=QueryParam(mode="hybrid", top_k=60)
)
```

#### Async Usage with Custom Parameters

```python
import asyncio

async def main():
    rag = HyperGraphRAG(
        working_dir="./kb",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embedding
    )

    # Async insert
    await rag.ainsert(documents)

    # Async query with entity and hyperedge specification
    result = await rag.aquery(
        query="What interventions reduce hypertension risk?",
        param=QueryParam(
            mode="hybrid",
            top_k=60,
            only_need_context=False  # Get full generation
        ),
        # Pre-computed entity matches (optional)
        entity_match={"query": ["HYPERTENSION", "CARDIOVASCULAR DISEASE"]},
        # Pre-computed hyperedge matches (optional)
        hyperedge_match={"query": ["<hyperedge>Hypertension increases cardiovascular risk..."]}
    )

    print(result)

asyncio.run(main())
```

#### Using Different Storage Backends

```python
# Use Milvus for vector storage + Neo4j for graph storage
rag = HyperGraphRAG(
    working_dir="./kb",
    vector_storage="MilvusVectorDBStorge",
    graph_storage="Neo4JStorage",
    kv_storage="MongoKVStorage"
)
```

### J.16 KEY DIFFERENCES: PAPER VS IMPLEMENTATION

| Aspect | Paper Description | Reference Implementation |
|--------|------------------|-------------------------|
| **Hyperedge Naming** | Not specified | Prefix with `"<hyperedge>"` to distinguish from entities |
| **Entity Merging** | Not mentioned | Concatenate descriptions with `<SEP>`, auto-summarize if too long |
| **Weight Aggregation** | Not specified | **Sum** weights for recurring hyperedges |
| **Default Weights** | 0-100 (entities), 0-10 (hyperedges) | Default 50.0 (entities), 1.0 (hyperedges) |
| **Extraction Passes** | Single pass | Two-pass gleaning for better recall |
| **Vector Databases** | Two (entities + hyperedges) | **Three** (entities + hyperedges + chunks) |
| **Retrieval Scoring** | `sim × score` | `cosine_similarity × weight` with thresholding |
| **Query Modes** | Not mentioned | Four modes: local, global, **hybrid** (default), naive |
| **Bipartite Storage** | Abstract Φ transformation | NetworkX/Neo4j with node type markers |
| **Chunk Retrieval** | Mentioned briefly | Parallel retrieval with top-6, threshold=0.5 |

### J.17 PERFORMANCE TUNING GUIDELINES

**For Large-Scale Medical/Legal Applications:**

1. **Use scalable backends:**
   ```python
   rag = HyperGraphRAG(
       vector_storage="MilvusVectorDBStorge",  # Better for 100k+ entities
       graph_storage="Neo4JStorage",           # Better for complex traversals
       kv_storage="MongoKVStorage"             # Distributed storage
   )
   ```

2. **Tune retrieval parameters:**
   ```python
   param = QueryParam(
       top_k=60,  # Paper recommendation
       max_token_for_local_context=4000,  # Entity descriptions
       max_token_for_global_context=4000  # Hyperedge descriptions
   )
   ```

3. **Enable embedding cache for repeated queries:**
   ```python
   rag = HyperGraphRAG(
       embedding_cache_config={
           "enabled": True,
           "similarity_threshold": 0.95
       }
   )
   ```

**For Low-Resource Environments:**

1. Use defaults (JSON + NanoVectorDB + NetworkX)
2. Reduce `top_k` to 20-30
3. Set `entity_extract_max_gleaning=1` (single pass)
4. Use smaller `chunk_token_size=800`

### J.18 DEBUGGING & LOGGING

The implementation includes extensive logging:

```python
import logging

# Enable debug logging
rag = HyperGraphRAG(
    log_level="DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR
)

# Check log file
# Logs saved to: hypergraphrag.log
```

**Common log messages:**
- `[New Docs] inserting X docs` - Document insertion
- `[New Chunks] inserting X chunks` - Chunk creation
- `[Entity Extraction]...` - Starting extraction
- `Trigger summary: ENTITY_NAME` - Auto-summarization
- `No new hyperedges and entities found` - All content already indexed

### J.19 CUSTOM KNOWLEDGE GRAPH INSERTION

**Advanced Feature (Not in Paper):**

You can insert pre-constructed knowledge graphs:

```python
custom_kg = {
    "chunks": [
        {"content": "Hypertension is...", "source_id": "doc1"}
    ],
    "entities": [
        {
            "entity_name": "Hypertension",
            "entity_type": "Disease",
            "description": "High blood pressure condition",
            "source_id": "doc1"
        }
    ],
    "hyperedges": [
        {
            "description": "Hypertension is defined as BP ≥140/90 mmHg",
            "entities": ["Hypertension", "Blood Pressure"],
            "source_id": "doc1",
            "weight": 9.0
        }
    ]
}

rag.insert_custom_kg(custom_kg)
```

This is useful for integrating external knowledge bases or databases.

### J.20 COMPARISON WITH GRAPH-R1

**Similarities:**
- Both use bipartite graph storage for n-ary relations
- Same storage backend options
- Same async processing architecture
- Two-pass gleaning for entity extraction

**Key Differences:**
- HyperGraphRAG uses THREE vector DBs (entities, hyperedges, chunks)
- Graph-R1 uses TWO vector DBs (entities, hyperedges)
- HyperGraphRAG explicitly implements hybrid chunk fusion
- Graph-R1 focuses more on agentic reasoning
- HyperGraphRAG emphasizes the bidirectional expansion more explicitly

**Architecture Comparison:**
```
Graph-R1:           HyperGraphRAG:
- entities_vdb      - entities_vdb
- hyperedges_vdb    - hyperedges_vdb
                    - chunks_vdb (ADDITIONAL)
- bipartite_graph   - bipartite_graph
```

---

**End of Document**

*For code, datasets, and additional resources, visit:*
*https://github.com/LHRLAB/HyperGraphRAG*
