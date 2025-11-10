# GRAPH-R1: TOWARDS AGENTIC GRAPHRAG FRAMEWORK VIA END-TO-END REINFORCEMENT LEARNING

**Preprint**

arXiv:2507.21892v1 [cs.CL] 29 Jul 2025

---

## ABSTRACT

Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by incorporating external knowledge, but relies on chunk-based retrieval that lacks structural semantics. GraphRAG methods improve RAG by modeling knowledge as entity-relation graphs, but still face challenges in high construction cost, fixed one-time retrieval, and reliance on long-context reasoning and prompt design. To address these challenges, we propose **Graph-R1**, an agentic GraphRAG framework via end-to-end reinforcement learning (RL). It introduces lightweight knowledge hypergraph construction, models retrieval as a multi-turn agent-environment interaction, and optimizes the agent process via an end-to-end reward mechanism. Experiments on standard RAG datasets show that Graph-R1 outperforms traditional GraphRAG and RL-enhanced RAG methods in reasoning accuracy, retrieval efficiency, and generation quality. Our code is publicly available at https://github.com/LHRLAB/Graph-R1.

---

## 1. INTRODUCTION

### Figure 1: An illustration of Graph-R1

**Description of Figure 1:**
The figure shows an example query: "Who is the spouse of the director of film In Memory Of Sergo Ordzhonikidze?"

It illustrates the Graph-R1 agent performing multi-turn interaction with a Knowledge HyperGraph containing nodes representing:
- Films: "In Memory of Sergo Ordzhonikidze" (1937), "The Harsh Life of Veronica" (2009)
- People: Dziga Vertov, Yelizaveta Svilova, Nika Agiashvili, Boris Kaufman
- Occupations: Cinema Theorist

The nodes are connected by hyperedges (Hyperedge_1, Hyperedge_2, Hyperedge_3) with coherence scores (1.8, 1.2, 0.5, 1.5, 1.0).

The agent follows a reasoning process with three "Think" steps:
1. First query to find the director of the film
2. Second query to find the spouse of the director
3. Final answer: Yelizaveta Svilova

The figure demonstrates the multi-turn interaction loop of "Think-Query-Retrieve-Rethink-Generate" within the knowledge hypergraph environment.

---

Large Language Models (LLMs) (Zhao et al., 2025) have achieved widespread success in NLP tasks. However, when applied to knowledge-intensive or proprietary knowledge-dependent applications, they still suffer from the hallucination problem (Zhang et al., 2023), generating inaccurate content. To improve credibility and factual consistency, Retrieval-Augmented Generation (RAG) (Lewis et al., 2020) introduces external knowledge sources as references, alleviating the knowledge bottleneck of pure language modeling. Nevertheless, existing RAG methods mostly rely on chunk-based text blocks (Gao et al., 2024), which makes it difficult to capture complex knowledge structures among entities. To address this, GraphRAG methods (Edge et al., 2025; Guo et al., 2025; Luo et al., 2025a) represent knowledge as entity-relation graphs, enhancing retrieval efficiency and generation quality.

Generally, GraphRAG methods consist of three processes: **knowledge graph construction**, **graph retrieval**, and **answer generation**. First, knowledge graphs are typically constructed by LLMs to extract entities and relations from text, forming a graph structure (Xu et al., 2024). Second, the retrieval process queries relevant subgraphs or paths through subgraph retrieval or path pruning strategies (Chen et al., 2025; Gutiérrez et al., 2025). Finally, the generation process prompts LLMs to generate answers based on the retrieved graph-based knowledge (Xiao et al., 2025).

### Key Challenges

However, current GraphRAG methods still face three key challenges:

**(i) High cost and semantic loss in knowledge construction process.** Compared to standard RAG, GraphRAG methods convert natural language knowledge into graph structures using LLMs, which results in high cost and often causes semantic loss relative to the original content (Luo et al., 2024; 2025a).

**(ii) Fixed retrieval process with only one-time interaction in graph retrieval process.** Although existing GraphRAG methods design various retrieval strategies to improve efficiency, they all aim to gather sufficient knowledge in a single fixed retrieval (Chen et al., 2025), which limits performance in complex queries.

**(iii) Dependence on large LLMs for long-context analysis and prompt quality in answer generation process.** Generation based on retrieved graph-structured knowledge often requires strong long-context reasoning ability, making the output quality highly dependent on the LLM's parameter size and prompt design (Guo et al., 2025), resulting in unstable reasoning and generation.

### Our Contribution: Graph-R1

To address these challenges, we propose **Graph-R1**, as illustrated in Figure 1, an agentic GraphRAG framework enhanced by end-to-end reinforcement learning (RL), inspired by DeepSeek-R1 (DeepSeek-AI et al., 2025).

First, we propose a **lightweight knowledge hypergraph construction** method to establish a standard agent environment for the query action space.

Moreover, we model the **retrieval process as a multi-turn agentic interaction process**, enabling LLMs to repeatedly perform the reasoning loop of "think-retrieve-rethink-generate" within the knowledge hypergraph environment.

Furthermore, we design an **end-to-end reward mechanism** that integrates generation quality, retrieval relevance, and structural reliability of graph paths into a unified optimization objective. Using RL, the agent learns a generalizable graph reasoning strategy and achieves tighter alignment between structured graph-based knowledge and language generation.

We perform experiments on various standard RAG datasets (Jin et al., 2025b). Experimental results demonstrate that Graph-R1 outperforms traditional GraphRAG methods and RAG combined with RL methods (Jin et al., 2025a; Song et al., 2025) in reasoning accuracy, retrieval efficiency, and generation quality. As shown in Figure 2, the end-to-end RL strategy guides the agent through multiple turns of interaction and goal-driven exploration in the graph, effectively bridging the gap between knowledge representation and language generation. This work lays a foundation for building the next generation of knowledge-driven and strategy-optimized agent-based generation systems.

---

## Figure 2: Comparison of F1 scores across RAG benchmarks

**Description of Figure 2:**
This bar chart compares F1 scores across six RAG benchmarks (2WikiMultiHopQA, HotpotQA, Musique, NQ, PopQA, TriviaQA, and Average) for seven methods:

1. **NaiveGeneration** (light blue): Performs poorest across all datasets, with F1 scores ranging from 11.4% (Musique) to 47.7% (TriviaQA), averaging 25.9%

2. **StandardRAG** (green striped): Shows improvement over naive generation, with scores from 17.3% (Musique) to 48.5% (TriviaQA), averaging 32.0%

3. **GraphRAG** (purple): Performance similar to StandardRAG, ranging from 15.1% to 45.1%, averaging 24.9%

4. **HyperGraphRAG** (blue): Moderate performance, ranging from 20.4% to 45.0%, averaging 29.4%

5. **Search-R1** (pink striped): Strong performance, ranging from 22.4% to 66.0%, averaging 46.2%

6. **R1-Searcher** (orange striped): Similar to Search-R1, ranging from 16.6% to 64.8%, averaging 42.3%

7. **Graph-R1 (Ours)** (dark purple): Best overall performance, ranging from 46.2% (Musique) to 71.9% (TriviaQA), averaging 57.8%

The chart demonstrates that using a graph as the knowledge environment enables RL to achieve a higher performance ceiling compared to chunk-based knowledge.

---

## 2. RELATED WORK

### RAG and GraphRAG

Retrieval-Augmented Generation (RAG) (Lewis et al., 2020) improves LLM factuality by retrieving external knowledge, but suffers from data silos and limited structural understanding. GraphRAG (Edge et al., 2025) addresses these limitations by leveraging graph-structured knowledge to enhance retrieval and reasoning. Based on this, enterprise-oriented systems (Wu et al., 2024; Liang et al., 2025; Wang et al., 2025a) and efficient variants like LightRAG (Guo et al., 2025) are proposed. Recent efforts further extend representation power via hypergraphs, causal graphs, or heterogeneous graphs (Luo et al., 2025a; Feng et al., 2025b; Wang et al., 2025b; Xu et al., 2025), while retrieval is optimized using path-based exploration and pruning techniques (Chen et al., 2025; Gutiérrez et al., 2025; Liu et al., 2025; Wang, 2025). Our work introduces **Graph-R1**, the first agentic GraphRAG framework with end-to-end reinforcement learning.

### Reinforcement Learning for LLMs

Reinforcement learning (RL) is increasingly adopted to enhance LLM reasoning (Wu, 2025; Luo et al., 2025b), as demonstrated by OpenAI's o1/o3/o4 (OpenAI et al., 2024b). DeepSeek-R1 (DeepSeek-AI et al., 2025) achieves comparable capabilities and further introduces the Group Relative Policy Optimization (GRPO) (Shao et al., 2024) for scalable end-to-end training. GRPO-based reasoning has been extended to tasks such as visual understanding (Shen et al., 2025), logical reasoning (Xie et al., 2025), and program synthesis (Ma et al., 2025). RL-enhanced agents have also shown strong performance in multi-turn interaction (Lu et al., 2025; Feng et al., 2025a) and open-domain retrieval (Jin et al., 2025a; Song et al., 2025; Zheng et al., 2025; Sun et al., 2025), highlighting RL's potential in agentic GraphRAG frameworks (Gao et al., 2025).

---

## 3. PRELIMINARIES

We formalize the GraphRAG pipeline into three stages as detailed below:

### (a) Knowledge Graph Construction

This stage extracts structured relational facts from raw text. Given a knowledge collection K = {d₁, d₂, ..., dₙ}, the goal is to extract facts fₐ from each semantic unit d ∈ K and aggregate them into a unified graph Gₖ:

```
Gₖ ~ Σ πₑₓₜ(fₐ | d)    (Equation 1)
    d∈K
```

where πₑₓₜ denotes an LLM-based extractor that parses each d into a set of relation-entity pairs fₐ = {(rᵢ, Vᵣᵢ)}, with rᵢ as the relation and Vᵣᵢ = {v₁, ..., vₙ} the participating entities.

### (b) Graph Retrieval

Graph retrieval is formulated as a two-step process over Gₖ: (1) retrieving candidate reasoning paths and (2) pruning irrelevant ones. Conditioned on a query q, the model first retrieves a candidate set Xq = {x₁, ..., xₘ} and then selects a relevant subset Zq ⊆ Xq. The overall objective is to maximize the expected joint likelihood of the two steps:

```
max Eₖ~P(Zq|q,Gₖ) [∏ Pθ(xₜ | x<ₜ, q, Gₖ) · ∏ Pθ(zₜ | z<ₜ, Xq, q)]    (Equation 2)
 θ                  t=1                        t=1
```

where Tₓ and Tᵤ denote the number of retrieved and selected paths, respectively.

### (c) Answer Generation

Given a query q and selected paths Zq, answer generation produces a natural language answer y grounded in graph-based evidence, formulated as:

```
P(y | q, Gₖ) = Σ P(y | q, Zq) · P(Zq | q, Gₖ)    (Equation 3)
             Zq⊆Xq
```

where P(y | q, Zq) is generation likelihood and P(Zq | q, Gₖ) is retrieval-pruning distribution.

---

## 4. METHODOLOGY: GRAPH-R1

In this section, as illustrated in Figure 3, we introduce Graph-R1, including agent initialization, multi-turn graph interaction, and outcome-directed end-to-end reinforcement learning.

### Figure 3: Overview of the Graph-R1 framework

**Description of Figure 3:**
This diagram shows the complete Graph-R1 framework workflow:

**Left side - Knowledge Construction:**
- Domain Knowledge K (shown as document icons)
- Extractor (LLM icon) processes documents
- Produces N-ary Facts

**Center - Knowledge HyperGraph:**
A graph showing medical knowledge with nodes:
- "Hypertension" (central node)
- Connected to: "Systolic BP ≥140 mmHg", "Diastolic BP ≥90 mmHg"
- "Cardiovascular Disease"
- "Low-sodium Diet"
- "Regular Exercise"

Connected by two hyperedges (Hyperedge_1 and Hyperedge_2)

**Right side - Agent Interaction:**
- User Question input
- Graph-R1 Agent performs:
  - <Think> (reflection)
  - Knowledge Query
  - <Retrieved Knowledge>
  - Choice to either:
    - Continue (Rethink and query again)
    - Terminate (<Answer>)

**Bottom - End-to-end Reinforcement Learning:**
- Process: JGRPO(θ) with reward calculation
- F1+Format Reward feedback loop

The diagram illustrates an RL-enhanced reasoning trajectory over knowledge hypergraph, where the agent iteratively decides to think, query, retrieve knowledge, and answer.

---

## 4.1 KNOWLEDGE CONSTRUCTION AND AGENT INITIALIZATION

Graph-R1 adopts an LLM-driven agent, initialized with a knowledge hypergraph environment Gₕ, the action space A, the state space S, and the answer target yq for the given query q.

### Graph Environment Gₕ

To support agentic reasoning, we propose a lightweight method for constructing a knowledge hypergraph Gₕ from given domain knowledge K = {d₁, d₂, ..., dₙ}. For each chunk unit d ∈ K, an LLM-based extractor πₑₓₜ identifies m n-ary relational facts, where each comprises a semantic segment hᵢ and a set of participating entities Vₕᵢ = {v₁, ..., vₙ}. A shared encoder φ(·) is then used to generate semantic embeddings for both entities and relations:

```
Gₕ = (V, Eₕ, φ), where πₑₓₜ(d) → {(hᵢ, Vₕᵢ)}ᵐᵢ₌₁, φ(v) = Enc(v), φ(hᵢ) = Enc(hᵢ)    (Equation 4)
```

where each hᵢ defines a hyperedge hᵢ ∈ Eₕ connecting its associated entities Vₕᵢ as v ∈ V. The resulting hypergraph Gₕ encodes high-order relational structures with rich semantic grounding.

### The Agent Action Space A

In Graph-R1, each agent action aₜ ∈ A comprises four sub-actions:

- **Thinking** aᵗʰⁱⁿᵏ: which decides whether to continue or terminate reasoning
- **Query Generation** aᵍᵘᵉʳʸ: which formulates a retrieval query
- **Graph Retrieval** aʳᵉᵗ: which extracts relevant knowledge from the hypergraph
- **Answering** aᵃⁿˢ: which produces a final response if reasoning ends

The agent action aₜ has two compositional forms, and the joint action log-likelihood is defined as:

```
           ⎧ log Gₕ(aʳᵉᵗ | sₜ, aᵗʰⁱⁿᵏ, aᵍᵘᵉʳʸ) + log π(aᵍᵘᵉʳʸ | sₜ, aᵗʰⁱⁿᵏ) +
log π(aₜ | sₜ) = ⎨ log π(aᵗʰⁱⁿᵏ | sₜ),  if aᵗʰⁱⁿᵏ → continue,
           ⎩ log π(aᵃⁿˢ | sₜ, aᵗʰⁱⁿᵏ) + log π(aᵗʰⁱⁿᵏ | sₜ),  if aᵗʰⁱⁿᵏ → terminate,

(Equation 5)
```

where, at each step, the agent first performs Thinking, and then conditionally chooses between continuing reasoning (Query Generation and Graph Retrieval) or terminating via Answering.

### The Agent State Space S and Target yq

At each step t, the state sₜ ∈ S is defined as sₜ = (s₁, a₁, ..., aₜ₋₁), with s₁ initialized from the input query q. Once a termination action aₜ is issued, the agent reaches final state sₜ, where T is the total number of reasoning steps, and an answer yq ~ aᵃⁿˢₜ is produced to address q.

**Proposition 1.** Graph-structured knowledge boosts agent accuracy by richer representation.

*Proof.* We provide experimental results in Section 5.2 and theoretical proofs in Appendix B.1.

---

## 4.2 KNOWLEDGE REASONING VIA MULTI-TURN GRAPH INTERACTION

We model reasoning as a multi-turn interaction between an agent πθ and a hypergraph Gₕ. We first define the step-wise policy πθ(· | sₜ) prompted by Table 1, then describe how to retrieve knowledge Gₕ(aʳᵉᵗ | ·, aᵍᵘᵉʳʸ) based on aᵍᵘᵉʳʸ in each step, and finally present the objective to optimize P(yq | ·).

### Table 1: Template for Graph-R1

```
You are a helpful assistant. Answer the given question. You can query from knowledge base
provided to you to answer the question. You can query knowledge as many times as you
want. You must first conduct reasoning inside <think>...</think>. If you need to query
knowledge, you can set a query statement between <query>...</query> to query from
knowledge base after <think>...</think>. When you have the final answer, you can
output the answer inside <answer>...</answer>. Question: question. Assistant:
```

Note: `question` will be replaced with the specific user query. The knowledge retrieved is placed within `<knowledge>...</knowledge>` after `</query>`.

### Modeling the Step-wise Reasoning Policy

At each reasoning step t, the LLM governs the agent's behavior by generating a structured output consisting of:
- (i) a thinking reflection aᵗʰⁱⁿᵏ that summarizes the current state and highlights potential knowledge gaps
- (ii) a composition indicator αₜ ∈ Aₜᵧₚₑ = {(query, retrieve), (answer)} that determines the sub-action structure
- (iii) a content output aᵒᵘᵗ ∈ Aᶜᵒⁿᵗᵉⁿᵗ, representing either a retrieval query or a final answer

We model this decision-making process as a hierarchical policy conditioned on the agent state sₜ ∈ S, which encodes the history of prior actions and retrieved information. The policy is factorized as:

```
πθ(aᵗʰⁱⁿᵏ, αₜ, aᵒᵘᵗ | sₜ) = πθ(aᵒᵘᵗ | αₜ, aᵗʰⁱⁿᵏ, sₜ) · πθ(αₜ | aᵗʰⁱⁿᵏ, sₜ) · πθ(aᵗʰⁱⁿᵏ | sₜ)    (Equation 6)
```

where πθ denotes the LLM-parameterized policy, which encourages three aligned behaviors: generating reflections aᵗʰⁱⁿᵏ that assess knowledge sufficiency, selecting αₜ to balance exploration and termination, and producing aᵒᵘᵗ that advances retrieval aᵍᵘᵉʳʸ or yields a direct answer aᵃⁿˢ.

### Knowledge Interaction via Hypergraph Retrieval

Given a query aᵍᵘᵉʳʸ generated by the reasoning LLM, we retrieve relevant knowledge aʳᵉᵗ from the hypergraph Gₕ = (V, Eₕ) through a dual-path interaction process: entity-based retrieval and direct hyperedge retrieval. The resulting n-ary relational facts are then aggregated via rank-based fusion to support downstream reasoning.

#### (i) Entity-based Hyperedge Retrieval

We first identify a set of top-ranked entities based on their similarity to the extracted entities Vₐᵍᵘᵉʳʸ, and collect hyperedges that connect to any retrieved entity:

```
Rᵥ(aᵍᵘᵉʳʸ) = kᵥargmax sim(φ(Vₐᵍᵘᵉʳʸ), φ(v)),  F*ᵥ = ⋃ {(eₕ, Vₑₕ) | vᵢ ∈ Vₑₕ, eₕ ∈ Eₕ}    (Equation 7)
              v∈V                                    vᵢ∈Rᵥ
```

where φ(Vₐᵍᵘᵉʳʸ) is the aggregated embedding of entities extracted from aᵍᵘᵉʳʸ, φ(v) is the entity embedding, kᵥ is the number of retrieved entities, and Vₑₕ denotes the entity set of hyperedge eₕ.

#### (ii) Direct Hyperedge Retrieval

In parallel, we directly retrieve hyperedges based on query-hyperedge similarity, and collect their associated relational facts:

```
Rₕ(aᵍᵘᵉʳʸ) = kₕargmax sim(φ(aᵍᵘᵉʳʸ), φ(eₕ)),  F*ₕ = ⋃ {(eᵢ, Vₑᵢ) | Vₑᵢ ⊆ V}    (Equation 8)
              eₕ∈Eₕ                                    eᵢ∈Rₕ
```

where φ(aᵍᵘᵉʳʸ) is the query embedding, φ(eₕ) is the hyperedge embedding, kₕ is the number of retrieved hyperedges, and Vₑᵢ denotes the entity set of hyperedge eᵢ.

#### (iii) Fusion via Reciprocal Rank Aggregation

To produce the final knowledge set, we merge results from both retrieval paths using reciprocal rank aggregation over hyperedges:

```
aʳᵉᵗ = F*ₐᵍᵘᵉʳʸ = Top-k(F*ᵥ ∪ F*ₕ, RankScore(f) = 1/rᵥ + 1/rₕ | ₐᵍᵘᵉʳʸ)    (Equation 9)
```

where rᵥ and rₕ are the ranks of n-ary relational fact f in F*ᵥ and F*ₕ respectively (set to ∞ if absent), and k is the number of retrieved facts aʳᵉᵗ returned to the agent.

### Optimization Objective for Agent Trajectories

The agent aims to learn a reasoning trajectory τ ∈ Tq that yields a faithful and contextually grounded answer yq. Each trajectory τ = ((s₁, a₁), (s₂, a₂), ..., (sₜ, aₜ)) comprises a sequence of actions executed over Gₕ, defined as:

```
max Eτ~πθ(Tq|q;Gₕ)[log P(yq | τ)]    (Equation 10)
 θ
```

where P(yq | τ) denotes the likelihood of the correct answer yq ~ aᵃⁿˢ under trajectory τ, guiding πθ toward answer-consistent reasoning.

**Proposition 2.** Multi-turn interaction with the graph environment improves retrieval efficiency.

*Proof.* We provide experimental results in Section 5.5 and theoretical proofs in Appendix B.2.

---

## 4.3 OUTCOME-DIRECTED END-TO-END REINFORCEMENT LEARNING

To optimize the reasoning policy πθ toward generating faithful and well-structured answers, we adopt an end-to-end reinforcement learning objective based on Group Relative Policy Optimization (GRPO) (Shao et al., 2024) Jɢʀᴘᴏ(θ) and design an outcome-directed reward function R(τ).

### End-to-end RL Objective Jɢʀᴘᴏ(θ)

Given a dataset question q ∈ Dǫ, the agent interacts with the knowledge hypergraph Gₕ to generate a group of multi-turn reasoning trajectories {τᵢ}ᴺᵢ₌₁ ⊆ Tq, where each τᵢ = ((s⁽ⁱ⁾₁, a⁽ⁱ⁾₁), ..., (s⁽ⁱ⁾ₜ, a⁽ⁱ⁾ₜ)) denotes a sequence of state-action pairs sampled from the environment. We optimize the policy πθ using the GRPO-based objective, which is defined as:

```
Jɢʀᴘᴏ(θ) = E[s₁~{P(q)|q∈Dǫ}, {τᵢ}ᴺᵢ₌₁~πθₒₗₐ(Tq|s₁;Gₕ)]
           [1/N Σᴺᵢ₌₁ 1/|τᵢ| Σ|τᵢ|ₜ₌₁ min(ρθ(a⁽ⁱ⁾ₜ)Â(τᵢ), clip(ρθ(a⁽ⁱ⁾ₜ), 1±ε)Â(τᵢ)) - β Dᴋʟ(πθ∥πʀᴇꜰ)]

(Equation 11)
```

where:

```
ρθ(a⁽ⁱ⁾ₜ) = πθ(a⁽ⁱ⁾ₜ | s⁽ⁱ⁾ₜ₋₁; Gₕ) / πθₒₗₐ(a⁽ⁱ⁾ₜ | s⁽ⁱ⁾ₜ₋₁; Gₕ)

Â(τᵢ) = (R(τᵢ) - mean{R(τⱼ)}ᴺⱼ₌₁) / Fₙₒᵣₘ({R(τⱼ)}ᴺⱼ₌₁)    (Equation 12)
```

Here, πθ is the current policy, and πθₒₗₐ is the behavior policy used for sampling. The importance ratio ρθ(a⁽ⁱ⁾ₜ) adjusts for distribution shift, while the advantage Â(τᵢ) normalizes the reward using a scaling function Fₙₒᵣₘ(·) (e.g., standard deviation). The clip(·) operator stabilizes updates by constraining policy shifts. A KL term Dᴋʟ(πθ ∥ πʀᴇꜰ) regularizes toward a reference policy πʀᴇꜰ, with β controlling its strength. This objective encourages high-reward, stable reasoning over Gₕ.

### Outcome-directed Reward Function R(τ)

To meet outcome requirements, we define a reward function R(τ) composed of two parts: a format reward Rꜰᴏʀᴍᴀᴛ(τ) and an answer reward Rᴀɴsᴡᴇʀ(aᵃⁿˢₜ), promoting both thoughtful retrieval and accurate answer generation.

#### (i) Format Reward

The format reward Rꜰᴏʀᴍᴀᴛ(τ) encourages the agent to follow the intended reasoning structure. At each step (sₜ, aₜ), we check whether the output includes a well-formed block (aᵗʰⁱⁿᵏ, αₜ, aᵒᵘᵗ). Each valid step receives 0.5 reward, capped at 1.0 overall:

```
Rꜰᴏʀᴍᴀᴛ(τ) = min(1.0, 0.5 · Σᵀₜ₌₁ I{(aᵗʰⁱⁿᵏ, αₜ, aᵒᵘᵗ) is well-formed})    (Equation 13)
```

where I{·} is an indicator function that returns 1 if the step output matches the expected format.

#### (ii) Answer Reward

The answer reward Rᴀɴsᴡᴇʀ(aᵃⁿˢₜ) measures the semantic correctness of the generated answer aᵃⁿˢₜ by comparing it with the ground-truth answer y*q using a token-level F1 score:

```
Rᴀɴsᴡᴇʀ(aᵃⁿˢₜ) = (2 · |tokens(aᵃⁿˢₜ) ∩ tokens(y*q)|) / (|tokens(aᵃⁿˢₜ)| + |tokens(y*q)|)    (Equation 14)
```

where | · | denotes multiset cardinality. The function tokens(·) applies standard preprocessing including lowercasing and whitespace-based tokenization.

#### (iii) Overall Outcome Reward

The total reward for a reasoning trajectory τ is defined as:

```
R(τ) = -1.0 + Rꜰᴏʀᴍᴀᴛ(τ) + I{Rꜰᴏʀᴍᴀᴛ(τ) = 1.0} · Rᴀɴsᴡᴇʀ(aᵃⁿˢₜ), where aᵃⁿˢₜ ∈ τ    (Equation 15)
```

ensuring that answer correctness is only rewarded when the format is structurally valid. With the outcome-directed reward R(τ), high answer quality aᵃⁿˢₜ is attainable through structurally coherent and reasoning-complete trajectories τ with multi-turn iteration with knowledge hypergraph Gₕ.

**Proposition 3.** End-to-end RL bridges the gap between graph-based knowledge and language.

*Proof.* We provide experimental results in Section 5.6 and theoretical proofs in Appendix B.3.

---

## Figure 4: Step-wise F1 score on HotpotQA

**Description of Figure 4:**
Three line graphs showing F1 score progression over 40 training steps for different model sizes:

**(a) Qwen2.5-1.5B-Instruct:**
- Graph-R1 (1.5B) reaches ~0.4 F1, outperforming all baselines
- R1-Searcher (1.5B) reaches ~0.35 F1
- Search-R1 (1.5B) reaches ~0.33 F1
- R1 (1.5B) reaches ~0.2 F1
- NaiveGeneration, StandardRAG, and HyperGraphRAG (GPT-4o-mini) shown as reference lines

**(b) Qwen2.5-3B-Instruct:**
- Graph-R1 (3B) reaches ~0.55 F1, significantly better than all baselines
- R1-Searcher (3B) and Search-R1 (3B) reach ~0.42 F1
- R1 (3B) reaches ~0.25 F1
- GPT-4o-mini baselines shown for comparison

**(c) Qwen2.5-7B-Instruct:**
- Graph-R1 (7B) reaches ~0.63 F1, best overall performance
- Search-R1 (7B) reaches ~0.5 F1
- R1-Searcher (7B) reaches ~0.46 F1
- R1 (7B) reaches ~0.37 F1
- GPT-4o-mini baselines shown for comparison

All graphs show that Graph-R1 consistently outperforms baselines and even GPT-4o-mini variants across all model sizes, with the performance gap increasing as model size increases.

---

## 5. EXPERIMENTS

This section presents the experimental setup, main results, and analysis. We answer the following research questions (RQs):
- **RQ1:** Does Graph-R1 outperform other methods?
- **RQ2:** Does the main component of Graph-R1 work, and how is its comparative analysis?
- **RQ3-6:** How are construction cost, retrieval efficiency, generation quality, and generalizability of Graph-R1, respectively?

### 5.1 EXPERIMENTAL SETUP

#### Datasets

To evaluate the performance of Graph-R1, we conduct experiments across six standard RAG datasets (Jin et al., 2025b):

1. **2WikiMultiHopQA (2Wiki.)** (Ho et al., 2020): A multi-hop dataset requiring reasoning across two Wikipedia documents
2. **HotpotQA** (Yang et al., 2018): A challenging multi-hop QA dataset with sentence-level supporting facts
3. **Musique** (Trivedi et al., 2022): Multi-hop questions needing chains of inference
4. **Natural Questions (NQ)** (Kwiatkowski et al., 2019): Large-scale single-hop QA grounded in real Google searches
5. **PopQA** (Mallen et al., 2023): Open-domain QA focused on popular culture
6. **TriviaQA** (Joshi et al., 2017): Large-scale trivia-style questions

More details are in Appendix D.

#### Baselines

We mainly compare Graph-R1 with:
- **NaiveGeneration, StandardRAG** (Lewis et al., 2020)
- **SFT** (Zheng et al., 2024)
- **R1** (Shao et al., 2024)
- **Search-R1** (Jin et al., 2025a)
- **R1-Searcher** (Song et al., 2025)

at three Qwen2.5 (Qwen et al., 2025) scales: 1.5B, 3B, and 7B.

We also compare:
- **GraphRAG** (Edge et al., 2025)
- **LightRAG** (Guo et al., 2025)
- **PathRAG** (Chen et al., 2025)
- **HippoRAG2** (Gutiérrez et al., 2025)
- **HyperGraphRAG** (Luo et al., 2025a)

based on GPT-4o-mini (OpenAI et al., 2024a) as a reference. More details are in Appendix E.

#### Evaluation Metrics

We evaluate Graph-R1 and baselines with four metrics:
- **Exact Match (EM)**
- **F1**
- **Retrieval Similarity (R-S)**
- **Generation Evaluation (G-E)**

More details are in Appendix F.

#### Implementation Details

We use GPT-4o-mini for knowledge construction in Graph-R1 and GraphRAG baselines. For retrieval, we use bge-large-en-v1.5 (Chen et al., 2023) in all variants. All experiments are done on 4 NVIDIA A100 GPUs (80GB). More details are in Appendix G.

---

## 5.2 MAIN RESULTS (RQ1)

### Table 2: Main results with best in bold

**Table Description:**
The table compares multiple methods across six datasets (2Wiki., HotpotQA, Musique, NQ, PopQA, TriviaQA) using F1, G-E (Generation Evaluation), EM (Exact Match), and R-S (Retrieval Similarity) metrics.

**Key Findings:**

**GPT-4o-mini baselines:**
- NaiveGeneration: Average F1 = 25.92
- StandardRAG: Average F1 = 32.05
- GraphRAG: Average F1 = 24.87
- LightRAG: Average F1 = 23.57
- PathRAG: Average F1 = 20.02
- HippoRAG2: Average F1 = 25.49
- HyperGraphRAG: Average F1 = 29.40 (best GPT-4o-mini method)

**Qwen2.5-1.5B-Instruct:**
- NaiveGeneration: Average F1 = 6.43
- StandardRAG: Average F1 = 11.08
- SFT: Average F1 = 14.23
- R1: Average F1 = 20.68
- Search-R1: Average F1 = 29.53
- R1-Searcher: Average F1 = 32.26
- **Graph-R1 (ours): Average F1 = 40.09** ✓

**Qwen2.5-3B-Instruct:**
- NaiveGeneration: Average F1 = 8.85
- StandardRAG: Average F1 = 13.03
- SFT: Average F1 = 15.85
- R1: Average F1 = 26.40
- Search-R1: Average F1 = 35.69
- R1-Searcher: Average F1 = 34.91
- **Graph-R1 (ours): Average F1 = 51.26** ✓

**Qwen2.5-7B-Instruct:**
- NaiveGeneration: Average F1 = 13.87
- StandardRAG: Average F1 = 15.89
- SFT: Average F1 = 24.01
- R1: Average F1 = 33.12
- Search-R1: Average F1 = 46.19
- R1-Searcher: Average F1 = 42.29
- **Graph-R1 (ours): Average F1 = 57.82** ✓

As shown in Table 2, we compare Graph-R1 with baselines across different base models, and observe that Graph-R1 consistently outperforms all baselines. In addition, we have two key observations.

### RL Unlocks the Power of Graph Representations

Prompt-only GraphRAG methods often underperform StandardRAG, showing that graph structures alone are not sufficient. Graph-R1, with multi-turn RL optimization, fully exploits structural signals, achieving 57.28 F1 under Qwen2.5-7B-Instruct, surpassing StandardRAG (32.05), HyperGraphRAG (29.40) and Search-R1 (46.19).

### Larger Base Model Further Enhances Performance

As base model size increases from 1.5B to 3B and 7B, Graph-R1 achieves steadily higher F1 scores: 40.09, 51.26, and 57.82. Moreover, its gap over other RL-enhanced baselines such as Search-R1 and R1-Searcher becomes increasingly evident. This shows that larger models better exploit the synergy between graph structures and RL.

---

## 5.3 ABLATION STUDY AND COMPARATIVE ANALYSIS (RQ2)

### Figure 5: Ablation and Comparative Analysis

**Description of Figure 5:**

The figure contains 6 subfigures:

**(a) Ablation Study Table:**
Shows results on 2Wiki. and HotpotQA datasets for Qwen2.5-3B and 7B models:

**Qwen2.5-3B-Instruct:**
- Graph-R1 (full): EM=50.00, F1=57.56, R-S=55.78, G-E=76.45 (2Wiki); EM=50.78, F1=56.75, R-S=54.74, G-E=77.46 (HotpotQA)
- w/o K.C. (without Knowledge Construction): Significant drops across all metrics
- w/o M.I. (without Multi-turn Interaction): Further performance degradation
- w/o R.L. (without Reinforcement Learning): Dramatic drop to near-zero EM

**Qwen2.5-7B-Instruct:**
- Graph-R1 (full): EM=55.47, F1=65.04, R-S=55.24, G-E=82.42 (2Wiki); EM=57.03, F1=62.69, R-S=56.27, G-E=80.03 (HotpotQA)
- Similar ablation patterns showing all components are critical

**(b) Representations Comparison:**
Line graph comparing F1 scores over training steps for different knowledge representations:
- Graph-R1 (3B) with hypergraph: Best performance (~0.57 F1)
- R1+LightRAG (3B) with binary graph: Moderate performance (~0.32 F1)
- Search-R1 (3B) and R1-Searcher (3B) with chunks: Lower performance (~0.42 and ~0.44 F1)
- R1 (3B) without knowledge: Lowest performance (~0.25 F1)
- GPT-4o-mini baselines shown for reference

**(c) Datasets Comparison:**
F1 scores across datasets for Graph-R1 (3B):
- HotpotQA: ~0.57 F1
- 2WikiMultiHopQA: ~0.58 F1
- Musique: ~0.41 F1
- Compared with GPT-4o-mini baselines

**(d) Parameters Comparison:**
F1 progression for different model sizes:
- Graph-R1 (7B): Best, reaching ~0.63 F1
- Graph-R1 (3B): Moderate, reaching ~0.57 F1
- Graph-R1 (1.5B): Lower, reaching ~0.41 F1
- HyperGraphRAG (GPT-4o-mini) reference line at ~0.37 F1

**(e) Qwen3 Comparison:**
Comparing Qwen versions:
- Graph-R1 (Qwen2.5-3B-Instruct): ~0.57 F1
- Graph-R1 (Qwen3-4B): Slightly lower ~0.54 F1
- HyperGraphRAG (GPT-4o-mini) reference

**(f) Algorithms Comparison:**
Comparing RL algorithms:
- GRPO (Graph-R1 3B): Best, ~0.57 F1
- PPO (Graph-R1 3B): Moderate, ~0.52 F1
- REINFORCE++ (Graph-R1 3B): Lower, ~0.48 F1
- HyperGraphRAG (GPT-4o-mini) reference

As shown in Figures 5, we conduct an ablation study and comparative analysis on Graph-R1.

### Ablation Study

We remove three core components of Graph-R1: knowledge construction (K.C.), multi-turn interaction (M.I.), and reinforcement learning (R.L.), to assess their individual contributions. As shown in Figure 5(a), removing any module leads to performance degradation.

### Comparison with Different Knowledge Representations

As shown in Figures 4 and 5(b), models without external knowledge (green) perform the worst. Chunk-based knowledge with RL (blue) performs better, but is still inferior to graph-based methods using binary relations (pink), while hypergraph-based knowledge with RL (red) achieves the highest ceiling. This demonstrates that, when combined with RL, stronger knowledge representations yield higher performance potential.

### Comparison across Datasets and Base Models

As shown in Figures 5(c) and 5(d), Graph-R1 consistently outperforms baselines across different datasets and parameter sizes, showcasing strong scalability. Interestingly, Figure 5(e) shows that when Graph-R1 is trained on Qwen3 (4B) (Yang et al., 2025), which is already well trained by RL, the model tends to over-rely on its own internal reasoning. Despite a stronger starting point, its overall performance ceiling appears slightly lower.

### Comparison with Different RL Algorithms

Figure 5(f) compares different RL strategies. GRPO significantly outperforms REINFORCE++ (Hu et al., 2025) and PPO (Schulman et al., 2017), achieving the highest F1. This confirms that GRPO facilitates more stable training and stronger multi-turn graph reasoning, making it a favorable choice for training agentic GraphRAG models.

---

## 5.4 ANALYSIS OF GRAPH-R1'S CONSTRUCTION COST (RQ3)

### Table 3: Time & Cost Comparisons on 2Wiki.

**Table Description:**

| Method | Knowledge Construction | | | | Retrieval & Generation | | |
|--------|---------|---------|-------|-------|---------|---------|-----|
| | TP1KT | CP1MT | #Node | #Edge | TPQ | CP1KQ | F1 |
| NaiveGeneration | 0 s | 0 $ | - | - | 3.7 s | 0.16 $ | 17.0 |
| StandardRAG | 0 s | 0 $ | - | - | 4.1 s | 1.35 $ | 22.3 |
| GraphRAG | 8.04 s | 3.35 $ | 7,771 | 4,863 | 7.4 s | 3.97 $ | 16.0 |
| LightRAG | 6.84 s | 4.07 $ | 59,197 | 24,596 | 12.2 s | 8.11 $ | 16.6 |
| PathRAG | 6.84 s | 4.07 $ | 59,197 | 24,596 | 15.8 s | 8.28 $ | 12.4 |
| HippoRAG2 | 3.25 s | 1.26 $ | 11,819 | 40,654 | 8.8 s | 7.68 $ | 16.3 |
| HyperGraphRAG | 6.76 s | 4.14 $ | 173,575 | 114,426 | 9.6 s | 8.76 $ | 21.1 |
| **Graph-R1 (7B)** | **5.69 s** | **2.81 $** | **120,499** | **98,073** | **7.0 s** | **0 $** | **65.0** |

**Metrics:**
- TP1KT: Time per 1K tokens
- CP1MT: Cost per 1M tokens
- TPQ: Time per query
- CP1KQ: Cost per 1K queries

As shown in Table 3, we utilize metrics: time per 1K tokens (TP1KT), cost per 1M tokens (CP1MT), number of nodes & edges, time per query (TPQ), cost per 1K queries (CP1KQ), and final F1 score.

### Construction Cost

Graph-R1 requires only 5.69 seconds and $2.81 per 1K tokens for knowledge construction, lower than GraphRAG (8.04s, $3.35) and HyperGraphRAG (6.76s, $4.14). Generating over 120K nodes and 98K edges, Graph-R1 maintains a semantically rich structure.

### Generation Cost

By leveraging end-to-end RL and localized knowledge retrieval, Graph-R1 achieves not only the best F1 but also a response time of 7.0s per query and a generation cost of $0, outperforming baselines such as HyperGraphRAG (9.6s, $8.76), highlighting its superior potential for real-world deployment.

---

## 5.5 ANALYSIS OF GRAPH-R1'S RETRIEVAL EFFICIENCY (RQ4)

### Figure 6: Step-wise response length & turns of interaction, and efficiency comparison on HotpotQA

**Description of Figure 6:**

The figure contains three subplots:

**(a) Response Length:**
Line graph showing response length (number of tokens) over training steps:
- R1-Searcher (3B): Peaks around 2800 tokens, very high variance
- Search-R1 (3B): Fluctuates between 1500-2000 tokens
- Graph-R1 (3B) (Ours): Stabilizes around 1200-1500 tokens, most concise

**(b) Turns of Interaction:**
Line graph showing number of interaction turns over training steps:
- All three methods start around 1.0 turn
- R1-Searcher (3B): Increases to ~4.5 turns with high variance
- Search-R1 (3B): Stabilizes around 2.0-2.5 turns
- Graph-R1 (3B) (Ours): Converges to ~2.3-2.5 turns, most stable

**(c) Efficiency Comparison:**
Scatter plot of F1 Score vs. Average Content Length for all methods:
- X-axis: Average Content Length (0-2000 tokens)
- Y-axis: F1 Score (0-70%)

Methods plotted (with approximate positions):
- **Graph-R1 variants (1.5B, 3B, 7B):** Highest F1 scores (40-65%) with moderate content length (800-1400 tokens)
- **Search-R1 and R1-Searcher variants:** Moderate F1 (30-46%) with higher content length (1200-1800 tokens)
- **R1 variants:** Lower F1 (20-33%) with minimal content
- **GPT-4o-mini baselines:** Lower F1 (16-32%) with varying content lengths
- **HyperGraphRAG:** F1 ~29%, moderate content length

As shown in Figure 6, to evaluate Graph-R1's retrieval efficiency, we analyze it from (a) response length, (b) number of interaction turns, and (c) performance with average retrieval content lengths.

### Tendency toward More Concise Thinking and Adequate Interaction

As shown in Figures 6(a) and 6(b), Graph-R1 generates shorter responses and conducts more interaction turns, averaging around 1200-1500 tokens and 2.3-2.5 turns, leading to more stable and accurate retrieval.

### Balancing Performance and Retrieved Content Length

As shown in Figure 6(c), Graph-R1 achieves the highest F1 scores with a moderate amount of average retrieved content compared to other methods, balancing input length and performance through its multi-turn interaction strategy.

---

## 5.6 ANALYSIS OF GRAPH-R1'S GENERATION QUALITY (RQ5)

### Figure 7: Generation Evaluations

**Description of Figure 7:**

Radar/spider chart comparing seven methods across seven generation quality dimensions (0-100 scale):

**Dimensions:**
1. Overall
2. Comprehensiveness
3. Knowledgeability
4. Correctness
5. Relevance
6. Diversity
7. Logical Coherence
8. Factuality

**Methods compared:**
1. **NaiveGeneration** (innermost): Scores 41-67 across dimensions
2. **StandardRAG**: Scores 59-80
3. **HyperGraphRAG**: Scores 60-84
4. **R1 (7B)**: Scores 59-86
5. **R1-Searcher (7B)**: Scores 70-95
6. **Search-R1 (7B)**: Scores 70-88
7. **Graph-R1 (7B) (Ours)** (outermost): Highest scores 82-95

**Specific scores for Graph-R1 (7B):**
- Overall: 82.4
- Comprehensiveness: 83.2
- Knowledgeability: 84.8
- Correctness: 86.9
- Relevance: 95.2
- Diversity: 51.7
- Logical Coherence: 88.5
- Factuality: 86.6

As shown in Figure 7, we evaluate the generation quality in seven dimensions and present a case study in Table 4.

### High-Quality Generation Performance

Graph-R1 outperforms all RL-based baselines and achieves generation quality comparable to GPT-4o-mini-based methods like HyperGraphRAG, with strong results in Correctness (86.9), Relevance (95.2), and Logical Coherence (88.5).

### RL Bridges the Gap Between Graph & Language

HyperGraphRAG performs similarly to StandardRAG, indicating limited gains from graph structure alone. In contrast, Graph-R1 achieves a much higher Overall score (82.4 vs. 70.3) than Search-R1, showing that graph-based reasoning becomes truly effective when combined with RL.

---

## Table 4: Case study on generation quality

**Query:** When was the director of film Ingmar's Inheritance born?
**Golden Answers:** ['18 November 1888']

### GPT-4o-mini Methods:

**NaiveGeneration:**
```
<think>
To answer this question, I first need to identify who the director of "Ingmar's Inheritance" is.
The film "Ingmar's Inheritance" is directed by a notable filmmaker. Researching the film, it
turns out that it was directed by the Swedish director, "Ingmar Bergman." I need to check the
birth date of Ingmar Bergman to provide the correct answer. Ingmar Bergman was born on July 14,
1918. Since the question is specifically asking about the birth of the director of the film rather
than the film's release date or other details, I will now formulate the answer based on this
information.
</think>

<answer>
Ingmar Bergman was born on July 14, 1918.
</answer>
```
**Evaluation:** F1=0.00, R-S=-, G-E=55.71

**StandardRAG:**
```
<think>
The question asks for the birth year of the director of the film "Ingmar's Inheritance." However,
the provided knowledge does not include information about the film "Ingmar's Inheritance" or its
director...
</think>

<answer>
The information necessary to answer the question about the director of "Ingmar's Inheritance" is
not available in the provided knowledge. Therefore, I cannot provide a specific birth date for that
director. However, Bille August, a noted director with connections to Ingmar Bergman, was born on
November 9, 1948. It's unclear if he is associated with "Ingmar's Inheritance."
</answer>
```
**Evaluation:** F1=3.70, R-S=39.52, G-E=55.71

**HyperGraphRAG:**
```
<think>
To answer the question, I need to identify the director of the film "Ingmar's Inheritance." The
information available mentions that "Ingmar's Inheritance" is a silent drama film directed by Gustaf
Molander. The provided data does not specifically give a birthdate for Gustaf Molander. However, I
find that the knowledge base also contains information about Ernst Ingmar Bergman, a prominent
director, with dates of birth and death, but he is not related to "Ingmar's Inheritance." Since only
Gustaf Molander is mentioned as the director of that specific film, and I don't have his birthdate
recorded in the provided knowledge, I cannot provide an answer to this question.
</think>

<answer>
The birthdate of Gustaf Molander, the director of "Ingmar's Inheritance," is not provided in the
knowledge base, so I cannot answer the question.
</answer>
```
**Evaluation:** F1=0.00, R-S=38.93, G-E=60.00

### Qwen2.5-7B-Instruct Methods:

**R1 (7B):**
```
<think>
To answer when the director of the film "Ingmar's Inheritance" was born, I first need to find out
who the director of this film is. The film "Ingmar's Inheritance" was directed by Garth Davis. Now I
need to find out when Garth Davis was born. Garth Davis was born on December 26, 1970. Therefore,
the director of the film "Ingmar's Inheritance" was born on December 26, 1970.
</think>

<answer>
December 26, 1970
</answer>
```
**Evaluation:** F1=0.00, R-S=-, G-E=41.43

**Search-R1 (7B):**
```
<think>
To answer this question, I need to find the director of the film "Ingmar's Inheritance" and then
determine their birth date.
</think>

<query>
director of Ingmar's Inheritance
</query>

<knowledge>
{"results": ["Weizäcker studied at the University of Tübingen. While serving in the administration
of King William II of Württemberg, he was raised to personal nobility on 24 February 1897...", ...]}
</knowledge>

<think>
From the information provided, Baldvin Zophóníasson is the director of "Ingmar's Inheritance" and he
was born in 1978 in Akureyri.
</think>

<answer>
1978
</answer>
```
**Evaluation:** F1=0.00, R-S=40.02, G-E=64.29

**Graph-R1 (7B) (Ours):**
```
<think>
To answer this question, I need to find out the director of the film "Ingmar's Inheritance" and then
determine the director's birth year. Let's start by finding the director of the film.
</think>

<query>
director of Ingmar's Inheritance
</query>

<knowledge>
{"results": [{"<knowledge>": "Ernst Ingmar Bergman (14 July 1918 – 30 July 2007) was a Swedish
director, writer, and producer who worked in film, television, theatre and radio.", "<coherence>":
1.25}, {...}, ...]}
</knowledge>

<think>
From the knowledge query, it seems that "Ingmar's Inheritance" was directed by Gustaf Molander. Now,
I need to find out Gustaf Molander's birth year to answer the question.
</think>

<query>
Gustaf Molander birth year
</query>

<knowledge>
{"results": [{"<knowledge>": "Gustaf Molander was born in Helsingfors (now Helsinki) in the Grand
Duchy of Finland (in the Russian Empire), where his father was working at the Swedish Theatre.",
"<coherence>": 1.5}, {...}, ...]}
</knowledge>

<think>
From the knowledge query, we found that Gustaf Molander was born on November 18, 1888. This is the
director of the film "Ingmar's Inheritance".
</think>

<answer>
The director of the film "Ingmar's Inheritance", Gustaf Molander, was born on November 18, 1888.
</answer>
```
**Evaluation:** F1=37.50, R-S=45.83, G-E=88.57

### Case Study Analysis

As shown in Table 4, NaiveGeneration and R1 fail to provide the correct answer, and both StandardRAG and HyperGraphRAG also fall short despite using structured prompts. Search-R1, though RL-enhanced, shows limited improvement due to weak retrieval grounding. In contrast, **Graph-R1 accurately identifies both the director and birthdate**, achieves the highest G-E score (88.57), and demonstrates that RL is more effective with graph-based knowledge interaction.

---

## 5.7 ANALYSIS OF GRAPH-R1'S GENERALIZABILITY ON O.O.D. SETTINGS (RQ6)

### Figure 8: F1 comparison and performance ratios across six datasets under O.O.D. cross-validation

**Description of Figure 8:**

Four heatmaps showing out-of-distribution (O.O.D.) generalization:

**(a) Search-R1 (F1):**
- Rows: Training datasets
- Columns: Test datasets
- Diagonal (I.I.D.): 38.04, 43.84, 7.65, 37.96, 38.67, 47.99
- Off-diagonal (O.O.D.): Shows significant performance drops
- Color scale: 0-70% F1

**(b) Graph-R1 (F1):**
- Diagonal (I.I.D.): 57.56, 56.75, 40.51, 44.75, 45.65, 62.31
- Off-diagonal (O.O.D.): Better maintained performance than Search-R1
- Most O.O.D. values above 35%, many above 45%
- Color scale: 0-70% F1

**(c) Search-R1 (Ratio):**
- O.O.D. to I.I.D. performance ratios
- Diagonal: 100% (by definition)
- Off-diagonal: Ranges from 40% to 96%
- Many values below 70%, indicating poor O.O.D. generalization
- Color scale: 40-100%

**(d) Graph-R1 (Ratio):**
- O.O.D. to I.I.D. performance ratios
- Diagonal: 100% (by definition)
- Off-diagonal: Most values above 80%, many above 90%
- Much better O.O.D. generalization than Search-R1
- Color scale: 40-100%

As shown in Figure 8, to verify generalization, we conduct O.O.D. cross-validation for Search-R1 (3B) & Graph-R1 (3B) across six datasets: (a-b) F1 comparison, and (c-d) O.O.D.-to-I.I.D. ratios.

### F1 Performance Across Datasets

Figures 8(a) and 8(b) show that Graph-R1 outperforms Search-R1 on six datasets in O.O.D. validation, with notable gains on NQ and TriviaQA. Its multi-turn interaction with hypergraph retrieval ensures more stable performance under distribution shifts.

### Robust Generalization Ability

Figures 8(c) and 8(d) show that Graph-R1 achieves higher O.O.D.-to-I.I.D. ratios than Search-R1, often above 85% and exceeding 90% in some cases, reflecting its strong robustness and cross-domain generalizability via end-to-end RL over knowledge hypergraph.

---

## 6. CONCLUSION

In this work, we introduce **Graph-R1**, an agentic GraphRAG framework powered by end-to-end RL. By introducing lightweight knowledge hypergraph construction and modeling retrieval as a multi-turn interaction process, Graph-R1 bridges graph-structured knowledge with natural language generation. A unified reward mechanism enables outcome-directed reasoning that outperforms prior GraphRAG methods and RL-enhanced baselines. Experiments across six benchmarks demonstrate Graph-R1's superiority in accuracy, retrieval efficiency, generation quality, and generalizability.

---

## REFERENCES

[Full reference list from pages 11-14 of the PDF, containing all citations]

Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang. Pathrag: Pruning graph-based retrieval augmented generation with relational paths, 2025.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation, 2023.

DeepSeek-AI, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.

Darren Edge, et al. From local to global: A graph rag approach to query-focused summarization, 2025.

[... continues with all references]

---

## APPENDIX

## A. PROMPTS USED IN GRAPH-R1

### A.1 KNOWLEDGE HYPERGRAPH CONSTRUCTION PROMPT

**Figure 9: Prompt for n-ary relation extraction πₑₓₜ in Equation 4**

```
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types,
identify all entities of those types from the text and all relationships among the identified
entities. Use {language} as output language.

-Steps-
1. Divide the text into several complete knowledge segments. For each knowledge segment, extract
   the following information:
   -- knowledge_segment: A sentence that describes the context of the knowledge segment.
   Format each knowledge segment as ("hyper-relation"{tuple_delimiter}<knowledge_segment>)

2. Identify all entities in each knowledge segment. For each identified entity, extract the
   following information:
   - entity_name: Name of the entity, use same language as input text. If English, capitalized
     the name.
   - entity_type: Type of the entity.
   - entity_description: Comprehensive description of the entity's attributes and activities.
   Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>
   {tuple_delimiter}<entity_description>)

3. Return output in {language} as a single list of all the entities and relationships identified
   in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

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
```

### A.2 AGENTIC KNOWLEDGE REASONING PROMPT

**Figure 10: Prompt for agentic knowledge reasoning πθ in Equation 6**

[Shows a detailed multi-turn conversation example with the Graph-R1 agent answering the question "Who is the spouse of the director of film In Memory Of Sergo Ordzhonikidze?" through multiple query-retrieval-reasoning cycles, ultimately arriving at the correct answer: Yelizaveta Svilova]

---

## B. THEORETICAL PROOF

### B.1 PROOF OF PROPOSITION 1

**Proposition 1.** Graph-structured knowledge boosts agent accuracy by richer representation.

**Proof.** Let the knowledge base K be encoded into two forms: a graph Rɢ and a linear chunk set Rᴄ, where Rᴄ = g(Rɢ) is a deterministic transformation that discards edge information. For a query Q and ground-truth answer A*, the agent's internal belief at step t is hₜ. Each step performs retrieval Eₜ(R) and Bayesian update, forming the recurrence:

```
hₜ₊₁ = f(hₜ, R)    (Equation 16)
```

Define the Lyapunov function as Vᴿ(hₜ) = -log P(A* | hₜ), measuring how far the agent is from certainty. Its update is:

```
ΔVᴿ(hₜ) = -log [P(Eₜ(R) | A*) / Σₐ P(a | hₜ)P(Eₜ(R) | a)]    (Equation 17)
```

Graphs can capture more relevant facts in shorter contexts due to explicit edges, leading to higher information density δᴿ and more negative ΔVᴿ(hₜ) in expectation. Thus, Vᴿ(hₜ) decreases faster with graphs, indicating faster convergence. From an information-theoretic view, the mutual information evolves as:

```
I(A*; hₜ₊₁ | Q) = I(A*; hₜ | Q) + I(A*; Eₜ(R) | hₜ, Q)    (Equation 18)
```

and since graphs provide denser evidence, we have Iᴿɢ(A*; hₜ | Q) ≥ Iᴿᴄ(A*; hₜ | Q). Then by Fano's inequality,

```
Pₑ(R) ≤ [H(A* | Q) - Iᴿ(A*; hₜ | Q) + 1] / log|A|    (Equation 19)
```

which implies Pₑ(Rɢ) ≤ Pₑ(Rᴄ), i.e., Acc(Rɢ) ≥ Acc(Rᴄ), with strict inequality when the graph contains structural relations not recoverable from text.

In summary, the graph-structured representation offers higher information density per retrieval, accelerates belief convergence via Lyapunov descent, and accumulates more mutual information, leading to provably higher answer accuracy. □

### B.2 PROOF OF PROPOSITION 2

**Proposition 2.** Multi-turn interaction with the graph environment improves retrieval efficiency.

**Proof.** [Full proof from page 16-17 showing how multi-turn adaptive retrieval achieves higher information gain per token than static single-turn retrieval] □

### B.3 PROOF OF PROPOSITION 3

**Proposition 3.** End-to-end RL bridges the gap between graph-based knowledge and language.

**Proof.** [Full proof from page 17 showing how RL aligns graph retrieval with answer generation through trajectory-level rewards] □

---

## C. GRAPH-R1 ALGORITHM DETAILS

### Algorithm 1: Graph-R1: Agentic GraphRAG via End-to-end RL

```
Require: Query q, knowledge corpus K = {d₁, ..., dₙ}, policy πθ, reward function R(τ)
Ensure: Final answer yq

1: // 1: Knowledge Hypergraph Construction
2: Initialize hypergraph Gₕ = (V, Eₕ, φ)
3: for each document d ∈ K do
4:     Extract relational facts: {(hᵢ, Vₕᵢ)} ~ πₑₓₜ(d)
5:     for each (hᵢ, Vₕᵢ) do
6:         Eₕ ← Eₕ ∪ {hᵢ}, V ← V ∪ Vₕᵢ
7:         φ(hᵢ) ← Enc(hᵢ), φ(v) ← Enc(v) for v ∈ Vₕᵢ
8:     end for
9: end for

10: // 2: Multi-turn Graph Reasoning
11: Initialize state s₁ ← q, trajectory τ ← ∅
12: for t = 1 to T do
13:     Generate reasoning plan: aᵗʰⁱⁿᵏ ~ πθ(· | sₜ)
14:     Choose intent: αₜ ~ πθ(· | aᵗʰⁱⁿᵏ, sₜ)
15:     if αₜ = (answer) then
16:         Output answer: aᵃⁿˢ ~ πθ(· | aᵗʰⁱⁿᵏ, sₜ)
17:         τ ← τ ∪ {(sₜ, aᵍᵘᵉʳʸ, aᵃⁿˢ)}; return yq = aᵃⁿˢ
18:     else if αₜ = (query, retrieve) then
19:         Generate query: aᵍᵘᵉʳʸ ~ πθ(· | aᵗʰⁱⁿᵏ, sₜ)
20:         Entity retrieval: Rᵥ = argmaxᵏⱽᵥ∈V sim(φ(v), φ(Vₐᵍᵘᵉʳʸ))
21:         Hyperedge retrieval: Rₕ = argmaxᵏᴴₕ∈Eₕ sim(φ(h), φ(aᵍᵘᵉʳʸ))
22:         Rank fusion: aʳᵉᵗ = Top-k(F*ᵥ ∪ F*ₕ, Score(f) = 1/rᵥ(f) + 1/rₕ(f))
23:         Update state sₜ₊₁ ← sₜ ∪ {(sₜ, aᵗʰⁱⁿᵏ, aᵍᵘᵉʳʸ, aʳᵉᵗ)}
24:         τ ← τ ∪ {(sₜ, aᵗʰⁱⁿᵏ, aᵍᵘᵉʳʸ, aʳᵉᵗ)}
25:     end if
26: end for

27: // 3: End-to-end Policy Optimization (GRPO)
28: Sample N trajectories {τᵢ} ~ πθₒₗₐ
29: for each τᵢ do
30:     Compute reward: R(τᵢ) = -1 + Rꜰᴏʀᴍᴀᴛ(τᵢ) + I{Rꜰᴏʀᴍᴀᴛ = 1} · Rᴀɴsᴡᴇʀ(yₜ, y*q)
31:     Compute advantage: Â(τᵢ) = [R(τᵢ) - mean({R(τⱼ)})] / std({R(τⱼ)})
32: end for
33: Update policy via GRPO: Jɢʀᴘᴏ ~ ΣᴺᵢΣ|τᵢ|ₜ min(ρθ(a⁽ⁱ⁾ₜ)Â(τᵢ), clip(ρθ(a⁽ⁱ⁾ₜ), 1±ε)Â(τᵢ))
34: where ρθ(a⁽ⁱ⁾ₜ) = πθ(a⁽ⁱ⁾ₜ | s⁽ⁱ⁾ₜ₋₁) / πθₒₗₐ(a⁽ⁱ⁾ₜ | s⁽ⁱ⁾ₜ₋₁)
```

### Complexity Analysis

Graph-R1 involves three computational components corresponding to phases:

1. **Hypergraph construction:** Scales with total token count Tₖ and number of extracted facts F, yielding O(Tₖ) + O(F)

2. **Multi-turn reasoning:** Agent performs T steps with similarity computations over |V| nodes and |Eₕ| hyperedges with embedding dimension d, yielding O((|V|+|Eₕ|)d) per step

3. **Policy optimization:** GRPO processes N sampled trajectories of max length T, with gradient updates costing O(NTd)

Each component is computationally tractable and benefits from parallelization and localized retrieval over compact hypergraph subsets.

---

## D. DATASET DETAILS

We conduct experiments on six widely-used RAG benchmarks selected from the FlashRAG toolkit (Jin et al., 2025b):

1. **2WikiMultiHopQA (2Wiki.)** (Ho et al., 2020): Multi-hop dataset requiring reasoning across two Wikipedia documents

2. **HotpotQA** (Yang et al., 2018): Challenging multi-hop QA dataset with sentence-level supporting facts and diverse question types

3. **Musique** (Trivedi et al., 2022): Multi-hop questions needing chains of inference, often involving three or more reasoning steps

4. **Natural Questions (NQ)** (Kwiatkowski et al., 2019): Large-scale single-hop QA dataset grounded in real Google search questions with Wikipedia passages

5. **PopQA** (Mallen et al., 2023): Open-domain QA dataset focused on popular culture questions sourced from Wikipedia

6. **TriviaQA** (Joshi et al., 2017): Large-scale dataset containing trivia-style questions with distantly supervised evidence documents

To ensure consistency across datasets and maintain manageable training and evaluation workloads, we uniformly sample **5,120 instances per dataset for training** and **128 instances for testing**.

---

## E. BASELINE DETAILS

### E.1 BASELINES WITH GPT-4O-MINI

- **NaiveGeneration (GPT-4o-mini):** Zero-shot generation without retrieval
- **StandardRAG (GPT-4o-mini)** (Lewis et al., 2020): Chunk-based RAG
- **GraphRAG** (Edge et al., 2025): Graph-structured retrieval baseline
- **LightRAG** (Guo et al., 2025): Lightweight GraphRAG variant
- **PathRAG** (Chen et al., 2025): Path-based pruning on entity graphs
- **HippoRAG2** (Gutiérrez et al., 2025): Hierarchical path planner
- **HyperGraphRAG** (Luo et al., 2025a): N-ary relational hypergraphs

### E.2 BASELINES WITH QWEN2.5 (1.5B, 3B, 7B)

- **NaiveGeneration:** Direct generation without retrieval
- **StandardRAG** (Lewis et al., 2020): Chunk-based RAG
- **SFT** (Zheng et al., 2024): Supervised fine-tuning
- **R1** (Shao et al., 2024): GRPO-trained without retrieval
- **Search-R1** (Jin et al., 2025a): Multi-turn chunk-based retrieval with GRPO
- **R1-Searcher** (Song et al., 2025): Two-stage GRPO with chunk-based retrieval

---

## F. EVALUATION DETAILS

### Figure 11: Seven Dimensions for Generation Evaluation

The seven evaluation dimensions are:

1. **Comprehensiveness** (0-10): Whether the thinking considers all important aspects and is thorough

2. **Knowledgeability** (0-10): Whether the thinking is rich in insightful, domain-relevant knowledge

3. **Correctness** (0-10): Whether the reasoning and answer are logically and factually correct

4. **Relevance** (0-10): Whether the reasoning and answer are highly relevant and helpful to the question

5. **Diversity** (0-10): Whether the reasoning is thought-provoking, offering varied or novel perspectives

6. **Logical Coherence** (0-10): Whether the reasoning is internally consistent, clear, and well-structured

7. **Factuality** (0-10): Whether the reasoning and answer are based on accurate and verifiable facts

We evaluate model performance using four metrics:

**(i) Exact Match (EM):**
```
EM = (1/N) Σᴺᵢ₌₁ I{norm(yᵢ) = norm(y*ᵢ)}    (Equation 33)
```

**(ii) F1 Score:**
```
F1 = (1/N) Σᴺᵢ₌₁ [2·|tokens(yᵢ) ∩ tokens(y*ᵢ)|] / [|tokens(yᵢ)| + |tokens(y*ᵢ)|]    (Equation 34)
```

**(iii) Retrieval Similarity (R-S):**
```
R-S = (1/N) Σᴺᵢ₌₁ cos(Enc(k⁽ⁱ⁾ʳᵉᵗʳ), Enc(k⁽ⁱ⁾ɢᴏʟᴅ))    (Equation 35)
```

**(iv) Generation Evaluation (G-E):**
```
G-E = (1/N) Σᴺᵢ₌₁ [(1/7) Σ⁷ᵈ₌₁ sᵢ,ᵈ]    (Equation 36)
```

---

## G. IMPLEMENTATION DETAILS

### Table 5: Hyperparameter settings for baselines and Graph-R1

| Method | Backbone | Batch Size | Max Length | Top-K | Algo | Epochs |
|--------|----------|------------|------------|-------|------|--------|
| NaiveGeneration | Qwen2.5 / GPT-4o-mini | - | ∞ | N/A | - | - |
| StandardRAG | Qwen2.5 / GPT-4o-mini | - | ∞ | 5 Chunks | - | - |
| GraphRAG | GPT-4o-mini | - | ∞ | 60 | - | - |
| LightRAG | GPT-4o-mini | - | ∞ | 60 | - | - |
| PathRAG | GPT-4o-mini | - | ∞ | 60 | - | - |
| HippoRAG2 | GPT-4o-mini | - | ∞ | 60 | - | - |
| HyperGraphRAG | GPT-4o-mini | - | ∞ | 60 | - | - |
| SFT | Qwen2.5 (1.5B, 3B, 7B) | 16 | 4096 | N/A | LoRA | 3 |
| R1 | Qwen2.5 (1.5B, 3B, 7B) | 128 | 4096 | N/A | GRPO | 3 |
| Search-R1 | Qwen2.5 (1.5B, 3B, 7B) | 128 | 4096 | 5 Chunks/Turn | GRPO | 3 |
| R1-Searcher | Qwen2.5 (1.5B, 3B, 7B) | 128 | 4096 | 5 Chunks/Turn | GRPO | 3 |
| **Graph-R1 (ours)** | Qwen2.5 (1.5B, 3B, 7B) | 128 | 4096 | 5/Turn | GRPO | 3 |

We use GPT-4o-mini for knowledge construction in Graph-R1 and GraphRAG baselines. For retrieval, we use **bge-large-en-v1.5** (Chen et al., 2023) in all variants. All experiments are done on **4 NVIDIA A100 GPUs (80GB)**.

---

## H. LIMITATIONS AND FUTURE WORK

While Graph-R1 achieves strong performance, several limitations remain:

1. **Construction cost:** The cost of hypergraph construction, especially relation extraction and encoding, remains non-trivial. Future work may explore more efficient methods for zero-cost extraction.

2. **Structural reasoning:** Current retrieval lacks structural reasoning. Integrating GNNs or trainable message-passing could improve both accuracy and scalability.

3. **Multi-modal extension:** Graph-R1 currently supports only textual knowledge; extending it to multi-modal inputs is a promising direction.

4. **Domain applications:** We aim to further apply Graph-R1 in knowledge-intensive domains such as healthcare, law, and finance, where robust and interpretable reasoning is essential.

---

## I. IMPLEMENTATION GUIDE (BASED ON REFERENCE CODE)

This section provides detailed implementation guidance based on the official Graph-R1 reference implementation in the `graphr1/` folder.

### I.1 CORE ARCHITECTURE & CLASS STRUCTURE

#### GraphR1 Main Class

The `GraphR1` class (in `graphr1/graphr1.py`) is the main entry point with the following key components:

**Configuration Parameters:**
```python
@dataclass
class GraphR1:
    # Working directory
    working_dir: str = "graphr1_cache_{timestamp}"

    # Storage backends
    kv_storage: str = "JsonKVStorage"           # Key-value storage
    vector_storage: str = "NanoVectorDBStorage" # Vector database
    graph_storage: str = "NetworkXStorage"       # Graph storage

    # Text chunking parameters
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o-mini"

    # Entity extraction parameters
    entity_extract_max_gleaning: int = 2
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
```

**Storage Components:**

The implementation uses three storage layers:
1. **Key-Value Storage** (`entities_vdb`, `hyperedges_vdb`, `chunks_vdb`): Stores entity descriptions, hyperedge contents, and text chunks
2. **Vector Storage**: Enables similarity search over embedded entities and hyperedges
3. **Graph Storage** (`chunk_entity_relation_graph`): Stores the chunk-entity-relation graph structure

### I.2 KNOWLEDGE HYPERGRAPH CONSTRUCTION DETAILS

#### Step 1: Document Chunking

```python
def chunking_by_token_size(
    content: str,
    overlap_token_size=128,
    max_token_size=1024,
    tiktoken_model="gpt-4o"
):
    """
    Chunks documents with token-based overlapping windows
    - Uses tiktoken for precise token counting
    - Creates overlapping chunks for context continuity
    - Returns chunks with metadata (tokens, content, chunk_order_index)
    """
```

**Key Implementation Detail**: The paper mentions chunking but doesn't specify the overlap strategy. The reference code uses **sliding window with overlap** to maintain context across chunks.

#### Step 2: Entity & N-ary Relation Extraction

The extraction prompt (from `graphr1/prompt.py`) extracts:

1. **Hyper-relations**: N-ary semantic segments connecting multiple entities
   - Format: `("hyper-relation"{tuple_delimiter}<knowledge_segment>)`

2. **Entities**: Named entities with type and description
   - Format: `("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)`

**Critical Implementation Detail**: The paper describes n-ary facts as `(hᵢ, Vₕᵢ)` but doesn't detail the extraction format. The reference code uses:
- **Two-pass extraction**: First pass extracts entities and relations, second pass (gleaning) refines missing entities
- **Entity normalization**: All entity names are uppercased and cleaned
- **Weight calculation**: Uses `float(record_attributes[4]) if is_float_regex(record_attributes[4]) else 1.0`

#### Step 3: Hypergraph Storage Structure

**Storage Schema** (missing from paper):
```python
# Entity storage
entities_vdb = {
    "entity_id": {
        "entity_name": str,        # Normalized entity name
        "entity_type": str,        # Entity type (PERSON, ORG, etc.)
        "description": str,        # Entity description
        "source_id": str,          # Chunk that mentions this entity
        "content": str             # Full description for embedding
    }
}

# Hyperedge storage
hyperedges_vdb = {
    "hyperedge_id": {
        "content": str,            # Semantic description of n-ary relation
        "entities": List[str],     # Entity names in this hyperedge
        "source_id": str,          # Source chunk
        "weight": float            # Importance weight
    }
}

# Chunk-Entity-Relation Graph
chunk_entity_relation_graph = {
    nodes: {chunk_id, entity_id},
    edges: {
        (chunk, entity): {"weight": float, "description": str},
        (entity, entity): {"hyperedge_id": str, "weight": float}
    }
}
```

### I.3 MULTI-TURN RETRIEVAL IMPLEMENTATION

#### Dual-Path Retrieval (Section 4.2)

The paper describes two retrieval paths, but the reference implementation adds crucial details:

**Entity-Based Retrieval**:
```python
# Step 1: Extract entities from query using LLM
query_entities = extract_entities(query, llm_func)

# Step 2: Find similar entities using vector similarity
entity_embeddings = await embedding_func(query_entities)
similar_entities = await entities_vdb.query(entity_embeddings, top_k=5)

# Step 3: Get hyperedges connected to these entities
connected_hyperedges = []
for entity in similar_entities:
    edges = await graph.get_node_edges(entity["entity_name"])
    connected_hyperedges.extend(edges)
```

**Direct Hyperedge Retrieval**:
```python
# Query hyperedges directly by semantic similarity
query_embedding = await embedding_func(query)
similar_hyperedges = await hyperedges_vdb.query(query_embedding, top_k=5)
```

**Reciprocal Rank Fusion** (Equation 9 implementation):
```python
def reciprocal_rank_fusion(entity_results, hyperedge_results, k=60):
    """
    Combines results from both retrieval paths
    RankScore(f) = 1/rank_entity + 1/rank_hyperedge
    """
    scores = defaultdict(float)

    for rank, item in enumerate(entity_results, 1):
        scores[item['id']] += 1 / rank

    for rank, item in enumerate(hyperedge_results, 1):
        scores[item['id']] += 1 / rank

    # Sort by combined score and take top-k
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, score in ranked[:k]]
```

### I.4 AGENTIC REASONING LOOP

#### QueryParam Configuration

The paper doesn't specify the query parameter structure. The reference code defines:

```python
@dataclass
class QueryParam:
    mode: Literal["local", "global", "hybrid", "naive"] = "hybrid"
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int = 60                              # Graph-R1 paper uses 5 initial, 10 multi-hop
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000
```

**Important**: The paper mentions "5 initial candidates" and "10-hop traversal" but the reference code uses flexible `top_k` parameter.

#### Agent Query Function

```python
async def aquery(
    query: str,
    param: QueryParam = QueryParam(),
    entity_match: dict = None,
    hyperedge_match: dict = None
):
    """
    Main query function that implements multi-turn interaction

    Args:
        query: User question
        param: Query parameters
        entity_match: Pre-computed entity matches (optional)
        hyperedge_match: Pre-computed hyperedge matches (optional)

    Returns:
        Retrieved contexts with coherence scores
    """
    # 1. Entity matching (if not provided)
    if entity_match is None:
        entity_match = await retrieve_entities(query, param.top_k)

    # 2. Hyperedge matching (if not provided)
    if hyperedge_match is None:
        hyperedge_match = await retrieve_hyperedges(query, param.top_k)

    # 3. Reciprocal rank fusion
    combined_results = reciprocal_rank_fusion(
        entity_match[query],
        hyperedge_match[query]
    )

    # 4. Expand via graph traversal (multi-hop)
    expanded_contexts = await expand_via_graph(
        combined_results,
        max_hops=param.top_k  # Paper uses 10
    )

    # 5. Compute coherence scores
    scored_contexts = compute_coherence_scores(expanded_contexts, query)

    return scored_contexts
```

### I.5 STORAGE BACKEND IMPLEMENTATIONS

The paper doesn't mention storage options. The reference code supports:

**Key-Value Storage**:
- `JsonKVStorage`: Local JSON file storage
- `OracleKVStorage`: Oracle Database
- `MongoKVStorage`: MongoDB
- `TiDBKVStorage`: TiDB

**Vector Storage**:
- `NanoVectorDBStorage`: Lightweight local vector DB
- `OracleVectorDBStorage`: Oracle Vector Search
- `MilvusVectorDBStorge`: Milvus vector database
- `ChromaVectorDBStorage`: ChromaDB
- `TiDBVectorDBStorage`: TiDB Vector Search

**Graph Storage**:
- `NetworkXStorage`: In-memory NetworkX graph (default)
- `Neo4JStorage`: Neo4j graph database
- `OracleGraphStorage`: Oracle Graph

### I.6 ASYNC PROCESSING & CONCURRENCY

**Critical Implementation Detail** (not in paper):

The reference code uses extensive async processing with concurrency limits:

```python
# Limit concurrent LLM calls
llm_model_func = limit_async_func_call(llm_model_max_async)(llm_model_func)

# Limit concurrent embedding calls
embedding_func = limit_async_func_call(embedding_func_max_async)(embedding_func)

# Event loop management
def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensures event loop availability in both sync and async contexts
    Critical for running Graph-R1 in different environments
    """
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop
```

### I.7 CACHING MECHANISMS

**LLM Response Caching** (not detailed in paper):
```python
# Enable caching to avoid redundant LLM calls
enable_llm_cache: bool = True

# Cache structure
llm_response_cache = {
    "hash(prompt)": {
        "response": str,
        "timestamp": datetime,
        "model": str
    }
}
```

**Embedding Caching**:
```python
embedding_cache_config = {
    "enabled": False,              # Default disabled
    "similarity_threshold": 0.95,  # Reuse if similarity > 0.95
    "use_llm_check": False        # Use LLM to verify semantic equivalence
}
```

### I.8 ENTITY EXTRACTION GLEANING

**Two-Pass Extraction** (missing from paper):

The reference code implements a gleaning mechanism:

1. **First Pass**: Extract obvious entities and relations
2. **Second Pass**: LLM reviews text again to find missed entities
3. **Max Gleaning Rounds**: `entity_extract_max_gleaning = 2`

This significantly improves entity extraction recall.

### I.9 ENTITY/RELATION SUMMARIZATION

**Long Description Handling** (not in paper):

When entity descriptions exceed `entity_summary_to_max_tokens` (default 500):

```python
async def _handle_entity_relation_summary(
    entity_name: str,
    description: str,
    global_config: dict
):
    """
    Summarizes long entity descriptions to fit token limits
    Uses LLM to condense while preserving key information
    """
    if len(tokens(description)) < summary_max_tokens:
        return description

    # Use LLM to summarize
    prompt = f"Summarize the following description of {entity_name}:\n{description}"
    summary = await llm_func(prompt, max_tokens=summary_max_tokens)
    return summary
```

### I.10 PRACTICAL USAGE EXAMPLES

#### Basic Usage

```python
from graphr1 import GraphR1, QueryParam

# Initialize
rag = GraphR1(
    working_dir="./my_knowledge_base",
    chunk_token_size=1200,
    entity_extract_max_gleaning=2
)

# Insert documents
documents = [
    "Albert Einstein was a theoretical physicist...",
    "The theory of relativity was published in 1905...",
]
rag.insert(documents)

# Query
result = rag.query(
    "Who developed the theory of relativity?",
    param=QueryParam(top_k=10)
)
```

#### Async Usage

```python
import asyncio

async def main():
    rag = GraphR1(working_dir="./kb")

    # Async insert
    await rag.ainsert(documents)

    # Async query
    result = await rag.aquery(
        "Who developed the theory of relativity?",
        param=QueryParam(top_k=10, only_need_context=True)
    )

    print(result)

asyncio.run(main())
```

#### Using Different Storage Backends

```python
# Use Neo4j for graph storage
rag = GraphR1(
    working_dir="./kb",
    graph_storage="Neo4JStorage",
    vector_storage="MilvusVectorDBStorge",
    kv_storage="MongoKVStorage"
)
```

### I.11 KEY DIFFERENCES: PAPER VS IMPLEMENTATION

| Aspect | Paper Description | Reference Implementation |
|--------|------------------|-------------------------|
| **Initial Retrieval** | 5 entities + 5 hyperedges | Configurable via `top_k` parameter |
| **Multi-hop Depth** | 10-hop traversal | Configurable, default uses `top_k` |
| **Storage** | Not specified | Multiple backends (JSON, Neo4j, MongoDB, etc.) |
| **Chunking** | Token-based | Sliding window with overlap |
| **Entity Extraction** | Single pass | Two-pass with gleaning |
| **Caching** | Not mentioned | LLM response cache + optional embedding cache |
| **Concurrency** | Not specified | Async with configurable limits |
| **Entity Names** | Not specified | Uppercase normalized |
| **Summarization** | Not mentioned | Auto-summarize long descriptions (>500 tokens) |

### I.12 PERFORMANCE TUNING GUIDELINES

**For Large-Scale Deployment**:
1. Use `MilvusVectorDBStorge` or `ChromaVectorDBStorage` instead of `NanoVectorDBStorage`
2. Use `Neo4JStorage` for graph storage with millions of entities
3. Enable embedding cache with `similarity_threshold=0.95`
4. Increase `embedding_func_max_async` and `llm_model_max_async` based on API limits
5. Use `MongoKVStorage` or `TiDBKVStorage` for distributed storage

**For Low-Resource Environments**:
1. Reduce `chunk_token_size` to 800
2. Set `entity_extract_max_gleaning=1` to reduce LLM calls
3. Use smaller `top_k` values (e.g., 5-10)
4. Enable `llm_cache` to avoid redundant calls

---

**END OF DOCUMENT**
