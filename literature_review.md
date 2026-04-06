# Literature Review: Compositionality of Linear Directions in LLMs

## Research Area Overview
The **Linear Representation Hypothesis (LRH)** posits that high-level concepts are represented as linear directions in the residual stream of Transformer-based Large Language Models (LLMs). While foundational work has successfully identified directions for binary concepts (e.g., gender, truth), recent research (2024-2026) has begun to uncover the limitations of this hypothesis, particularly concerning **compositionality**—the ability to combine these directions to represent complex or novel concepts.

## Key Papers

### 1. The Linear Representation Hypothesis and the Geometry of Large Language Models (Park et al., 2023)
- **Contribution:** Formally defines the LRH through three lenses: subspace, measurement (probing), and intervention (steering).
- **Methodology:** Introduces the **Causal Inner Product**, a non-Euclidean metric that respects language structure and unifies embedding and unembedding spaces.
- **Relevance:** Establishes that "causally separable" concepts (those that can be varied independently) are represented as orthogonal directions under this inner product.

### 2. The Geometries of Truth Are Orthogonal Across Tasks (Azizian et al., 2025)
- **Contribution:** Demonstrates that "truthfulness" is not a universal direction but is **task-specific** and largely **orthogonal** across different datasets.
- **Key Finding:** Linear probes trained on one task (e.g., TriviaQA) fail to generalize to others (e.g., BioASQ), even if both represent factual truth.
- **Relevance:** Suggests that "truth" directions do not compose globally; they are context-dependent clusters.

### 3. Algorithmic Primitives and Compositional Geometry of Reasoning (Lippl et al., 2025)
- **Contribution:** Identifies **Algorithmic Primitives** (e.g., "retrieving nearest neighbor") as linear "primitive vectors."
- **Key Finding:** Unlike semantic concepts (like truth), these algorithmic directions **are compositional** and can be combined via vector addition/subtraction to steer complex reasoning tasks.
- **Relevance:** Provides a counterpoint to Azizian, suggesting that structural/procedural directions may be more compositional than semantic ones.

### 4. Stop Probing, Start Coding: Why Linear Probes and SAEs Fail at Compositional Generalisation (Pacela et al., 2026)
- **Contribution:** Distinguishes between linear **encoding** (concepts as directions) and linear **accessibility** (recoverability via linear probes).
- **Key Finding:** Under **superposition** (more concepts than dimensions), the decision boundary becomes non-linear. Linear probes fail to generalize to novel combinations of latent factors (compositional shifts).
- **Relevance:** Directly supports the hypothesis that linear directions only appear linear in limited contexts and fail during composition.

### 5. PolySAE: Modeling Feature Interactions via Polynomial Decoding (Koromilas et al., 2026)
- **Contribution:** Extends Sparse Autoencoders (SAEs) with **polynomial terms** (quadratic/cubic) to capture multiplicative feature interactions.
- **Key Finding:** Standard linear SAEs cannot distinguish between additive co-occurrence and **multiplicative composition** (e.g., "star" + "coffee" = "Starbucks").
- **Relevance:** Confirms that semantic composition is inherently non-linear and requires higher-order interactions.

## Synthesis of Findings
The literature suggests a hierarchy of compositionality:
1. **Structural/Algorithmic Primitives:** Highly compositional and transferable (Lippl 2025).
2. **Related Semantic Concepts (Additive):** Approximately linear and composable when the linear approximation holds (Park 2023).
3. **Task-Dependent Semantic Concepts:** Context-specific and orthogonal (Azizian 2025).
4. **Complex/Phonetic Compositions:** Inherently non-linear; linear models fail to generalize (Pacela 2026, Koromilas 2026).

## Recommendations for Experiment Design
- **Datasets:** Use `Park 2023` word pairs for related concepts (additive baseline) and `Azizian 2025` truth datasets for task-specific/orthogonal tests.
- **Evaluation:** Compare linear probes (measurement) vs. vector addition (intervention) across in-distribution and compositional shift settings.
- **Methods:** Test if adding vectors for related concepts (e.g., "red" + "blue") preserves linearity better than unrelated concepts.
- **Higher-order Analysis:** Use the `tuned-lens` to trace how composition evolves layer-by-layer and check for non-linear "jumps" using `PolySAE`-style analysis.
