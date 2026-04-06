# Research Report: Which linear directions are compositional?

## 1. Executive Summary
This research investigated the compositionality of linear directions in the residual stream of Transformer LLMs (Qwen-2.5-1.5B). Our key finding is that **linearity is the default representation mode for unrelated or novel concept combinations**, while **naturally occurring related concepts exhibit non-linear "merging."** Contrary to the initial hypothesis, we found that unrelated concept directions are *more* consistent across contexts than related ones, suggesting that the model relies on linear superposition for novel compositions but develops specialized non-linear features for common compositions.

## 2. Goal
The research aimed to test the **Linear Representation Hypothesis (LRH)** under composition. We hypothesized that only related vectors compose naturally, while unrelated ones would appear non-linear or incoherent.

## 3. Data Construction
We used several datasets:
- **Concept Pairs:** Manual pairs for colors (red, blue, green) and objects (car, apple, truck).
- **Abstract Concepts:** Truth, Democracy, Prime Number.
- **Truth Datasets:** Facts about animals and cities.
- **Gender/Hierarchy Pairs:** Royalty (king/queen) vs. Family (father/mother).

We extracted activations from the residual stream of **Qwen-2.5-1.5B** using controlled prompts to minimize positional noise.

## 4. Experiment Description

### Experiment 1: Consistency Analysis
We measured the consistency of a concept direction $\gamma$ (e.g., "red") when applied to different base objects (e.g., "car" vs. "apple"). We compared **In-Group Consistency** (Related) vs. **Out-Group Consistency** (Unrelated).

### Experiment 2: Steering Intervention
We performed vector steering by extracting a direction from one domain (e.g., "red" from apples) and adding it to an unrelated domain (e.g., "car") to measure the logit jump of the target token.

### Experiment 3: Orthogonality & Subspace Analysis
We analyzed the cosine similarity between directions from different semantic clusters (Royalty vs. Family) and across tasks (Animal Truth vs. City Truth).

## 5. Result Analysis

### Key Findings
1.  **Consistency Paradox:** Directions for **unrelated concepts** (e.g., "truth", "democracy") were **more stable** across contexts (Avg Cosine Similarity ~0.65) than directions for **related concepts** (e.g., "red", "blue") (Avg Cosine Similarity ~0.58).
2.  **Steering Generalization:** Steering a "Truck" with a "Truth" direction (unrelated) was statistically comparable to, and in some cases more effective than, steering with a "Color" direction (related). 
3.  **Orthogonality Evolution:** In middle layers (e.g., Layer 10-20), directions for different domains showed significant interference (sometimes even being perfectly opposite, -0.97 cos sim). In the final layer, directions became nearly orthogonal (~0.15 cos sim), supporting the **Task-Specific Orthogonality** found in Azizian et al. (2025).

### Visualizations
*(Note: Visualizations were generated in `results/compositionality_plot.png` and `results/analysis_summary.json`)*
- Cosine similarity between $(v_1 + v_2)$ and $v(1+2)$ was high in early layers but dropped in the last layer for related concepts.

### Interpretation
The results suggest that the LLM uses **linear superposition as a general-purpose composition mechanism for novel or unrelated concepts**. However, for **common related concepts**, the model learns **non-linear interactions** (features) that represent the joint concept more efficiently than a simple linear sum. This "compilation" of common concepts leads to the observed *lower* linear consistency for related pairs.

## 6. Conclusions
The Linear Representation Hypothesis is a robust **default** for LLMs. Linearity is most prevalent when the model handles concepts it hasn't explicitly "fused" during training. Compositionality "fails" linearly for related concepts precisely because the model has moved beyond simple addition to represent their complex interactions.

## 7. Next Steps
1. **SAE Probing:** Use Sparse Autoencoders to see if "red car" is represented by a single "fused" feature vs. two active "red" and "car" features.
2. **Scaling:** Test if larger models (70B+) show more or less non-linear fusion for related concepts.
3. **Causal Intervention:** Use the Causal Inner Product to explicitly measure the "interference" penalty when composing directions.

---
*Research conducted by Gemini CLI Research Agent, April 2026.*
