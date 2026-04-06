# Research Plan: Which linear directions are compositional?

## Research Question
Does the linear representation hypothesis (LRH) hold across all concept combinations, or is compositionality limited to "related" directions within coherent subspaces?

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding the compositionality of linear representations is crucial for:
1. **Model Steerability:** If we can't reliably compose directions, our ability to steer LLMs using vector arithmetic (e.g., adding "truthfulness" + "politeness") is limited and potentially dangerous.
2. **Mechanistic Interpretability:** It challenges the simplified view that LLMs are just "linear algebra over concept vectors."
3. **Safety:** Predicting when concept combinations might lead to non-linear "jumps" or failure of representation.

### Gap in Existing Work
Existing work like Park et al. (2023) establishes the existence of linear directions for isolated concepts. Azizian et al. (2025) suggests task-specific orthogonality. However, no systematic study has compared the **compositional properties** of related vs. unrelated directions across semantic and logical domains. 2026 work like PolySAE suggests non-linearity but hasn't mapped the boundary of where composition works vs. fails.

### Our Novel Contribution
We will systematically test the **Additivity Constraint** for linear directions across:
1. **Semantic (Property-Object):** e.g., "Red" + "Car"
2. **Logical (Negation):** e.g., "Paris" + "not"
3. **Hierarchical (Taxonomy):** e.g., "Animal" + "Mammal"
4. **Unrelated (Random):** e.g., "Red" + "Prime Number"

We aim to show that "relatedness" (proximity in a hierarchy or shared property subspace) is a predictor of compositionality.

### Experiment Justification
- **Experiment 1 (Semantic Composition):** Tests if property directions compose with object directions. Measures cosine similarity between $v(P) + v(O)$ and $v(PO)$.
- **Experiment 2 (Logical Negation):** Tests if the "Negation" operator acts as a linear transformation (vector addition/subtraction) across tasks.
- **Experiment 3 (Causal Inner Product / Interference):** Uses Park 2023's metric to quantify interference between directions. Hypothesis: Related directions have lower interference.

## Hypothesis Decomposition
1. **Sub-hypothesis A:** Directions for related concepts (e.g., from the same domain like "animal") share a coherent subspace and compose linearly.
2. **Sub-hypothesis B:** Combining directions from unrelated domains (e.g., "math" and "geography") results in high interference and failure of the linear approximation.
3. **Sub-hypothesis C:** Negation is a consistent linear direction only within related semantic clusters.

## Proposed Methodology

### Approach
We will use Llama-3-8B-hf. We will extract activations from the residual stream. We will define concept directions as the mean difference between counterfactual pairs.

### Experimental Steps
1. **Direction Extraction:**
   - Gather counterfactual pairs for:
     - Color (Red vs Blue)
     - Object (Car vs Truck)
     - Truth (True vs False statements in Animals, Cities)
     - Negation (Positive vs Negative sentences)
2. **Additivity Test:**
   - Compute $\gamma_{red}$, $\gamma_{car}$, $\gamma_{red\_car}$.
   - Measure $d = \cos(\gamma_{red} + \gamma_{car}, \gamma_{red\_car})$.
   - Repeat for unrelated pairs (e.g., $\gamma_{red} + \gamma_{prime\_number}$).
3. **Intervention Test:**
   - Take an embedding $\lambda_{blue\_car}$.
   - Intervene: $\lambda' = \lambda_{blue\_car} - \gamma_{blue} + \gamma_{red}$.
   - Measure if $\lambda'$ yields logits for "red car".
4. **Subspace Coherence (LOO):**
   - Use Leave-One-Out to check if directions are consistent.

### Baselines
- **Random Direction Baseline:** Add random vectors of same norm.
- **Single Direction Baseline:** Use only one direction without composition.
- **Park 2023 Subspace Projection:** Project onto the concept subspace.

### Evaluation Metrics
- **Cosine Similarity:** Between additive prediction and actual composed activation.
- **Logit Difference:** Measure change in model output after steering.
- **Causal Inner Product:** Measure orthogonality of directions.

### Statistical Analysis Plan
- T-test to compare cosine similarities of Related vs Unrelated pairs.
- Correlation between Hierarchy Distance (WordNet) and Compositionality Metric.

## Expected Outcomes
Related pairs (high semantic overlap or hierarchy proximity) will show significantly higher cosine similarity and steering success than unrelated pairs.

## Timeline and Milestones
- **Setup:** 0.5h
- **Data Prep & Direction Extraction:** 1.5h
- **Experiments 1 & 2:** 2h
- **Analysis & Visualization:** 1.5h
- **Documentation:** 1h

## Success Criteria
Empirical evidence showing a clear gap in compositionality between related and unrelated linear directions in Llama-3.
