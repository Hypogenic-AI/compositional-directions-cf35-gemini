# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Which linear directions are compositional?". It includes foundational and state-of-the-art papers, relevant datasets, and code repositories for experimentation.

## Papers
Total papers downloaded: 8

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Linear Repr. Hypothesis | Park et al. | 2023 | papers/park2023_linear_representation_hypothesis.pdf | Foundational LRH definition |
| Geometries of Truth Orthogonal | Azizian et al. | 2025 | papers/azizian2025_geometries_of_truth_orthogonal.pdf | Task-specific orthogonality |
| Algorithmic Primitives | Lippl et al. | 2025 | papers/lippl2025_algorithmic_primitives.pdf | Compositional reasoning vectors |
| Stop Probing, Start Coding | Pacela et al. | 2026 | papers/stop_probing_start_coding_2026.pdf | Linear probe failure in OOD |
| PolySAE | Koromilas et al. | 2026 | papers/polysae2026_compositional_sae.pdf | Non-linear feature interactions |
| Categorical/Hierarchical | Park et al. | 2024 | papers/park2024_geometry_categorical_hierarchical.pdf | Polytopes & hierarchy geometry |
| Geometry of Truth | Marks & Tegmark | 2023 | papers/marks2023_geometry_of_truth.pdf | Emergent linear structure in truth |
| Comp. Gen. in Vision | Koromilas et al. | 2026 | papers/compositional_generalization_linear_orthogonal_2026.pdf | Orthogonality in vision embeddings |

## Datasets
Total datasets gathered: 5 (plus several in code repos)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TruthfulQA | HuggingFace | 817 | Truthfulness | datasets/truthful_qa.csv | Generative benchmark |
| TriviaQA | HuggingFace | 100+ | Fact recall | datasets/trivia_qa_sample.csv | Used for truth directions |
| Concept Pairs | code/park_geometry | 20+ files | Semantic | code/park_geometry/word_pairs/ | Gender, language, etc. |
| Logic/Truth | code/truthfulness_probes | 50+ files | Logic | code/truthfulness_probes/data/ | Negation/Conjunction tests |
| WordNet | code/park_categorical | 900+ | Hierarchical | code/park_categorical/data/ | Polytope/Hierarchy tests |

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| park_geometry | github.com/KihoPark/linear_rep_geometry | Subspace/Steering | code/park_geometry | Park 2023 official code |
| truthfulness_probes | github.com/colored-dye/truthfulness_probe_generalization | Truth/Logic | code/truthfulness_probes | Bao et al. 2025 official code |
| tuned_lens | github.com/AlignmentResearch/tuned-lens | Layer decoding | code/tuned_lens | Essential for analysis |
| park_categorical | github.com/KihoPark/LLM_Categorical_Hierarchical_Representations | Hierarchy | code/park_categorical | Park 2024 official code |
| mlsae | github.com/tim-lawson/mlsae | Multi-layer SAEs | code/mlsae | For cross-layer analysis |

## Resource Gathering Notes

### Search Strategy
- Used `paper-finder` for high-relevance academic search.
- Used `arxiv` API for precise ID matching and downloading.
- Searched HuggingFace for benchmark datasets mentioned in recent literature.
- Cloned official repositories from Semantic Scholar and paper links.

### Selection Criteria
- Focused on very recent work (2025-2026) that specifically addresses **compositional failure** of linear probes.
- Selected foundational papers (LRH, Geometry of Truth) as baselines.
- Prioritized code repositories that provide both data generation and analysis tools.

### Challenges Encountered
- Initial `arxiv` search results were sometimes incorrect; resolved by searching for specific author combinations.
- Some 2026 papers are very recent and required careful title matching.

## Recommendations for Experiment Design

1. **Primary Dataset**: Use the word pairs from `park_geometry` as the "control" group for linear composition.
2. **Stress Test**: Apply the logical negation/conjunction tests from `truthfulness_probes` to see if semantic directions (e.g., "Paris" + "not") break linearity.
3. **Metric**: Use **Cosine Similarity** between composed vectors and probe directions, and **Logit Difference** after steering interventions.
4. **Tool**: Use `tuned_lens` to observe if composition happens early (first few layers) or late (closer to the head), which might indicate different mechanisms (direct vs. compositional as in Khandelwal 2025).
