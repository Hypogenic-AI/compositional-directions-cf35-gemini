# Downloaded Datasets

## Dataset 1: TruthfulQA
- **Source**: HuggingFace `truthfulqa/truthful_qa`
- **Location**: `datasets/truthful_qa.csv`
- **Task**: Evaluate truthfulness and hallucination.

## Dataset 2: TriviaQA (Sample)
- **Source**: HuggingFace `trivia_qa`
- **Location**: `datasets/trivia_qa_sample.csv`
- **Task**: Used for extracting truth directions (correct vs. incorrect fact recall).

## Dataset 3: Word Pairs (Semantic)
- **Source**: `code/park_geometry/word_pairs/`
- **Task**: Provides directions for gender, language, countries, etc.

## Dataset 4: Logical Transformations
- **Source**: `code/truthfulness_probes/data/`
- **Task**: Specifically negation, conjunction, and disjunction pairs for truth.

## Dataset 5: WordNet Hierarchy
- **Source**: `code/park_categorical/data/`
- **Task**: 900+ concepts with hierarchical relationships.

## .gitignore for Datasets
A `.gitignore` has been created to exclude large data files from git while keeping documentation and small samples.
