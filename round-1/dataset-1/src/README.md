# Dataset Processing Summary

## Overview
Successfully acquired and standardized three benchmark datasets for quorum-sensing consensus evaluation.

## Datasets Processed

### 1. MMLU-Hard (2197 examples)
- **Source**: HuggingFace `cais/mmlu`
- **Subjects**: 10 hard subjects (professional_accounting, conceptual_physics, formal_logic, machine_learning, global_facts, abstract_algebra, college_physics, college_mathematics, electrical_engineering, moral_scenarios)
- **Split**: test
- **Difficulty**: 0.7 (high)
- **Contestedness**: 0.6 (moderate-high)

### 2. GPQA (448 examples)
- **Source**: HuggingFace `Idavidrein/gpqa` (config: gpqa_main)
- **Domains**: Biology, Physics, Chemistry
- **Split**: train
- **Difficulty**: 0.66 (high)
- **Contestedness**: 0.7 (high)

### 3. TruthfulQA (817 examples)
- **Source**: HuggingFace `truthfulqa/truthful_qa` (config: multiple_choice)
- **Split**: validation
- **Categories**: 38 topics
- **Difficulty**: 0.5 (moderate)
- **Contestedness**: Variable (computed from ratio of plausible incorrect answers)

## Output Files

| File | Size | Description |
|------|------|-------------|
| `data_out.json` | 2.93 MB | Full dataset (3462 examples) |
| `data_out_mini.json` | 0.03 MB | Mini dataset (30 examples, 10 per source) |
| `data_out_preview.json` | 0.01 MB | Preview dataset (9 examples, 3 per source) |

## Schema Validation
✓ All examples have required fields (id, question, choices, correct_answer, category, difficulty, contestedness, metadata)
✓ All fields have correct types
✓ Total size <300MB (well under limit)
✓ Unified schema across all three datasets

## Next Steps
The processed datasets are ready for use in evaluating hierarchical LLM consensus protocols with confidence-calibrated veto mechanisms.
