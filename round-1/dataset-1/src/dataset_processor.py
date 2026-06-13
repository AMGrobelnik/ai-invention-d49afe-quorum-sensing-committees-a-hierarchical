#!/usr/bin/env python3
"""Process MMLU-Hard, GPQA, and TruthfulQA datasets for quorum-sensing consensus evaluation."""

from loguru import logger
from pathlib import Path
import json
import sys
import pandas as pd
from typing import List, Dict, Any

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def download_mmlu_hard() -> List[Dict[str, Any]]:
    """Download and process MMLU-Hard subset (subjects with lowest accuracy)."""
    logger.info("Downloading MMLU dataset...")
    
    try:
        from datasets import load_dataset
        
        # Hard subjects based on Hendrycks et al. results (lowest accuracy)
        hard_subjects = [
            'professional_accounting',
            'conceptual_physics', 
            'formal_logic',
            'machine_learning',
            'global_facts',
            'abstract_algebra',
            'college_physics',
            'college_mathematics',
            'electrical_engineering',
            'moral_scenarios'
        ]
        
        examples = []
        example_id = 0
        
        for subject in hard_subjects:
            try:
                logger.info(f"Loading MMLU subject: {subject}")
                dataset = load_dataset('cais/mmlu', subject, split='test')
                
                for i, row in enumerate(dataset):
                    examples.append({
                        "id": f"mmlu_{example_id}",
                        "question": row['question'],
                        "choices": row['choices'],
                        "correct_answer": row['answer'],
                        "category": subject,
                        "difficulty": 0.7,  # Hard subset = high difficulty
                        "contestedness": 0.6,  # Estimated based on low baseline accuracy
                        "metadata": {
                            "source_dataset": "mmlu",
                            "original_split": "test",
                            "num_choices": len(row['choices'])
                        }
                    })
                    example_id += 1
            except Exception as e:
                logger.warning(f"Failed to load subject {subject}: {e}")
                continue
        
        logger.info(f"MMLU-Hard: Loaded {len(examples)} examples")
        return examples
        
    except Exception as e:
        logger.error(f"Failed to download MMLU: {e}")
        raise

@logger.catch(reraise=True)
def download_gpqa() -> List[Dict[str, Any]]:
    """Download and process GPQA dataset."""
    logger.info("Downloading GPQA dataset...")
    
    try:
        from datasets import load_dataset
        
        # Try multiple possible GPQA dataset IDs
        dataset_ids = [
            ('Idavidrein/gpqa', 'gpqa_main'),
            ('Idavidrein/gpqa', 'gpqa_diamond'),
            ('nmayorga7/gpqa_diamond', None),
            ('jeggers/gpqa_formatted', None)
        ]
        
        examples = []
        loaded = False
        
        for dataset_id, config in dataset_ids:
            try:
                logger.info(f"Trying GPQA dataset: {dataset_id} (config: {config})")
                if config:
                    dataset = load_dataset(dataset_id, config, split='train')
                else:
                    dataset = load_dataset(dataset_id, split='train')
                
                for i, row in enumerate(dataset):
                    # GPQA structure: Question, Correct Answer, Incorrect Answer 1, etc.
                    question = row.get('Question', '')
                    
                    if not question:
                        continue
                    
                    # Build choices: correct answer + incorrect answers
                    correct_answer = row.get('Correct Answer', '')
                    incorrect_1 = row.get('Incorrect Answer 1', '')
                    incorrect_2 = row.get('Incorrect Answer 2', '')
                    incorrect_3 = row.get('Incorrect Answer 3', '')
                    
                    choices = [correct_answer, incorrect_1, incorrect_2, incorrect_3]
                    choices = [c for c in choices if c]  # Remove empty
                    
                    if len(choices) < 2:
                        continue
                    
                    # Correct answer is always index 0 (first choice)
                    correct_idx = 0
                    
                    # Get subdomain for category
                    subdomain = row.get('Subdomain', 'unknown')
                    
                    # Estimate difficulty from paper (expert: ~65%, non-expert: ~34%)
                    difficulty = 0.66  # High difficulty
                    
                    examples.append({
                        "id": f"gpqa_{i}",
                        "question": question,
                        "choices": choices,
                        "correct_answer": correct_idx,
                        "category": subdomain,
                        "difficulty": difficulty,
                        "contestedness": 0.7,  # High contestedness for graduate-level
                        "metadata": {
                            "source_dataset": "gpqa",
                            "original_split": "train",
                            "num_choices": len(choices),
                            "explanation": row.get('Explanation', '')
                        }
                    })
                
                loaded = True
                logger.info(f"GPQA: Loaded {len(examples)} examples from {dataset_id}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {dataset_id}: {e}")
                continue
        
        if not loaded:
            raise ValueError("Could not load GPQA from any known source")
        
        return examples
        
    except Exception as e:
        logger.error(f"Failed to download GPQA: {e}")
        raise

@logger.catch(reraise=True)
def download_truthfulqa() -> List[Dict[str, Any]]:
    """Download and process TruthfulQA dataset (multiple choice config)."""
    logger.info("Downloading TruthfulQA dataset...")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation')
        
        examples = []
        
        for i, row in enumerate(dataset):
            # TruthfulQA mc format
            question = row['question']
            correct_answers = row.get('mc1_targets', {}).get('labels', [])
            incorrect_answers = row.get('mc2_targets', {}).get('labels', [])
            
            # Build choices list: correct answers first, then incorrect
            choices = []
            correct_indices = []
            
            # Add correct answers
            for j, (ans, label) in enumerate(zip(row.get('mc1_targets', {}).get('choices', []), 
                                                  row.get('mc1_targets', {}).get('labels', []))):
                if label == 1:
                    correct_indices.append(len(choices))
                choices.append(ans)
            
            # Add incorrect but plausible answers
            for ans, label in zip(row.get('mc2_targets', {}).get('choices', []), 
                                  row.get('mc2_targets', {}).get('labels', [])):
                if label == 0:  # Incorrect
                    choices.append(ans)
            
            if not choices or not question:
                continue
            
            # Contestedness: ratio of plausible incorrect to total
            num_incorrect = len([a for a, l in zip(row.get('mc2_targets', {}).get('choices', []), 
                                                     row.get('mc2_targets', {}).get('labels', [])) 
                                if l == 0])
            contestedness = num_incorrect / len(choices) if choices else 0.5
            
            examples.append({
                "id": f"truthfulqa_{i}",
                "question": question,
                "choices": choices,
                "correct_answer": correct_indices[0] if correct_indices else 0,
                "category": row.get('category', 'unknown'),
                "difficulty": 0.5,  # Moderate difficulty
                "contestedness": contestedness,
                "metadata": {
                    "source_dataset": "truthful_qa",
                    "original_split": "validation",
                    "num_choices": len(choices)
                }
            })
        
        logger.info(f"TruthfulQA: Loaded {len(examples)} examples")
        return examples
        
    except Exception as e:
        logger.error(f"Failed to download TruthfulQA: {e}")
        raise

@logger.catch(reraise=True)
def main():
    """Main processing function."""
    logger.info("Starting dataset processing...")
    
    # Download all three datasets
    mmlu_examples = download_mmlu_hard()
    gpqa_examples = download_gpqa()
    truthfulqa_examples = download_truthfulqa()
    
    # Combine into unified dataset
    all_examples = mmlu_examples + gpqa_examples + truthfulqa_examples
    
    logger.info(f"Total examples: {len(all_examples)}")
    logger.info(f"  MMLU-Hard: {len(mmlu_examples)}")
    logger.info(f"  GPQA: {len(gpqa_examples)}")
    logger.info(f"  TruthfulQA: {len(truthfulqa_examples)}")
    
    # Save full dataset
    output_path = Path("data_out.json")
    output = {"examples": all_examples}
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved full dataset to {output_path}")
    
    # Generate mini dataset (10 per source)
    mini_examples = (
        mmlu_examples[:10] + 
        gpqa_examples[:10] + 
        truthfulqa_examples[:10]
    )
    mini_path = Path("data_out_mini.json")
    mini_path.write_text(json.dumps({"examples": mini_examples}, indent=2))
    logger.info(f"Saved mini dataset to {mini_path}")
    
    # Generate preview dataset (3 per source)
    preview_examples = (
        mmlu_examples[:3] + 
        gpqa_examples[:3] + 
        truthfulqa_examples[:3]
    )
    preview_path = Path("data_out_preview.json")
    preview_path.write_text(json.dumps({"examples": preview_examples}, indent=2))
    logger.info(f"Saved preview dataset to {preview_path}")
    
    # Validate schema
    logger.info("Validating output schema...")
    try:
        import jsonschema
        
        schema = {
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "question": {"type": "string"},
                            "choices": {"type": "array", "items": {"type": "string"}},
                            "correct_answer": {"type": ["integer", "array"]},
                            "category": {"type": "string"},
                            "difficulty": {"type": "number", "minimum": 0, "maximum": 1},
                            "contestedness": {"type": "number", "minimum": 0, "maximum": 1},
                            "metadata": {"type": "object"}
                        },
                        "required": ["id", "question", "choices", "correct_answer", "category"]
                    }
                }
            },
            "required": ["examples"]
        }
        
        jsonschema.validate(output, schema)
        logger.info("✓ Schema validation PASSED")
        
    except ImportError:
        logger.warning("jsonschema not installed, skipping validation")
    except Exception as e:
        logger.error(f"Schema validation FAILED: {e}")
        raise
    
    logger.info("Dataset processing complete!")

if __name__ == "__main__":
    main()
