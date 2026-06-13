#!/usr/bin/env python3
"""Process and standardize datasets to exp_sel_data_out.json schema."""

from loguru import logger
from pathlib import Path
import json
import sys

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add("logs/run.log", rotation="30 MB", level="DEBUG")

@logger.catch(reraise=True)
def main():
    """Main processing function."""
    logger.info("Starting dataset standardization...")
    
    # Load the processed datasets
    with open("data_out.json") as f:
        data = json.load(f)
    
    examples = data["examples"]
    
    # Group examples by dataset
    datasets = {}
    for ex in examples:
        src = ex["metadata"]["source_dataset"]
        if src not in datasets:
            datasets[src] = []
        datasets[src].append(ex)
    
    # Create output in exp_sel_data_out.json format
    output = {"datasets": []}
    
    for dataset_name, dataset_examples in datasets.items():
        logger.info(f"Processing {dataset_name}: {len(dataset_examples)} examples")
        
        formatted_examples = []
        for i, ex in enumerate(dataset_examples):
            # Convert to required schema format
            formatted_ex = {
                "input": ex["question"],
                "output": str(ex["correct_answer"]),
                "metadata_dataset": dataset_name,
                "metadata_category": ex["category"],
                "metadata_difficulty": ex["difficulty"],
                "metadata_contestedness": ex["contestedness"],
                "metadata_num_choices": ex["metadata"]["num_choices"],
                "metadata_choices": ex["choices"],
                "metadata_original_split": ex["metadata"]["original_split"],
                "metadata_example_index": i
            }
            formatted_examples.append(formatted_ex)
        
        output["datasets"].append({
            "dataset": dataset_name,
            "examples": formatted_examples
        })
    
    # Save output
    output_path = Path("full_data_out.json")
    output_path.write_text(json.dumps(output, indent=2))
    logger.info(f"Saved {len(output['datasets'])} datasets to {output_path}")
    
    # Print summary
    for ds in output["datasets"]:
        logger.info(f"  {ds['dataset']}: {len(ds['examples'])} examples")
    
    logger.info("Dataset standardization complete!")

if __name__ == "__main__":
    main()
