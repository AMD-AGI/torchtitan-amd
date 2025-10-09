#!/usr/bin/env python3
"""
Script to download the NeelNanda/c4_test dataset from Hugging Face
and save it locally for use with torchtitan.
"""

import argparse
import os
from datasets import load_dataset, load_from_disk


def download_c4_test_dataset(
    save_dir: str = "../assets/hf/c4_test",
    streaming: bool = False,
):
    """
    Download the NeelNanda/c4_test dataset from Hugging Face and save it to disk.
    
    Args:
        save_dir (str): Directory to save the dataset (default: .assets/hf_datasets/c4_test)
        streaming (bool): If True, uses streaming mode (doesn't download entire dataset)
    """
    
    repo_id = "NeelNanda/c4_test"
    
    print(f"Downloading dataset: {repo_id}")
    print(f"Save directory: {save_dir}")
    print(f"Streaming mode: {streaming}")
    
    try:
        # Load the dataset
        print("\nLoading dataset from Hugging Face...")
        dataset = load_dataset(
            repo_id,
            split="train",
            streaming=streaming,
        )
        
        print(f"Dataset loaded successfully!")
        
        if not streaming:
            # Print dataset information
            print(f"\nDataset information:")
            print(f"  Number of examples: {len(dataset)}")
            print(f"  Features: {dataset.features}")
            
            # Show a sample
            print("\nSample data (first example):")
            print(f"  Text preview: {dataset[0]['text'][:200]}...")
            
            # Save to disk
            print(f"\nSaving dataset to disk at: {save_dir}")
            os.makedirs(save_dir, exist_ok=True)
            dataset.save_to_disk(save_dir)
            print("Dataset saved successfully!")
            
            # Verify saved dataset
            print(f"\nVerifying saved dataset...")
            loaded_dataset = load_from_disk(save_dir)
            print(f"  Verification successful! Loaded {len(loaded_dataset)} examples")
            
        else:
            print("\nStreaming mode enabled.")
            print("Note: Cannot save streaming datasets. Remove --streaming flag to save to disk.")
        
        return dataset
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the NeelNanda/c4_test dataset and save it locally"
    )
    
    parser.add_argument(
        "--save_dir",
        type=str,
        default=".assets/hf_datasets/c4_test",
        help="Directory to save the dataset (default: .assets/hf_datasets/c4_test)",
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (won't save to disk)",
    )
    
    args = parser.parse_args()
    
    dataset = download_c4_test_dataset(
        save_dir=args.save_dir,
        streaming=args.streaming,
    )
    
    print("\n" + "="*60)
    print("Download complete!")
    print(f"Dataset saved to: {args.save_dir}")
    print("You can now use it with: dataset_name='c4_test_local'")
    print("="*60)
