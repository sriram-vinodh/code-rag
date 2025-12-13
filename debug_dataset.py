#!/usr/bin/env python3

import os
import sys
sys.path.append('/Users/sriram-14910/code/code-rag')

# Set up environment to use test data
os.environ['TRAIN_CSV_PATH'] = 'tests/test_cypher_dataset.csv'

from helper_models.nl_to_cypher import prepare_data

# Test the actual prepare_data function with test CSV
df, vocab_size, char_to_int, int_to_char, prompts_encoded, queries_encoded = prepare_data()

print(f"Dataset loaded with {len(prompts_encoded)} samples")
print(f"Vocabulary size: {vocab_size}")

# Check first few samples
for i in range(min(5, len(prompts_encoded))):
    print(f"Sample {i}: prompt len={len(prompts_encoded[i])}, query len={len(queries_encoded[i])}")

# Check if all sequences have the same length
all_prompt_lens = [len(seq) for seq in prompts_encoded]
all_query_lens = [len(seq) for seq in queries_encoded]

print(f"Unique prompt lengths: {set(all_prompt_lens)}")
print(f"Unique query lengths: {set(all_query_lens)}")

if len(set(all_prompt_lens)) == 1 and len(set(all_query_lens)) == 1 and all_prompt_lens[0] == all_query_lens[0]:
    print(f"✓ All sequences have consistent length: {all_prompt_lens[0]}")
else:
    print("✗ Sequences have inconsistent lengths!")
    print(f"Prompt length range: {min(all_prompt_lens)}-{max(all_prompt_lens)}")
    print(f"Query length range: {min(all_query_lens)}-{max(all_query_lens)}")
