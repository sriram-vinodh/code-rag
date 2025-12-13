#!/usr/bin/env python3

import os
import sys
sys.path.append('/Users/sriram-14910/code/code-rag')

from helper_models.nl_to_cypher import prepare_data
import torch

# Set up environment
os.environ['TRAIN_CSV_PATH'] = 'tests/test_cypher_dataset.csv'

# Prepare data
df, vocab_size, char_to_int, int_to_char, prompts_encoded, queries_encoded = prepare_data()

print(f"Number of samples: {len(prompts_encoded)}")
print(f"Vocabulary size: {vocab_size}")

# Look at first few samples
for i in range(min(3, len(prompts_encoded))):
    print(f"Sample {i}:")
    print(f"  Prompt length: {len(prompts_encoded[i])}")
    print(f"  Query length: {len(queries_encoded[i])}")

# Check a batch
batch_size = 2
prompt_seqs = prompts_encoded[:batch_size]
query_seqs = queries_encoded[:batch_size]

# Make sure all sequences in the batch have the same length
max_prompt_len = max(len(seq) for seq in prompt_seqs)
max_query_len = max(len(seq) for seq in query_seqs)
max_batch_len = max(max_prompt_len, max_query_len)

print(f"\nBatch analysis:")
print(f"  Max prompt length: {max_prompt_len}")
print(f"  Max query length: {max_query_len}")
print(f"  Max batch length: {max_batch_len}")

# Pad sequences to max length for this batch
prompt_seqs_padded = [seq + [0] * (max_batch_len - len(seq)) for seq in prompt_seqs]
query_seqs_padded = [seq + [0] * (max_batch_len - len(seq)) for seq in query_seqs]

prompt_tensor = torch.tensor(prompt_seqs_padded, dtype=torch.long)
target_tensor = torch.tensor(query_seqs_padded, dtype=torch.long)

print(f"  Prompt tensor shape: {prompt_tensor.shape}")
print(f"  Target tensor shape: {target_tensor.shape}")

# Test model forward pass
from helper_models.nl_to_cypher import Seq2SeqRNN

model = Seq2SeqRNN(vocab_size, embedding_dim=128, hidden_size=256, output_size=vocab_size)
output, _ = model(prompt_tensor, target_tensor)

print(f"  Model output shape: {output.shape}")
print(f"  Expected shape: {target_tensor.shape + (vocab_size,)}")

# Test reshaping for loss
output_reshaped = output.contiguous().view(-1, vocab_size)
target_reshaped = target_tensor.contiguous().view(-1)

print(f"  Output reshaped: {output_reshaped.shape}")
print(f"  Target reshaped: {target_reshaped.shape}")
