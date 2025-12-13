#!/usr/bin/env python3

import os
import sys
sys.path.append('/Users/sriram-14910/code/code-rag')

from helper_models.nl_to_cypher import prepare_data, Seq2SeqRNN
import torch
import torch.nn as nn

# Set up environment
os.environ['TRAIN_CSV_PATH'] = 'tests/test_cypher_dataset.csv'

# Prepare data
df, vocab_size, char_to_int, int_to_char, prompts_encoded, queries_encoded = prepare_data()

print(f"Vocabulary size: {vocab_size}")

# Set up training data (subset)
train_prompts = prompts_encoded[:8]  # Small batch
train_queries = queries_encoded[:8]

# Model setup
model = Seq2SeqRNN(vocab_size, embedding_dim=128, hidden_size=256, output_size=vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

# Simulate one batch step
batch_size = 2
b = 0
batch_end = min(b + batch_size, len(train_prompts))
batch_size_actual = batch_end - b

# Prepare batch tensors
prompt_seqs = train_prompts[b:batch_end]
query_seqs = train_queries[b:batch_end]

print(f"Prompt lengths: {[len(seq) for seq in prompt_seqs]}")
print(f"Query lengths: {[len(seq) for seq in query_seqs]}")

# Pad both prompt and query sequences to the same max length
max_len = max(max(len(seq) for seq in prompt_seqs), max(len(seq) for seq in query_seqs))
print(f"Max length: {max_len}")

prompt_seqs_padded = [seq + [0] * (max_len - len(seq)) for seq in prompt_seqs]
query_seqs_padded = [seq + [0] * (max_len - len(seq)) for seq in query_seqs]

prompt_tensor = torch.tensor(prompt_seqs_padded, dtype=torch.long)
target_tensor = torch.tensor(query_seqs_padded, dtype=torch.long)

print(f"Prompt tensor shape: {prompt_tensor.shape}")
print(f"Target tensor shape: {target_tensor.shape}")

# Forward pass with teacher forcing
output, _ = model(prompt_tensor, target_tensor)

print(f"Output shape: {output.shape}")

# Calculate loss - reshape the output to match target shape
output_reshaped = output.contiguous().view(-1, vocab_size)  # [B*T, V]
target_reshaped = target_tensor.contiguous().view(-1)  # [B*T]

print(f"Output reshaped: {output_reshaped.shape}")
print(f"Target reshaped: {target_reshaped.shape}")

try:
    loss = loss_fn(output_reshaped, target_reshaped)
    print(f"Loss calculation successful: {loss.item()}")
except Exception as e:
    print(f"Loss calculation failed: {e}")

# Now test the full training function with limited epochs
print("\n--- Testing full training function ---")

# Copy the training logic but with 1 epoch
train_prompts = prompts_encoded[:8]  
train_queries = queries_encoded[:8]

model = Seq2SeqRNN(vocab_size, embedding_dim=128, hidden_size=256, output_size=vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

batch_size = 32
epochs = 1

for epoch in range(epochs):
    total_train_loss = 0
    model.train()
    
    for b in range(0, len(train_prompts), batch_size):
        batch_end = min(b + batch_size, len(train_prompts))
        batch_size_actual = batch_end - b
        
        # Prepare batch tensors
        prompt_seqs = train_prompts[b:batch_end]
        query_seqs = train_queries[b:batch_end]

        # Pad both prompt and query sequences to the same max length
        max_len = max(max(len(seq) for seq in prompt_seqs), max(len(seq) for seq in query_seqs))
        prompt_seqs_padded = [seq + [0] * (max_len - len(seq)) for seq in prompt_seqs]
        query_seqs_padded = [seq + [0] * (max_len - len(seq)) for seq in query_seqs]

        prompt_tensor = torch.tensor(prompt_seqs_padded, dtype=torch.long)
        target_tensor = torch.tensor(query_seqs_padded, dtype=torch.long)
        
        print(f"Batch {b//batch_size}: prompt shape {prompt_tensor.shape}, target shape {target_tensor.shape}")
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with teacher forcing
        output, _ = model(prompt_tensor, target_tensor)
        
        print(f"Output shape: {output.shape}")
        
        # Calculate loss - reshape the output to match target shape
        output_reshaped = output.contiguous().view(-1, vocab_size)  # [B*T, V]
        target_reshaped = target_tensor.contiguous().view(-1)  # [B*T]
        
        print(f"Reshaped: output {output_reshaped.shape}, target {target_reshaped.shape}")
        
        try:
            loss = loss_fn(output_reshaped, target_reshaped)
            print(f"Loss: {loss.item()}")
        except Exception as e:
            print(f"Loss calculation failed: {e}")
            break
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_train_loss += loss.item() * batch_size_actual

print("Mini training completed successfully!")
