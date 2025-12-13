
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import logging
import pickle
from tqdm import tqdm


# --- Setup Logging ---
logger = logging.getLogger(__name__)


# --- 0. Setup ---
MODEL_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(MODEL_DIR, exist_ok=True)  # Create model directory if it doesn't exist
MODEL_PATH = os.path.join(MODEL_DIR, 'rnn_embedding_model.pth')
VOCAB_PATH = os.path.join(MODEL_DIR, 'rnn_vocab.pkl')
CSV_PATH = os.path.join(MODEL_DIR, 'data', 'cypher_dataset.csv')

def pad_sequence(seq, max_len, pad_token=0):
    """Pad a sequence to max_len with pad_token"""
    return seq + [pad_token] * (max_len - len(seq))

def prepare_data(csv_path=None):
    """
    Prepare training data from CSV file.
    
    Args:
        csv_path: Optional path to CSV file. If None, uses default CSV_PATH
    """
    data_path = csv_path if csv_path else CSV_PATH
    if not os.path.exists(data_path):
        logging.error(f"Error: {data_path} not found. Please create it first.")
        return None, None, None, None, None, None
        
    df = pd.read_csv(data_path)
    logging.info(f"Successfully loaded data from {data_path}")
    
    # Find max sequence lengths
    max_prompt_len = max(len(text) for text in df['Human Prompt'])
    max_query_len = max(len(text) for text in df['Cypher Query'])
    # Use the same max length for both prompts and queries to ensure consistency
    max_len = max(max_prompt_len, max_query_len)
    # Build vocabulary
    all_text = ''.join(df['Human Prompt']) + ''.join(df['Cypher Query'])
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    logging.info(f"Vocabulary Size: {vocab_size}")
    char_to_int = {ch: i+1 for i, ch in enumerate(chars)}  # Reserve 0 for padding
    int_to_char = {i+1: ch for i, ch in enumerate(chars)}
    int_to_char[0] = '<PAD>'  # Add padding token
    
    # Encode and pad sequences to the same length
    prompts_encoded = [
        pad_sequence([char_to_int[c] for c in prompt], max_len)
        for prompt in df['Human Prompt']
    ]
    queries_encoded = [
        pad_sequence([char_to_int[c] for c in query], max_len)
        for query in df['Cypher Query']
    ]
    
    logging.info(f"Max prompt length: {max_prompt_len}")
    logging.info(f"Max query length: {max_query_len}")
    logging.info(f"Using unified max length: {max_len}")
    
    return df, vocab_size + 1, char_to_int, int_to_char, prompts_encoded, queries_encoded

# --- 2. Building the RNN Model with Embeddings ---
# (This section remains the same)

class Seq2SeqRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size):
        super(Seq2SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Encoder
        self.embedding_encoder = nn.Embedding(input_size, embedding_dim)
        self.lstm_encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        
        # Decoder
        self.embedding_decoder = nn.Embedding(output_size, embedding_dim)
        self.project_decoder = nn.Linear(embedding_dim, hidden_size)  # Project decoder embeddings
        self.lstm_decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Attention
        self.attention_combine = nn.Linear(hidden_size * 2, hidden_size)  # Combine context and hidden
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def encode(self, input_seq):
        # Input shape: (batch_size, seq_len)
        embedded = self.dropout(self.embedding_encoder(input_seq))  # [B, S, E]
        hidden = self.init_hidden(input_seq.size(0))  # Initialize hidden state
        encoder_outputs, hidden = self.lstm_encoder(embedded, hidden)  # outputs: [B, S, H], hidden: ([1, B, H], [1, B, H])
        return encoder_outputs, hidden  # Return LSTM outputs for attention
        
    def attention_mechanism(self, decoder_state, encoder_outputs):
        # decoder_state: [B, H]
        # encoder_outputs: [B, S, H]
        
        # Add sequence length dimension to decoder state
        decoder_state = decoder_state.unsqueeze(1)  # [B, 1, H]
        
        # Calculate attention scores using dot product
        attention_scores = torch.bmm(decoder_state, encoder_outputs.transpose(1, 2))  # [B, 1, S]
        attention_weights = F.softmax(attention_scores, dim=2)  # [B, 1, S]
        
        # Apply attention to encoder outputs
        context = torch.bmm(attention_weights, encoder_outputs)  # [B, 1, H]
        return context, attention_weights
        
    def decode(self, input_seq, hidden, encoder_outputs):
        # Input shape: [B, 1]
        # Embed input
        embedded = self.dropout(self.embedding_decoder(input_seq))  # [B, 1, E]
        
        # Project embedded to hidden size
        embedded_proj = self.project_decoder(embedded)  # [B, 1, H]
        
        # Get attention context
        # Take last layer's hidden state and convert to [B, H]
        decoder_state = hidden[0][-1]  # [B, H]
        
        context, attention_weights = self.attention_mechanism(decoder_state, encoder_outputs)  # [B, 1, H]
        
        # Combine projected embedded input and attention context
        rnn_input = self.attention_combine(torch.cat((embedded_proj, context), dim=-1))  # [B, 1, H]
        
        # LSTM decoder step
        output, hidden = self.lstm_decoder(rnn_input, hidden)  # [B, 1, H]
        
        # Project to vocabulary size
        output = self.out(output)  # [B, 1, V]
        return output, hidden, attention_weights
        
    def forward(self, input_seq, target_seq=None):
        # Get batch size and device
        batch_size = input_seq.size(0)
        device = input_seq.device
        
        # Encode input sequence
        encoder_outputs, hidden = self.encode(input_seq)
        
        max_output_length = target_seq.size(1) if target_seq is not None else 200
        outputs = []
        decoder_input = torch.zeros((batch_size, 1), dtype=torch.long, device=device)  # Start token

        # Always generate exactly max_output_length steps during training
        for i in range(max_output_length):
            output, hidden, _ = self.decode(decoder_input, hidden, encoder_outputs)
            outputs.append(output)

            if target_seq is not None:
                # Teacher forcing: use target as next input
                decoder_input = target_seq[:, i:i+1]
            else:
                # Inference: use predicted token
                decoder_input = output.argmax(dim=-1)
                decoder_input = decoder_input.view(batch_size, 1)
                if (decoder_input == 0).all():
                    break

        # Stack all outputs along the sequence dimension
        outputs = torch.cat(outputs, dim=1)  # [B, T, V]
        # If in inference mode, pad output to max_output_length for consistency
        if target_seq is not None:
            outputs = outputs[:, :max_output_length, :]
        else:
            if outputs.size(1) < max_output_length:
                pad_size = max_output_length - outputs.size(1)
                pad = torch.zeros((batch_size, pad_size, outputs.size(2)), device=outputs.device)
                outputs = torch.cat([outputs, pad], dim=1)
        return outputs, hidden
        
    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


def train_and_save(csv_path=None, validation_split=0.2, early_stop_patience=5, batch_size=32):
    """
    Train the model with validation and early stopping.
    
    Args:
        csv_path: Optional path to training data
        validation_split: Fraction of data to use for validation
        early_stop_patience: Number of epochs to wait for improvement before stopping
        batch_size: Training batch size
    """
    global char_to_int, int_to_char  # Make these available to the model
    df, vocab_size, char_to_int, int_to_char, prompts_encoded, queries_encoded = prepare_data(csv_path)
    if df is None:
        return None, None, None, None

    # Split data into train and validation sets
    total_samples = len(prompts_encoded)
    val_size = max(1, int(total_samples * validation_split))  # Ensure at least 1 validation sample
    train_size = total_samples - val_size
    
    indices = torch.randperm(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_prompts = [prompts_encoded[i] for i in train_indices]
    train_queries = [queries_encoded[i] for i in train_indices]
    val_prompts = [prompts_encoded[i] for i in val_indices]
    val_queries = [queries_encoded[i] for i in val_indices]

    # Model parameters - increased for better learning
    embedding_dim = 128  # Increased from 64
    hidden_size = 256    # Increased from 128
    learning_rate = 0.001  # Reduced for stability
    epochs = 500        # Increased for better convergence
    
    model = Seq2SeqRNN(vocab_size, embedding_dim, hidden_size, vocab_size)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    logging.info("Starting training with validation and early stopping...")
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Training phase
        model.train()
        total_train_loss = 0
        num_batches = len(train_prompts) // batch_size + (1 if len(train_prompts) % batch_size != 0 else 0)
        
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
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with teacher forcing
        output, _ = model(prompt_tensor, target_tensor)
        
        # Calculate loss - reshape the output to match target shape
        output_reshaped = output.contiguous().view(-1, vocab_size)  # [B*T, V]
        target_reshaped = target_tensor.contiguous().view(-1)  # [B*T]
        
        loss = loss_fn(output_reshaped, target_reshaped)
        
        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        total_train_loss += loss.item() * batch_size_actual
        
        avg_train_loss = total_train_loss / len(train_prompts)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for b in range(0, len(val_prompts), batch_size):
                batch_end = min(b + batch_size, len(val_prompts))
                batch_size_actual = batch_end - b
                
                # Prepare batch tensors
                prompt_seqs = val_prompts[b:batch_end]
                query_seqs = val_queries[b:batch_end]

                # Pad both prompt and query sequences to the same max length
                max_len = max(max(len(seq) for seq in prompt_seqs), max(len(seq) for seq in query_seqs))
                prompt_seqs_padded = [seq + [0] * (max_len - len(seq)) for seq in prompt_seqs]
                query_seqs_padded = [seq + [0] * (max_len - len(seq)) for seq in query_seqs]

                prompt_tensor = torch.tensor(prompt_seqs_padded, dtype=torch.long)
                target_tensor = torch.tensor(query_seqs_padded, dtype=torch.long)
                
                # Generate complete sequences
                output, _ = model(prompt_tensor)
                
                # Calculate validation loss - reshape the output to match target shape
                output_reshaped = output.contiguous().view(-1, vocab_size)  # [B*T, V]
                target_reshaped = target_tensor.contiguous().view(-1)  # [B*T]
                loss = loss_fn(output_reshaped, target_reshaped)
                total_val_loss += loss.item() * batch_size_actual
                
                # Check prediction accuracy
                predicted_seqs = torch.argmax(output, dim=-1)
                
                # Check each sequence in the batch
                for i in range(batch_size_actual):
                    # Convert sequences back to strings, ignoring padding
                    pred_tokens = predicted_seqs[i]
                    target_tokens = target_tensor[i]
                    
                    pred_str = ''.join([int_to_char[idx.item()] for idx in pred_tokens if idx.item() != 0])
                    target_str = ''.join([int_to_char[idx.item()] for idx in target_tokens if idx.item() != 0])
                    
                    # Check if the core query structure matches
                    if "MATCH" in pred_str and "RETURN" in pred_str:
                        if pred_str.strip().lower() == target_str.strip().lower():
                            correct_predictions += 1
        
        avg_val_loss = total_val_loss / len(val_prompts)
        accuracy = correct_predictions / len(val_prompts)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f'Epoch [{epoch+1}/{epochs}]')
            logging.info(f'Training Loss: {avg_train_loss:.4f}')
            logging.info(f'Validation Loss: {avg_val_loss:.4f}')
            logging.info(f'Validation Accuracy: {accuracy:.2%}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    torch.save(model.state_dict(), MODEL_PATH)
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump({
            'char_to_int': char_to_int,
            'int_to_char': int_to_char,
            'vocab_size': vocab_size
        }, f)
    
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Model weights saved to {MODEL_PATH}")
    logging.info(f"Vocab saved to {VOCAB_PATH}")
    
    return model, char_to_int, int_to_char, vocab_size


def load_model_and_vocab():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)):
        return None, None, None, None
    with open(VOCAB_PATH, 'rb') as f:
        vocab_data = pickle.load(f)
    char_to_int = vocab_data['char_to_int']
    int_to_char = vocab_data['int_to_char']
    vocab_size = vocab_data['vocab_size']
    model = Seq2SeqRNN(vocab_size, 64, 128, vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model, char_to_int, int_to_char, vocab_size

def generate_cypher(prompt_text, validate=True, max_length=200):
    """
    Given a natural language prompt, generate a Cypher query using the trained RNN model.
    
    Args:
        prompt_text: Natural language query description
        validate: Whether to validate the generated query
        max_length: Maximum length of generated query
        
    Returns:
        Generated Cypher query or None if validation fails
    """
    logger.info("\n=== 🤖 Natural Language to Cypher ===")
    logger.info("Input prompt: %s", prompt_text)
    
    model, char_to_int, int_to_char, vocab_size = load_model_and_vocab()
    if model is None:
        logger.warning("⚠️ Model or vocab not found. Training from scratch...")
        model, char_to_int, int_to_char, vocab_size = train_and_save()
        if model is None:
            logger.error("❌ Failed to train model")
            return None
            
    logger.info("✅ Model and vocabulary loaded successfully")
    
    with torch.no_grad():
        # Clean and validate input
        prompt_text = prompt_text.strip()
        if not prompt_text:
            logger.error("Empty prompt received")
            return None
        
        # Prepare input
        prompt_encoded = [char_to_int.get(c, 0) for c in prompt_text]
        prompt_tensor = torch.tensor(prompt_encoded, dtype=torch.long).unsqueeze(0)
        
        # Generate sequence
        model.eval()
        output, _ = model(prompt_tensor)
        
        # Get predicted sequence
        predicted = torch.argmax(output, dim=-1).squeeze(0)
        generated_query = ''.join([int_to_char[idx.item()] for idx in predicted])
        
        # Post-process query
        generated_query = generated_query.strip()
        if not generated_query.startswith("MATCH"):
            generated_query = "MATCH" + generated_query
        if not generated_query.endswith(";"):
            generated_query += ";"
            
        # Basic validation
        if validate:
            if not all(keyword in generated_query for keyword in ["MATCH", "RETURN"]):
                logger.warning("Generated query missing required keywords")
                return None
                
            # Check for balanced parentheses and brackets
            if generated_query.count('(') != generated_query.count(')'):
                logger.warning("Unbalanced parentheses in generated query")
                return None
                
            if generated_query.count('[') != generated_query.count(']'):
                logger.warning("Unbalanced brackets in generated query")
                return None
        
        logger.info("\n=== 🎯 Generated Cypher Query ===\n%s\n===============================", generated_query)
        return generated_query
        decoder_input = torch.tensor([[char_to_int[start_char]]], dtype=torch.long)
        generated_query = start_char
        
        logger.info("🔄 Generating Cypher query...")
        for _ in range(200):
            output, hidden = model(decoder_input, hidden)
            _, top_i = output.topk(1)
            predicted_char = int_to_char[top_i.item()]
            generated_query += predicted_char
            decoder_input = top_i.squeeze(0)
            if predicted_char == ';':
                break
        
        logger.info("\n=== 🎯 Generated Cypher Query ===\n%s\n===============================", generated_query)
        return generated_query

# Command-line usage
if __name__ == "__main__":
    import argparse
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'  # Simplified format for cleaner output
    )
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Generate Cypher queries from natural language descriptions.'
    )
    parser.add_argument(
        'query',
        nargs='?',  # Makes the argument optional
        help='Natural language description of the query'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    try:
        # Ensure model is trained
        if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)):
            print("🔄 Model not found. Training from scratch...")
            train_and_save()
            
        if args.interactive:
            print("\n🔄 Interactive Mode - Type 'exit' to quit")
            print("="*50)
            while True:
                try:
                    prompt = input("\n📝 Enter query description: ")
                    if prompt.lower() in ['exit', 'quit', ':q']:
                        break
                    if not prompt.strip():
                        continue
                        
                    print("\n🔍 Generating Cypher query...")
                    result = generate_cypher(prompt)
                    
                    if result:
                        print("\n✅ Generated Cypher Query:")
                        print("="*50)
                        print(result)
                        print("="*50)
                    else:
                        print("\n❌ Failed to generate query")
                        
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"\n❌ Error: {e}")
            
            print("\n👋 Goodbye!")
            
        elif args.query:
            result = generate_cypher(args.query)
            if result:
                # Print only the query for easy piping
                print(result)
            else:
                print("Failed to generate query", file=sys.stderr)
                sys.exit(1)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
