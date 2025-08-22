
import torch
import torch.nn as nn
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
MODEL_PATH = os.path.join(MODEL_DIR, 'rnn_embedding_model.pth')
VOCAB_PATH = os.path.join(MODEL_DIR, 'rnn_vocab.pkl')
CSV_PATH = os.path.join(MODEL_DIR, 'data', 'cypher_dataset.csv')

def prepare_data():
    if not os.path.exists(CSV_PATH):
        logging.error(f"Error: {CSV_PATH} not found. Please create it first.")
        return None, None, None, None, None, None
    df = pd.read_csv(CSV_PATH)
    logging.info("Successfully loaded data from CSV.")
    all_text = ''.join(df['Human Prompt']) + ''.join(df['Cypher Query'])
    chars = sorted(list(set(all_text)))
    vocab_size = len(chars)
    logging.info(f"Vocabulary Size: {vocab_size}")
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    prompts_encoded = [[char_to_int[c] for c in prompt] for prompt in df['Human Prompt']]
    queries_encoded = [[char_to_int[c] for c in query] for query in df['Cypher Query']]
    return df, vocab_size, char_to_int, int_to_char, prompts_encoded, queries_encoded

# --- 2. Building the RNN Model with Embeddings ---
# (This section remains the same)

class Seq2SeqRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(Seq2SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden):
        embedded = self.embedding(input_seq)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


def train_and_save():
    df, vocab_size, char_to_int, int_to_char, prompts_encoded, queries_encoded = prepare_data()
    if df is None:
        return None, None, None, None
    embedding_dim = 64
    hidden_size = 128
    learning_rate = 0.005
    epochs = 200
    model = Seq2SeqRNN(vocab_size, embedding_dim, hidden_size, vocab_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    logging.info("Starting training with Embeddings, Progress Bars, and Logging...")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        total_loss = 0
        for i in range(len(prompts_encoded)):
            prompt_seq = prompts_encoded[i]
            query_seq = queries_encoded[i]
            prompt_tensor = torch.tensor(prompt_seq, dtype=torch.long).unsqueeze(0)
            target_tensor = torch.tensor(query_seq, dtype=torch.long)
            hidden = model.init_hidden()
            optimizer.zero_grad()
            query_tensor_input = target_tensor[:-1].unsqueeze(0)
            output, _ = model(query_tensor_input, hidden)
            target_for_loss = target_tensor[1:]
            loss = loss_fn(output.view(-1, vocab_size), target_for_loss.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(prompts_encoded)
        if (epoch + 1) % 20 == 0:
            logging.info(f'Epoch [{epoch+1}/{epochs}], Average Loss: {average_loss:.4f}')
    logging.info("Training finished.")
    torch.save(model.state_dict(), MODEL_PATH)
    with open(VOCAB_PATH, 'wb') as f:
        pickle.dump({'char_to_int': char_to_int, 'int_to_char': int_to_char, 'vocab_size': vocab_size}, f)
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

def generate_cypher(prompt_text):
    """
    Given a natural language prompt, generate a Cypher query using the trained RNN model.
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
        prompt_encoded = [char_to_int.get(c, 0) for c in prompt_text]
        prompt_tensor = torch.tensor(prompt_encoded, dtype=torch.long).unsqueeze(0)
        hidden = model.init_hidden()
        
        # Use a default start char if not available
        start_char = list(char_to_int.keys())[0]
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

# Example usage (for testing):
if __name__ == "__main__":
    # Only train if model/vocab are missing
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)):
        train_and_save()
    # Test generation
    test_prompt = "Get all classes that inherit from BaseClass"
    print(generate_cypher(test_prompt))
