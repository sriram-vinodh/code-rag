import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import logging
from nl_to_cypher import Seq2SeqRNN, prepare_data, CSV_PATH
import itertools
import json
import os

logger = logging.getLogger(__name__)

def evaluate_model(model, val_prompts, val_queries, char_to_int, vocab_size, criterion):
    """Evaluate model performance on validation set"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for prompt_seq, query_seq in zip(val_prompts, val_queries):
            prompt_tensor = torch.tensor(prompt_seq, dtype=torch.long).unsqueeze(0)
            target_tensor = torch.tensor(query_seq, dtype=torch.long)
            hidden = model.init_hidden()
            query_tensor_input = target_tensor[:-1].unsqueeze(0)
            output, _ = model(query_tensor_input, hidden)
            target_for_loss = target_tensor[1:]
            loss = criterion(output.view(-1, vocab_size), target_for_loss.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_prompts)

def train_with_params(params, train_prompts, train_queries, val_prompts, val_queries, vocab_size, char_to_int):
    """Train model with given hyperparameters"""
    model = Seq2SeqRNN(vocab_size, params['embedding_dim'], params['hidden_size'], vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    patience = 5  # Number of epochs to wait for improvement
    
    for epoch in range(params['epochs']):
        model.train()
        total_loss = 0
        for i in range(len(train_prompts)):
            prompt_seq = train_prompts[i]
            query_seq = train_queries[i]
            prompt_tensor = torch.tensor(prompt_seq, dtype=torch.long).unsqueeze(0)
            target_tensor = torch.tensor(query_seq, dtype=torch.long)
            hidden = model.init_hidden()
            optimizer.zero_grad()
            query_tensor_input = target_tensor[:-1].unsqueeze(0)
            output, _ = model(query_tensor_input, hidden)
            target_for_loss = target_tensor[1:]
            loss = criterion(output.view(-1, vocab_size), target_for_loss.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        val_loss = evaluate_model(model, val_prompts, val_queries, char_to_int, vocab_size, criterion)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
    
    return best_val_loss

def tune_hyperparameters():
    """Find optimal hyperparameters using cross-validation"""
    logger.info("Loading data for hyperparameter tuning...")
    df, vocab_size, char_to_int, int_to_char, prompts_encoded, queries_encoded = prepare_data()
    if df is None:
        return None
    
    # Define hyperparameter search space
    param_grid = {
        'embedding_dim': [32, 64, 128],
        'hidden_size': [64, 128, 256],
        'learning_rate': [0.001, 0.005, 0.01],
        'epochs': [100, 200, 300]
    }
    
    # Generate all combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                        for v in itertools.product(*param_grid.values())]
    
    # Prepare for k-fold cross validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    results = []
    
    logger.info(f"Starting hyperparameter tuning with {len(param_combinations)} combinations...")
    
    for params in tqdm(param_combinations, desc="Parameter Combinations"):
        fold_scores = []
        
        # Perform k-fold cross validation
        for fold, (train_idx, val_idx) in enumerate(kf.split(prompts_encoded)):
            train_prompts = [prompts_encoded[i] for i in train_idx]
            train_queries = [queries_encoded[i] for i in train_idx]
            val_prompts = [prompts_encoded[i] for i in val_idx]
            val_queries = [queries_encoded[i] for i in val_idx]
            
            val_loss = train_with_params(params, train_prompts, train_queries, 
                                       val_prompts, val_queries, vocab_size, char_to_int)
            fold_scores.append(val_loss)
        
        avg_score = np.mean(fold_scores)
        results.append({
            'params': params,
            'avg_val_loss': avg_score,
            'std_val_loss': np.std(fold_scores)
        })
        
        logger.info(f"Params: {params}")
        logger.info(f"Average Validation Loss: {avg_score:.4f}")
        logger.info("-" * 50)
    
    # Find best parameters
    best_result = min(results, key=lambda x: x['avg_val_loss'])
    
    # Save results
    output_file = os.path.join(os.path.dirname(CSV_PATH), 'hyperparameter_tuning_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'all_results': results,
            'best_params': best_result['params'],
            'best_val_loss': best_result['avg_val_loss']
        }, f, indent=2)
    
    logger.info("\n" + "="*50)
    logger.info("Best Hyperparameters:")
    logger.info(f"Embedding Dimension: {best_result['params']['embedding_dim']}")
    logger.info(f"Hidden Size: {best_result['params']['hidden_size']}")
    logger.info(f"Learning Rate: {best_result['params']['learning_rate']}")
    logger.info(f"Epochs: {best_result['params']['epochs']}")
    logger.info(f"Validation Loss: {best_result['avg_val_loss']:.4f}")
    logger.info("="*50)
    
    return best_result['params']

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    logger.info("Starting hyperparameter tuning...")
    best_params = tune_hyperparameters()
    if best_params:
        logger.info("Hyperparameter tuning completed successfully!")
        logger.info("You can now update the parameters in nl_to_cypher.py with these values.")
