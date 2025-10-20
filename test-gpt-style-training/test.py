#!/usr/bin/env python3
"""
🤖 Urdu Conversational Chatbot with Transformer Architecture
Complete implementation from scratch using PyTorch

Features:
- Custom Transformer Encoder-Decoder architecture
- BPE tokenization with Urdu-specific handling
- Teacher forcing during training
- Comprehensive evaluation metrics (BLEU, ROUGE-L, chrF, Perplexity)
- Interactive chat interface

Usage:
    python urdu_chatbot_complete.py --mode train    # Train the model
    python urdu_chatbot_complete.py --mode chat     # Interactive chat
    python urdu_chatbot_complete.py --mode eval     # Evaluate model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import re
import json
import random
import math
import time
import sys
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# 1. DATA PREPROCESSING & NORMALIZATION
# ============================================================================

def normalize_urdu_text(text):
    """Normalize Urdu text by removing diacritics and standardizing forms"""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove Arabic diacritical marks
    diacritics = [
        '\u0610', '\u0611', '\u0612', '\u0613', '\u0614', '\u0615', '\u0616',
        '\u0617', '\u0618', '\u0619', '\u061A', '\u064B', '\u064C', '\u064D',
        '\u064E', '\u064F', '\u0650', '\u0651', '\u0652', '\u0653', '\u0654',
        '\u0655', '\u0656', '\u0657', '\u0658', '\u0659', '\u065A', '\u065B',
        '\u065C', '\u065D', '\u065E', '\u065F', '\u0670', '\u06D6', '\u06D7',
        '\u06D8', '\u06D9', '\u06DA', '\u06DB', '\u06DC', '\u06DF', '\u06E0',
        '\u06E1', '\u06E2', '\u06E3', '\u06E4', '\u06E7', '\u06E8', '\u06EA',
        '\u06EB', '\u06EC', '\u06ED'
    ]
    
    for diacritic in diacritics:
        text = text.replace(diacritic, '')
    
    # Standardize Alef and Yeh forms
    text = re.sub('[أإآٱ]', 'ا', text)
    text = re.sub('[يىئ]', 'ی', text)
    
    # Remove zero-width characters
    text = text.replace('\u200c', '').replace('\u200d', '')
    
    # Normalize punctuation
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = text.replace('\u2026', '...').replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('`', "'")
    
    return text.strip()

# ============================================================================
# 2. BPE TOKENIZER
# ============================================================================

class BPETokenizer:
    """Byte Pair Encoding tokenizer with RTL support for Urdu"""
    
    def _init_(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = []
        self.token_to_id = {}
        self.id_to_token = {}
    
    def get_stats(self, words: Dict[str, int]) -> Counter:
        """Count frequency of adjacent pairs in the corpus"""
        pairs = Counter()
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_pair(self, pair: Tuple[str, str], words: Dict[str, int]) -> Dict[str, int]:
        """Merge all occurrences of the most frequent pair"""
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in words:
            new_word = word.replace(bigram, replacement)
            new_words[new_word] = words[word]
        return new_words
    
    def train(self, corpus: List[str], verbose: bool = True):
        """Train BPE on corpus"""
        word_freqs = Counter()
        for text in corpus:
            word_freqs.update(text.split())
        
        # RTL-aware: </w> at beginning
        words = {}
        for word, freq in word_freqs.items():
            words['</w> ' + ' '.join(list(word))] = freq
        
        vocab = set()
        for word in words.keys():
            vocab.update(word.split())
        
        if verbose:
            print(f"Initial vocab: {len(vocab)} characters")
        
        num_merges = self.vocab_size - len(vocab)
        for i in range(num_merges):
            pairs = self.get_stats(words)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            words = self.merge_pair(best_pair, words)
            self.merges.append(best_pair)
        
        final_vocab = set()
        for word in words.keys():
            final_vocab.update(word.split())
        
        self.vocab = sorted(list(final_vocab))
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        if verbose:
            print(f"Final vocab: {len(self.vocab)} tokens")
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using BPE"""
        words = text.split()
        tokens = []
        for word in words:
            word_tokens = ['</w>'] + list(word)
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if (word_tokens[i], word_tokens[i + 1]) == pair:
                        word_tokens = word_tokens[:i] + [''.join(pair)] + word_tokens[i + 2:]
                    else:
                        i += 1
            tokens.extend(word_tokens)
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.token_to_id.get('<UNK>', 0)) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [self.id_to_token.get(idx, '<UNK>') for idx in token_ids]
        text = ''.join(tokens).replace('</w> ', ' ').replace('</w>', ' ')
        return text.strip()

# ============================================================================
# 3. DATASET PREPARATION
# ============================================================================

class UrduChatbotDataset(Dataset):
    """Dataset for Urdu chatbot with teacher forcing"""
    
    def _init_(self, sentence_groups, tokenizer, max_len=50):
        self.sentence_groups = sentence_groups
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.unk_token = '<UNK>'
        
        # Add special tokens to vocabulary
        for token in [self.pad_token, self.start_token, self.end_token, self.unk_token]:
            if token not in self.tokenizer.token_to_id:
                idx = len(self.tokenizer.token_to_id)
                self.tokenizer.token_to_id[token] = idx
                self.tokenizer.id_to_token[idx] = token
        
        self.pad_idx = self.tokenizer.token_to_id[self.pad_token]
        self.start_idx = self.tokenizer.token_to_id[self.start_token]
        self.end_idx = self.tokenizer.token_to_id[self.end_token]
        self.unk_idx = self.tokenizer.token_to_id[self.unk_token]
    
    def _len_(self):
        return len(self.sentence_groups)
    
    def _getitem_(self, idx):
        # Split based on word count (2/5th for input, 3/5th for target)
        group = self.sentence_groups[idx]
        words = group.split()
        
        # Calculate split point based on word count
        total_words = len(words)
        split_point = max(1, total_words * 2 // 5)  # At least 1 word for input
        
        input_text = ' '.join(words[:split_point])
        target_text = ' '.join(words[split_point:])
        
        # Tokenize input and target
        input_tokens = self.tokenizer.tokenize(input_text)
        target_tokens = self.tokenizer.tokenize(target_text)
        
        # Convert to IDs
        input_ids = [self.tokenizer.token_to_id.get(t, self.unk_idx) for t in input_tokens]
        target_ids = [self.tokenizer.token_to_id.get(t, self.unk_idx) for t in target_tokens]
        
        # Truncate and pad
        input_ids = input_ids[:self.max_len] + [self.pad_idx] * (self.max_len - len(input_ids))
        target_ids = target_ids[:self.max_len] + [self.pad_idx] * (self.max_len - len(target_ids))
        
        # Teacher forcing: decoder input is [START] + target[:-1], decoder target is target
        decoder_input_ids = [self.start_idx] + target_ids[:-1]
        decoder_target_ids = target_ids
        
        return {
            'encoder_input': torch.tensor(input_ids, dtype=torch.long),
            'decoder_input': torch.tensor(decoder_input_ids, dtype=torch.long),
            'decoder_target': torch.tensor(decoder_target_ids, dtype=torch.long)
        }

# ============================================================================
# 4. TRANSFORMER ARCHITECTURE
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine"""
    def _init_(self, d_model, max_len=5000, dropout=0.1):
        super()._init_()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    def _init_(self, d_model, num_heads, dropout=0.1):
        super()._init_()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def _init_(self, d_model, d_ff, dropout=0.1):
        super()._init_()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""
    def _init_(self, d_model, num_heads, d_ff, dropout=0.1):
        super()._init_()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class DecoderLayer(nn.Module):
    """Single Transformer Decoder Layer"""
    def _init_(self, d_model, num_heads, d_ff, dropout=0.1):
        super()._init_()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        return x

class Transformer(nn.Module):
    """Complete Transformer Encoder-Decoder Model"""
    def _init_(self, vocab_size, d_model=256, num_heads=2, d_ff=1024,
                 num_encoder_layers=2, num_decoder_layers=2, max_len=512,
                 dropout=0.1, pad_idx=0):
        super()._init_()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.encoder_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.decoder_pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # Encoder
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.encoder_pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        encoder_output = x
        
        # Decoder
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.decoder_pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        output = self.output_projection(x)
        return output

# ============================================================================
# 5. TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch with teacher forcing"""
    model.train()
    epoch_loss = 0
    pbar = tqdm(train_loader, desc="Training")
    
    for batch in pbar:
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        decoder_target = batch['decoder_target'].to(device)
        
        optimizer.zero_grad()
        output = model(encoder_input, decoder_input)
        output = output.reshape(-1, output.size(-1))
        target = decoder_target.reshape(-1)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            decoder_target = batch['decoder_target'].to(device)
            
            output = model(encoder_input, decoder_input)
            output = output.reshape(-1, output.size(-1))
            target = decoder_target.reshape(-1)
            loss = criterion(output, target)
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return epoch_loss / len(val_loader)

# ============================================================================
# 6. EVALUATION METRICS
# ============================================================================

def calculate_bleu_score(predicted, reference):
    """Calculate BLEU score"""
    smoothing = SmoothingFunction().method1
    reference_tokens = reference.split()
    predicted_tokens = predicted.split()
    
    if len(reference_tokens) == 0 or len(predicted_tokens) == 0:
        return 0.0
    
    return sentence_bleu([reference_tokens], predicted_tokens, smoothing_function=smoothing)

def calculate_rouge_l(predicted, reference):
    """Calculate ROUGE-L score"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, predicted)
    return scores['rougeL'].fmeasure

def calculate_chrf(predicted, reference):
    """Calculate chrF score (character-level F-score)"""
    def get_char_ngrams(text, n):
        return [text[i:i+n] for i in range(len(text)-n+1)]
    
    def f_score(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    total_f = 0.0
    for n in range(1, 7):  # 1-6 character n-grams
        pred_ngrams = get_char_ngrams(predicted, n)
        ref_ngrams = get_char_ngrams(reference, n)
        
        if len(pred_ngrams) == 0 and len(ref_ngrams) == 0:
            continue
        
        precision = len(set(pred_ngrams) & set(ref_ngrams)) / max(len(pred_ngrams), 1)
        recall = len(set(pred_ngrams) & set(ref_ngrams)) / max(len(ref_ngrams), 1)
        
        total_f += f_score(precision, recall)
    
    return total_f / 6

def calculate_perplexity(model, test_loader, device):
    """Calculate perplexity on test set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in test_loader:
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            decoder_target = batch['decoder_target'].to(device)
            
            output = model(encoder_input, decoder_input)
            output = output.reshape(-1, output.size(-1))
            target = decoder_target.reshape(-1)
            
            loss = F.cross_entropy(output, target, reduction='sum')
            total_loss += loss.item()
            total_tokens += (target != 0).sum().item()  # Exclude padding tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

# ============================================================================
# 7. TEXT GENERATION
# ============================================================================

def generate_text(model, tokenizer, dataset, input_text, max_length=30, temperature=0.8, device='cpu'):
    """Generate text continuation using beam search"""
    model.eval()
    
    with torch.no_grad():
        tokens = tokenizer.tokenize(input_text)
        token_ids = [tokenizer.token_to_id.get(t, dataset.unk_idx) for t in tokens]
        encoder_ids = token_ids[:50] + [dataset.pad_idx] * (50 - len(token_ids))
        encoder_input = torch.tensor([encoder_ids], dtype=torch.long).to(device)
        decoder_input = torch.tensor([[dataset.start_idx]], dtype=torch.long).to(device)
        
        generated_tokens = []
        for _ in range(max_length):
            output = model(encoder_input, decoder_input)
            next_token_logits = output[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == dataset.end_idx:
                break
            
            generated_tokens.append(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        
        output_tokens = []
        for idx in generated_tokens:
            if idx not in [dataset.pad_idx, dataset.start_idx, dataset.unk_idx]:
                token = tokenizer.id_to_token.get(idx, '<UNK>')
                output_tokens.append(token)
        
        text = ''.join(output_tokens).replace('</w> ', ' ').replace('</w>', ' ')
        return text.strip()

# ============================================================================
# 8. MAIN FUNCTIONS
# ============================================================================

def load_data():
    """Load sentence groups from file"""
    print("Loading sentence groups...")
    with open('cleaned_urdu_text.txt', 'r', encoding='utf-8') as f:
        sentence_groups = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(sentence_groups)} sentence groups")
    return sentence_groups

def train_model(resume=False):
    """Train the Urdu chatbot model"""
    if resume:
        print("🔄 Resuming Urdu Chatbot Training")
        print("=" * 60)
    else:
        print("🚀 Starting Urdu Chatbot Training")
        print("=" * 60)
    
    # Configuration following project specs
    config = {
        'vocab_size': 800,
        'd_model': 512,  # Embedding dimensions
        'num_heads': 2,  # Number of attention heads
        'd_ff': 1024,    # Feed-forward dimensions
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'max_len': 50,
        'dropout': 0.1,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-4,  # Adam optimizer learning rate
    }
    
    # Load data
    sentence_groups = load_data()
    
    # Normalize text
    print("Normalizing text...")
    normalized_groups = [normalize_urdu_text(group) for group in sentence_groups]
    
    # Train tokenizer
    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
    tokenizer.train(normalized_groups, verbose=True)
    
    # Split dataset: 80% train, 10% validation, 10% test
    print("Splitting dataset...")
    random.shuffle(normalized_groups)
    train_size = int(0.8 * len(normalized_groups))
    val_size = int(0.1 * len(normalized_groups))
    
    train_groups = normalized_groups[:train_size]
    val_groups = normalized_groups[train_size:train_size + val_size]
    test_groups = normalized_groups[train_size + val_size:]
    
    print(f"Train: {len(train_groups)}, Val: {len(val_groups)}, Test: {len(test_groups)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = UrduChatbotDataset(train_groups, tokenizer, config['max_len'])
    val_dataset = UrduChatbotDataset(val_groups, tokenizer, config['max_len'])
    test_dataset = UrduChatbotDataset(test_groups, tokenizer, config['max_len'])
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    vocab_size = len(tokenizer.token_to_id)
    
    # Create model
    print("Creating Transformer model...")
    model = Transformer(
        vocab_size=vocab_size,
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        max_len=config['max_len'],
        dropout=config['dropout'],
        pad_idx=train_dataset.pad_idx
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.9, 0.98))
    
    # Load checkpoint if resuming
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    if resume:
        try:
            checkpoint = torch.load('best_model_allinone.pt', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        except FileNotFoundError:
            print("❌ No checkpoint found! Starting fresh training.")
            resume = False
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("Starting fresh training.")
            resume = False
    
    print("\n🎯 Starting Training Loop")
    print("-" * 40)
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'tokenizer': tokenizer,
                'config': config,
                'vocab_size': vocab_size,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, 'best_model_allinone.pt')
            print("★ Best model saved!")
    
    print("\n✅ Training Complete!")
    
    # Final evaluation
    print("\n📊 Final Evaluation")
    print("-" * 30)
    
    # Load best model for evaluation
    checkpoint = torch.load('best_model_allinone.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, test_loader, device)
    print(f"Test Perplexity: {perplexity:.2f}")
    
    # Test generation
    print("\n🎭 Test Generation Examples")
    print("-" * 35)
    test_inputs = ["یہ ایک", "پاکستان میں", "اچھا"]
    
    for input_text in test_inputs:
        generated = generate_text(model, tokenizer, train_dataset, input_text, max_length=20, device=device)
        print(f"Input:  {input_text}")
        print(f"Output: {generated}")
        print()

def chat_mode():
    """Interactive chat mode"""
    print("💬 Urdu Chatbot - Interactive Mode")
    print("=" * 40)
    print("Type 'quit' to exit")
    print()
    
    # Load model
    try:
        checkpoint = torch.load('best_model_allinone.pt', map_location=device)
        config = checkpoint['config']
        tokenizer = checkpoint['tokenizer']
        
        # Create model
        model = Transformer(
            vocab_size=checkpoint['vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            max_len=config['max_len'],
            dropout=config['dropout'],
            pad_idx=0
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create dataset for special tokens
        dataset = UrduChatbotDataset([], tokenizer, config['max_len'])
        
        print("Model loaded successfully!")
        print()
        
    except FileNotFoundError:
        print("❌ Model file not found! Please train the model first.")
        return
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() == 'quit':
                print("Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            # Generate response
            response = generate_text(model, tokenizer, dataset, user_input, max_length=30, device=device)
            print(f"Bot: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")

def evaluate_model():
    """Evaluate the trained model"""
    print("📊 Model Evaluation")
    print("=" * 30)
    
    try:
        checkpoint = torch.load('best_model_allinone.pt', map_location=device)
        config = checkpoint['config']
        tokenizer = checkpoint['tokenizer']
        
        # Load test data
        sentence_groups = load_data()
        normalized_groups = [normalize_urdu_text(group) for group in sentence_groups]
        
        # Use last 10% as test set
        test_size = int(0.1 * len(normalized_groups))
        test_groups = normalized_groups[-test_size:]
        
        # Create test dataset
        test_dataset = UrduChatbotDataset(test_groups, tokenizer, config['max_len'])
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = Transformer(
            vocab_size=checkpoint['vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            d_ff=config['d_ff'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            max_len=config['max_len'],
            dropout=config['dropout'],
            pad_idx=0
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Calculate metrics
        print("Calculating metrics...")
        perplexity = calculate_perplexity(model, test_loader, device)
        
        print(f"📈 Evaluation Results:")
        print(f"Perplexity: {perplexity:.2f}")
        
        # Proper evaluation with test data pairs
        print("\n🔍 Detailed Evaluation Metrics:")
        print("-" * 35)
        
        # Evaluate on actual test data pairs
        bleu_scores = []
        rouge_scores = []
        chrf_scores = []
        
        print(f"\n🎭 Evaluating on Test Data:")
        print("-" * 30)
        
        # Sample a subset of test data for evaluation
        eval_samples = min(100, len(test_groups))  # Evaluate on 100 samples or all if less
        
        for i in range(eval_samples):
            # Get input and target from test data
            group = test_groups[i]
            words = group.split()
            
            # Split same way as training (2/5th input, 3/5th target)
            total_words = len(words)
            split_point = max(1, total_words * 2 // 5)
            
            input_text = ' '.join(words[:split_point])
            target_text = ' '.join(words[split_point:])
            
            # Generate response
            generated = generate_text(model, tokenizer, test_dataset, input_text, max_length=20, device=device)
            
            # Calculate metrics against actual target
            bleu = calculate_bleu_score(generated, target_text)
            rouge = calculate_rouge_l(generated, target_text)
            chrf = calculate_chrf(generated, target_text)
            
            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            chrf_scores.append(chrf)
            
            # Show first few examples
            if i < 5:
                print(f"Sample {i+1}:")
                print(f"Input:     {input_text}")
                print(f"Generated: {generated}")
                print(f"Target:    {target_text}")
                print(f"BLEU: {bleu:.3f} | ROUGE-L: {rouge:.3f} | chrF: {chrf:.3f}")
                print()
        
        # Average metrics
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        avg_rouge = sum(rouge_scores) / len(rouge_scores)
        avg_chrf = sum(chrf_scores) / len(chrf_scores)
        
        print(f"📊 Evaluation Results (on {eval_samples} samples):")
        print(f"BLEU Score:    {avg_bleu:.3f}")
        print(f"ROUGE-L Score: {avg_rouge:.3f}")
        print(f"chrF Score:    {avg_chrf:.3f}")
        print(f"Perplexity:    {perplexity:.2f}")
        
    except FileNotFoundError:
        print("❌ Model file not found! Please train the model first.")
    except Exception as e:
        print(f"Error during evaluation: {e}")

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Urdu Conversational Chatbot')
    parser.add_argument('--mode', choices=['train', 'chat', 'eval'], required=True,
                       help='Mode: train, chat, or eval')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from saved checkpoint')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_model(resume=args.resume)
    elif args.mode == 'chat':
        chat_mode()
    elif args.mode == 'eval':
        evaluate_model()

if _name_ == "_main_":
    main()