import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as sp
import math

# Set page config
st.set_page_config(
    page_title="Urdu Span Corruption Model",
    page_icon="🔤",
    layout="centered"
)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model Architecture
class PositionalEncoding(nn.Module):
    """Positional encoding using sine and cosine"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
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
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
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
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
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
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
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
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
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
    def __init__(self, vocab_size, d_model=256, num_heads=2, d_ff=1024,
                 num_encoder_layers=2, num_decoder_layers=2, max_len=512,
                 dropout=0.1, pad_idx=0):
        super().__init__()
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

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    # Load tokenizer
    tokenizer = sp.SentencePieceProcessor()
    tokenizer.Load('bpe_tokenizer.model')
    
    vocab_size = tokenizer.get_piece_size()
    pad_id = tokenizer.piece_to_id('<pad>')
    sos_id = tokenizer.piece_to_id('<s>')
    eos_id = tokenizer.piece_to_id('</s>')
    
    # Load model
    model = Transformer(
        vocab_size=vocab_size,
        d_model=512,
        num_heads=2,
        d_ff=1024,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_len=50,
        dropout=0.1,
        pad_idx=pad_id
    ).to(device)
    
    checkpoint = torch.load('model_state.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, pad_id, sos_id, eos_id

# Generation function with temperature-based sampling
def generate_text(model, tokenizer, pad_id, sos_id, eos_id, input_text, max_length=30, temperature=0.8):
    """Generate text continuation using temperature-based sampling"""
    model.eval()
    
    with torch.no_grad():
        # Encode input
        token_ids = tokenizer.encode(input_text)
        encoder_ids = token_ids[:50] + [pad_id] * (50 - len(token_ids))
        encoder_input = torch.tensor([encoder_ids], dtype=torch.long).to(device)
        decoder_input = torch.tensor([[sos_id]], dtype=torch.long).to(device)
        
        generated_tokens = []
        for _ in range(max_length):
            output = model(encoder_input, decoder_input)
            next_token_logits = output[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            if next_token.item() == eos_id:
                break
            
            generated_tokens.append(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        
        # Decode output
        output_tokens = []
        for idx in generated_tokens:
            if idx not in [pad_id, sos_id, eos_id]:
                token = tokenizer.id_to_piece(idx)
                output_tokens.append(token)
        
        text = ''.join(output_tokens).replace('▁', ' ')
        return text.strip()

# Streamlit UI
st.title("🔤 Urdu Span Corruption Transformer")
st.markdown("### Fill in the masked spans in Urdu text")

# Load model
try:
    model, tokenizer, pad_id, sos_id, eos_id = load_model_and_tokenizer()
    st.success(f"✓ Model loaded successfully! (Device: {device})")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Input section
st.markdown("---")
st.markdown("#### Input Text")

input_text = st.text_area(
    "Corrupted Text",
    height=100,
    placeholder="Enter Urdu text with masked spans...",
)

col1, col2 = st.columns(2)
with col1:
    max_length = st.slider("Maximum output length", 1, 50, 30)
with col2:
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)

st.caption("Higher temperature = more random/creative output")

# Prediction
if st.button("Generate", type="primary"):
    if input_text.strip():
        with st.spinner("Generating..."):
            try:
                output = generate_text(
                    model, tokenizer, pad_id, sos_id, eos_id, 
                    input_text, max_length=max_length, temperature=temperature
                )
                
                st.markdown("---")
                st.markdown("#### Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Input:**")
                    st.info(input_text)
                
                with col2:
                    st.markdown("**Output:**")
                    st.success(output)
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please enter some text first!")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Urdu Span Corruption Model | Built with PyTorch & Streamlit</div>",
    unsafe_allow_html=True
)