import streamlit as st
import torch
import torch.nn as nn
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

# Model Architecture (copied from your code)
def get_positional_encoding(max_seq_len, dm):
    pos = torch.arange(max_seq_len).unsqueeze(1)
    denom = 10000 ** (2 * torch.arange(0, dm//2) / dm)
    angles = pos / denom
    PE = torch.zeros(max_seq_len, dm)
    PE[:, 0::2] = torch.sin(angles)
    PE[:, 1::2] = torch.cos(angles)
    return PE

def create_padding_mask(seq, pad_id):
    return seq == pad_id

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask.bool()

def FullyConnected(embedding_dim, fully_connected_dim, dropout=0.3):
    return nn.Sequential(
        nn.Linear(embedding_dim, fully_connected_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fully_connected_dim, embedding_dim)
    )

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, 
                                        dropout=dropout_rate, batch_first=True)
        self.ffn = FullyConnected(embedding_dim, ffn_dim)
        self.layernorm1 = nn.LayerNorm(embedding_dim, eps=layernorm_eps)
        self.layernorm2 = nn.LayerNorm(embedding_dim, eps=layernorm_eps)
        self.dropout_attn = nn.Dropout(dropout_rate)
        self.dropout_ffn = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output, _ = self.mha(query=x, value=x, key=x, key_padding_mask=mask)
        attn_output = self.dropout_attn(attn_output)
        skip_x_attn = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(skip_x_attn)
        ffn_output = self.dropout_ffn(ffn_output)
        out = self.layernorm2(ffn_output + skip_x_attn)
        return out

class Encoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, num_heads, ffn_dim, input_vocab_size, 
                 max_seq_len, dropout_rate=0.1, padding_id=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim, padding_idx=padding_id)
        self.pos_encoding = get_positional_encoding(max_seq_len, embedding_dim)
        self.enc_layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, ffn_dim, dropout_rate) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.scale_dm = torch.sqrt(torch.tensor(self.embedding_dim))

    def forward(self, x, padding_mask):
        seq_len = x.shape[1]
        x = self.embedding(x) * self.scale_dm
        pos_enc = self.pos_encoding[:seq_len, :].to(x.device)
        x = self.dropout(x + pos_enc)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, padding_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, dropout_rate=0.1, layernorm_eps=1e-6):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = FullyConnected(embedding_dim, ffn_dim)
        self.layernorm1 = nn.LayerNorm(embedding_dim, eps=layernorm_eps)
        self.layernorm2 = nn.LayerNorm(embedding_dim, eps=layernorm_eps)
        self.layernorm3 = nn.LayerNorm(embedding_dim, eps=layernorm_eps)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, enc_out, look_ahead_mask, padding_mask):
        masked_att, _ = self.mha1(query=x, key=x, value=x, attn_mask=look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout1(masked_att))
        cross_att, _ = self.mha2(query=out1, key=enc_out, value=enc_out, key_padding_mask=padding_mask)
        out2 = self.layernorm2(out1 + self.dropout2(cross_att))
        ffn_out = self.ffn(out2)
        out3 = self.layernorm3(out2 + self.dropout3(ffn_out))
        return out3

class Decoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, num_heads, ffn_dim, target_vocab_size, 
                 max_seq_len, dropout_rate=0.1, padding_id=0):
        super().__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(target_vocab_size, embedding_dim, padding_idx=padding_id)
        self.pos_encoding = get_positional_encoding(max_seq_len, embedding_dim)
        self.dec_layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, ffn_dim, dropout_rate) 
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.scale_dm = torch.sqrt(torch.tensor(self.embedding_dim))

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        seq_len = x.shape[1]
        x = self.embedding(x) * self.scale_dm
        pos_enc = self.pos_encoding[:seq_len, :].to(x.device)
        x = self.dropout(x + pos_enc)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size=1024, max_seq_len=100, padding_idx=0):
        super().__init__()
        self.encoder = Encoder(num_layers=2, embedding_dim=256, num_heads=4, ffn_dim=1024, 
                              input_vocab_size=vocab_size, max_seq_len=max_seq_len, 
                              padding_id=padding_idx)
        self.decoder = Decoder(num_layers=2, embedding_dim=256, num_heads=4, ffn_dim=1024, 
                              target_vocab_size=vocab_size, max_seq_len=max_seq_len, 
                              padding_id=padding_idx)
        self.final_layer = nn.Linear(in_features=256, out_features=vocab_size)

    def forward(self, src, tgt, padding_mask, teacher_forcing=0):
        batch_len, tgt_len = tgt.shape
        dec_input = tgt[:, 0:1]
        enc_output = self.encoder(src, padding_mask)
        outputs = []
        
        for i in range(0, tgt_len):
            tgt_look_ahead_mask = create_look_ahead_mask(dec_input.size(1)).to(device)
            dec_out = self.decoder(dec_input, enc_output, tgt_look_ahead_mask, padding_mask)
            pred = self.final_layer(dec_out)
            outputs.append(pred[:, -1:, :])
            
            if i < tgt_len - 1:
                tf = torch.rand(batch_len, 1, device=device) < teacher_forcing
                pred_t = pred[:, -1:, :].argmax(dim=-1)
                ground_t = tgt[:, i+1:i+2]
                next_t = torch.where(tf, ground_t, pred_t)
                dec_input = torch.cat([dec_input, next_t], dim=1)
        
        output = torch.cat(outputs, dim=1)
        return output

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    # Load tokenizer
    tokenizer = sp.SentencePieceProcessor()
    tokenizer.Load('d:/Workspace/PYTHON/NLP/Project2/test-gpt-style-training/bpe_tokenizer.model')
    
    vocab_size = tokenizer.get_piece_size()
    pad_id = tokenizer.piece_to_id('<pad>')
    
    # Load model
    model = Transformer(vocab_size=vocab_size, padding_idx=pad_id).to(device)
    checkpoint = torch.load('d:/Workspace/PYTHON/NLP/Project2/test-gpt-style-training/model_state.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer, pad_id

# Prediction function
def predict(model, tokenizer, pad_id, input_text, max_len=50):
    model.eval()
    with torch.no_grad():
        # Encode input
        input_ids = tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        
        # Create padding mask
        padding_mask = create_padding_mask(input_tensor, pad_id).to(device)
        
        # Start with BOS token
        decoder_input = torch.tensor([[tokenizer.piece_to_id('<s>')]], dtype=torch.long).to(device)
        
        # Generate output
        for _ in range(max_len):
            tgt_look_ahead_mask = create_look_ahead_mask(decoder_input.size(1)).to(device)
            enc_output = model.encoder(input_tensor, padding_mask)
            dec_out = model.decoder(decoder_input, enc_output, tgt_look_ahead_mask, padding_mask)
            pred = model.final_layer(dec_out)
            
            next_token = pred[:, -1:, :].argmax(dim=-1)
            
            # Check for EOS token
            if next_token.item() == tokenizer.piece_to_id('</s>'):
                break
                
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # Decode output
        output_ids = decoder_input[0].cpu().tolist()
        output_text = tokenizer.decode(output_ids)
        
        return output_text

# Streamlit UI
st.title("🔤 Urdu Span Corruption Transformer")
st.markdown("### Fill in the masked spans in Urdu text")

# Load model
try:
    model, tokenizer, pad_id = load_model_and_tokenizer()
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
    placeholder="Enter Urdu text ...",
)

max_length = st.slider("Maximum output length", 1, 20, 5)

# Prediction
if st.button("Generate", type="primary"):
    if input_text.strip():
        with st.spinner("Generating..."):
            try:
                output = predict(model, tokenizer, pad_id, input_text, max_len=max_length)
                
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