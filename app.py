import streamlit as st
import torch
import torch.nn as nn
import sentencepiece as sp
import math
from typing import List, Tuple

# Set page config for RTL support
st.set_page_config(
    page_title="اردو چیٹ بوٹ | Urdu Chatbot",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for Urdu RTL support and styling
st.markdown("""
    <style>
    .urdu-text {
        direction: rtl;
        text-align: right;
        font-family: 'Jameel Noori Nastaleeq', 'Noto Nastaliq Urdu', Arial;
        font-size: 18px;
        line-height: 2;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
    }
    .bot-message {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        direction: rtl;
        text-align: right;
    }
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== MODEL ARCHITECTURE ====================

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

def FullyConnected(embedding_dim, fully_connected_dim, dropout=0.1):
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
    def __init__(self, vocab_size=512, max_seq_len=100, padding_idx=0):
        super().__init__()
        self.encoder = Encoder(num_layers=2, embedding_dim=256, num_heads=4, ffn_dim=1024,
                               input_vocab_size=vocab_size, max_seq_len=max_seq_len, padding_id=padding_idx)
        self.decoder = Decoder(num_layers=2, embedding_dim=256, num_heads=4, ffn_dim=1024,
                               target_vocab_size=vocab_size, max_seq_len=max_seq_len, padding_id=padding_idx)
        self.final_layer = nn.Linear(in_features=256, out_features=vocab_size)

    def forward(self, src, tgt, padding_mask, teacher_forcing=0):
        batch_len, tgt_len = tgt.shape
        dec_input = tgt[:, 0:1]
        enc_output = self.encoder(src, padding_mask)
        outputs = []
        
        for i in range(0, tgt_len):
            tgt_look_ahead_mask = create_look_ahead_mask(dec_input.size(1)).to(src.device)
            dec_out = self.decoder(dec_input, enc_output, tgt_look_ahead_mask, padding_mask)
            pred = self.final_layer(dec_out)
            outputs.append(pred[:, -1:, :])
            
            if i < tgt_len - 1:
                tf = torch.rand(batch_len, 1, device=src.device) < teacher_forcing
                pred_t = pred[:, -1:, :].argmax(dim=-1)
                ground_t = tgt[:, i+1:i+2]
                next_t = torch.where(tf, ground_t, pred_t)
                dec_input = torch.cat([dec_input, next_t], dim=1)
        
        output = torch.cat(outputs, dim=1)
        return output

# ==================== GENERATION FUNCTIONS ====================

def greedy_decode(model, src, src_mask, max_len, sos_id, eos_id, pad_id, device):
    """Greedy decoding"""
    model.eval()
    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)
        dec_input = torch.tensor([[sos_id]], device=device)
        
        for _ in range(max_len):
            tgt_mask = create_look_ahead_mask(dec_input.size(1)).to(device)
            dec_out = model.decoder(dec_input, enc_output, tgt_mask, src_mask)
            pred = model.final_layer(dec_out)
            next_token = pred[:, -1, :].argmax(dim=-1, keepdim=True)
            
            if next_token.item() == eos_id:
                break
            
            dec_input = torch.cat([dec_input, next_token], dim=1)
    
    return dec_input[0].cpu().tolist()

def beam_search_decode(model, src, src_mask, max_len, sos_id, eos_id, pad_id, beam_width, device):
    """Beam search decoding"""
    model.eval()
    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)
        
        # Initialize beam with SOS token
        beams = [([sos_id], 0.0)]  # (sequence, score)
        
        for _ in range(max_len):
            candidates = []
            
            for seq, score in beams:
                if seq[-1] == eos_id:
                    candidates.append((seq, score))
                    continue
                
                dec_input = torch.tensor([seq], device=device)
                tgt_mask = create_look_ahead_mask(dec_input.size(1)).to(device)
                dec_out = model.decoder(dec_input, enc_output, tgt_mask, src_mask)
                pred = model.final_layer(dec_out)
                log_probs = torch.log_softmax(pred[:, -1, :], dim=-1)
                
                # Get top k tokens
                top_log_probs, top_indices = torch.topk(log_probs[0], beam_width)
                
                for log_prob, idx in zip(top_log_probs, top_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + log_prob.item()
                    candidates.append((new_seq, new_score))
            
            # Select top beam_width candidates
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Check if all beams ended
            if all(seq[-1] == eos_id for seq, _ in beams):
                break
        
        # Return best sequence
        best_seq = beams[0][0]
        return best_seq

# ==================== INITIALIZATION ====================

@st.cache_resource
def load_model_and_tokenizer(tokenizer_path, model_path):
    """Load tokenizer and model"""
    # Load tokenizer
    tokenizer = sp.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)
    
    # Get special token IDs
    vocab_size = tokenizer.get_piece_size()
    pad_id = tokenizer.piece_to_id('<pad>')
    sos_id = tokenizer.piece_to_id('<s>')
    eos_id = tokenizer.piece_to_id('</s>')
    mask_id = tokenizer.piece_to_id('<mask>')
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(vocab_size=vocab_size, max_seq_len=100, padding_idx=pad_id).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return tokenizer, model, device, pad_id, sos_id, eos_id, mask_id

# ==================== STREAMLIT APP ====================

def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>💬 اردو چیٹ بوٹ | Urdu Conversational Chatbot</h1>", 
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Powered by Transformer with Span Corruption Training</p>", 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    tokenizer_path = st.sidebar.text_input(
        "Tokenizer Path",
        value="unigram_tokenizer.model"
    )   
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="span_15_3.pth"
    )
    
    # Load model
    if st.sidebar.button("🔄 Load Model"):
        with st.spinner("Loading model and tokenizer..."):
            try:
                st.session_state.tokenizer, st.session_state.model, st.session_state.device, \
                st.session_state.pad_id, st.session_state.sos_id, st.session_state.eos_id, \
                st.session_state.mask_id = load_model_and_tokenizer(tokenizer_path, model_path)
                st.sidebar.success("✅ Model loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading model: {str(e)}")
                return
    
    # Check if model is loaded
    if 'model' not in st.session_state:
        st.info("👆 Please load the model from the sidebar to begin.")
        return
    
    st.sidebar.markdown("---")
    
    # Mode selection
    mode = st.sidebar.selectbox(
        "🎯 Interaction Mode",
        [
            "Fill-in Corrupted Text",
            "Conversational Response",
            "Mask-based Generation"
        ]
    )
    
    # Decoding strategy
    decoding = st.sidebar.radio(
        "🔍 Decoding Strategy",
        ["Greedy", "Beam Search"]
    )
    
    beam_width = 3
    if decoding == "Beam Search":
        beam_width = st.sidebar.slider("Beam Width", 3, 5, 3)
    
    max_length = st.sidebar.slider("Max Generation Length", 20, 100, 50)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.history = []
    
    # Initialize history
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💭 Input")
        user_input = st.text_area(
            "Enter Urdu text:",
            height=100,
            placeholder="یہاں اردو متن لکھیں...",
            key="input_box"
        )
        
        if st.button("🚀 Generate", type="primary"):
            if not user_input.strip():
                st.warning("Please enter some text!")
                return
            
            with st.spinner("Generating response..."):
                # Prepare input based on mode
                if mode == "Fill-in Corrupted Text":
                    # User provides corrupted text directly
                    input_text = user_input
                elif mode == "Conversational Response":
                    # Treat input as normal text
                    input_text = user_input
                else:  # Mask-based Generation
                    # Add mask token at the end
                    input_text = user_input + " <mask>"
                
                # Tokenize
                input_ids = st.session_state.tokenizer.encode(input_text)
                src = torch.tensor([input_ids], device=st.session_state.device)
                src_mask = create_padding_mask(src, st.session_state.pad_id).to(st.session_state.device)
                
                # Generate
                if decoding == "Greedy":
                    output_ids = greedy_decode(
                        st.session_state.model, src, src_mask, max_length,
                        st.session_state.sos_id, st.session_state.eos_id,
                        st.session_state.pad_id, st.session_state.device
                    )
                else:
                    output_ids = beam_search_decode(
                        st.session_state.model, src, src_mask, max_length,
                        st.session_state.sos_id, st.session_state.eos_id,
                        st.session_state.pad_id, beam_width, st.session_state.device
                    )
                
                # Decode
                output_text = st.session_state.tokenizer.decode(output_ids)
                output_text = output_text.replace("<s>", "").replace("</s>", "").replace("<mask>", "").strip()
                
                # Add to history
                st.session_state.history.append({
                    'input': user_input,
                    'output': output_text,
                    'mode': mode,
                    'decoding': decoding
                })
    
    with col2:
        st.subheader("ℹ️ Mode Info")
        if mode == "Fill-in Corrupted Text":
            st.info("🔧 The model will fill in masked portions of your text. Use <mask> tokens in your input.")
        elif mode == "Conversational Response":
            st.info("💬 The model generates a response based on your input text.")
        else:
            st.info("✨ A <mask> token is automatically added to your input for generation.")
    
    # Display latest result
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📤 Latest Output")
        latest = st.session_state.history[-1]
        st.markdown(f"<div class='bot-message urdu-text'>{latest['output']}</div>", 
                   unsafe_allow_html=True)
    
    # Conversation history
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Conversation History")
        
        for i, entry in enumerate(reversed(st.session_state.history[-5:])):
            with st.expander(f"Exchange {len(st.session_state.history) - i} - {entry['mode']} ({entry['decoding']})"):
                st.markdown("<p><strong>Input:</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<div class='user-message urdu-text'>{entry['input']}</div>", 
                           unsafe_allow_html=True)
                st.markdown("<p><strong>Output:</strong></p>", unsafe_allow_html=True)
                st.markdown(f"<div class='bot-message urdu-text'>{entry['output']}</div>", 
                           unsafe_allow_html=True)

if __name__ == "__main__":
    main()