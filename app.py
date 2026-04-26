import http.server
import socketserver
import json
import os
import torch
import pandas as pd
from config import Config
from data_loader import SimpleTokenizer

PORT = 8000
static_dir = os.path.join(os.path.dirname(__file__), 'static')

# Global state for model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config()
tokenizer = None
model = None

def init_system():
    global tokenizer, model
    print("Initializing system... (please wait)")
    
    # 1. Load Tokenizer
    tokenizer = SimpleTokenizer(config)
    if not os.path.exists(tokenizer.vocab_file):
        print("Vocabulary not found. Building from data.csv...")
        if os.path.exists('data.csv'):
            df = pd.read_csv('data.csv').dropna(subset=['question', 'answer'])
            tokenizer.build_vocab(df)
        else:
            print("Error: data.csv not found to build vocabulary.")
    else:
        tokenizer.build_vocab(None)
        
    import numpy as np
    # Using dummy embeddings for memory efficiency if model weights are fully saved,
    # or you can load the actual ones if needed. The checkpoint usually overrides it.
    dummy_embeddings = np.zeros((config.vocab_size, config.hidden_dim), dtype=np.float32)
    
    try:
        from transformer import TransformerQA
    except ImportError:
        print("TransformerQA unavailable.")
        return False
        
    # 2. Init Model
    model = TransformerQA(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        max_seq_len=getattr(config, 'rag_max_seq_len', config.max_seq_len),
        embeddings=dummy_embeddings
    ).to(device)
    
    # 3. Load Weights
    if os.path.exists(config.model_save_path):
        state = torch.load(config.model_save_path, map_location=device)
        state.pop("pos_encoding.pe", None)
        state.pop("decoder.pos_encoding.pe", None)
        own_state  = model.state_dict()
        compatible = {k: v for k, v in state.items() if k in own_state and v.shape == own_state[k].shape}
        model.load_state_dict(compatible, strict=False)
        print(f"Loaded trained weights from '{config.model_save_path}'!")
    else:
        print(f"WARNING: '{config.model_save_path}' not found! Model is untrained.")
        
    model.eval()
    return True


class ChatHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=static_dir, **kwargs)

    def do_POST(self):
        if self.path == '/api/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                question = data.get('question', '').strip()
                
                if not question:
                    self.send_json_response({"error": "Empty question"})
                    return
                
                if model is None or tokenizer is None:
                    self.send_json_response({"error": "System not initialized completely"})
                    return
                
                # Tokenize
                src_ids = torch.tensor([tokenizer.encode(question)], device=device)
                
                # Generate
                with torch.no_grad():
                    result_ids = model.generate(src_ids, tokenizer, max_length=20, beam_width=1)
                    if isinstance(result_ids, tuple):
                        result_ids = result_ids[0]
                    answer = tokenizer.decode(result_ids)
                
                self.send_json_response({"answer": answer})
                
            except Exception as e:
                self.send_json_response({"error": str(e)})
        else:
            self.send_error(404, "Endpoint not found")

    def send_json_response(self, payload):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode('utf-8'))


if __name__ == "__main__":
    success = init_system()
    if not success:
        print("Failed to start server.")
        exit(1)
        
    print(f"\n======================================")
    print(f" Chatbot UI is running!")
    print(f" Open http://localhost:{PORT} in your browser")
    print(f"======================================")
    
    with socketserver.TCPServer(("", PORT), ChatHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
