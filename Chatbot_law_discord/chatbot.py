import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from underthesea import word_tokenize
import faiss
import torch.nn as nn

with open(r"data_labeled.json", "r", encoding="utf-8") as file:
    data = json.load(file)
    
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")

#vector embeddings from law documents
def get_phobert_embedding(text):
    text = word_tokenize(text, format="text")  # Word separation
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()  # Take the average vector

# Convert all law documents into embeddings
law_embeddings = np.array([get_phobert_embedding(item["text"]) for item in data], dtype=np.float32)

# FAISS index
dimension = law_embeddings.shape[1]  # Dimension of vector
index = faiss.IndexFlatL2(dimension)  
index.add(law_embeddings)  #Add embeddings to FAISS

conversation_history = [] 
MAX_HISTORY = 3

class ContextLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ContextLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context_embedding = self.fc(lstm_out[:, -1, :])
        return context_embedding
    
input_dim = law_embeddings.shape[1]
hidden_dim = 256
num_layers = 1

context_model = ContextLSTM(input_dim, hidden_dim, num_layers)

def get_law_answer(user_input, alpha=0.2):
    global conversation_history

    # Convert current question to PhoBERT vector
    user_embedding = get_phobert_embedding(user_input)

    # Add to conversation history
    conversation_history.append(user_embedding)
    if len(conversation_history) > MAX_HISTORY:
        conversation_history.pop(0)

    # Convert conversation history into tensor to feed into LSTM
    context_tensor = torch.tensor([conversation_history], dtype=torch.float32)

    # Get contextual embedding from LSTM
    with torch.no_grad():
        context_embedding = context_model(context_tensor).numpy()

    # Combined with PhoBERT alpha-weighted embedding
    final_embedding = alpha * context_embedding + (1 - alpha) * user_embedding

    # Search in FAISS with contextual vectors
    D, I = index.search(final_embedding, 1)
    best_match_idx = I[0][0]

    if D[0][0] < 55.0:
        return f"{data[best_match_idx]['label']}:\n{data[best_match_idx]['text']}"
    else:
        return "Xin lỗi, tôi chưa tìm thấy điều luật phù hợp."