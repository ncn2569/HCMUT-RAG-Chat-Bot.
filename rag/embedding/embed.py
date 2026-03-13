from dotenv import load_dotenv
import os
import json
import numpy as np  
load_dotenv('config/.env')
os.environ["HF_HOME"] = os.getenv("HF_HOME")
token= os.getenv('HF_TOKEN')

# from sentence_transformers import SentenceTransformer
# def load_embedder():
#     model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",token=token,trust_remote_code=True)
#     return model
from google import genai

client = genai.Client(api_key=os.getenv('API_KEY'))

def load_embedder():
    return client  # genai client

# def embedding():# nhớ bỏ
#     model=load_embedder()
#     chunks = []
#     with open('data/processed/data-cleaned.jsonl', 'r', encoding='utf-8') as f:
#         for line in f:
#             chunks.append(json.loads(line.strip()))

#     texts = [chunk['text'] for chunk in chunks]

#     embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
#     np.save('data/vectors/vectors.npy', embeddings)
    
#     with open('data/vectors/vectors.json', 'w', encoding='utf-8') as f:
#         for chunk in chunks:
#             f.write(json.dumps({"text": chunk['text']}, ensure_ascii=False) + '\n')
    
#     print(f"✅ Saved: vectors.npy ({embeddings.shape})")
#     print(f"✅ Saved: vectors.json ({len(chunks)} items)")
def embedding():
    chunks = []
    with open('data/processed/data-cleaned.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line.strip()))

    texts = [chunk['text'] for chunk in chunks]

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = client.models.embed_content(
            model=os.getenv('model_embedding_name'),
            contents=batch,
            config={"task_type": "RETRIEVAL_DOCUMENT"}  # ✅ documents
        )
        for emb in result.embeddings:
            all_embeddings.append(emb.values)
        print(f"✅ Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    embeddings = np.array(all_embeddings)
    np.save('data/vectors/vectors1.npy', embeddings)

    with open('data/vectors/vectors1.jsonl', 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps({"text": chunk['text']}, ensure_ascii=False) + '\n')

    print(f"\n✅ Saved: vectors1.npy {embeddings.shape}")
    print(f"✅ Saved: vectors1.jsonl ({len(chunks)} items)")
