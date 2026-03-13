import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# import json
# _data = None
# def _load_data():
#     global _data
#     if _data is None:
#         with open('data/vectors/vectors1.jsonl', 'r', encoding='utf-8') as f:
#             _data = [json.loads(line) for line in f]
#     return _data

def dense_search(query: str, embedder, embeddings: np.ndarray, top_k: int = 20):
    """
    Dense search đơn giản - chỉ 1 query, trả về [(idx, rank), ...]
    """
    # Encode query
    # query_vec = embedder.encode(query, convert_to_numpy=True).reshape(1, -1)
    
    # # Tính similarity
    # scores = cosine_similarity(query_vec, embeddings)[0]
    
    # # Lấy top_k
    # top_indices = scores.argsort()[-top_k:][::-1]
    
    # # Format: [(idx, rank), ...]
    # results = [(int(idx), i + 1) for i, idx in enumerate(top_indices)]
    # data = _load_data()

    result = embedder.models.embed_content(
        model=os.getenv('model_embedding_name'),
        contents=query,
        config={"task_type": "QUESTION_ANSWERING"}
    )
    vector = np.array(result.embeddings[0].values).reshape(1, -1)
    scores_original = cosine_similarity(vector, embeddings)[0]
    top_orig = scores_original.argsort()[-top_k:][::-1]

    res_orig = [(int(i), r + 1) for r, i in enumerate(top_orig)]
    
    # data = _load_data()
    # print(f"\n{'='*60}")
    # print(f"[DENSE SEARCH] Query: '{query}'")
    # print('='*60)
    # print(f"\n--- Top 5 Retrieved ---")
    # for i, (idx, rank) in enumerate(res_orig[:20], 1):  
    #     text = data[idx]['text']  
    #     q_part = text.split('A: ')[0].replace('Q: ', '').strip()
    #     a_part = text.split('A: ')[1].strip() if 'A: ' in text else ""
    #     score = scores_original[idx]  
    #     print(f"\n{i}. [Doc {idx} | Score: {score:.4f}]")
    #     print(f"   Q: {q_part}")
    #     print(f"   A: {a_part[:150]}...")
    # print(f"\n{'='*60}")
    
    #return results
    return res_orig