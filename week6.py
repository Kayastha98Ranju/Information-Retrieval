import re
import math
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List

def parse_documents(file_path: str) -> Dict[int, str]:
    docs = {}
    current_id = None
    current_text = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('.I '):
                if current_id:
                    docs[current_id] = ' '.join(current_text)
                current_id = int(line[3:])
                current_text = []
            elif line.startswith('.T ') or line.startswith('.W '):
                continue
            elif line:
                current_text.append(line)
        if current_id:
            docs[current_id] = ' '.join(current_text)
    return docs

def parse_queries(file_path: str) -> Dict[int, str]:
    queries = {}
    current_id = None
    current_text = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('.I '):
                if current_id:
                    queries[current_id] = ' '.join(current_text)
                current_id = int(line[3:])
                current_text = []
            elif line.startswith('.W '):
                continue
            elif line:
                current_text.append(line)
        if current_id:
            queries[current_id] = ' '.join(current_text)
    return queries

def parse_relevance(file_path: str) -> Dict[int, set]:
    rel = defaultdict(set)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                qid = int(parts[0])
                did = int(parts[1])
                rel[qid].add(did)
    return rel


stopwords = set(['a', 'an', 'the', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])  # Basic stopwords

def preprocess(text: str) -> List[str]:
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [token for token in tokens if token not in stopwords and len(token) > 1]

def build_index(docs: Dict[int, str]) -> tuple:
    inverted_index = defaultdict(list)
    doc_terms = {}
    vocab = set()
    for doc_id, text in docs.items():
        tokens = preprocess(text)
        doc_terms[doc_id] = Counter(tokens)
        for term in set(tokens):
            inverted_index[term].append(doc_id)
            vocab.add(term)
    N = len(docs)
    df = {term: len(docs_list) for term, docs_list in inverted_index.items()}
    return doc_terms, inverted_index, vocab, df, N

def tfidf_vector(doc_terms: Dict, query_tokens: List[str], vocab: set, df: Dict, N: int, doc_id: int = None) -> np.ndarray:
    vec = np.zeros(len(vocab))
    term_list = list(vocab)
    if doc_id:
        tf = doc_terms[doc_id]
        for i, term in enumerate(term_list):
            if term in tf:
                tf_val = tf[term]
                idf = math.log(N / (df.get(term, 1)))
                vec[i] = tf_val * idf
    else:  # Query vector
        q_counter = Counter(query_tokens)
        for i, term in enumerate(term_list):
            if term in q_counter:
                tf_val = q_counter[term]
                idf = math.log(N / (df.get(term, 1)))
                vec[i] = tf_val * idf
    return vec

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0


def rank_documents(query_tokens: List[str], doc_terms: Dict, vocab: set, df: Dict, N: int) -> List[tuple]:
    q_vec = tfidf_vector(doc_terms, query_tokens, vocab, df, N)
    scores = []
    for doc_id in doc_terms:
        d_vec = tfidf_vector(doc_terms, [], vocab, df, N, doc_id)
        sim = cosine_similarity(q_vec, d_vec)
        if sim > 0:
            scores.append((doc_id, sim))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def average_precision(ranked_docs: List[int], relevant: set) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total_rel = len(relevant)
    precisions = []
    for rank, doc_id in enumerate(ranked_docs, 1):
        if doc_id in relevant:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / total_rel

def compute_map(queries: Dict, rels: Dict, doc_terms: Dict, vocab: set, df: Dict, N: int) -> float:
    aps = []
    for qid, qtext in queries.items():
        q_tokens = preprocess(qtext)
        ranked_list = [doc for doc, _ in rank_documents(q_tokens, doc_terms, vocab, df, N)]
        ap = average_precision(ranked_list, rels[qid])
        aps.append(ap)
    return sum(aps) / len(queries)


docs = parse_documents('CISI.ALL')
queries = parse_queries('CISI.QRY')
rels = parse_relevance('CISI.REL')

doc_terms, inv_index, vocab, df, N = build_index(docs)

print("Starting retrieval for all 112 queries...\n")
print("="*80)

total_ap = 0.0
num_queries = len(queries)

for qid in sorted(queries.keys()):
    qtext = queries[qid]
    q_tokens = preprocess(qtext)
  
    ranked = rank_documents(q_tokens, doc_terms, vocab, df, N)

    relevant_docs = rels[qid]
    ap = average_precision([doc_id for doc_id, _ in ranked], relevant_docs)
    total_ap += ap

    print(f"QUERY {qid:3d} | AP = {ap:.4f}")
    print(f"Text: {qtext.strip()}")
    print("Top 5 documents:")
    for rank, (doc_id, score) in enumerate(ranked[:5], 1):
        print(f"   {rank:2d}. Doc {doc_id:4d} → Score: {score:.4f}")
    print("-" * 80)

map_score = total_ap / num_queries if num_queries > 0 else 0
print(f"\nFINAL RESULT")
print(f"Mean Average Precision (MAP) @ all queries: {map_score:.4f}")
print(f"Total queries processed: {num_queries}")
print("="*80)