import re
import math
import numpy as np
from collections import defaultdict, Counter

def parse_documents(file_path):
    docs = {}
    current_id = None
    current_text = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('.I '):
                if current_id:
                    docs[current_id] = ' '.join(current_text)
                current_id = int(line.split()[1])
                current_text = []
            elif line.startswith('.T') or line.startswith('.W') or line.startswith('.A') or line.startswith('.X'):
                continue
            elif line:
                current_text.append(line)
        if current_id:
            docs[current_id] = ' '.join(current_text)
    return docs

def parse_queries(file_path):
    queries = {}
    current_id = None
    current_text = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('.I '):
                if current_id:
                    queries[current_id] = ' '.join(current_text)
                current_id = int(line.split()[1])
                current_text = []
            elif line.startswith('.W'):
                continue
            elif line:
                current_text.append(line)
        if current_id:
            queries[current_id] = ' '.join(current_text)
    return queries

def parse_relevance(file_path):
    rel = defaultdict(set)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                qid = int(parts[0])
                did = int(parts[1])
                rel[qid].add(did)
    return rel

stopwords = set(['a','an','the','is','are','was','were','and','or','but','in','on','at','to','for','of','with','by','from','as','i','we','you','it','this','that','be','have','had','has','not','which','what','when','where','who','will','s','t','can','could','should','would','may','might','must','do','does','did'])

def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    return [t for t in tokens if t not in stopwords and len(t) > 1]

def build_index(docs):
    doc_terms = {}
    inverted_index = defaultdict(list)
    vocab = set()
    for doc_id, text in docs.items():
        tokens = preprocess(text)
        doc_terms[doc_id] = Counter(tokens)
        for term in set(tokens):
            inverted_index[term].append(doc_id)
            vocab.add(term)
    N = len(docs)
    df = {term: len(postings) for term, postings in inverted_index.items()}
    return doc_terms, inverted_index, vocab, df, N

def tfidf_vector(doc_terms, query_tokens, vocab, df, N, doc_id=None):
    vec = np.zeros(len(vocab))
    term_list = list(vocab)
    if doc_id:
        tf = doc_terms[doc_id]
        for i, term in enumerate(term_list):
            if term in tf:
                tf_val = tf[term]
                idf = math.log(N / (df.get(term, 1)))
                vec[i] = tf_val * idf
    else:
        q_counter = Counter(query_tokens)
        for i, term in enumerate(term_list):
            if term in q_counter:
                tf_val = q_counter[term]
                idf = math.log(N / (df.get(term, 1)))
                vec[i] = tf_val * idf
    return vec

def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

def rank_documents(query_tokens, doc_terms, vocab, df, N):
    q_vec = tfidf_vector(doc_terms, query_tokens, vocab, df, N)
    scores = []
    for doc_id in doc_terms:
        d_vec = tfidf_vector(doc_terms, [], vocab, df, N, doc_id)
        sim = cosine_similarity(q_vec, d_vec)
        if sim > 0:
            scores.append((doc_id, sim))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def average_precision(ranked_docs, relevant):
    if not relevant:
        return 0.0
    hits = 0
    precisions = []
    for rank, doc_id in enumerate(ranked_docs, 1):
        if doc_id in relevant:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / len(relevant) if relevant else 0.0

docs = parse_documents('CISI.ALL')
queries = parse_queries('CISI.QRY')
rels = parse_relevance('CISI.REL')

doc_terms, inv_index, vocab, df, N = build_index(docs)

print("Starting retrieval for all 112 queries...\n")
print("="*90)

total_ap = 0.0
for qid in sorted(queries.keys()):
    qtext = queries[qid]
    q_tokens = preprocess(qtext)
    ranked = rank_documents(q_tokens, doc_terms, vocab, df, N)
    ranked_ids = [doc_id for doc_id, _ in ranked]
    ap = average_precision(ranked_ids, rels[qid])
    total_ap += ap
    
    print(f"QUERY {qid:3d} | AP = {ap:.4f}")
    print(f"Text: {qtext.strip()}")
    print("Top 5 documents:")
    for rank, (doc_id, score) in enumerate(ranked[:5], 1):
        print(f"   {rank:2d}. Doc {doc_id:4d} → Score: {score:.4f}")
    print("-" * 90)

map_score = total_ap / len(queries)
print(f"\nFINAL RESULT")
print(f"Mean Average Precision (MAP) over 112 queries: {map_score:.4f}")
print(f"Total queries processed: {len(queries)}")
print("="*90)