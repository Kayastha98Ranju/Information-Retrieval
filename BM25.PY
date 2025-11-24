import numpy as np
from collections import Counter
import math
import os

# Load dataset
with open("Dataset.txt", "r", encoding="utf-8") as f:
    documents = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(documents)} documents\n")
query = "fox dog brown"

def preprocess(text):
    return text.lower().replace('.', '').replace(',', '').split()

query_tokens = preprocess(query)
corpus = [preprocess(doc) for doc in documents]
lengths = [len(doc) for doc in corpus]
N = len(documents)
avgdl = sum(lengths) / N if N > 0 else 1

# TF and DF
tf_docs = [Counter(doc) for doc in corpus]
df = Counter()
for tf in tf_docs:
    df.update(tf.keys())
idf = {term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1) for term in df}

# Collection model
total_terms = sum(lengths)
coll_freq = Counter()
for tf in tf_docs:
    coll_freq.update(tf)
p_coll = {t: coll_freq[t]/total_terms for t in coll_freq}

# BM25
def bm25(i):
    score = 0
    tf = tf_docs[i]
    doc_len = lengths[i]
    for term in query_tokens:
        if term not in tf: continue
        f = tf[term]
        score += idf.get(term, 0) * (f * 2.2) / (f + 1.2 * (1 - 0.75 + 0.75 * doc_len / avgdl))
    return score

# Jelinek-Mercer
def lm(i, lmbda=0.5):
    score = 0
    tf = tf_docs[i]
    doc_len = lengths[i] or 1
    for term in query_tokens:
        p_doc = tf.get(term, 0) / doc_len
        p_c = p_coll.get(term, 0)
        p = (1 - lmbda) * p_doc + lmbda * p_c
        score += math.log(p + 1e-12)
    return score

# Results
print("="*70)
print(f"QUERY: {query.upper()}")
print("="*70)
print("BM25 RESULTS".center(70))
for rank, (idx, sc) in enumerate(sorted(enumerate([bm25(i) for i in range(N)]), key=lambda x: x[1], reverse=True), 1):
    print(f"Rank {rank}: Score {sc:.4f} → {documents[idx][:100]}")

print("\nJELINEK-MERCER RESULTS (λ=0.5)".center(70))
for rank, (idx, sc) in enumerate(sorted(enumerate([lm(i) for i in range(N)]), key=lambda x: x[1], reverse=True), 1):
    print(f"Rank {rank}: Score {sc:.4f} → {documents[idx][:100]}")