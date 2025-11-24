import os
import re
from collections import Counter
import math

def preprocess(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())

docs = {}
lengths = {}
folder = "Trump Speechs"

if not os.path.exists(folder):
    print("ERROR: Folder 'Trump Speeches' not found!")
    exit()

for file in os.listdir(folder):
    path = os.path.join(folder, file)
    if os.path.isfile(path) and file.lower().endswith('.txt'):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            tokens = preprocess(f.read())
            docs[file] = tokens
            lengths[file] = len(tokens)

if not os.path.exists("queries.txt"):
    print("ERROR: File named 'queries' not found! Make sure it's exactly named 'queries' (no .txt)")
    exit()

with open("queries.txt", "r", encoding="utf-8") as f:
    queries = [line.strip() for line in f if line.strip()]

if not docs:
    print("No documents found!")
    exit()

N = len(docs)
avg_dl = sum(lengths.values()) / N
all_tokens = [t for tokens in docs.values() for t in tokens]
collection_freq = Counter(all_tokens)
total_terms = len(all_tokens) if all_tokens else 1

df = Counter()
for tokens in docs.values():
    df.update(set(tokens))

print("\n" + "="*110)
print(" " * 42 + "BM25 + JELINEK-MERCER RANKING")
print("="*110 + "\n")

for query in queries:
    q_terms = preprocess(query)
    if not q_terms:
        continue

    print(f"QUERY: {query.upper()}")
    print("-" * 110)
    print(f"{'RANK':<4} {'BM25 SCORE':<14} {'DOCUMENT':<50} {'JM SCORE (λ=0.5)':<18} {'DOCUMENT'}")
    print("-" * 110)

    bm25_scores = {}
    jm_scores = {}

    for doc_id, tokens in docs.items():
        tf = Counter(tokens)
        doc_len = lengths[doc_id]

        bm25 = 0.0
        jm = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            df_term = df.get(term, 0)
            idf = math.log((N - df_term + 0.5) / (df_term + 0.5) + 1)
            bm25 += idf * (f * 2.2) / (f + 1.2 * (1 - 0.75 + 0.75 * doc_len / avg_dl))

            p_doc = f / doc_len if doc_len > 0 else 0
            p_coll = collection_freq.get(term, 0) / total_terms
            jm += math.log(0.5 * p_doc + 0.5 * p_coll + 1e-12)

        bm25_scores[doc_id] = bm25
        jm_scores[doc_id] = jm

    top10_bm25 = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    top10_jm = sorted(jm_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    for i in range(10):
        b_line = ""
        j_line = ""
        if i < len(top10_bm25):
            doc, score = top10_bm25[i]
            b_line = f"{i+1:<4} {score:<14.4f} {doc:<50}"
        if i < len(top10_jm):
            doc, score = top10_jm[i]
            j_line = f"{score:<18.4f} {doc}"
        print(b_line + j_line)
    print("\n")