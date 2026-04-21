import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


df = pd.read_csv("data.csv")

print("Total samples:", len(df))
print("Columns:", df.columns.tolist())

df["question"] = df["question"].astype(str)
df["answer"] = df["answer"].astype(str)


df["q_len"] = df["question"].apply(lambda x: len(str(x).split()))
df["a_len"] = df["answer"].apply(lambda x: len(str(x).split()))

print("\nQuestion Length Stats:")
print(df["q_len"].describe())

print("\nAnswer Length Stats:")
print(df["a_len"].describe())


all_text = " ".join(df["question"].astype(str) + " " + df["answer"].astype(str))
tokens = all_text.split()

vocab = Counter(tokens)

print("\nVocab size:", len(vocab))
print("Top 20 words:", vocab.most_common(20))


rare_words = [w for w, c in vocab.items() if c == 1]
print("Rare words:", len(rare_words))


print("\n📊 PERCENTILES (IMPORTANT)")
for p in [50, 75, 90, 95, 99]:
    print(f"Q_len p{p}: {np.percentile(df['q_len'], p)}")
    print(f"A_len p{p}: {np.percentile(df['a_len'], p)}")

# === RECOMMENDATION ===
q_95 = int(np.percentile(df["q_len"], 95))
a_95 = int(np.percentile(df["a_len"], 95))

print("\n🎯 SUGGESTED max_seq_len:")
print(f"Question max len ≈ {q_95}")
print(f"Answer max len   ≈ {a_95}")
print(f"Combined (safe)  ≈ {q_95 + a_95 + 5}")

# === VOCAB ===
all_text = " ".join(df["question"] + " " + df["answer"])
tokens = all_text.split()

vocab = Counter(tokens)

print("\n🔤 VOCAB INFO")
print("Vocab size:", len(vocab))
print("Top 20 words:", vocab.most_common(20))

rare_words = [w for w, c in vocab.items() if c == 1]
print("Rare words:", len(rare_words))


plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(df["q_len"], bins=50)
plt.title("Question Length Distribution")

plt.subplot(1, 2, 2)
plt.hist(df["a_len"], bins=50)
plt.title("Answer Length Distribution")

plt.tight_layout()
plt.show()