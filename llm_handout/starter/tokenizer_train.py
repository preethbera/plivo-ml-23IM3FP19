import json
import time

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def train():
    print("Training BPE tokenizer...")
    t0 = time.time()
    with open("../data/train_corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    ids = list(text.encode("utf-8"))
    
    vocab_size = 512
    num_merges = vocab_size - 256
    
    merges = {}
    for i in range(num_merges):
        counts = get_stats(ids)
        if not counts:
            break
        best = max(counts, key=counts.get)
        idx = 256 + i
        ids = merge(ids, best, idx)
        merges[best] = idx
        if (i+1) % 50 == 0 or i == num_merges - 1:
            print(f"merge {i+1}/{num_merges}: {best} -> {idx}")
    
    # Save merges. Convert keys from tuple of ints to string for JSON
    merges_json = {f"{p0},{p1}": idx for (p0, p1), idx in merges.items()}
    with open("merges.json", "w") as f:
        json.dump(merges_json, f)
        
    print(f"Saved merges.json in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    train()
