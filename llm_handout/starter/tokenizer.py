"""Baseline tokenizer: raw UTF-8 bytes, vocab of 256. Simple, never fails on
unseen text — and treats a Devanagari character as 3 tokens. Think about
what that does to your model's context window and your token budget on the
Hindi part of the corpus.

You may replace this with anything you train ON THE PROVIDED CORPUS ONLY
(e.g., BPE), as long as:
  1. it can encode ARBITRARY UTF-8 text (byte-level fallback) and it is
     LOSSLESS: decode(encode(text)) == text, exactly. The scorer and the
     graders both verify this round-trip — a lossy tokenizer makes bpb
     meaningless and disqualifies the run.
  2. this file keeps exposing:  load() -> tokenizer object with
     .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size.
     train.py and evaluate.py call load() with NO arguments — keep any
     extra parameters optional.
  3. anything it needs is saved under your submission folder and loaded by
     load() with no internet. Grading runs with cwd = your folder; resolve
     saved files relative to __file__ to be safe.
"""
import json
import os

class BPETokenizer:
    def __init__(self, merges_file="merges.json"):
        self.merges_list = []
        self.vocab_size = 256
        if os.path.exists(merges_file):
            with open(merges_file, "r") as f:
                merges_json = json.load(f)
            # sort by index to apply merges in order
            sorted_merges = sorted(merges_json.items(), key=lambda x: x[1])
            for k, v in sorted_merges:
                p0, p1 = map(int, k.split(","))
                self.merges_list.append(((p0, p1), v))
            self.vocab_size = 256 + len(self.merges_list)
            
        # Create reverse mapping for decode
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges_list:
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for pair, idx in self.merges_list:
            newids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    newids.append(idx)
                    i += 2
                else:
                    newids.append(ids[i])
                    i += 1
            ids = newids
        return ids

    def decode(self, ids):
        b = b"".join(self.vocab.get(i, b"") for i in ids)
        return b.decode("utf-8", errors="replace")

def load(path=None):
    """Return the tokenizer used by evaluate.py."""
    merges_path = os.path.join(os.path.dirname(__file__), "merges.json")
    return BPETokenizer(merges_path)
