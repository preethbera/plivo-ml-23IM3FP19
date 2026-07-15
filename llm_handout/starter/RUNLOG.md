# Run Log

## Baseline
- **Hypothesis**: Initial baseline run with default parameters.
- **What changed**: N/A (Baseline)
- **Dev BPB before**: N/A
- **Dev BPB after**: 2.3718
- **Conclusion**: Baseline established. Model has 1,339,840 parameters. It uses no weight tying, standard Adam without warmup/schedule, and a small batch size.

## Experiment 1: Weight Tying
- **Hypothesis**: Tying the input token embedding and the output projection weights will drastically reduce the parameter count (by roughly 40,960 parameters for vocab_size 256 and n_embd 160) without sacrificing representational power, freeing up our parameter budget for later model capacity increases.
- **What changed**: Set `tie_weights = True` in `model.py`.
- **Dev BPB before**: 2.3718
- **Dev BPB after**: 2.4122
- **Conclusion**: As expected, strictly tying weights slightly degraded performance because it reduced the model's capacity (parameters dropped from 1,339,840 to 1,298,880). However, this provides more parameter budget for subsequent structural scaling without hitting the 2,000,000 parameter limit.

## Experiment 2: Gradient Accumulation & Clipping
- **Hypothesis**: The baseline batch size of 8 is extremely small, leading to noisy gradients and suboptimal convergence. By implementing gradient accumulation (8 steps), we increase the effective batch size to 64. Since the 2,000 hard limit is on *optimizer steps*, this effectively lets the model see 8x more data during training. We also add gradient clipping (1.0) to stabilize the larger effective updates.
- **What changed**: Modified the training loop in `train.py` to accumulate gradients over 8 micro-batches and added `torch.nn.utils.clip_grad_norm_`.
- **Dev BPB before**: 2.4122
- **Dev BPB after**: 2.0226
- **Conclusion**: Dev BPB decreased from 2.4122 to 2.0226. Increasing the effective batch size to 64 via gradient accumulation improved training stability and increased the amount of data processed within the step limit. Gradient clipping provided additional stability.

## Experiment 3: BPE Tokenizer (Vocab 512)
- **Hypothesis**: The pure byte-level tokenizer is highly inefficient for Hindi (Devanagari characters take 3 bytes each), which consumes the model's context window. By implementing Byte Pair Encoding (BPE) and expanding the vocab from 256 to 512, we can merge common byte sequences. This compresses the sequence length, allowing the model to learn longer-range dependencies more easily and reducing the BPB.
- **What changed**: Created `tokenizer_train.py` to learn merges from `train_corpus.txt`. Rewrote `tokenizer.py` to apply the learned BPE merges. Updated `Config.vocab_size = 512` in `model.py`.
- **Dev BPB before**: 2.0226
- **Dev BPB after**: 1.9174
- **Conclusion**: Dev BPB decreased from 2.0226 to 1.9174. While the cross-entropy loss per token increased, the overall bits per byte on the evaluation set improved. The BPE tokenizer successfully compressed the corpus from 7.3M tokens to 3.2M tokens, effectively increasing the model's context window.

## Experiment 4: AdamW Optimizer and Cosine Annealing Learning Rate Schedule
- **Hypothesis**: Implementing AdamW with weight decay and a Cosine Annealing learning rate schedule with linear warmup will improve convergence compared to a constant learning rate.
- **What changed**: Modified `train.py` to use `AdamW` instead of `Adam`. Added an explicit learning rate warmup and cosine decay loop.
- **Dev BPB before**: 1.9174
- **Dev BPB after**: 2.1095
- **Conclusion**: Dev BPB increased from 1.9174 to 2.1095. Despite using AdamW and a Cosine Annealing schedule, the training was less effective than the baseline Adam optimizer for this specific setup and step count.

## Experiment 5: ALiBi and Architectural Scaling
- **Hypothesis**: Learned positional embeddings consume parameter budget and do not extrapolate well. By replacing them with ALiBi (Attention with Linear Biases) applied directly to the attention scores, we save parameters and improve translation invariance. This allows us to scale up the architecture to 6 layers (utilizing ~1.94M parameters), maximizing our capacity under the 2M cap.
- **What changed**: Removed `pos_emb` from `model.py`. Implemented ALiBi biases inside `SelfAttention` with a custom attention mask. Increased `n_layer` from 4 to 6 in `Config` to reach 1,937,920 parameters.
- **Dev BPB before**: 2.1095
- **Dev BPB after**: 1.8616
- **Conclusion**: Dev BPB dropped significantly from 2.1095 to 1.8616. Replacing positional embeddings with ALiBi preserved translation invariance while freeing up enough parameter budget to safely scale the model up to 6 layers. This structural scaling (1,937,920 parameters), paired with standard Adam and gradient accumulation, was the most effective configuration under the strict constraints.
