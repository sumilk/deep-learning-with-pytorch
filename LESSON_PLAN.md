# Deep Learning with PyTorch — Lesson Plan

Proposed curriculum: **65 notebooks**, numbered **00–64**, organized into **11 chapters**, plus one **problem-set notebook per chapter**.

Status: proposed plan awaiting review. No tutorial notebooks have been written yet.

## Goals and prerequisites

Start from basic Python and introduce the required mathematics as we go. No prior NumPy, calculus, or probability is assumed. Progress slowly from individual tensor operations to complete CNN and Transformer systems. Prioritize understanding, experiments, and debugging over quickly assembling models.

Teach the progression between architectures: what problem each addresses, how its mechanisms work, what limitations remain, and how later approaches address those limitations. Architectural families are alternatives with different tradeoffs, not a universal ranking.

Teach rigor as a habit: verify gradients, sanity-check losses, compare models across several seeds with matched budgets, and report uncertainty rather than single numbers.

## Compute tiers: CPU core, GPU scale-up

Every notebook has a **CPU core** that teaches the concept on small data in minutes. Many notebooks also have a **GPU scale-up** section that repeats the experiment at a size where the GPU matters: larger data, deeper models, real datasets, or pretrained weights.

| Tier | Hardware | Role |
| --- | --- | --- |
| **CPU core** | Any laptop | Required. Every concept, exercise, and check runs here. |
| **GPU scale-up** | NVIDIA CUDA (local, Colab, or cloud) | Optional sections marked **[GPU]**. Larger experiments, mixed precision, compilation, profiling, and pretrained models. |
| **Apple MPS** | Apple Silicon | Supported where PyTorch's MPS backend supports the operations. Notebooks state known gaps (unsupported ops, fp16/bf16 behavior, weaker determinism) and fall back to CPU explicitly. |

Chapter 5 teaches GPU computing directly. After it, GPU scale-up sections assume its tools: correct timing, mixed precision, profiling, and memory management.

Notebooks select devices through one shared helper, print the selected device, and never silently assume CUDA.

## Recurring datasets

A small set of datasets recurs throughout the course so that comparisons across chapters are meaningful.

| Dataset | Source | Used in |
| --- | --- | --- |
| **Synthetic 2D sets** (blobs, XOR, rings, spirals, checkerboards) | Generated in code | Chapters 2–4 |
| **Synthetic regression functions** with controllable noise | Generated in code | Chapters 1–4 |
| **Procedural shapes** (shapes, textures, backgrounds, controllable shortcuts, segmentation masks) | Generated in code | Chapters 4, 6 |
| **Synthetic sequence tasks** (copy, reverse, delayed recall, counting, a small translation grammar) | Generated in code | Chapters 7–9 |
| **Small text corpus** (public-domain text, a few MB) | Committed to the repo | Chapters 7, 9, 10 |
| **Small tabular dataset** with imbalance and leakage traps | Generated or committed | Chapter 4, capstone 61 |
| **Real datasets** (MNIST/Fashion-MNIST, CIFAR-10, Oxford-IIIT Pets, pretrained torchvision weights) | Downloaded **[GPU]/optional** | Scale-up sections only |

Download sections state the dataset size, the license, and the expected download and run time.

## In-chapter exercises

Exercises are spread through each notebook and each chapter, not saved for the end.

1. **Inline checkpoints:** after each concept, one or two short tasks such as predicting a shape, filling in one line, or explaining a plot. Each has an `assert`-based check that gives immediate feedback.
2. **End-of-notebook exercises:** progressing through **modify → implement → debug → investigate**. "Debug" exercises include deliberately broken code. "Investigate" exercises ask an open question answered with an experiment and a short written conclusion.
3. **Chapter problem set:** a separate notebook per chapter (`chNN_problems.ipynb`) that combines the chapter's ideas into larger problems. It includes a debugging problem, a small experiment with a written analysis, and optionally a **[GPU]** problem.

Implementation exercises ship with automated checks: shape and value tests, comparisons against PyTorch reference implementations, and `gradcheck` where applicable. Hints and worked solutions live in separate solution notebooks so learners can attempt problems first.

---

## Chapter 1. Foundations: tools and math

**Notebooks 00–06**

- **00 — Course guide and setup:** environments, pinned Python/PyTorch versions, Jupyter, CPU/CUDA/MPS device selection, Colab for GPU access, seeds and `torch.use_deterministic_algorithms`, and how to approach exercises.
- **01 — Tensors from first principles:** shapes, dtypes, indexing, broadcasting, views, copies, strides, contiguity, and device placement.
- **02 — Linear algebra through pictures:** vectors, dot products, matrix multiplication, projections, affine transformations, and batched matrix multiplication.
- **03 — Derivatives, gradients, and Jacobians:** slopes, partial derivatives, chain rule, finite differences, Jacobians, vector–Jacobian products, and gradient descent.
- **04 — Probability for learning:** random variables, Bernoulli/categorical/Gaussian distributions, likelihood, maximum likelihood estimation, entropy, cross-entropy, and KL divergence.
- **05 — Autograd:** computational graphs, autograd as vector–Jacobian products, gradient accumulation, leaf tensors, `detach`, `no_grad`, in-place traps, custom `torch.autograd.Function`, and `gradcheck`.
- **06 — First learning problem:** linear regression with manual gradients, then autograd, then `nn.Module`; mean squared error derived as Gaussian maximum likelihood.

**Visual intuition:** vector transformations, loss surfaces, gradient arrows, likelihood curves, and optimization trajectories.

**Chapter problem set:**
- Predict the output shapes of 20 broadcasting and view expressions, then verify them.
- Implement a custom `autograd.Function` for a numerically stable operation such as `softplus` and verify it with `gradcheck`.
- Debug a training loop that forgets `zero_grad`, then a second loop with an in-place modification that breaks autograd.
- Investigate: fit a Gaussian and a Laplace likelihood to data with outliers; explain how the loss (MSE vs. MAE) follows from the likelihood.

## Chapter 2. From one perceptron to networks

**Notebooks 07–12**

- **07 — The perceptron:** weighted sums, thresholds, decision boundaries, and the perceptron learning rule.
- **08 — Linear separability:** AND, OR, XOR, convergence assumptions, and why one perceptron fails.
- **09 — Logistic regression:** smooth decisions, probabilities, log-odds, and binary cross-entropy derived from Bernoulli likelihood.
- **10 — Softmax regression:** multiclass logits, softmax, categorical cross-entropy, the log-sum-exp trick, and sanity-checking the initial loss (≈ ln C for C classes).
- **11 — Networks of perceptrons:** construct logical functions and combine half-spaces into decision regions.
- **12 — Universal approximation:** build functions using ReLU pieces; explore width versus depth and approximation error.

Distinguish **representing a function**, **learning it through optimization**, and **generalizing to unseen data**. Universal approximation does not guarantee the latter two. State the theorem's assumptions and distinguish approximation of continuous functions from arbitrary exact classification.

**Tricky problems:** XOR with misleading features, concentric rings, checkerboards, and models that interpolate training data but fail between samples.

**Chapter problem set:**
- Hand-construct a three-layer step network that classifies a 16-tile checkerboard exactly, using parity as a linear function of ordered threshold units.
- Implement Newton's method for logistic regression; compare its quadratic convergence with gradient descent and explain why it does not scale to networks.
- Debug a multiclass classifier whose loss stays at ln K because shuffling decoupled inputs from labels.
- Investigate: constructed versus trained one-hidden-layer ReLU networks approximating `sin(x)`; how the gap grows with width, and how much seeds differ.

## Chapter 3. MLPs and backpropagation in depth

**Notebooks 13–20**

- **13 — Build an MLP from scratch:** tensor operations, parameters, forward passes, batch dimensions, and the debugging habits used for the rest of the course: initial-loss checks, overfitting a single batch, and monitoring gradient norms.
- **14 — Derive backpropagation:** manually differentiate a two-layer network, verify against autograd and `gradcheck`, and relate each step to vector–Jacobian products.
- **15 — Inside `nn.Module`:** parameter registration, buffers, `state_dict`, `train`/`eval` modes, and forward/backward hooks for inspecting activations and gradients.
- **16 — Activations:** sigmoid, tanh, ReLU, Leaky ReLU, GELU; saturation and dead neurons measured with hooks.
- **17 — Loss functions:** regression, binary, multiclass, and multilabel objectives derived from likelihoods; logits, reductions, and numerical stability.
- **18 — Initialization and gradient flow:** symmetry breaking, Xavier, Kaiming, and exploding/vanishing activations across depth.
- **19 — Optimization:** full-batch versus mini-batch noise; implement SGD, momentum, RMSProp, Adam, and AdamW; L2 penalty versus decoupled weight decay; learning-rate schedules and warmup.
- **20 — Depth and representation learning:** visualize hidden features and investigate when deeper networks help.

**Visual intuition:** changing decision boundaries, hidden-space transformations, activation distributions, and gradients across layers.

**Chapter problem set:**
- Derive and implement backpropagation for a three-layer MLP with a different activation per layer; pass `gradcheck`.
- Implement AdamW from scratch and match `torch.optim.AdamW` exactly over 100 steps.
- Debug three broken MLPs: a missing nonlinearity, softmax applied before `CrossEntropyLoss`, and a layer stored in a plain Python list so its parameters are never registered.
- Investigate: plot activation and gradient statistics per layer for a 20-layer MLP under three initializations and explain the differences.

## Chapter 4. Training models reliably

**Notebooks 21–27**

- **21 — Data pipelines:** `Dataset`, `DataLoader`, batching, shuffling, collate functions, preprocessing, and train-only normalization.
- **22 — Generalization:** underfitting, overfitting, learning curves, the bias–variance tradeoff, data leakage, and distribution shift.
- **23 — Regularization:** L1/L2, dropout, augmentation, early stopping, and label smoothing.
- **24 — Normalization and residual connections:** LayerNorm, an introduction to BatchNorm and its train/eval behavior, and skip connections as gradient highways. BatchNorm is revisited in Chapter 6.
- **25 — Evaluation:** confusion matrices, precision/recall, PR/ROC curves, imbalance, thresholds, and calibration.
- **26 — Rigorous experiments:** variance across seeds, confidence intervals, matched parameter and compute budgets, ablations, hyperparameter search (grid, random, learning-rate range tests), and honest baselines.
- **27 — A reliable training system:** a reusable training loop with checkpoints, optimizer and scheduler state, resuming, gradient clipping, logging, and a debugging checklist.

**Tricky problems:** random-label memorization, leaked preprocessing, accuracy on rare classes, incorrect loss scaling, silently missing parameters, and an "improvement" that disappears across seeds.

**Chapter problem set:**
- Find and fix the leakage in a provided pipeline whose validation score is suspiciously high.
- Run a 5-seed comparison of two regularization strategies; report means with confidence intervals and decide whether the difference is real.
- Kill a training run mid-epoch, resume from a checkpoint, and verify the loss curve matches an uninterrupted run.
- Investigate: train on random labels and real labels; compare learning curves and explain what memorization implies about generalization.

## Chapter 5. GPU computing

**Notebooks 28–30**

The concepts run on CPU with reduced demonstrations. The measurements are intended for a CUDA GPU (local or Colab), with MPS notes where behavior differs.

- **28 — How GPUs execute PyTorch:** devices and transfers, asynchronous kernel launches, synchronization, correct benchmarking (warmup, `torch.cuda.synchronize`, `torch.utils.benchmark`), pinned memory, the caching allocator, memory accounting, out-of-memory errors, and CUDA versus MPS differences.
- **29 — Throughput and memory techniques:** mixed precision (fp16 with `GradScaler`, bf16), `torch.compile`, DataLoader workers and prefetching, the PyTorch profiler, gradient accumulation, and activation checkpointing.
- **30 — Scaling an experiment:** port the Chapter 4 training system to the GPU; measure speedups versus model and batch size; batch-size and learning-rate scaling; when the GPU does not help (tiny models, input-bound pipelines); and reproducibility on GPUs.

Multi-GPU and distributed training are out of scope; see the scope boundary.

**Tricky problems:** benchmarks that time only the kernel launch, fp16 overflow without loss scaling, a "GPU" run whose data stays on the CPU, a DataLoader that starves the GPU, and nondeterministic results across runs.

**Chapter problem set:**
- Write a correct benchmark for matrix multiplication across sizes and dtypes; find the crossover point where GPU beats CPU.
- Profile a provided slow training loop, identify the bottleneck (host–device transfers, `.item()` calls, data loading), and speed it up at least 3×.
- Train the same model in fp32, fp16 + `GradScaler`, and bf16; compare speed, memory, and final accuracy across seeds.
- **[GPU]** Fit the largest MLP you can on your GPU using mixed precision, gradient accumulation, and activation checkpointing; explain each technique's memory saving.

## Chapter 6. CNNs: operations to architectural families

**Notebooks 31–38**

- **31 — Convolution from scratch:** sliding windows, channels, padding, stride, dilation, im2col, FLOP and parameter counts, and comparison with `Conv2d`.
- **32 — Spatial intuition:** edge filters, feature maps, pooling, receptive fields, and translation equivariance versus invariance.
- **33 — Classic CNNs (LeNet, AlexNet, VGG):** the architectural progression; implement a small VGG-style network; measure why stacked 3×3 convolutions beat large kernels in receptive field per parameter; BatchNorm in convolutional networks.
- **34 — Inception and ResNet:** implement the bottleneck and residual block; measure gradient flow and trainability with and without residual connections as depth grows.
- **35 — Efficient CNNs (MobileNet, EfficientNet):** implement depthwise separable convolutions and measure FLOPs, parameters, and actual latency on CPU and **[GPU]**. DenseNet feature reuse and EfficientNet compound scaling are covered as reading with diagrams.
- **36 — U-Net and dense prediction:** segmentation, transposed convolution versus upsampling, skip connections, Dice/IoU, and class imbalance.
- **37 — Transfer learning:** freezing versus fine-tuning and feature extraction. CPU core: pretrain on procedural shapes, then transfer to a shifted variant. **[GPU]**: fine-tune pretrained torchvision weights on a real dataset.
- **38 — Interpreting CNNs:** saliency maps and Grad-CAM implemented with hooks, plus sanity checks that reveal when explanations do not depend on the model.

Clearly distinguish scratch teaching versions from full reference architectures. Each architecture notebook implements and measures one key mechanism rather than training every full model.

**Tricky problems:** classify shape despite changing backgrounds, detect shortcut learning, diagnose pooling-induced information loss, and segment tiny objects.

**Chapter problem set:**
- Implement `conv2d` forward and backward with im2col and match `F.conv2d` and its gradients.
- Debug a model whose accuracy collapses in `eval()` mode (a BatchNorm statistics bug).
- Detect and fix a shortcut: a classifier that learned background color instead of shape; prove it with Grad-CAM and a counterfactual test set.
- **[GPU]** Train a small ResNet on CIFAR-10 with mixed precision; compare to the VGG-style baseline at a matched compute budget across 3 seeds.

## Chapter 7. Sequences and recurrent memory

**Notebooks 39–45**

- **39 — Representing sequences:** character, word, and subword tokenization; implement a small byte-pair encoding (BPE) tokenizer; embeddings, padding, lengths, masks, and sequence task types.
- **40 — Vanilla RNN from scratch:** unroll recurrence and build a sequence classifier.
- **41 — Backpropagation through time:** long-range dependencies, gradient flow, clipping, and truncated BPTT.
- **42 — LSTM from scratch:** derive gates and cell state; compare with PyTorch's implementation.
- **43 — GRU from scratch:** reset/update gates and controlled comparisons with RNNs and LSTMs.
- **44 — Stacked and bidirectional models:** packed sequences, variable lengths, and when bidirectionality leaks future information.
- **45 — Convolutions for sequences:** 1D, causal, and dilated convolutions as a parallel alternative to recurrence; receptive field versus sequence length.

**Visual intuition:** unrolled computation graphs, gate values over time, memory retention, and gradient magnitude versus distance.

**Tricky problems:** delayed copy, first-token recall, order-sensitive classification, and padding that changes predictions.

**Chapter problem set:**
- Train a BPE tokenizer on the course corpus and compare sequence lengths and vocabulary sizes against character and word tokenization.
- Implement an LSTM cell from scratch and match `nn.LSTM` outputs and gradients exactly, including weight layout.
- Debug a classifier whose predictions change when padding is added.
- Investigate: on the delayed-recall task, find the maximum delay each of RNN, LSTM, GRU, and dilated causal CNN can solve at a matched parameter count, across 3 seeds.

## Chapter 8. Seq2Seq and the introduction of attention

**Notebooks 46–49**

- **46 — Encoder–decoder Seq2Seq:** build a recurrent model for variable-length sequence transformation.
- **47 — Training versus generation:** shifted targets, teacher forcing, exposure bias, greedy decoding, and beam search.
- **48 — Attention as alignment:** the fixed-vector bottleneck; derive additive and dot-product attention over encoder states; visualize alignments and masks.
- **49 — Attentive Seq2Seq:** add attention to the previous model and compare performance as sequences grow longer.

**Hands-on tasks:** sequence reversal, copying, and a small synthetic translation grammar, with held-out lengths and compositions.

**Chapter problem set:**
- Implement beam search with length normalization and show a case where it beats greedy decoding and a case where it does not.
- Debug a decoder whose targets are not shifted, so training loss is near zero but generation fails.
- Investigate: plot accuracy versus sequence length for Seq2Seq with and without attention, including lengths longer than any seen in training.

## Chapter 9. Transformers, built gradually

**Notebooks 50–56**

- **50 — Queries, keys, and values:** from alignment attention (notebook 48) to learned Q/K/V projections; attention as content-based retrieval; the variance argument for 1/√d scaling.
- **51 — Self-attention and position:** permutation equivariance, sinusoidal and learned positional encodings, and padding/causal masks.
- **52 — Multi-head attention from scratch:** splitting heads, merging outputs, shape tracing, and equivalence checks against `nn.MultiheadAttention`.
- **53 — Transformer encoder:** feed-forward layers, residuals, normalization, and sequence classification.
- **54 — Decoder-only Transformer:** causal language modeling, shifted labels, weight tying, learning-rate warmup, and training a tiny GPT-style model on the course corpus with the BPE tokenizer from notebook 39.
- **55 — Encoder–decoder Transformer:** cross-attention and a complete sequence-transformation model.
- **56 — Generation and efficiency:** temperature, top-k/top-p, KV caching, and training versus inference parallelism.

**Tricky problems:** future-token leakage, reversed mask semantics, wrong softmax axes, missing positions, and generation that disagrees with cached inference.

**Chapter problem set:**
- Implement multi-head attention with a single fused QKV projection and prove it matches the three-projection version.
- Debug a tiny GPT that reaches suspiciously low training loss because of a causal-mask bug.
- Implement a KV cache and verify that cached and uncached generation produce identical logits.
- **[GPU]** Scale the tiny GPT in width, depth, and data; plot loss versus compute and discuss what changes.

## Chapter 10. Modern extensions and practical limits

**Notebooks 57–60**

- **57 — Modern Transformer components:** pre/post normalization, RMSNorm, SwiGLU, and RoPE.
- **58 — Attention efficiency:** FLOP and memory analysis of attention, `F.scaled_dot_product_attention`, the idea behind FlashAttention, and grouped-query attention. Benchmark on CPU and **[GPU]**, noting that CPU timings do not reflect GPU kernel behavior.
- **59 — Vision Transformers:** patch embeddings, positional information, and CNN inductive biases. CPU core: a ViT loses to a CNN on small procedural data, and the notebook explains why. **[GPU]**: compare on a larger dataset with augmentation.
- **60 — Adaptation and robustness:** implement LoRA and apply it to the tiny GPT from notebook 54 for a new task; compare with full fine-tuning; investigate spurious features and out-of-distribution evaluation.

**Chapter problem set:**
- Add RoPE to the tiny GPT and test generalization to longer contexts than seen in training.
- Measure attention memory versus sequence length with naive attention and `scaled_dot_product_attention`; find where the naive version runs out of memory **[GPU]**.
- Implement LoRA from scratch, verify that merging the adapter reproduces the adapted model exactly, and compare trainable parameter counts.

## Chapter 11. Independent capstones

**Notebooks 61–64**

Each capstone requires a baseline, at least 3 seeds per configuration, matched budgets for comparisons, and a written analysis of failures and uncertainty. Each has an optional **[GPU]** scale-up.

- **61 — Tabular classification:** imbalance, leakage, calibration, an MLP, and a gradient-boosted trees baseline the MLP must honestly be compared against.
- **62 — Vision:** compare CNN families on classification or segmentation under a fixed compute budget.
- **63 — Sequence modeling:** compare RNN, LSTM, GRU, causal CNN, and attention on matched tasks and evaluation.
- **64 — Tiny Transformer system:** tokenizer, training, checkpointing, generation, evaluation, GPU throughput tuning, and a written failure analysis.

Capstones provide task specifications, evaluation criteria, and baselines while leaving room for independent implementation.

---

## Teaching notebook requirements

Each teaching notebook will contain:

1. Prerequisites, learning objectives, estimated time, and a motivating problem.
2. Plain-language intuition, diagrams, and equations with explicit tensor shapes.
3. Small numerical examples before larger implementations.
4. Runnable implementations from scratch where instructive.
5. Comparison against PyTorch implementations where applicable.
6. Plots that expose behavior, including failures.
7. Inline checkpoint exercises with automated checks.
8. End-of-notebook exercises progressing through **modify → implement → debug → investigate**.
9. An optional **[GPU]** scale-up section where the GPU changes what can be learned.
10. A short explanation of what remains unsolved and motivates the next lesson.

## Time budget

| Item | Estimated time |
| --- | --- |
| Teaching notebook (CPU core + inline exercises) | 2–3 hours |
| End-of-notebook exercises | 1–2 hours |
| Chapter problem set | 3–5 hours |
| **[GPU]** scale-up section | 0.5–2 hours, plus run time |
| Capstone | 10–20 hours |

The full course is roughly 300–400 hours. For a shorter path, notebooks 35, 45, 58, and 59 and all **[GPU]** sections can be skipped without breaking later dependencies.

## Graphics and intuition

Use graphics wherever they materially aid understanding, including:

- Decision boundaries and hidden representation transformations.
- Computational graphs and loss-surface trajectories.
- Convolution animations or step plots, receptive fields, and feature maps.
- Recurrent gate traces and gradient propagation over time.
- Attention heatmaps, positional encodings, and causal masking.
- Architecture diagrams, compute/memory comparisons, and profiler timelines.
- Results across seeds with error bars, never single runs presented as conclusions.

Prefer notebook-native plotting and diagrams that can be regenerated from code. Annotate axes, tensor dimensions, and experimental conditions. Explain what each graphic demonstrates and where the analogy stops being accurate.

## Execution and validation goals

- Pin Python and PyTorch versions; record them in each notebook's first cell output.
- CPU core sections run using locally generated or committed data, with no downloads.
- **[GPU]** and download sections are clearly marked, guarded so they skip cleanly without a GPU, and state download size and compute requirements.
- Make random seeds, determinism settings, expected runtimes, and device selection explicit.
- Important scratch components receive mathematical or behavioral checks against reference implementations, including `gradcheck` for custom gradients.
- Notebooks should execute from a fresh kernel in order without hidden state.
- Add an execution/validation runner alongside notebook implementation that:
  - runs all CPU core sections and exercise checks against the solution notebooks;
  - runs **[GPU]** sections separately on a CUDA machine;
  - records actual validation results, runtimes, and hardware.

## Scope boundary

This is a broad foundation in supervised deep learning, GPU training on a single device, CNNs, sequence models, and Transformers. It is not an exhaustive treatment of every deep learning field. Autoencoders and variational autoencoders, self-supervised and contrastive learning, generative adversarial networks, diffusion models, graph neural networks, reinforcement learning, object detection, and multi-GPU/distributed training are potential later tracks.
