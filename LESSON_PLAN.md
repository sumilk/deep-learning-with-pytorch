# Deep Learning with PyTorch — Lesson Plan

Proposed curriculum: **56 notebooks**, numbered **00–55**.

Status: proposed plan awaiting review. No tutorial notebooks have been written yet.

## Goals and prerequisites

Start from basic Python and introduce the required mathematics as we go. Progress slowly from individual tensor operations to complete CNN and Transformer systems. Prioritize understanding, experiments, and debugging over quickly assembling models.

Teach the progression between architectures: what problem each addresses, how its mechanisms work, what limitations remain, and how later approaches address those limitations. Architectural families are alternatives with different tradeoffs, not a universal ranking.

## 1. Foundations: learn the tools and the math

**Notebooks 00–05**

- **00 — Course guide and setup:** environments, Jupyter, CPU/GPU, reproducibility, and how to approach the exercises.
- **01 — Tensors from first principles:** shapes, indexing, broadcasting, views, copies, strides, and device placement.
- **02 — Linear algebra through pictures:** vectors, dot products, matrix multiplication, projections, and affine transformations.
- **03 — Derivatives and gradients:** slopes, partial derivatives, chain rule, finite differences, and gradient descent.
- **04 — Autograd:** computational graphs, gradient accumulation, leaf tensors, `detach`, `no_grad`, and in-place operation traps.
- **05 — First learning problem:** fit linear regression using manual gradients, then autograd, then `nn.Module`.

**Visual intuition:** vector transformations, loss surfaces, gradient arrows, and optimization trajectories.

## 2. From one perceptron to networks

**Notebooks 06–10**

- **06 — The perceptron:** weighted sums, thresholds, decision boundaries, and the perceptron learning rule.
- **07 — Linear separability:** AND, OR, XOR, convergence assumptions, and why one perceptron fails.
- **08 — Logistic regression:** smooth decisions, probabilities, log-odds, and binary cross-entropy.
- **09 — Networks of perceptrons:** construct logical functions and combine half-spaces into decision regions.
- **10 — Universal approximation:** build functions using ReLU pieces; explore width versus depth and approximation error.

Distinguish **representing a function**, **learning it through optimization**, and **generalizing to unseen data**. Universal approximation does not guarantee the latter two. State the theorem's assumptions and distinguish approximation of continuous functions from arbitrary exact classification.

**Tricky problems:** XOR with misleading features, concentric rings, checkerboards, and models that interpolate training data but fail between samples.

## 3. MLPs and backpropagation in depth

**Notebooks 11–17**

- **11 — Build an MLP from scratch:** tensor operations, parameters, forward passes, and batch dimensions.
- **12 — Derive backpropagation:** manually differentiate a two-layer network and verify against autograd.
- **13 — Activations:** sigmoid, tanh, ReLU, Leaky ReLU, GELU; saturation and dead neurons.
- **14 — Loss functions:** regression, binary, multiclass, and multilabel objectives; logits and numerical stability.
- **15 — Initialization and gradient flow:** symmetry breaking, Xavier, Kaiming, and exploding/vanishing activations.
- **16 — Optimization:** implement SGD, momentum, RMSProp, and Adam; examine learning rates and weight decay.
- **17 — Depth and representation learning:** visualize hidden features and investigate when deeper networks help.

**Visual intuition:** changing decision boundaries, hidden-space transformations, activation distributions, and gradients across layers.

## 4. Training models reliably

**Notebooks 18–23**

- **18 — Data pipelines:** datasets, loaders, batching, shuffling, preprocessing, and train-only normalization.
- **19 — Generalization:** underfitting, overfitting, learning curves, data leakage, and distribution shift.
- **20 — Regularization:** L1/L2, dropout, augmentation, early stopping, and label smoothing.
- **21 — Normalization and residual connections:** BatchNorm versus LayerNorm, train/eval behavior, and skip connections.
- **22 — Evaluation:** confusion matrices, precision/recall, PR/ROC curves, imbalance, thresholds, and calibration.
- **23 — A reliable training system:** checkpoints, optimizer state, schedules, gradient clipping, mixed precision, and debugging.

**Tricky problems:** random-label memorization, leaked preprocessing, accuracy on rare classes, incorrect loss scaling, and silently missing parameters.

## 5. CNNs: operations to architectural families

**Notebooks 24–30**

- **24 — Convolution from scratch:** sliding windows, channels, padding, stride, dilation, and comparison with `Conv2d`.
- **25 — Spatial intuition:** edge filters, feature maps, pooling, receptive fields, and translation behavior.
- **26 — LeNet, AlexNet, and VGG:** architectural progression, training a small classifier, and parameter costs.
- **27 — Inception and ResNet:** parallel branches, bottlenecks, residual blocks, and depth.
- **28 — DenseNet, MobileNet, and EfficientNet:** feature reuse, depthwise separable convolutions, and scaling tradeoffs.
- **29 — U-Net and dense prediction:** segmentation, upsampling, skip connections, Dice/IoU, and class imbalance.
- **30 — Transfer learning and interpretation:** freezing versus fine-tuning, feature extraction, saliency, and Grad-CAM.

Clearly distinguish scratch teaching versions from full reference architectures.

**Tricky problems:** classify shape despite changing backgrounds, detect shortcut learning, diagnose pooling-induced information loss, and segment tiny objects.

## 6. Sequences and recurrent memory

**Notebooks 31–36**

- **31 — Representing sequences:** tokenization, embeddings, padding, lengths, masks, and sequence task types.
- **32 — Vanilla RNN from scratch:** unroll recurrence and build a sequence classifier.
- **33 — Backpropagation through time:** long-range dependencies, gradient flow, clipping, and truncated BPTT.
- **34 — LSTM from scratch:** derive gates and cell state; compare with PyTorch's implementation.
- **35 — GRU from scratch:** reset/update gates and controlled comparisons with RNNs and LSTMs.
- **36 — Stacked and bidirectional models:** packed sequences, variable lengths, and when bidirectionality leaks future information.

**Visual intuition:** unrolled computation graphs, gate values over time, memory retention, and gradient magnitude versus distance.

**Tricky problems:** delayed copy, first-token recall, order-sensitive classification, and padding that changes predictions.

## 7. Seq2Seq and the introduction of attention

**Notebooks 37–40**

- **37 — Encoder–decoder Seq2Seq:** build a recurrent model for variable-length sequence transformation.
- **38 — Training versus generation:** shifted targets, teacher forcing, exposure bias, greedy decoding, and beam search.
- **39 — Attention from scratch:** derive additive and dot-product attention; visualize alignments and masks.
- **40 — Attentive Seq2Seq:** add attention to the previous model and compare performance as sequences grow longer.

**Hands-on tasks:** sequence reversal, copying, and a small synthetic translation grammar, with held-out lengths and compositions.

## 8. Transformers, built gradually

**Notebooks 41–47**

- **41 — Queries, keys, and values:** attention as content-based retrieval; derive scaled dot-product attention.
- **42 — Self-attention and position:** permutation behavior, positional encodings, and padding/causal masks.
- **43 — Multi-head attention from scratch:** splitting heads, merging outputs, shape tracing, and equivalence checks.
- **44 — Transformer encoder:** feed-forward layers, residuals, normalization, and sequence classification.
- **45 — Decoder-only Transformer:** causal language modeling, shifted labels, and training a tiny GPT-style model.
- **46 — Encoder–decoder Transformer:** cross-attention and a complete sequence-transformation model.
- **47 — Generation and efficiency:** temperature, top-k/top-p, KV caching, and training versus inference parallelism.

**Tricky problems:** future-token leakage, reversed mask semantics, wrong softmax axes, missing positions, and generation that disagrees with cached inference.

## 9. Modern extensions and practical limits

**Notebooks 48–51**

- **48 — Modern Transformer components:** pre/post normalization, RMSNorm, SwiGLU, and RoPE.
- **49 — Attention efficiency:** compute/memory scaling, optimized attention, grouped-query attention, and benchmarking.
- **50 — Vision Transformers:** patch embeddings, positional information, and comparison with CNN inductive biases.
- **51 — Adaptation and robustness:** implement LoRA on a small model; investigate spurious features and out-of-distribution evaluation.

## 10. Independent capstones

**Notebooks 52–55**

- **52 — Tabular classification:** imbalance, leakage, calibration, and MLP baselines.
- **53 — Vision:** compare CNN families on classification or segmentation under a compute budget.
- **54 — Sequence modeling:** compare RNN, LSTM, GRU, and attention using matched tasks and evaluation.
- **55 — Tiny Transformer system:** training, checkpointing, generation, evaluation, and a written failure analysis.

## Teaching notebook requirements

Each teaching notebook will contain:

1. Prerequisites, learning objectives, and a motivating problem.
2. Plain-language intuition, diagrams, and equations with explicit tensor shapes.
3. Small numerical examples before larger implementations.
4. Runnable implementations from scratch where instructive.
5. Comparison against PyTorch implementations where applicable.
6. Plots that expose behavior, including failures.
7. Exercises progressing through **modify → implement → debug → investigate**.
8. Separate hints and worked solutions.
9. A short explanation of what remains unsolved and motivates the next lesson.

Capstones will provide task specifications, evaluation criteria, and baselines while leaving room for independent implementation.

## Graphics and intuition

Use graphics wherever they materially aid understanding, including:

- Decision boundaries and hidden representation transformations.
- Computational graphs and loss-surface trajectories.
- Convolution animations or step plots, receptive fields, and feature maps.
- Recurrent gate traces and gradient propagation over time.
- Attention heatmaps, positional encodings, and causal masking.
- Architecture diagrams and compute/memory comparisons.

Prefer notebook-native plotting and diagrams that can be regenerated from code. Annotate axes, tensor dimensions, and experimental conditions. Explain what each graphic demonstrates and where the analogy stops being accurate.

## Execution and validation goals

- Core labs run on CPU using small, locally generated datasets without downloads.
- Larger real-dataset experiments are optional, with download and compute requirements stated.
- Make random seeds, expected runtimes, and device selection explicit.
- Important scratch components receive mathematical or behavioral checks against reference implementations.
- Notebooks should execute from a fresh kernel in order without hidden state.
- Add an execution/validation runner alongside notebook implementation and record actual validation results.

## Scope boundary

This is a broad foundation in supervised deep learning, CNNs, sequence models, and Transformers. It is not an exhaustive treatment of every deep learning field. Generative adversarial networks, diffusion models, graph neural networks, reinforcement learning, and distributed training are potential later tracks.
