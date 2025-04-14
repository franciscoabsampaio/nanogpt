# nanogpt

Building a small language model to study modern LLM architectures.

## Network Initialization

Some care is advised when initializing neural network weights and biases. Nowadays, the most common way to initialize weights is via Kaiming initialization.

However, some advances have also made initialization less relevant:

- Residual connections;
- Normalization layers, such as batch normalization, group normalization, etc;
- Better optimizers, such as RMSProp and Adam.

## Observability

Things to monitor:

- Dead neurons.
- Layer histograms.

## Statistical Assumptions

### Softmax

Softmax is grounded on the following statistical assumptions:

- **The distribution is categorical.**
- **Independence among classes.** If certain classes are dependent (e.g. `puppy` and `dog`), softmax may be overconfident in assigning a single class the highest score, when multiple are likely.
- **Scores follow a Gumbel distribution.** It implicitly assumes that input scores are noisy estimates of a latent preference value.
- **Scale invariance.** Its outputs change significantly when the input logits are changed. Higher scores get disproportionately more weight. If scores are on different scales, softmax may overemphasize large scores, and becomes almost a step function. If logits are small, softmax approaches a uniform distribution. **Temperature scaling** can help adjust sensitivity.

## References

Below are most of the references I used for learning about LLMs.

WIP

- **2023**. Andrej Karpathy. *Let's build GPT: from scratch, in code, spelled out*. At 40m.
- **2023**. Meta AI. *LLaMA: Open and Efficient Foundation Language Models*. At pg 3/27.
- **2020**. Noah Shazeer. *GLU Variants Improve Transformers*. At 1/5.
- **2019**. Edward Yang. *PyTorch internals*. At 'Autograd'.

✅

- **2024**. Larry Du. *All the Activation Functions (and a history of deep learning)*.
- **2024**. J Carlos Roldán. *What is SwiGLU*.
- **2024**. Andrej Karpathy. *Let's build the GPT tokenizer*.
- **2022**. Andrej Karpathy. *The spelled out intro to language modeling: building makemore*.
- **2022**. Andrej Karpathy. *The spelled out intro to neural networks and backpropagation: building micrograd*.
- **2015**. Kaiming et. al. *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*.
- **2003**. Bengio et. al. *A Neural Probabilistic Language Model*.
