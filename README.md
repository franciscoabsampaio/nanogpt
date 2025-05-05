# nanogpt

Small project to study modern deep learning and LLM architectures. Here you'll find a few example models, references that I found useful, and learnings.

Training dataset was [**tiny_shakespeare**](https://www.tensorflow.org/datasets/catalog/tiny_shakespeare), with a 80/10/10 train/validation/test split.

## Tokenizer

`sentencepiece` was used to train a tokenizer on the input data with the Byte-Pair Encoding algorithm. Most settings were taken from the Llama 2 tokenization training, with the following notable options:

- Vocabulary size of 32k words. This was selected considering Shakespeare's work features a vocabulary of about 30k unique words and over 800k total words, and the "4 tokens per 3 words" rule of thumb. Additionally, over 34k tokens caused `sentencepiece` to raise an error.
- Character coverage of `0.999995`. A slightly higher percentage was considered, because Shakespeare's work is unlikely to have typos, "faulty" words, etc.
- Max token length of 27 tokens (Shakespeare's longest word is Honorificabilitudinitatibus, at 27 characters of length). Due to how uncommon it is, it's most definitely not going to be parsed as a single token, but it's an excellent Easter egg.

## Models

### MLP

MLP using pre-trained tokenizer (`p50k_base` - 50k tokens), trained over 1400 iterations with SCD and the following hyperparameters:

- `batch_size`: 100
- `block_size`: 10
- `embedding_dims`: 200
- `n_neurons`: 200

Batch normalization was commented out. Learning rate was reduced by a factor of 10 after 1000 iterations.

A key way in which this implementation of an MLP differs from a traditional MLP, is how the inputs were **flattened**, to maximize sequence understanding, instead of conducting the analysis per-token. That is, each sequence entered the linear layer (and subsequent layers) in the shape `(B, T * C)` (the entirety of the time dimension (T) is being passed between neuronal layers as if it were extra embedding dimensions), which differs from, e.g., the WaveNet model (see below), which implements its context-understanding from other mechanics (dilation, causal convolutions, etc).

#### Number of Parameters

- Total: 20.562.681
- Trainable: 20.562.681

#### Output of 100 tokens given token 0

```txt
 him, hisMA, unbelievably do
MENblueilia make, toMinoruh still quart him.

vis fearful: be'd!
My speak ',
You you us brother know the crimesICH UntilB
R Canter1950AR worldatur HELt may of,
 lighter will I:
Or prostitute thou what Evolution to toS the Norse

R heliumest the recalls; fast Surgery a 330 ofWilliams.

CU art:
Th is IISTORY?
```

#### Learnings

- Learning rate can be reduced much later.
- This tokenizer is not suitable, but model still learned something.

## Network Initialization

Some care is advised when initializing neural network weights and biases. Nowadays, the most common way to initialize weights is via Kaiming initialization.

However, some advances have also made initialization less relevant:

- Residual connections;
- Normalization layers, such as batch normalization, group normalization, etc;
- Better optimizers, such as RMSProp and Adam.

## Batch Normalization

Batch normalization, first introduced by Ioffe et. al., is a widely used technique for addressing initialization problems and learning instabilities, such as vanishing/exploding gradients.

However, most implementations of BatchNorm differ from the original paper, and instead use an EMA version of the batch statistics. Specifically, it weighs both past and present statistics through a momentum parameter. Despite showing good performance in later epochs of training, this can create instability and innacuracy in the earlier stages of training, due to how volatile statistics can be during warm-up, and taking past statistics into account becomes a source of instability.

PreciseBN, on the other hand, implements the .

Finally, there are some considerations that should be remembered when applying batch normalization in general:

- Mini-batches that are too small make aggregate batch statistics unreliable.
- For EMA BatchNorm, large batch sizes aggravate training instability, due to how infrequently statistics are updated.

## Observability

Things to monitor:

- Dead neurons.
- Layer histograms.
- Loss.
- Update to data ratios (how aggressively each parameter is being changed).

## Statistics

### Softmax

Softmax is grounded on the following statistical assumptions:

- **The distribution is categorical.**
- **Independence among classes.** If certain classes are dependent (e.g. `puppy` and `dog`), softmax may be overconfident in assigning a single class the highest score, when multiple are likely.
- **Scores follow a Gumbel distribution.** It implicitly assumes that input scores are noisy estimates of a latent preference value.
- **Scale invariance.** Its outputs change significantly when the input logits are changed. Higher scores get disproportionately more weight. If scores are on different scales, softmax may overemphasize large scores, and becomes almost a step function. If logits are small, softmax approaches a uniform distribution. **Temperature scaling** can help adjust sensitivity.

## References

Below are most of the references I used for learning about LLMs.

WIP

https://blog.ezyang.com/2019/05/pytorch-internals/
https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation

- **2023**. Andrej Karpathy. *Let's build GPT: from scratch, in code, spelled out*. At 40m.
- **2023**. Meta AI. *LLaMA: Open and Efficient Foundation Language Models*. At pg 3/27.
- **2020**. Noah Shazeer. *GLU Variants Improve Transformers*. At 1/5.
- **2019**. Edward Yang. *PyTorch internals*. At 'Autograd'.

✅

- **2024**. Larry Du. *All the Activation Functions (and a history of deep learning)*.
- **2024**. J Carlos Roldán. *What is SwiGLU*.
- **2024**. Andrej Karpathy. *Let's build the GPT tokenizer*.
- **2022**. Andrej Karpathy. *Building makemore Part 5: Building a WaveNet*.
- **2022**. Andrej Karpathy. *Building makemore Part 4: Becoming a Backprop Ninja*.
- **2022**. Andrej Karpathy. *Building makemore Part 3: Activations & Gradients, BatchNorm*.
- **2022**. Andrej Karpathy. *Building makemore Part 2: MLP*.
- **2022**. Andrej Karpathy. *The spelled out intro to language modeling: building makemore*.
- **2022**. Andrej Karpathy. *The spelled out intro to neural networks and backpropagation: building micrograd*.
- **2021**. Wu et. al. *Rethinking "Batch" in BatchNorm*.
- **2016**. Oord et. al. *WaveNet: A Generative Model for Raw Audio*. At 6/15.
- **2015**. Ioffe et. al. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*.
- **2015**. Kaiming et. al. *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*.
- **2003**. Bengio et. al. *A Neural Probabilistic Language Model*.
