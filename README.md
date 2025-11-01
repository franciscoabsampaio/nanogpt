# nanogpt

Small project to study modern deep learning and LLM architectures. Here you'll find a few example models, references that I found useful, and learnings.

Training dataset was [**tiny_shakespeare**](https://www.tensorflow.org/datasets/catalog/tiny_shakespeare), with a 80/10/10 train/validation/test split.

## Tokenizer

`sentencepiece` was used to train a tokenizer on the input data with the Byte-Pair Encoding algorithm. Most settings were taken from the Llama 2 tokenization training, with the following notable options:

- Vocabulary size of 32k words. This was selected considering Shakespeare's work features a vocabulary of about 30k unique words and over 800k total words, and the "4 tokens per 3 words" rule of thumb. Additionally, over 34k tokens caused `sentencepiece` to raise an error.
- Character coverage of `0.999995`. A slightly higher percentage was considered, because Shakespeare's work is unlikely to have typos, "faulty" words, etc.
- Max token length of 27 tokens (Shakespeare's longest word is Honorificabilitudinitatibus, at 27 characters of length). Due to how uncommon it is, it's most definitely not going to be parsed as a single token, but it's an excellent Easter egg.

## Models

### LLM

Trained on Oscar 2301 (~20TB).

MEGABYTE-like.
because megabyte's performance is best on earlier tokens \cite{megabyte_2023}, strided inference helps with prediction accuracy

Additional projection to latent space in the vein of Perceiver / PerceiverIO / PerceiverAR? Not really, the PerceiverAR architecture uses, by default, an embedding latent space, where the query matrix is obtained by taking the `N` last elements of the input.
This is unlike architectures such as the WaveNet, where fixed or hand-tuned sparsity require several layers, achieving long-range communication at the cost of significant fragility.

The authors of PerceiverAR also suggest substituting dropout by a random selection of inputs from the maximum input context.

GPT-3 used the same architecture as GPT-2, with the exception of alternating dense and locally banded sparse
attention patterns in the layers of the transformer, similar to the Sparse Transformer.

#### Efficiency

Codebook learning and quantization can be used to improve storage efficiency.

### Transformer

Decoder-only transformer architecture using pre-trained tokenizer (`p50k_base` - 50k tokens), trained over 3000 iterations with AdamW and the following hyperparameters:

- `batch_size`: 200
- `block_size`: 10
- `embedding_dims`: 2048
- `number_of_heads`: 4
- `number_of_multi_head_attention_blocks`: 2
- `dropout_rate`: 0.2

Similarly to the WaveNet, learning rate was warmed up linearly, over 1500 steps, then decayed with `CosineAnnealingLR`. Starting learning rate was `1e-5`, peak was `1.33e-4`, and final was ????????.

The key differences between this implementation of a decoder-only Transformer and the default one are:

- Pre-normalization was used instead of post-normalization. Nowadays, this is the most common choice, because it helps with gradient issues before even the inputs are fed to the layers.
- In each self-attention head, dropout was applied to the attention weights (the output of the softmax), before multiplying the masked weights matrix by the values matrix, whereas in the original paper it is applied after the final projection. This change teaches the model to not rely too heavily on any single input token's value V, encouraging it to spread its attention bets and learn more robust, distributed representations. It is, essentially, a regularization strategy that is more targeted at the attention mechanism itself. Additionally, if some attention links are stochastically removed, heads are more likely to find different, complementary information.
- Gradient accumulation was kept from previous architectures.

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

### WaveNet

WaveNet using pre-trained tokenizer (Autotokenizer's `bert-base-uncased` - 30k tokens), trained over ~50000 iteration steps, with AdamW, and the following hyperparameters:

- `batch_size`: 200
- `block_size`: 16
- `embedding_dims`: 2048
- `n_layers`: 4 (number of layers in the WaveNet block)
- `channels_gate`: 2048
- `channels_residual`: 2048
- `channels_skip`: 2048
- `kernel_size`: 2

The resulting WaveNet receptive field was 16.

Layer normalization was commented out. A single output conv 1x1 layer was used - the data weren't sufficiently large to get enough signal through. Learning rate was warmed up linearly, then decayed with `CosineAnnealingLR`.

The model converged much slower than the MLP. Significant instability warranted the implementation of LR warmup and gradient accumulation. Parameters were initialized with Xavier uniform.

#### Number of Parameters

- Total: 292.855.610

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

## Training

### Dropout

Dropout is the stochastic removal of hidden units from a network. It is a simple mechanism for controlling overfit in deep learning models, and is intued to break up co-adaptations that neurons would otherwise develop - making each unit more self-dependent.

#### Recommendations

Due to its network diminishing effect (because some units are being dropped out during training), the size of the network should be increased by `n/(1-d)`, where `n` is the otherwise ideal network size, and `d` is the dropout rate.

Since dropout increases the amount of noise in the gradients, the model takes longer to converge, and increasing the learning rate by at least one order of magnitude and the momentum by 5-10% may be warranted. To prevent network weights from exploding, Sristava et. al. (2014) recommend max-norm regularization.

### Batch Normalization

Batch normalization, first introduced by Ioffe et. al., is a widely used technique for addressing initialization problems and learning instabilities, such as vanishing/exploding gradients.

However, most implementations of BatchNorm differ from the original paper, and instead use an EMA version of the batch statistics. Specifically, it weighs both past and present statistics through a momentum parameter. Despite showing good performance in later epochs of training, this can create instability and innacuracy in the earlier stages of training, due to how volatile statistics can be during warm-up, and taking past statistics into account becomes a source of instability.

On the other hand, **PreciseBN**, as described by Wu et. al., calibrates statistics periodically, allowing for higher accuracy and stability.

Many other variations of BatchNorm have been proposed in the literature. Of particular interest is **FrozenNorm**, typically applied during the last training epochs or fine-tuning. In this method, population statistics are computed, frozen, and used for the remainder of the training schedule - greatly reducing train-test inconsistencies.

#### Recommendations

There are some considerations that should be remembered when applying batch normalization in general:

- Mini-batches that are too small make aggregate batch statistics unreliable.
- Mini-batches that are too large reduce training noise, and increase fit on the training set, which decreases the model's ability to generalize. In essence, some training noise caused by insufficient batch size can deliver some form of regularization during training. This gives way to using mini-batch statistics (instead of population statistics) at inference time - bridging the train-test gap while delivering similar accuracy.
- In addition, for EMA BatchNorm, large batch sizes aggravate training instability, due to how infrequently statistics are updated.

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

🌟 mark references that were of outstanding value to me.

WIP

- **2023**. Hendrycks et. Gimpel. *Gaussian Error Linear Units (GELU)*.
- **2024**. Andrej Karpathy. *Let's reproduce GPT-2 (124M)*. At 31:00.
- **2023**. Meta AI. *LLaMA: Open and Efficient Foundation Language Models*. At pg 3/27.
- **2020**. Noah Shazeer. *GLU Variants Improve Transformers*. At 1/5.

### Fundamentals

- **2022**. Andrej Karpathy. *Building makemore Part 4: Becoming a Backprop Ninja*.
- **2022**. Andrej Karpathy. *The spelled out intro to language modeling: building makemore*.
- **2022**. Andrej Karpathy. *The spelled out intro to neural networks and backpropagation: building micrograd*.

#### Activation Functions

- **2024**. Larry Du. *All the Activation Functions (and a history of deep learning)*.
- **2024**. J Carlos Roldán. *What is SwiGLU*.
- **2024**. Su et. al. *RoFormer: Enhanced transformer with Rotary Position Embedding*.

#### PyTorch

- **2019**. Edward Yang. *PyTorch internals*.

### Architecture

#### MoE (Mixture of Experts)

- **2025**. Meta AI. *The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation*.

#### MEGABYTE

- **2023**. Yu et. al. *MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers*.

#### Transformer

- 🌟 **2023**. Andrej Karpathy. *Let's build GPT: from scratch, in code, spelled out*.
- **2020**. Open AI. *Language Models are Few-Shot Learners*.
- **2019**. Radford et. al. *Language Models are Unsupervised Multitask Learners*.
- 🌟 **2017**. Vaswani et. al. *Attention Is All You Need*.

#### WaveNet

- **2022**. Andrej Karpathy. *Building makemore Part 5: Building a WaveNet*.
- **2016**. Oord et. al. *WaveNet: A Generative Model for Raw Audio*.

#### Sequence-to-Sequence Models

- **2014**. Sutskever et. al. *Sequence to Sequence Learning with Neural Networks*.

#### MLP

- **2022**. Andrej Karpathy. *Building makemore Part 2: MLP*.
- 🌟 **2003**. Bengio et. al. *A Neural Probabilistic Language Model*.

### Training

- 🌟 **2022**. Andrej Karpathy. *Building makemore Part 3: Activations & Gradients, BatchNorm*.
- 🌟 **2021**. Wu et. al. *Rethinking "Batch" in BatchNorm*.
- **2015**. Ioffe et. al. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*.
- **2015**. Kaiming et. al. *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*.
- **2014**. Sristava et. al. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.

### Tokenization

- 🌟 **2024**. Andrej Karpathy. *Let's build the GPT tokenizer*.
- **2023**. mwatkins, Jessica Rumbelow. *SolidGoldMagikarp III: Glitch token archaeology*.
- **2023**. mwatkins, Jessica Rumbelow. *SolidGoldMagikarp II: technical details and more recent findings*.
- **2023**. Jessica Rumbelow, mwatkins. *SolidGoldMagikarp (plus, prompt generation)*.

### Inference

- **2025**. Sulbha Jain. *LLM Inferencing strategies — Review of Greedy Search and Beam Search*.

### Interpretability

- **2023**. Harrison Pim. *Privileged vs non-privileged bases in machine learning*.
- **2017**. Lundberg and Lee. *A Unified Approach to Interpreting Model Predictions*.
