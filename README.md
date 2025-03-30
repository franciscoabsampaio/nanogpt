# nanogpt

Go back to:

- Intro to neural networks and backpropagation: DONE.
- makemore:
  - Part 1: 55:13.
- Let's build GPT: currently at 32:29.
- Let's build the GPT tokenizer.
- Train sentencepiece model.

## Statistical Assumptions

### Softmax

Softmax is grounded on the following statistical assumptions:

- **The distribution is categorical.**
- **Independence among classes.** If certain classes are dependent (e.g. `puppy` and `dog`), softmax may be overconfident in assigning a single class the highest score, when multiple are likely.
- **Scores follow a Gumbel distribution.** It implicitly assumes that input scores are noisy estimates of a latent preference value.
- **Scale invariance.** Its outputs change significantly when the input logits are changed. Higher scores get disproportionately more weight. If scores are on different scales, softmax may overemphasize large scores, and becomes almost a step function. If logits are small, softmax approaches a uniform distribution. **Temperature scaling** can help adjust sensitivity.
