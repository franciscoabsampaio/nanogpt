from nanogpt.models import *
from nanogpt.input import get_input_data, TINYSHAKESPEARE
from nanogpt.token import get_encoder, sentencepiece
from nanogpt.train import split_train_test, get_batch
from nanogpt.visualization import save_plot_histogram_of_tensors, save_plot_loss, save_plot_update_to_data_ratios
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_tokenizer():
    options = dict(
        # input
        input=TINYSHAKESPEARE,
        input_format="text",
        # output
        model_prefix="shakespeare35k",  # output filename prefix
        # algorithm
        model_type="bpe",  # Byte-Pair Encoding
        vocab_size=32000,
        # normalization
        normalization_rule_name='identity',
        remove_extra_whitespaces=False,
        input_sentence_size=2_000_000,  # max number of training sentences
        max_sentence_length=4192,  # max number of bytes per sequence
        seed_sentencepiece_size=1_000_000,
        shuffle_input_sentence=True,
        # rare word treatment
        character_coverage=0.999995,
        byte_fallback=True,
        # merge_rules
        split_digits=True,
        split_by_unicode_script=True,
        split_by_whitespace=True,
        max_sentencepiece_length=27,  # Shakespeare's longest word: Honorificabilitudinitatibus
        add_dummy_prefix=True,
        allow_whitespace_only_pieces=True,
        # special tokens
        unk_id=0,  # Unknown tokens
        bos_id=1,  # Beginning of sentence
        eos_id=2,  # End of sentence
        pad_id=-1,
        # systems
        num_threads=os.cpu_count(),  # use all system resources
    )
    sentencepiece.SentencePieceTrainer.train(**options)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available
    
    data = get_input_data()
    
    enc, vocabulary_size = get_encoder('autotokenizer', vocabulary=data['vocabulary'])
    print(f"Tokenizer vocab size: {vocabulary_size}")

    tensor = torch.tensor(
        enc.encode(data['text']),
        dtype=torch.int64  # We're using integers because each char will be mapped to an int
    )

    tensor_train, tensor_test = split_train_test(tensor, train_size=0.8)
    tensor_validation, tensor_test = split_train_test(tensor_test, train_size=0.5)

    torch.manual_seed(42)

    model = MLP(
        vocabulary_size=vocabulary_size,
        embedding_dims=2048,
        batch_size=200,  # The number of sequences to process in parallel
        block_size=10,  # The number of tokens used to predict the next token
        n_neurons=1024,
    )
    model = WaveNet(
        batch_size=200,
        block_size=15,
        n_layers=3,
        vocabulary_size=vocabulary_size,
        channels_embedding=128,
        channels_gate=128,
        channels_residual=128,
        channels_skip=256,
        kernel_size=3,
    )
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Model receptive field: {model.receptive_field()}")

    # Initial loss should be, at worst, equivalent to a random guess.
    # -log(p(x)) = -log(1 / vocabulary_size) = log(vocabulary_size)
    print(f"Initial loss shouldn't be greater than: {(
        max_initial_loss := torch.tensor(vocabulary_size).log().item()
    )}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    list_of_losses = [max_initial_loss]
    
    target_lr = 1.5e-2 # Your target learning rate
    optimizer = torch.optim.AdamW(
        [{
            'params': layer[0].parameters(),
            'lr': target_lr,
            'name': key,
            'named_params': dict(layer[0].named_parameters()),
        } for key, layer in model.layers_and_learning_rates.items() ],
        weight_decay=0.01,
    )
    
    # Steps, learning rate warm-up
    steps, warmup_steps, max_steps = 0, 500, 150000
    initial_lr = 1e-5 # Or even 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=1e-5
    )
    
    dict_of_update_to_data_ratios = {
        k: { m: [] for m, _ in v[0].named_parameters() }
        for k, v in model.layers_and_learning_rates.items()
    }

    # --- Gradient Accumulation Setup ---
    accumulation_steps = 8 # Calculate effective batch size: 10 * 8 = 80. Adjust as needed based on memory.

    # Make iteration count 1-based for easier modulo calculation
    for steps in range(1, max_steps + 1): 
        # --- Learning Rate Adjustment ---
        if steps <= warmup_steps:
            # Linear warmup calculation
            lr_scale = min(1.0, float(steps) / float(warmup_steps))
            current_lr = initial_lr + lr_scale * (target_lr - initial_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # --- Forward Pass ---
        tensor_x, tensor_y = get_batch(tensor_train, batch_size=model.batch_size, block_size=model.block_size)
        # logits: (batch, vocab_size, block_size)
        logits = model(tensor_x.to(device))
        # Reshape for CrossEntropyLoss:
        # Need (Batch * BlockSize, VocabSize) for logits
        # Need (Batch * BlockSize) for targets
        B, C, T = logits.shape
        logits_reshaped = logits.reshape(B * T, C) 
        targets_reshaped = tensor_y.reshape(B * T).to(device) # Use reshape or view

        # Calculate loss over all positions
        # Normalize the loss contribution over accumulation_steps
        loss = criterion(logits_reshaped, targets_reshaped) / accumulation_steps

        # --- Backward Pass ---
        loss.backward()
        # Store the average loss across the accumulation cycle for more stable logging
        current_loss_value = loss.item() * accumulation_steps

        # Only step optimizer and scheduler after accumulating gradients for accumulation_steps batches
        if steps % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # --- Step scheduler AFTER optimizer step, only AFTER warmup ---
            if steps > warmup_steps:
                scheduler.step() # Step the main scheduler

            # --- Logging / Plotting ---
            # Update-to-data ratio
            for param_group in optimizer.param_groups:
                for m, p in param_group['named_params'].items():
                    if p.grad is not None:
                        # Append the update-to-data ratio for the parameter
                        dict_of_update_to_data_ratios[param_group['name']][m].append(
                            (param_group['lr'] * p.grad.std() / (p.data.std() + 1e-5)).log10().item()
                        )
                    else:
                        print(f"Warning: Parameter name '{m}' not found in ratio dict for group '{param_group['name']}'")
    
            # Reset for next accumulation cycle
            optimizer.zero_grad(set_to_none=True)
            
            # Append the accumulation cycle loss
            list_of_losses.append(current_loss_value)
            print(
                f"Iter: {steps // accumulation_steps}"
                f"\tStep: {steps}"
                f"\tLR: {optimizer.param_groups[0]['lr']:.6f}"
                f"\tLoss: {current_loss_value:.4f}"
            )

            # Perform plotting less frequently if desired
            if (steps // accumulation_steps) % 10 == 0: # Plot every 10 effective batches
                save_plot_histogram_of_tensors({k: v for k, v in model.named_parameters()})
                save_plot_loss(list_of_losses)
                save_plot_update_to_data_ratios(dict_of_update_to_data_ratios)
        
        del tensor_x, tensor_y, logits, logits_reshaped, targets_reshaped, loss
    
    print(f"\nSmallest loss during training: {min(list_of_losses)}")

    # with torch.no_grad():
        # tensor_x, tensor_y = get_batch(tensor_train, batch_size=len(tensor_train) // 5, block_size=model.block_size)
        # logits = model(tensor_x.to(device))
        # loss = criterion(logits, tensor_y[:, -1].to(device))
        # print(f"training loss: {loss.item()}")

        # tensor_x, tensor_y = get_batch(tensor_validation, batch_size=len(tensor_validation), block_size=model.block_size)
        # logits = model(tensor_x.to(device))
        # loss = criterion(logits, tensor_y[:, -1].to(device))
        # print(f"validation loss: {loss.item()}")

    import time

    tensor_x, tensor_y = get_batch(tensor_validation, batch_size=model.batch_size, block_size=model.block_size)
    current_token = tensor_x[0, :]

    model.to('cpu')
    model.eval()

    for _ in range(100):
        with torch.no_grad():
            logits = model(current_token.unsqueeze(0))  # shape: (1, vocab_size, block_size)
            
            # Get logits from the last position in the sequence
            logits_last = logits[:, :, -1]  # shape: (1, vocab_size)
            probs = F.softmax(logits_last, dim=-1)  # shape: (1, vocab_size)

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(probs[0], num_samples=1)[0]

            print(enc.decode([next_token.item()]), end='', flush=True)

            # Append the token and keep only the last `block_size` tokens
            current_token = torch.cat([current_token, next_token.unsqueeze(0)])[-model.block_size:]

            time.sleep(0.5)
    return

    # NOTE: Because we're assuming that the output of each neuron is the logarithm of the count (log-counts),
    # we need to exponentiate the logits to get the counts.
    # Since x is a one-hot vector (ones and zeros),
    # the neuron weights are, too, log-counts.
    # This is equivalent to applying the softmax function to the logits,
    # essentially converting them to probabilities.
    # 
    # tensor_counts = tensor_logits.exp()
    # tensor_probabilities = tensor_counts / tensor_counts.sum(dim=-1, keepdim=True)
    tensor_probabilities = f.softmax(tensor_logits.float(), dim=1)

    # NOTE: GOAL: Maximize likelihood of the data wrt model parameters
    # equivalent to mazimizing the log-likelihood of the data (because log is monotonic)
    # equivalent to minimizing the negative log-likelihood of the data
    # equivalent to minimizing the average negative log-likelihood of the data
    # 
    # NOTE: The logarithm of 0 is infinity and thus undefined,
    # so we need to add a small value (epsilon) to avoid division by zero
    # This is called model smoothing or Laplace smoothing
    # 
    # NOTE: We can then compute the negative log loss,
    # which is what we want to minimize.
    # A measure of the variance of the weights can be added to the loss function,
    # effectively minimizing the counts of the most frequent tokens,
    # forcing the model to learn probabilities that are more uniform,
    # snoothing the model and preventing overfitting.
    # This is called regularization.
    tensor_negative_log_loss = (
        -(tensor_probabilities[0, torch.arange(block_size), tensor_y] + 1e-8).log().mean()
        + 0.01 * (tensor_neuron_weights ** 2).mean()  # 0.01 is the regularization strength, a hyperparameter
    )
    print(tensor_negative_log_loss.item())

    tensor_neuron_weights.grad = None
    tensor_negative_log_loss.backward()

    tensor_neuron_weights.data += -0.5 * tensor_neuron_weights.grad

    # forward
    tensor_logits = tensor_x_onehot @ tensor_neuron_weights  # log-counts
    tensor_probabilities = f.softmax(tensor_logits.float(), dim=1)
    tensor_negative_log_loss = -tensor_probabilities[0, torch.arange(block_size), tensor_y].log().mean()
    
    print(tensor_negative_log_loss.item())

    tensor_x_batch, tensor_y_batch = get_batch(tensor_train, batch_size=batch_size, block_size=block_size)
    
    print([(tensor_x_batch[i, j], tensor_y_batch[i, j]) for i in range(batch_size) for j in range(block_size)])

    m = BigramLanguageModel(vocabulary_size)
    logits, loss = m(tensor_x_batch, tensor_y_batch)
    print(loss)


if __name__ == "__main__":
    main()
