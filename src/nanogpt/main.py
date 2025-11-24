from nanogpt.models import *
from nanogpt import input, token, train
from nanogpt.visualization import save_plot_histogram_of_tensors, save_plot_loss, save_plot_update_to_data_ratios
import time
import torch
import torch.nn.functional as F


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available
    
    data = input.get_input_data()
    
    enc, vocabulary_size = token.get_encoder('autotokenizer', vocabulary=data['vocabulary'])
    print(f"Tokenizer vocab size: {vocabulary_size}")

    tensor = torch.tensor(
        enc.encode(data['text']),
        dtype=torch.int64  # We're using integers because each char will be mapped to an int
    )

    tensor_train, tensor_test = train.split_train_test(tensor, train_size=0.8)
    tensor_validation, tensor_test = train.split_train_test(tensor_test, train_size=0.5)

    torch.manual_seed(42)
    # highest: F32
    # high: TF32
    # medium: BF16
    torch.set_float32_matmul_precision('high')

    target_lr = 1e-3
    initial_lr = 1e-5 # Or even 0

    model = WaveNet(
        batch_size=200,
        block_size=16,
        n_layers=4,
        vocabulary_size=vocabulary_size,
        channels_embedding=2048,
        channels_gate=2048,
        channels_residual=2048,
        channels_skip=2048,
        kernel_size=2,
        learning_rate=target_lr
    )
    model = MLP(
        vocabulary_size=vocabulary_size,
        embedding_dims=2048,
        batch_size=200,  # The number of sequences to process in parallel
        block_size=10,  # The number of tokens used to predict the next token
        n_neurons=1024,
    )
    model = transformer.Transformer(transformer.Config(
        vocabulary_size,
        device,
        block_size=10,
        number_of_blocks=2,
        number_of_heads=4,
        channels_embedding=2048,
    ))
    model = GPT2(transformer.Config(
        vocabulary_size,
        device,
        batch_size=4,  # Micro-batch size
        block_size=1024,
        number_of_blocks=16,
        number_of_heads=8,
        channels_embedding=768
    ))
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"Model receptive field: {model.receptive_field()}")

    # Initial loss should be, at worst, equivalent to a random guess.
    # -log(p(x)) = -log(1 / vocabulary_size) = log(vocabulary_size)
    print(f"Initial loss should be around: {(
        max_initial_loss := torch.tensor(vocabulary_size).log().item()
    )}")
    model.loss = train.loss
    list_of_losses = [max_initial_loss]
    
    optimizer = torch.optim.AdamW(
        [{
            'params': layer[0].parameters(),
            'lr': target_lr,
            'name': key,
            'named_params': dict(layer[0].named_parameters()),
            'weight_decay': 0.01
            falta configurar weight decay por parametro com base em dimensionalidade
        } for key, layer in model.layers_and_learning_rates.items() ],
        betas=(0.9, 0.95)  # Following GPT-3
    )

    # Compile model
    model = torch.compile(model)
    
    # Steps, learning rate warm-up
    list_of_step_times = []
    list_of_tokens_per_sec = []
    steps, warmup_steps, max_steps = 0, 500, 1000 #10_000, 100
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=initial_lr / target_lr,
        end_factor=1,
        total_iters=warmup_steps
    )
    scheduler_decay = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,
        eta_min=1e-7
    )
    
    dict_of_update_to_data_ratios = {
        k: { m: [] for m, _ in v[0].named_parameters() }
        for k, v in model.layers_and_learning_rates.items()
    }

    # --- Gradient Accumulation Setup ---
    # Gradient accumulation makes larger batch sizes achievable
    # while still using a smaller micro-batch size.
    # The gradients are accumulated until the desired batch size is reached.
    # The total batch size is typically measured in tokens.
    total_batch_size = 524_288  # 2^19 which will play well with B and T
    assert (
        total_batch_size % (model.batch_size * model.block_size),
        "Assert total_batch_size is divisible by B * T"
    )
    accumulation_steps = total_batch_size // (model.batch_size * model.block_size)

    ! replace softmax by online softmax where possible
    # Make iteration count 1-based for easier modulo calculation
    for steps in range(1, max_steps + 1): 
        _time_at_step_start = time.time()
        # --- Forward Pass ---
        tensor_x, tensor_y = train.get_batch(tensor_train, batch_size=model.batch_size, block_size=model.block_size)
        # Execute the forward pass and loss computation with torch.autocast
        # which enables automatic mixed precision,
        # allowing torch to match each op to appropriate datatype,
        # yielding gains in network runtime and memory footprint.
        # Ref: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # logits: (batch, vocab_size, block_size)
            logits = model(tensor_x.to(device))

            # Cross-entropy loss is used to measure the difference between the predicted
            # distribution and the actual distribution of the target token.
            # The entropy of the correct token should be low,
            # because the model should be confident in its prediction.
            #
            # Most loss functions reduce the loss by computing its mean over the sample size.
            # Because we're accumulating the gradient, we want to keep the same behaviour
            # as if we were computing the gradient for the entire macro batch.
            # Consequently, we must normalize it over the number of accumulation steps,
            # so that once we accumulate the losses, the accumulated loss is the same.
            loss_training = model.loss(logits, tensor_y.to(device)) / accumulation_steps

        # --- Backward Pass ---
        loss_training.backward()
        
        # Measure time elapsed
        torch.cuda.synchronize()  # Wait for the GPU to finish the backward pass before timing
        list_of_step_times.append(time.time() - _time_at_step_start)
        list_of_tokens_per_sec.append(model.block_size * model.batch_size / list_of_step_times[-1])

        # Only step optimizer and scheduler after accumulating gradients for accumulation_steps batches
        if steps % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # --- Step scheduler ---
            if steps < warmup_steps:
                scheduler_warmup.step() # Step the warm-up scheduler
            else:
                scheduler_decay.step() # Step the main scheduler

            # --- Logging / Plotting ---
            # Update-to-data ratio
            tenho de refatorizar isto para ser async ou qq coisa
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
            
            # Append the train/val loss
            loss_validation = train.estimate_val_loss(
                model,
                tensor_validation,
                iterations_for_computing_loss=accumulation_steps,
                device=device
            )
            loss_mean = (loss_training.item() * accumulation_steps + loss_validation) / 2
            list_of_losses.append(loss_mean)
            ! change nomenclature: iter->step and step->micro_batch?
            print(
                f"iter: {steps // accumulation_steps}"
                f"\tstep: {steps}"
                f"\tlr: {optimizer.param_groups[0]['lr']:.7f}"
                f"\tloss: {loss_mean:.4f}"
                f"\tmean time/step: {sum(list_of_step_times) / accumulation_steps:.3f}"
                f"\ttokens/sec: {sum(list_of_tokens_per_sec) / accumulation_steps:.3f}"
            )
            list_of_step_times = []
            list_of_tokens_per_sec = []

            # Perform plotting less frequently if desired
            if (steps // accumulation_steps) % 10 == 0: # Plot every 10 effective batches
                save_plot_loss(list_of_losses)
                save_plot_update_to_data_ratios(dict_of_update_to_data_ratios)
                save_plot_histogram_of_tensors({k: v for k, v in model.named_parameters()})

        del tensor_x, tensor_y, logits, loss_training
    
    print(f"\nSmallest loss during training: {min(list_of_losses)}")


    initial_prompt = "\n"
    initial_token_ids = enc.encode(initial_prompt)
    current_token = train.pad(
        torch.tensor(initial_token_ids, dtype=torch.int64).to(device),
        model.block_size
    )  # (1, T)
    model.eval()
    for _ in range(100):
        with torch.no_grad():
            logits = model(current_token.unsqueeze(0))  # shape: (1, block_size, vocab_size) = (1, T, C)
            
            # Get logits from the last position in the sequence
            logits_last = logits[:, -1, :]  # shape: (1, vocab_size) = (1, C)
            # NOTE: Because we're assuming that the output of each neuron is the logarithm of the count (log-counts),
            # we need to exponentiate the logits to get the counts.
            # Since x is a one-hot vector (ones and zeros),
            # the neuron weights are, too, log-counts.
            # This is equivalent to applying the softmax function to the logits,
            # essentially converting them to probabilities.
            # 
            # tensor_counts = tensor_logits.exp()
            # tensor_probabilities = tensor_counts / tensor_counts.sum(dim=-1, keepdim=True)
            probs = F.softmax(logits_last / 4, dim=-1)  # shape: (1, vocab_size)

            # Top-k sampling (keep only the top 50 probabilities from the probability distribution)
            top_k_probs, top_k_indices = torch.topk(probs, 50, dim=-1)
            # Sample the next token from the probability distribution
            # (vocab_size collapses to a single token - the prediction)
            top_k_next_token = torch.multinomial(top_k_probs, num_samples=1)  # (1, 1)
            next_token = torch.gather(top_k_indices, 1, top_k_next_token)[0]  # (1, 1)

            print(enc.decode([next_token.item()]), end='', flush=True)

            # Append the token and keep only the last `block_size` tokens
            current_token = torch.cat([current_token, next_token])[-model.block_size:]

            time.sleep(0.5)


if __name__ == "__main__":
    main()
