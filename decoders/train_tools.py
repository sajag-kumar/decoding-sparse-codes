import torch
import numpy as np
from tqdm import tqdm

def sample_errors(sampler, n_errors):
    e = 0
    syndromes, logical_flips, errors = sampler.sample(shots=n_errors, return_errors=True)
    while e < n_errors:
        errorless_shots = np.argwhere(np.sum(errors, axis=1)==0).flatten()
        syndromes = np.delete(syndromes, errorless_shots, axis=0)
        logical_flips = np.delete(logical_flips, errorless_shots, axis=0)
        errors = np.delete(errors, errorless_shots, axis=0)
        e = errors.shape[0]
        if e < n_errors:
            temp_syndromes, temp_logical_flips, temp_errors = sampler.sample(shots=(n_errors-e), return_errors=True)
            syndromes = np.concatenate((syndromes, temp_syndromes), axis=0)
            logical_flips = np.concatenate((logical_flips, temp_logical_flips), axis=0)
            errors = np.concatenate((errors, temp_errors), axis=0)
    return syndromes, logical_flips, errors

def optimization_step(decoder, syndromes, errors, optimizer: torch.optim.Optimizer):
   loss = decoder.forward(syndromes, errors)
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   return loss.detach()

def training_loop(decoder, optimizer, mini_batches, path):
    loss = torch.zeros(mini_batches)
    idx = 0
    with tqdm(total=mini_batches) as pbar:
        for _ in range(mini_batches):
            sampler = decoder.dem.compile_sampler()
            # syndromes, logical_flips, errors = sampler.sample(shots=decoder.batch_size, return_errors=True)
            syndromes, logical_flips, errors = sample_errors(sampler=sampler, n_errors=decoder.batch_size)
            syndromes = torch.from_numpy(syndromes).int()
            syndromes = torch.reshape(syndromes, (len(syndromes), len(syndromes[0]), 1))
            logical_flips = torch.from_numpy(logical_flips).int()
            errors = torch.from_numpy(errors).int()
            loss[idx]= optimization_step(decoder, syndromes, errors, optimizer)
            pbar.update(1)
            pbar.set_description(f"loss {loss[idx]:.16f}")
            idx += 1
        decoder.save_weights(path)
    print('Training complete.\n')
    return loss