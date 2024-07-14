import torch
from tqdm import tqdm
from .nbp import Nbp

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
            syndromes, logical_flips, errors = sampler.sample(shots=decoder.batch_size, return_errors=True)
            
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