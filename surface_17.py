import torch
import stim
import os
import csv
import numpy as np
import time
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
from decoders import Nbp
from decoders.train_tools import optimization_step, training_loop
plt.style.use(['science'])

def circuit(p):
    circuit = stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                rounds=3,
                distance=3,
                after_clifford_depolarization=p,
                after_reset_flip_probability=p,
                before_measure_flip_probability=p,
                before_round_data_depolarization=p)
    return circuit

def main():

    # NBP parameters

    P = 0.005
    LEARNING_RATE = 0.001
    LAYERS = 20
    BATCH_SIZE = 120
    MINIBATCHES = 600
    CODE = 'surface_17'
    NOISE_MODEL = 'circuit'
    LOSS_FUNCTION = 'He=s'
    WEIGHTS_PATH = f'{CODE}_weights/{NOISE_MODEL}_{LOSS_FUNCTION}'
    OUTFILE = f'{CODE}_weights/{NOISE_MODEL}_{LOSS_FUNCTION}.csv'

    # Monte carlo parameters

    ERROR_SAMPLES = 1000

    # NBP Training
    
    if not os.path.exists(WEIGHTS_PATH):
        decoder = Nbp(circuit=circuit(P),
                  layers=LAYERS,
                  batch_size=BATCH_SIZE,
                  loss_function=LOSS_FUNCTION)
    

        parameters = decoder.weights_llr + decoder.weights_de + decoder.marg_weights_llr + decoder.marg_weights_de + decoder.rhos + decoder.residual_weights
        optimiser = torch.optim.Adam(parameters, lr=LEARNING_RATE)
        loss = torch.zeros(MINIBATCHES)
        idx = 0

        ps = np.arange(0.001, 0.01, 0.002)

        print('Traning NBP')
        print('')

        with tqdm(total=MINIBATCHES) as pbar:
            for _ in range(MINIBATCHES):

                sampler = decoder.dem.compile_sampler()
                syndromes, logical_flips, errors = sampler.sample(shots=int(decoder.batch_size/(len(ps)+1)), return_errors=True)

                for p in ps:
                    dem = circuit(p).detector_error_model(decompose_errors=False)
                    sampler = dem.compile_sampler()
                    temp_syndromes, temp_logical_flips, temp_errors = sampler.sample(shots=int(decoder.batch_size/(len(ps)+1)), return_errors=True)
                    syndromes = np.concatenate((syndromes, temp_syndromes), axis=0)
                    logical_flips = np.concatenate((logical_flips, temp_logical_flips), axis=0)
                    errors = np.concatenate((errors, temp_errors), axis=0)

                syndromes = torch.from_numpy(syndromes).int()
                syndromes = torch.reshape(syndromes, (len(syndromes), len(syndromes[0]), 1))
                logical_flips = torch.from_numpy(logical_flips).int()
                errors = torch.from_numpy(errors).int()
                loss[idx]= optimization_step(decoder, syndromes, errors, optimiser)

                pbar.update(1)
                pbar.set_description(f"loss {loss[idx]:.16f}")
                idx += 1

            decoder.save_weights(WEIGHTS_PATH)
        print('Training complete.\n')
        
        plt.figure(figsize=(5, 5))
        plt.plot(loss)
        plt.xlabel(r'Minibatch')
        plt.ylabel(r'$\mathcal{L}$')
        plt.savefig(f'{WEIGHTS_PATH}_loss.pdf')
        plt.clf()

    # NBP Decoding
    
    decoder = Nbp(circuit=circuit(P),
                  layers=LAYERS,
                  batch_size=1,
                  loss_function=LOSS_FUNCTION,
                  weights=WEIGHTS_PATH)

    params = {}

    ps = np.arange(0.001, 0.011, 0.001)

    for p in ps:

        print('Error rate - ', p)

        dem = circuit(p).detector_error_model(decompose_errors=False)
        sampler = dem.compile_sampler()
        syndromes, logical_flips, errors = sampler.sample(shots = ERROR_SAMPLES, return_errors=True)
        sampled_errors = np.argwhere(np.sum(errors,axis=1)>0).flatten()

        n_fails = 0

        decoding_start_time = time.perf_counter_ns()

        print('')
        print(f'{len(sampled_errors)} errors sampled.')
        print('')

        for sample in tqdm(sampled_errors):

            syndrome = syndromes[sample]
            logical_flip = logical_flips[sample]

            if np.sum(syndrome) != 0:
                syndrome = torch.from_numpy(syndrome).int()
                syndrome = torch.reshape(syndrome, (1, len(syndrome), 1))
                prediction = decoder.decode(syndrome)
            else:
                prediction = np.zeros_like(logical_flip)

            logical_flip = torch.from_numpy(logical_flip).int()
            logical_flip = torch.reshape(logical_flip, (1, len(logical_flip)))
            lers = logical_flip != prediction
            lers = lers.detach().numpy()
            lers = np.reshape(lers, (len(lers)))

            if np.any(lers):
                n_fails += 1

        decoding_duration = (time.perf_counter_ns() - decoding_start_time)
        dec_time_per_shot = (decoding_duration/ERROR_SAMPLES)/1E9

        print('')
        print(f'p_ler = {n_fails}/{ERROR_SAMPLES} = {n_fails/ERROR_SAMPLES}')
        print('')

        params |= {'error_rate': p, 'shots' : ERROR_SAMPLES, 'fails' : n_fails, 'dec_time_per_shot': dec_time_per_shot}

        if os.path.exists(OUTFILE):
                with open(OUTFILE, 'a') as file:
                    w = csv.DictWriter(file, params.keys())
                    w.writerow(params)
        else:
            with open(OUTFILE, 'w+') as file:
                w = csv.DictWriter(file, params.keys())
                w.writeheader()
                w.writerow(params)
                
if __name__ == '__main__':
    main()