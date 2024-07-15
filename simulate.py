import stim
import numpy as np
import torch
from tqdm import tqdm
from decoders import Bp, BpOsd, Nbp
from decoders.train_tools import optimization_step, training_loop
from decoders.tensor_tools import DEM_to_matrices
import csv
import time
import argparse
import os

def load_circuits(folder):
    
    noisy_circuits =[]
    
    stim_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    
    for file in stim_files:
        arr = []
        arr.append(stim.Circuit.from_file(f'{folder}/{file}'))
        
        params_dict = {}
        ps = file.split('_')
        params_dict['code'] = ps[0]
        params_dict['distance'] = ps[1]
        params_dict['rounds'] = ps[2]
        params_dict['noise_model'] = ps[3]
        
        error_rate = ps[4][:-5]
        params_dict['error_rate'] = error_rate
        
        arr.append(params_dict)
        
        noisy_circuits.append(arr)
    
    return noisy_circuits

torch.autograd.set_detect_anomaly(True)

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dec','--decoder', default='bposd', type=str)
    parser.add_argument('-e','--experiment', type=str)
    
    parser.add_argument('-bpm','--bp_method', default='product_sum')
    parser.add_argument('-bpi','--max_bp_iters', default=19, type=int)
    parser.add_argument('-mss','--ms_scaling_factor', default=0.0, type=float)
    parser.add_argument('-om','--osd_method', default='OSD_CS')
    parser.add_argument('-oo','--osd_order', default=20, type=int)
    
    parser.add_argument('-bs', '--batch_size', default=120, type=int)
    parser.add_argument('-l', '--layers', default=20, type=int)
    parser.add_argument('-lf', '--loss_function', default='He=s', type=str)
    parser.add_argument('-ms', '--minibatches', default=100, type=int)
    parser.add_argument('-lr', '--learning_rate', default=100, type=int)
    
    parser.add_argument('-s','--shots', default=1000, type=int)
    
    parser.add_argument('-d','--dir',default='circuits')
    parser.add_argument('-o','--out',default='data')
    
    args = parser.parse_args()
    
    DECODER = args.decoder
    EXPERIMENT = args.experiment
    
    BP_METHOD = args.bp_method
    MAX_ITERATIONS = args.max_bp_iters
    SCALING_FACTOR = args.ms_scaling_factor
    OSD_METHOD = args.osd_method
    OSD_ORDER = args.osd_order
    
    BATCH_SIZE = args.batch_size
    LAYERS = args.layers
    LOSS_FUNCTION = args.loss_function
    MINIBATCHES = args.minibatches
    LEARNING_RATE = args.learning_rate
    
    SHOTS = args.shots
    
    IN_DIR = args.dir
    OUT_DIR = args.out
    
    FOLDER = f'{IN_DIR}/{EXPERIMENT}'
    OUTFILE = f'{OUT_DIR}/{EXPERIMENT}.csv'
    
    noisy_circuits = load_circuits(FOLDER)
    
    for noisy_circuit in noisy_circuits:
        
        circuit = noisy_circuit[0]
        params = noisy_circuit[1]
        
        detector_error_model = circuit.detector_error_model(decompose_errors=False)
        sampler = detector_error_model.compile_sampler()
        syndromes, logical_flips, errors = sampler.sample(shots = SHOTS, return_errors=True)
        sampled_errors = np.argwhere(np.sum(errors,axis=1)>0).flatten()
        
        if DECODER == 'bposd':
            
            decoder = BpOsd(model = circuit,
                            bp_method = BP_METHOD,
                            max_bp_iters = MAX_ITERATIONS,
                            ms_scaling_factor = SCALING_FACTOR,
                            osd_method = OSD_METHOD,
                            osd_order = OSD_ORDER)
            
        if DECODER == 'bp':
            
            decoder = Bp(model = circuit,
                         bp_method = BP_METHOD,
                         max_bp_iters = MAX_ITERATIONS,
                         ms_scaling_factor = SCALING_FACTOR)
            
        if DECODER == 'nbp':
            
            WEIGHTS_PATH = f'data/weights/{params['code']}_{params['distance']}_{params['rounds']}_{params['noise_model']}'
            
            if os.path.exists(WEIGHTS_PATH):
            
                decoder = Nbp(circuit=circuit,
                              layers = LAYERS,
                              batch_size = BATCH_SIZE,
                              loss_function = LOSS_FUNCTION,
                              weights = WEIGHTS_PATH)
                
            else:
                
                print('Training NBP')
                
                decoder = Nbp(circuit=circuit,
                              layers = LAYERS,
                              batch_size = BATCH_SIZE,
                              loss_function = LOSS_FUNCTION)
                
                parameters = decoder.weights_llr + decoder.weights_de + decoder.marg_weights_llr + decoder.marg_weights_de + decoder.rhos + decoder.residual_weights
                optimiser = torch.optim.Adam(parameters, lr=LEARNING_RATE)
                loss = training_loop(decoder=decoder, 
                                     optimizer=optimiser, 
                                     mini_batches=MINIBATCHES, 
                                     path=WEIGHTS_PATH)
            
        n_fails = 0
        
        decoding_start_time = time.perf_counter_ns()
        
        print('')
        for key in params:
            print(f'{key}: {params[key]}')
        print(f'{len(sampled_errors)} errors sampled.')
        print('')
        
        for sample in tqdm(sampled_errors):
            
            syndrome = syndromes[sample]
            logical_flip = logical_flips[sample]
            
            if np.sum(syndrome) != 0:
                prediction = decoder.decode(syndrome)
            else:
                prediction = np.zeros_like(logical_flip)
            
            lers = logical_flip != prediction
            
            if np.any(lers):
                n_fails += 1

        decoding_duration = (time.perf_counter_ns() - decoding_start_time)
        dec_time_per_shot = (decoding_duration/SHOTS)/1E9

        params |= {'decoder': DECODER, 'shots' : SHOTS, 'fails' : n_fails, 'dec_time_per_shot': dec_time_per_shot}
        
        print('')
        print(f'p_ler = {n_fails}/{SHOTS} = {n_fails/SHOTS}')
        print('')
        
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