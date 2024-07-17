import stim
import numpy as np
import torch
from tqdm import tqdm
from decoders import Bp, BpOsd, Nbp
from decoders.train_tools import training_loop
import csv
import time
import argparse
import os
torch.autograd.set_detect_anomaly(True)

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
        params_dict['error_rate'] = ps[4][:-5]
        
        arr.append(params_dict)
        noisy_circuits.append(arr)
    
    return noisy_circuits

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dec','--decoder', type=str)
    
    parser.add_argument('-bpm','--bp_method', default='product_sum')
    parser.add_argument('-bpi','--max_bp_iters', default=20, type=int)
    parser.add_argument('-mss','--ms_scaling_factor', default=0.0, type=float)
    parser.add_argument('-om','--osd_method', default='OSD_CS')
    parser.add_argument('-oo','--osd_order', default=20, type=int)
    
    parser.add_argument('-bs', '--batch_size', default=120, type=int)
    parser.add_argument('-l', '--layers', default=20, type=int)
    parser.add_argument('-lf', '--loss_function', default='He=s', type=str)
    parser.add_argument('-ms', '--minibatches', default=1000, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=int)
    
    parser.add_argument('-s','--shots', default=10000, type=int)
    
    parser.add_argument('-d','--in_dir', default='circuits')
    parser.add_argument('-c', '--circuit_dir', type=str)
    parser.add_argument('-o','--out_dir', default='data')
    
    args = parser.parse_args()
    
    DECODER = args.decoder
    
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
    
    IN_DIR = args.in_dir
    CIRQ_DIR = args.circuit_dir
    OUT_DIR = args.out_dir
    
    FOLDER = f'{IN_DIR}/{CIRQ_DIR}'
    if os.path.exists(f'{OUT_DIR}/decoding_data'):
        OUTFILE = f'{OUT_DIR}/decoding_data/{CIRQ_DIR}.csv'
    else:
        os.makedirs(f'{OUT_DIR}/decoding_data')
        OUTFILE = f'{OUT_DIR}/decoding_data/{CIRQ_DIR}.csv'
    
    noisy_circuits = load_circuits(FOLDER)
    
    for noisy_circuit in noisy_circuits:
        
        circuit = noisy_circuit[0]
        params = noisy_circuit[1]
        params |= {'decoder': DECODER}
         
        detector_error_model = circuit.detector_error_model(decompose_errors=False)
        sampler = detector_error_model.compile_sampler()
        syndromes, logical_flips, errors = sampler.sample(shots = SHOTS, return_errors=True)
        sampled_errors = np.argwhere(np.sum(errors,axis=1)>0).flatten()
        
        if DECODER == 'bposd':
            params |= {'bp_method': BP_METHOD, 'max_bp_iterations': MAX_ITERATIONS, 'ms_scaling_factor': SCALING_FACTOR}
            params |= {'osd_method': OSD_METHOD, 'osd_order': OSD_ORDER}
            decoder = BpOsd(model = circuit,
                            bp_method = BP_METHOD,
                            max_bp_iters = MAX_ITERATIONS,
                            ms_scaling_factor = SCALING_FACTOR,
                            osd_method = OSD_METHOD,
                            osd_order = OSD_ORDER)
            
        if DECODER == 'bp':
            params |= {'bp_method': BP_METHOD, 'max_bp_iterations': MAX_ITERATIONS, 'ms_scaling_factor': SCALING_FACTOR}
            decoder = Bp(model = circuit,
                         bp_method = BP_METHOD,
                         max_bp_iters = MAX_ITERATIONS,
                         ms_scaling_factor = SCALING_FACTOR)
            
        if DECODER in ('nbp', 'nbposd'):
            
            if os.path.exists(f'{OUT_DIR}/weights'):
                WEIGHTS_PATH = f'{OUT_DIR}/weights/{CIRQ_DIR}/{params['error_rate']}_{LAYERS}_{LOSS_FUNCTION}'
            else:
                os.makedirs(f'{OUT_DIR}/weights')
                WEIGHTS_PATH = f'{OUT_DIR}/weights/{CIRQ_DIR}/{params['error_rate']}_{LAYERS}_{LOSS_FUNCTION}'
            if os.path.exists(WEIGHTS_PATH):
                decoder = Nbp(circuit=circuit,
                              layers = LAYERS,
                              batch_size = BATCH_SIZE,
                              loss_function = LOSS_FUNCTION,
                              weights = WEIGHTS_PATH,
                              osd_method = OSD_METHOD,
                              osd_order = OSD_ORDER)
            else:
                print('Training NBP')
                print('')
                for key in params:
                    print(f'{key}: {params[key]}')
                print(f'Loss function: {LOSS_FUNCTION}')
                print(f'Layers: {LAYERS}')
                print('')
                decoder = Nbp(circuit=circuit,
                              layers = LAYERS,
                              batch_size = BATCH_SIZE,
                              loss_function = LOSS_FUNCTION,
                              osd_method = OSD_METHOD,
                              osd_order = OSD_ORDER)
                parameters = decoder.weights_llr + decoder.weights_de + decoder.marg_weights_llr + decoder.marg_weights_de + decoder.rhos + decoder.residual_weights
                optimiser = torch.optim.Adam(parameters, lr=LEARNING_RATE)
                loss = training_loop(decoder=decoder, optimizer=optimiser, mini_batches=MINIBATCHES, path=WEIGHTS_PATH)
                np.savetxt(f'{WEIGHTS_PATH}/loss.txt', loss)
                
        if DECODER in ('bp', 'bposd'):
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
        elif DECODER == 'nbp':
            params |= {'loss_function': LOSS_FUNCTION, 'layers': LAYERS}
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
                    syndrome = torch.from_numpy(syndrome).int()
                    syndrome = torch.reshape(syndrome, (1, len(syndrome), 1))
                    prediction = decoder.decode(syndrome)
                else:
                    prediction = torch.zeros_like(torch.from_numpy(logical_flip).int())
                logical_flip = torch.from_numpy(logical_flip).int()
                logical_flip = torch.reshape(logical_flip, (1, len(logical_flip)))
                lers = logical_flip != prediction
                lers = lers.detach().numpy()
                lers = np.reshape(lers, (len(lers)))
                if np.any(lers):
                    n_fails += 1
        elif DECODER == 'nbposd':
            params |= {'osd_method': OSD_METHOD, 'osd_order': OSD_ORDER}
            params |= {'loss_function': LOSS_FUNCTION, 'layers': LAYERS}
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
                    syndrome = torch.from_numpy(syndrome).int()
                    syndrome = torch.reshape(syndrome, (1, len(syndrome), 1))
                    prediction = decoder.decode_osd(syndrome)
                else:
                    prediction = torch.zeros_like(torch.from_numpy(logical_flip).int())
                logical_flip = torch.from_numpy(logical_flip).int()
                logical_flip = torch.reshape(logical_flip, (1, len(logical_flip)))
                lers = logical_flip != prediction
                lers = lers.detach().numpy()
                lers = np.reshape(lers, (len(lers)))
                if np.any(lers):
                    n_fails += 1
            
        decoding_duration = (time.perf_counter_ns() - decoding_start_time)
        dec_time_per_shot = (decoding_duration/SHOTS)/1E9
        params |= {'shots' : SHOTS, 'fails' : n_fails, 'dec_time_per_shot': dec_time_per_shot}
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