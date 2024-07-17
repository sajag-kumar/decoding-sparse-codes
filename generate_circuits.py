import stim
import numpy as np
import os

def generate_circuit(code, distance, rounds, noise_model, error_rate, folder):
    
    file = f'{folder}/{code}_{distance}_{rounds}_{noise_model}_{str(error_rate)[2:5]}.stim'
    
    if code == 'surface':
        
        if noise_model == 'cl':
            
            circuit = stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                rounds=rounds,
                distance=distance,
                after_clifford_depolarization=error_rate,
                after_reset_flip_probability=error_rate,
                before_measure_flip_probability=error_rate,
                before_round_data_depolarization=error_rate)
        
        elif noise_model == 'ph':
            
            circuit = stim.Circuit.generated(
                "surface_code:unrotated_memory_z",
                rounds=rounds,
                distance=distance,
                after_clifford_depolarization=0,
                after_reset_flip_probability=0,
                before_measure_flip_probability=error_rate,
                before_round_data_depolarization=error_rate)
    
    elif code == 'repetition':
        
        if noise_model == 'cl':
            
            circuit = stim.Circuit.generated(
                "repetition_code:memory",
                rounds=rounds,
                distance=distance,
                after_clifford_depolarization=error_rate,
                after_reset_flip_probability=error_rate,
                before_measure_flip_probability=error_rate,
                before_round_data_depolarization=error_rate)
        
        elif noise_model == 'ph':
            
            circuit = stim.Circuit.generated(
                "repetition_code:memory",
                rounds=rounds,
                distance=distance,
                after_clifford_depolarization=0,
                after_reset_flip_probability=0,
                before_measure_flip_probability=error_rate,
                before_round_data_depolarization=error_rate)
    
    circuit.to_file(file)    
    
    return None

def main(code, distances, error_rates, noise_model):
    
    for distance in distances:
        folder = f'circuits/{code}_{distance}_{distance}_{noise_model}'
        if not os.path.exists(folder):
            os.makedirs(folder)
        for error_rate in error_rates:
            generate_circuit(code, distance, distance, noise_model, error_rate, folder)

if __name__ == '__main__':
    
    distances = [3, 5, 7]
    error_rates = np.arange(0.001, 0.011, 0.002)
    
    code = 'surface'
    noise_model = 'cl'
    
    main(code, distances, error_rates, noise_model)