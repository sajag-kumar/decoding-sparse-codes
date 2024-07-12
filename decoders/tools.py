import stim
import numpy as np
from typing import List
from dataclasses import dataclass
from scipy.sparse import csc_matrix

@dataclass
class DEM_matrices:
    check_matrix: csc_matrix
    logical_matrix: csc_matrix
    priors: np.ndarray

def DEM_to_matrices(DEM: stim.DetectorErrorModel) -> DEM_matrices:

    priors = np.zeros(DEM.num_errors)
    check_matrix = np.zeros((DEM.num_detectors, DEM.num_errors))
    logical_matrix = np.zeros((DEM.num_observables, DEM.num_errors))
    
    e = 0
    
    for instruction in DEM.flattened():
        
        if instruction.type == "error":
            
            detectors: List[int] = []
            logicals: List[int] = []
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    detectors.append(t.val)
                elif t.is_logical_observable_id():
                    logicals.append(t.val)

            priors[e] = p
            check_matrix[detectors, e] = 1
            logical_matrix[logicals, e] = 1
            
            e += 1
            
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()
    
    check_matrix = csc_matrix(check_matrix)
    logical_matrix = csc_matrix(logical_matrix)
    
    return DEM_matrices(check_matrix = check_matrix,
                        logical_matrix = logical_matrix,
                        priors = priors)