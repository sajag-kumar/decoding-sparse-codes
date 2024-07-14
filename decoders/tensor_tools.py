import torch
import stim
import numpy as np
from typing import List
from dataclasses import dataclass

@dataclass
class DEM_Matrices:
    check_matrix: torch.Tensor
    logical_matrix: torch.Tensor
    llrs: torch.Tensor

def DEM_to_matrices(DEM: stim.DetectorErrorModel) -> DEM_Matrices:

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
    
    check_matrix = torch.tensor(check_matrix, dtype=torch.int)
    logical_matrix = torch.tensor(logical_matrix, dtype=torch.int)
    priors = torch.tensor(priors, dtype=torch.float32)
    
    llrs = torch.log((1 - priors) / priors)
    
    return DEM_Matrices(
        check_matrix=check_matrix,
        logical_matrix=logical_matrix,
        llrs=llrs
    )