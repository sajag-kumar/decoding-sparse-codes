from ldpc.bposd_decoder import BpOsdDecoder
import numpy as np
import stim
from .tools import DEM_to_matrices

    
class BpOsd:
    
    def __init__(self,
                 model: stim.Circuit,
                 bp_method : str = 'product_sum',
                 max_bp_iters : int = 19,
                 ms_scaling_factor : int = 0,
                 osd_method : str = 'OSD_CS',
                 osd_order : int = 20) -> None:
               
            
        self._model = model.detector_error_model(decompose_errors=False)
                                
        self._matrices = DEM_to_matrices(self._model)
        priors = self._matrices.priors

        self._decoder = BpOsdDecoder(self._matrices.check_matrix,
                                      channel_probs = priors,
                                      bp_method = bp_method,
                                      max_iter = max_bp_iters,
                                      ms_scaling_factor = ms_scaling_factor,
                                      osd_method = osd_method,
                                      osd_order = osd_order)
                
    def decode(self, syndrome : np.ndarray) -> np.ndarray:
            
        correction = self._decoder.decode(syndrome)
        prediction = (self._matrices.logical_matrix @ correction) % 2
        
        return prediction