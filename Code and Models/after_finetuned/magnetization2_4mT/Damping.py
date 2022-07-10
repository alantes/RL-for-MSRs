import numpy as np
import numba
from elastica.external_forces import NoForces

class DampingUD(NoForces):
    def __init__(
            self,
            nu_v=2e4,
            level_v=4,
            **kwargs,
    ):
        super(DampingUD, self).__init__()
        self.nu_v = nu_v
        self.level_v = level_v

    def apply_torques(self, system, time: np.float64 = 0.0):

        element_number =  system.lengths.shape[0]
        node_number = element_number + 1
        for i in range(node_number - self.level_v):
            v0 = system.velocity_collection[..., i]
            v1 = system.velocity_collection[..., i+self.level_v]
            delta_v = v1 - v0
            system.external_forces[..., i] += self.nu_v * system.lengths[i] * delta_v
            system.external_forces[..., i+self.level_v] += self.nu_v * system.lengths[i] * (-delta_v)
