import numpy as np
import numba
from elastica.external_forces import NoForces

class MagneticTorques(NoForces):
    def __init__(
            self,
            base_length,
            magnetization,  # (3, n_elem), Eulerian Coordiante (x,y,z)
            external_magnetic_field,  # (3,), Eulerian Coordinate (x,y,z)
            step_skip,
            max_rate_of_change_of_activation=np.inf,
            **kwargs,
    ):
        super(MagneticTorques, self).__init__()
        self.base_length = base_length
        self.magnetic_field = (
            external_magnetic_field
            if hasattr(external_magnetic_field, "__call__")
            else lambda time_v: external_magnetic_field
        )

        self.field_and_torque_profile_recorder = kwargs.get("field_and_torque_profile_recorder", None)

        self.step_skip = step_skip
        self.counter = 0  # for recording data from the muscles

        self.initial_magnetization_Euler = magnetization
        self.magnetization_Lag = np.zeros(self.initial_magnetization_Euler.shape)
        self.magnetic_field_cached = np.zeros(np.array(self.magnetic_field(0)).shape[0])

        self.max_rate_of_change_of_activation = max_rate_of_change_of_activation

        self.initial_call_flag = 0

    def apply_torques(self, system, time: np.float64 = 0.0):

        assert system.lengths.shape[0] == self.initial_magnetization_Euler.shape[
            1], "Element number is not correct in magnetization setting!"

        self.filter_activation(
            self.magnetic_field_cached[:],
            np.array((self.magnetic_field(time))),
            self.max_rate_of_change_of_activation,
        )

        # calculate the element torques along the rod
        for element_index in np.arange(self.initial_magnetization_Euler.shape[1]):
            # lagrangian vectors
            normal_vec = system.director_collection[..., element_index][0]
            binormal_vec = system.director_collection[..., element_index][1]
            tangent_vec = system.director_collection[..., element_index][2]
            if self.initial_call_flag == 0:
                self.magnetization_Lag[..., element_index] = self.Euler_to_Lag(
                    self.initial_magnetization_Euler[..., element_index], normal_vec, binormal_vec, tangent_vec)
            self.magnetic_field_cached_Lag = self.Euler_to_Lag(self.magnetic_field_cached, normal_vec, binormal_vec,
                                                               tangent_vec)
            torque_in_element_Lag = system.volume[..., element_index] * np.cross(
                self.magnetization_Lag[..., element_index], self.magnetic_field_cached_Lag)
            system.external_torques[..., element_index] += torque_in_element_Lag

        if self.counter % self.step_skip == 0:
            if self.field_and_torque_profile_recorder is not None:
                self.field_and_torque_profile_recorder["time"].append(time)
                self.field_and_torque_profile_recorder["magnetic_field"].append(
                    self.magnetic_field_cached.copy()
                )
                self.field_and_torque_profile_recorder["torque"].append(
                    system.external_torques.copy()
                )
                self.field_and_torque_profile_recorder["element_position"].append(
                    np.cumsum(system.lengths)
                )

        self.counter += 1
        self.initial_call_flag = 1

    @staticmethod
    def Euler_to_Lag(vector, normal, binormal, tangent):
        return np.array([np.dot(vector, normal), np.dot(vector, binormal), np.dot(vector, tangent)])

    @staticmethod
    @numba.njit()
    def filter_activation(signal, input_signal, max_signal_rate_of_change):
        for i in np.arange(input_signal.shape[0]):
            signal_difference = input_signal[i] - signal[i]
            signal[i] += np.sign(np.array(signal_difference)) * np.minimum(
                max_signal_rate_of_change, np.abs(signal_difference)
            )