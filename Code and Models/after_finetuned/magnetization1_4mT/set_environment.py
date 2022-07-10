import gym
from gym import spaces
import numpy as np

import copy
from post_processing import plot_video_with_sphere, plot_video_with_sphere_2D
from MagneticTorques import MagneticTorques
from Damping import DampingUD

from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *

from magnetic_field import MagneticField

# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks):
    pass


class Environment(gym.Env):
    # Required for OpenAI Gym interface
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        final_time,
        num_steps_per_update,
        move_direction = 0, # 0,1,2 for x,y,z
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        sim_dt=1.0e-4,
        frequency = 5,
        n_elem=40,
        *args,
        **kwargs,
    ):

        super(Environment, self).__init__()

        # Integrator type
        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        self.final_time = final_time
        self.h_time_step = sim_dt  # this is a stable time step
        self.total_steps = int(self.final_time / self.h_time_step)
        self.time_step = np.float64(float(self.final_time) / self.total_steps)
        print("Total steps", self.total_steps)

        # Video speed
        self.rendering_fps = 60
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # target position
        # self.target_position = target_position

        # learning step define through num_steps_per_update
        self.num_steps_per_update = num_steps_per_update
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)
        print("Total learning steps per simulation", self.total_learning_steps)

        # magnitude of external field
        self.maximum_magnetic_field_amp = kwargs.get("maximum_magnetic_field_amplitude", 7.0) # base maximum
        self.refinement_magnetic_field_amp = kwargs.get("refinement_magnetic_field_amp", 2.0) # base maximum
        self.magnetic_field_generator = MagneticField(self.refinement_magnetic_field_amp)

        # action_space
        self.action_space = spaces.Box(
            low=-1, # low and high bound of external magnetic field
            high=1,
            shape=(2,), # x,y direction of external magnetic field
            dtype=np.float64,
        )
        self.action = np.zeros(2)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(325,),
            dtype=np.float64,
        )

        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.
        self.COLLECT_DATA_FOR_POSTPROCESSING = COLLECT_DATA_FOR_POSTPROCESSING

        self.time_tracker = np.float64(0.0)

        self.max_rate_of_change_of_activation = kwargs.get(
            "max_rate_of_change_of_activation", np.inf
        )

        self.n_elem = n_elem
        self.move_direction = int(move_direction)

    def reset(self):
        # reset magnetic field generator
        # self.policy_generator.reset()
        # self.magnetic_field_generator.reset()

        self.same_status_cntr = 0

        # reset the simulation environment
        self.simulator = BaseSimulator()

        # setting up test params
        n_elem = self.n_elem
        start = (0, 0 ,0)
        direction = np.array([1.0, 0.0, 0.0])  # rod direction: pointing +x
        normal = np.array([0.0, 1.0, 0.0])
        binormal = np.cross(direction, normal)

        density = 1.860e3 / 1000
        gravitational_acc = -9.80665e0 * 1000
        nu_1 = 150  # disable the original damping model
        E = 84500000
        poisson_ratio = 0.48

        # Set the arm properties after defining rods
        base_length = 20  # rod base length
        height = 0.8 # 磁软体的厚度
        base_radius_rod = height * np.sqrt(3)/3 #analytical: np.sqrt(3)/3
        self.base_radius_rod = base_radius_rod
        radius_tip = base_radius_rod  # radius of the arm at the tip
        radius_base = base_radius_rod  # radius of the arm at the base

        radius_along_rod = np.linspace(radius_base, radius_tip, n_elem)
        # or there will be an shape mismatch error

        # Arm is shearable Cosserat rod
        self.shearable_rod = CosseratRod.straight_rod(
            n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius=radius_along_rod,
            alpha_c = 1.0,
            density=density,
            nu=nu_1,
            youngs_modulus=E,
            poisson_ratio=poisson_ratio,
        )

        # Now rod is ready for simulation, append rod to simulation
        self.simulator.append(self.shearable_rod)

        # magnetization pattern
        magnetization = np.zeros((3, n_elem))
        period = 2 * np.pi
        for i in range(0, 32):
            magnetization[:, i:i+1] = 61.3E3 * np.array([[-1], [0], [0]])
        for i in range(32, 64):
            magnetization[:, i:i+1] = 61.3E3 * np.array([[1], [0], [0]])
        # Add magnetic torques acting on the rod for actuation
        self.external_magnetic_field = [0,0,0]
        self.external_magnetic_field[:] = self.magnetic_field_generator.reset()
        self.field_and_torque_profile_list_dir = defaultdict(list)

        # Apply magnetic torques
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MagneticTorques,
            base_length=base_length,
            magnetization = magnetization,
            external_magnetic_field=self.external_magnetic_field,
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            field_and_torque_profile_recorder=self.field_and_torque_profile_list_dir,
        )

        # Add User Defined Damping
        # level for coarse tuning, nu_ud for fine tuning
        self.simulator.add_forcing_to(self.shearable_rod).using(
            DampingUD
        )

        # Add gravitational forces
        self.simulator.add_forcing_to(self.shearable_rod).using(
            GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
        )

        # Define frictional force parameters
        origin_plane = np.array([0.0, -base_radius_rod, 0.0])
        normal_plane = np.array([0.0, 1.0, 0.0])
        slip_velocity_tol = 1e-20
        mu = 0.6  # need tuning
        k_2 = 8E4 # tuned
        nu_2 = 6000  # tuned
        kinetic_mu_array = np.array([1.0 * mu, 1.0 * mu, 1.0 * mu])
        static_mu_array = 8/6 * kinetic_mu_array

        # Add friction forces to the rod
        self.simulator.add_forcing_to(self.shearable_rod).using(
            AnisotropicFrictionalPlane,
            k=k_2,
            nu=nu_2,
            plane_origin=origin_plane,
            plane_normal=normal_plane,
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )
        """
        # Add additional contact model
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ContactPlane,
            plane_origin=origin_plane,
            plane_normal=normal_plane,
        )  
        """ 
        self.tunnel_depth = 6
        
        """
        # Add upper bound plane
        self.simulator.add_forcing_to(self.shearable_rod).using(
            AnisotropicFrictionalPlane,
            k=k_2,
            nu=nu_2,
            plane_origin=np.array([0.0, -base_radius_rod+self.tunnel_depth, 0.0]),
            plane_normal=np.array([0.0, -1.0, 0.0]),
            slip_velocity_tol=slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )
        """
        
        """
        self.simulator.add_forcing_to(self.shearable_rod).using(
            ContactPlane,
            plane_origin=np.array([0.0, -base_radius_rod+1, 0.0]),
            plane_normal=np.array([0.0, -1.0, 0.0]),
        )  
        """

        # Call back function to collect arm data from simulation
        class MagneticSoftRodBasisCallBack(CallBackBaseClass):
            """
            Call back function for Elastica rod
            """
            def __init__(
                self, step_skip: int, callback_params: dict,
            ):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["com"].append(
                        system.compute_position_center_of_mass()
                    )
                    return

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            # Collect data using callback function for postprocessing
            self.post_processing_dict_rod = defaultdict(list)
            # list which collected data will be append
            # set the diagnostics for rod and collect data
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                MagneticSoftRodBasisCallBack,
                step_skip=self.step_skip,
                callback_params=self.post_processing_dict_rod,
            )

        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

        # set state
        state = self.get_state()

        # reset current_step
        self.current_step = 0
        # reset time_tracker
        self.time_tracker = np.float64(0.0)


        # last_rod_position is used in reward engineering
        self.last_rod_position = self.shearable_rod.position_collection[..., 32].copy()

        # After resetting the environment return state information
        return state

    def sampleAction(self):
        random_action = (np.random.rand(1 * 2) - 0.5) * 2
        # shape:(2,) cuz this is 2D mag field,
        # uniform distribution from -1 to 1
        return random_action

    def get_state(self):
        # ground contact indicator
        self.ground_contanct_dir = [1.0 + 10 * np.abs(self.shearable_rod.position_collection.copy()[..., i][1]) \
                            if self.shearable_rod.position_collection.copy()[..., i][1] <= 0 else 0.0 \
                            for i in range(self.shearable_rod.position_collection.copy()[..., :][1].shape[0])]

        self.height_indicator = [0.1 * self.shearable_rod.position_collection.copy()[..., i][1]     \
                            if self.shearable_rod.position_collection.copy()[..., i][1] > 0 else 0.0       \
                            for i in range(self.shearable_rod.position_collection.copy()[..., :][1].shape[0])]
        
        
        # upper bound contact indicator
        self.upperbound_contanct_dir = [1.0 + 0.05 * np.abs(self.shearable_rod.position_collection.copy()[..., i][1] - (-2*self.base_radius_rod+self.tunnel_depth))# 可能需要改变 0.1414 的值 \ 
                            if self.shearable_rod.position_collection.copy()[..., i][1] >= -2*self.base_radius_rod+self.tunnel_depth else 0.0 \
                            for i in range(self.shearable_rod.position_collection.copy()[..., :][1].shape[0])]
        
        # angle pointer
        self.anlges_dir = [np.angle(self.shearable_rod.director_collection.copy()[..., i][2][0] + \
                        self.shearable_rod.director_collection.copy()[...,i][2][1] * 1j) \
                        for i in range(self.shearable_rod.director_collection.copy()[..., :][2][0].shape[0])]

        # angle velocity pointer
        self.angle_vel_dir = [0.25 * self.shearable_rod.omega_collection.copy()[..., i][1] \
                for i in range(self.shearable_rod.omega_collection.copy()[..., :][1].shape[0])]

        mag_amp, mag_angle = self.magnetic_field_generator.get_mag_field()
        #base_magnetic_field_angle = self.policy_generator.get_angle()

        state = [
            *self.angle_vel_dir, #64
            *self.anlges_dir, #64
            *self.height_indicator, #65
            *self.upperbound_contanct_dir, #65
            *self.ground_contanct_dir, #65
            #base_magnetic_field_angle,
            0.2 * mag_amp,
            mag_angle
            ]
        assert len(state) == 325 #261
        return state


    def step(self, action):
        self.action = np.array(action)
        self.external_magnetic_field[:] = self.magnetic_field_generator.update(0.3 * self.action) #np.add(self.policy_generator.sample_action(), self.magnetic_field_generator.update(0.3 * self.action))
        prev_state = self.get_state()
        # Do multiple time step of simulation for <one learning step>
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )
        self.current_step += 1
        state = self.get_state()

        state_difference = np.array(state[64:193]) - np.array(prev_state[64:193])
        #print(self.external_magnetic_field)
        if np.max(np.abs(state_difference)/(np.abs(np.array(prev_state[64:193])) + 0.0001)) < 0.02:
            self.same_status_cntr += 1
            if self.same_status_cntr % 10 == 0:
                print("MSR gets stuck for 0.1s.")
        else:
             self.same_status_cntr = 0

        current_mid_position = self.shearable_rod.position_collection[...,32].copy()
        forward_distance = current_mid_position - self.last_rod_position 
        reward = 0
        """
        if abs(np.linalg.norm(forward_distance)) > 0.003:
            reward += 10 * forward_distance[self.move_direction]
            #print("valid motion")
        """
        reward += 10 * forward_distance[self.move_direction]
        
        #reward -= 0.00002 * np.abs(self.shearable_rod.velocity_collection.copy()[..., 32][1])
        #print(self.upperbound_contanct_dir)

        if self.current_step%100 == 1:
            print("Current position: {}".format(np.round(current_mid_position,2)))

        done = False
        is_too_many_timesteps = False
        # Position of the rod cannot be NaN, it is not valid, stop the simulation
        invalid_values_condition = _isnan_check(self.shearable_rod.position_collection)
        if invalid_values_condition == True:
            print(" Nan detected, exiting simulation now")
            self.shearable_rod.position_collection = np.zeros(
                self.shearable_rod.position_collection.shape
            )
            reward += -100000
            state = self.get_state()
            done = True

        if self.current_step >= self.total_learning_steps:
            done = True
            is_too_many_timesteps = True
        
        """
        if self.same_status_cntr >= 100:
            print("Quit because the MSR gets stuck.")
            done = True
            #reward -= 1000
        """

        self.last_rod_position = current_mid_position.copy()
        return state, reward, done, {"ctime": self.time_tracker, "overtime": is_too_many_timesteps, "magnetic_field": self.external_magnetic_field.copy()}

    def render(self, mode="human"):
        return

    def post_processing(self, filename_video, SAVE_DATA=False, **kwargs):

        if self.COLLECT_DATA_FOR_POSTPROCESSING:

            plot_video_with_sphere_2D(
                [self.post_processing_dict_rod],
                video_name="2d_" + filename_video,
                fps=self.rendering_fps,
                step=1,
                vis2D=False,
                **kwargs,
            )

            plot_video_with_sphere(
                [self.post_processing_dict_rod],
                video_name="3d_" + filename_video,
                fps=self.rendering_fps,
                step=1,
                vis2D=False,
                **kwargs,
            )

            if SAVE_DATA == True:
                import os

                save_folder = os.path.join(os.getcwd(), "data")
                os.makedirs(save_folder, exist_ok=True)

                # Transform nodal to elemental positions
                position_rod = np.array(self.post_processing_dict_rod["position"])
                position_rod = 0.5 * (position_rod[..., 1:] + position_rod[..., :-1])

                np.savez(
                    os.path.join(save_folder, "arm_data.npz"),
                    position_rod=position_rod,
                    radii_rod=np.array(self.post_processing_dict_rod["radius"]),
                    n_elems_rod=self.shearable_rod.n_elems,
                )

                np.savez(
                    os.path.join(save_folder, "arm_activation.npz"),
                    magnetic_field=np.array(
                        self.field_and_torque_profile_list_dir["magnetic_field"]
                    ),
                )

        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )
