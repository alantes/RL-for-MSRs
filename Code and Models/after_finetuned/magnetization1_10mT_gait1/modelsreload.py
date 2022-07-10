import gym
import numpy as np 
import pandas as pd
from td3_torch import Agent
from utils import plot_learning_curve
from set_environment import Environment

final_time = 10
num_steps_per_update = 1250 # may need to change to 1000
sim_dt = 0.8e-5
maximum_magnetic_field_amplitude = 0
refinement_magnetic_field = 10
max_rate_of_change_of_activation = np.inf
env = Environment(
	final_time=final_time,
	num_steps_per_update=num_steps_per_update,
	COLLECT_DATA_FOR_POSTPROCESSING=True,
	move_direction=0,
	frequency=5,
	sim_dt=sim_dt,
	n_elem=64,
	max_rate_of_change_of_activation=max_rate_of_change_of_activation,
	maximum_magnetic_field_amplitude=maximum_magnetic_field_amplitude, # maximum value of base
	refinement_magnetic_field_amp=refinement_magnetic_field # maximum value of refinement
)

agent = Agent(alpha=3e-4, beta=3e-4,
			input_dims=env.observation_space.shape, tau=0.005,
			env=env, batch_size=256, layer1_size=256, layer2_size=256,
			n_actions=env.action_space.shape[0], warmup=0,noise=0)

n_games = 1
index_last_game = n_games - 1
filename = 'MSR_' + str(n_games) 
figure_file = 'plots/' + filename + '.png'

agent.load_models() # load from disk
steps = 0
magnetic_field = []
reward_history = []
for i in range(n_games):
	observation = env.reset()
	done = False
	score = 0
	while not done:
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		steps += 1
		score += reward
		if i == index_last_game:
			reward_history.append(reward)
			magnetic_field.append(info["magnetic_field"])
		observation = observation_

	print('episode ', i, 'score %.2f' % score,
			'steps %d' % steps)
env.post_processing(
    filename_video="video" + filename + ".mp4", SAVE_DATA=True,
)
df = pd.DataFrame(magnetic_field, columns=["mag_x", "mag_y", "mag_z"])
df.to_csv("./magnetic_field.csv")
df_reward = pd.DataFrame(reward_history, columns=["reward"])
df_reward.to_csv("./reward_reload.csv")
