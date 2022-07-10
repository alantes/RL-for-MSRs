import gym
import numpy as np 
import pandas as pd
from td3_torch import Agent
from utils import plot_learning_curve
from set_environment import Environment

final_time = 10
num_steps_per_update = 200
sim_dt = 5e-5
maximum_magnetic_field_amplitude = 10
refinement_magnetic_field = 10

max_rate_of_change_of_activation = np.inf
env = Environment(
    final_time=final_time,
    num_steps_per_update=num_steps_per_update,
    COLLECT_DATA_FOR_POSTPROCESSING=True,
    move_direction=0,
    sim_dt=sim_dt,
    n_elem=64,
    max_rate_of_change_of_activation=max_rate_of_change_of_activation,
    maximum_magnetic_field_amplitude=maximum_magnetic_field_amplitude, # maximum value of base
	refinement_magnetic_field_amp=refinement_magnetic_field # maximum value of refinement
)

n_games = 1
filename = 'MSR_' + str(n_games) 
figure_file = 'plots/' + filename + '.png'

best_score = env.reward_range[0]
score_history = []

steps = 0

magnetic_field = []
for i in range(n_games):
	observation = env.reset()
	done = False
	score = 0
	episode_steps = 0
	while not done:
		action = [1, 1]
		observation_, reward, done, info = env.step(action)
		magnetic_field.append(info["magnetic_field"])
		steps += 1
		episode_steps += 1
		score += reward
		observation = observation_
	score_history.append(score)
	avg_score = np.mean(score_history[-10:])

	print('episode ', i, 'score %.2f' % score,
			'trailing 10 games avg %.3f' % avg_score,
			'steps %d' % steps)
env.post_processing(
    filename_video="video" + filename + ".mp4", SAVE_DATA=True,
)
df = pd.DataFrame(magnetic_field, columns=["mag_x", "mag_y", "mag_z"])
df.to_csv("./magnetic_field.csv")