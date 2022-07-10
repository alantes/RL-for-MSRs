import gym
import numpy as np 
import pandas as pd
from td3_torch import Agent
from utils import plot_learning_curve
from set_environment import Environment

if __name__ == '__main__':
	final_time = 10
	num_steps_per_update = 1250
	sim_dt = 0.8e-5
	maximum_magnetic_field_amplitude = 0
	refinement_magnetic_field = 4
	max_rate_of_change_of_activation = np.inf
	warmup = 2500
	env = Environment(
		final_time=final_time,
		num_steps_per_update=num_steps_per_update,
		COLLECT_DATA_FOR_POSTPROCESSING=False,
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
				n_actions=env.action_space.shape[0], warmup=0,noise=0.2)
	
	n_games = 1
	filename = 'MSR_' + str(n_games) + '.png'
	figure_file = 'plots/' + filename

	best_score = env.reward_range[0]
	score_history = []
	reward_history = []

	agent.load_models() 
	# load from disk
	steps = 0
	for i in range(n_games):
		reward_window = []
		observation = env.reset()
		done = False
		score = 0
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			reward_window.append(reward)
			steps += 1
			if info["overtime"] == False:
				agent.remember(observation, action, np.mean(reward_window[-1:]), observation_, done)
				if steps > warmup: # consistent with the Fujimoto version (https://github.com/sfujim/TD3/blob/master/main.py)
					agent.learn()
			score += reward
			reward_history.append(reward)
			observation = observation_
		score_history.append(score)
		avg_score = np.mean(score_history[-5:])
		if steps > 100000: # 改为200000
			break

		if avg_score > best_score:
			best_score = avg_score
			agent.save_models()

		print('episode ', i, 'score %.2f' % score,
				'trailing 10 games avg %.3f' % avg_score,
				'steps %d' % steps)
	
	x = [i+1 for i in range(steps)]
	plot_learning_curve(x, reward_history, figure_file)
	df = pd.DataFrame(reward_history, columns=["reward"])
	df.to_csv("./reward.csv")
	
