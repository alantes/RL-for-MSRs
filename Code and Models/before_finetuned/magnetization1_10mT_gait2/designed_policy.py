import numpy as np

class DesignedPolicy():
	"""designed moving forward policy"""
	def __init__(self, period, amplitude):
		self.timestep = 0
		self.angle = 0
		self.period = period # period means how many total timesteps per period
		self.amplitude = amplitude

	def sample_action(self): # sample actions ranging from -1 to 1
		self.timestep += 1
		component_x = np.cos(2*3.14/self.period*self.timestep) * self.amplitude
		component_y = np.sin(2*3.14/self.period*self.timestep) * self.amplitude
		self.angle = np.angle(component_x + component_y * 1j)
		return [component_x, component_y, 0]

	def get_angle(self):
		return self.angle

	def reset(self):
		self.timestep = 0
		self.angle = 0