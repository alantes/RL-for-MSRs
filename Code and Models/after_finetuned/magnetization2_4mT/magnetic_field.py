import numpy as np

class MagneticField():
	"""designed moving forward policy"""
	def __init__(self, maximum):
		self.component_x = 0.0
		self.component_y = 0.0
		self.angle = 0.0
		self.amplitude = 0.0
		self.maximum = maximum

	def update(self, actions): 
		actions = np.array(actions)
		self.component_x = np.clip(self.component_x + actions[0], -self.maximum, self.maximum)
		self.component_y = np.clip(self.component_y + actions[1], -self.maximum, self.maximum)
		self.angle = np.angle(self.component_x + self.component_y * 1j)
		self.amplitude = np.linalg.norm([self.component_x, self.component_y])
		if self.amplitude > self.maximum:
			ratio = self.maximum/self.amplitude
			self.component_x *= ratio
			self.component_y *= ratio
			self.amplitude *= ratio
		return [self.component_x, self.component_y, 0]

	def get_mag_field(self):
		return self.amplitude, self.angle

	def get_components(self):
		return [self.component_x, self.component_y, 0]

	def reset(self):
		self.component_x = 0.0
		self.component_y = 0.0
		self.angle = 0.0
		self.amplitude = 0.0
		return [self.component_x, self.component_y, 0]

	def random_reset(self):
		self.component_x = np.random.uniform(-0.7 * self.maximum, 0.7 * self.maximum)
		self.component_y = np.random.uniform(-0.7 * self.maximum, 0.7 * self.maximum)
		self.angle = np.angle(self.component_x + self.component_y * 1j)
		self.amplitude = np.linalg.norm([self.component_x, self.component_y])
		assert self.amplitude < self.maximum, f"amplitude should be smaller than {self.maximum}"
		return [self.component_x, self.component_y, 0]

