import os

# Set the environment variable for JAX platform to CPU
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from jax import default_backend

print("Current JAX platform:", default_backend())


import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import jit

import jaxley as jx
from jaxley.channels import Na, K, Leak
from jaxley.synapses import IonotropicSynapse
from jaxley.connect import fully_connect

class FlowRunner:

	def __init__(self):
		self.build_cell()
		self.init_stimulation_params()
	
	
	def build_cell(self):
		# Build the cell.
		comp = jx.Compartment()
		branch = jx.Branch(comp, nseg=4)
		self.cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
		
	def init_stimulation_params(self):
		self.dt = 0.025
		self.delay = 1.0
		self.dur = 1.0
		self.amp = 0.1
		self.t_max = 10.0
	
	def insert_channels(self):
		# Insert channels.
		if self.cell:
			self.cell.insert(Leak())
			self.cell.branch(0).insert(Na())
			self.cell.branch(0).insert(K())
	
	def visualize_morphology(self):
		# Visualize the morphology.
		if self.cell:
			self.cell.compute_xyz()
			cell_fig, cell_ax = plt.subplots(1, 1, figsize=(4, 4))
			_ = self.cell.vis(ax=cell_ax, col="k")
			_ = self.cell.branch(0).vis(ax=cell_ax, col="r")
			_ = self.cell.branch(0).loc(0.0).vis(ax=cell_ax, col="b")
			plt.show()
	
	def stimulate(self):
		# Stimulate.
		current = jx.step_current(i_delay=self.delay, i_dur=self.dur, i_amp=self.amp, delta_t=self.dt, t_max=self.t_max)
		self.cell.delete_stimuli()
		self.cell.branch(0).loc(0.0).stimulate(current)
		return current
	
	def visualize_stimulation(self, current):
		# Stimulation visualization
		time_vec = np.arange(0, self.t_max + self.dt, self.dt)
		current_fig, ax = plt.subplots(1, 1, figsize=(4, 2))
		_ = plt.plot(time_vec, current)
		plt.show()
	
	def record(self):
		# Record.
		self.cell.delete_recordings()
		self.cell.branch(0).loc(0.0).record("v")
		self.cell.branch(3).loc(1.0).record("v")
	
	# Simulate and plot.
	def visualize_cell_responce(self):
		fig, response_ax = plt.subplots(1, 1)
		_ = response_ax.plot(self.voltages[0], c="b")
		_ = response_ax.plot(self.voltages[1], c="orange")
		plt.plot(self.voltages.T)
		
		plt.title("Voltage vs Time")
		plt.xlabel("Time (ms)")
		plt.ylabel("Voltage (mV)")
		plt.grid()
		plt.show()
	
	def run(self):
		self.insert_channels()
		self.visualize_morphology()
		current = self.stimulate()
		self.visualize_stimulation(current)
		self.record()
		self.voltages = jx.integrate(self.cell)
		self.visualize_cell_responce()

if __name__ == '__main__':
	runner = FlowRunner()
	runner.run()