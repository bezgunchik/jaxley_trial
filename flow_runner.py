from utils import *

class FlowRunner:
	
	def __init__(self, cell=None, stimulation_params=None):
		if cell:
			self.cell = cell
		else:
			self.build_default_cell()
		if stimulation_params:
			self.stimulation_params = stimulation_params
		else:
			self.init_default_stimulation_params()
		
	def build_default_cell(self):
		# Build the cell.
		comp = jx.Compartment()
		branch = jx.Branch(comp, nseg=4)
		self.cell = jx.Cell(branch, parents=[-1, 0, 0, 1, 1])
		
	def init_default_stimulation_params(self):
		self.stimulation_params = {
			'dt': 0.025,
			'delay': 1.0,
			'dur': 1.0,
			'amp': 0.1,
			't_max': 10.0
		}
	
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
		current = jx.step_current(i_delay=self.stimulation_params['delay'],
		                          i_dur=self.stimulation_params['dur'],
		                          i_amp=self.stimulation_params['amp'],
		                          delta_t=self.stimulation_params['dt'],
		                          t_max=self.stimulation_params['t_max'])
		self.cell.delete_stimuli()
		self.cell.branch(0).loc(0.0).stimulate(current)
		return current
	
	def visualize_stimulation(self, current):
		# Stimulation visualization
		time_vec = np.arange(0,
		                     self.stimulation_params['t_max'] + self.stimulation_params['dt'],
		                     self.stimulation_params['dt'])
		current_fig, ax = plt.subplots(1, 1, figsize=(4, 2))
		_ = plt.plot(time_vec, current)
		plt.show()
	
	def record(self):
		# Record.
		self.cell.delete_recordings()
		self.cell.branch(0).loc(0.0).record("v")
		# self.cell.branch(3).loc(1.0).record("v")
	
	# Simulate and plot.
	def visualize_cell_responce(self):
		fig, response_ax = plt.subplots(1, 1)
		_ = response_ax.plot(self.voltages[0], c="b")
		# _ = response_ax.plot(self.voltages[1], c="orange")
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

