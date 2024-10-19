import numpy as np

from main import *
from load_data import load_dataset
from jax import vmap

def flow():
	layer0_size = 128
	layer1_size = 10
	layer2_size = 1
	
	input_cell = build_cell_type1()
	intermidiate_cell = build_cell_type2()
	total_cells_num = 139
	cells = ([input_cell for _ in range(layer0_size)] +
	         [intermidiate_cell for _ in range(layer1_size + layer2_size)])
	net = jx.Network(cells)
	
	

	
	layer0 = net.cell(range(layer0_size))
	layer1 = net.cell(range(layer0_size, layer0_size + layer1_size))
	layer2 = net.cell(range(layer0_size + layer1_size, layer0_size + layer1_size + layer2_size))
	fully_connect(layer0, layer1, IonotropicSynapse())
	fully_connect(layer1, layer2, IonotropicSynapse())
	
	# Channels
	net.insert(Na())
	net.insert(K())
	net.insert(Leak())
	
	# Visualize net
	net.compute_xyz()
	net.rotate(180)
	fig, ax = plt.subplots(1, 1)
	_ = net.vis(ax=ax, detail="full", layers=[layer0_size, layer1_size, layer2_size], layer_kwargs={"within_layer_offset": 15, "between_layer_offset": 25})
	# _ = net.vis(ax=ax, detail="full", layers=[layer0_size, layer1_size, layer2_size], layer_kwargs={"within_layer_offset": 8, "between_layer_offset": 25})
	plt.show()
	
	# Currently returns 1 sample
	# TODO handle multimple inputs
	x, y, d = load_dataset(0, 0)
	
	i_amp = 0.05
	delta_t = 0.025
	pixel_duration = 1
	data_stimuli = None
	for row_index, row in enumerate(x):
		t_max = len(row)
		currents = jx.datapoint_to_step_currents(i_delay=0.0, i_dur=1.0, i_amp=row * i_amp, delta_t=delta_t, t_max=t_max)
		overall_current = sum_row_currents(currents, pixel_duration, delta_t)
		# TODO to debug the difference between overall_current_np and overall_current
		overall_current_np = np.repeat(row * i_amp, pixel_duration / delta_t)
		if row_index % 30 == 0:
			visualize_row_currents(overall_current, t_max=currents.shape[0], delta_t=delta_t, row_index=row_index)
			visualize_row_currents(np.append(overall_current_np, [0]), t_max=currents.shape[0], delta_t=delta_t, row_index=row_index)
		data_stimuli = net.cell(row_index).branch(0).loc(0.0).data_stimulate(overall_current, data_stimuli=data_stimuli)
		if row_index == 50:
			break
	net.delete_recordings()
	net.cell(total_cells_num - 1).branch(0).loc(0.0).record()
	
	recs = jx.integrate(net, data_stimuli=data_stimuli)
	o = 0

def visualize_row_currents(overall_current, t_max, delta_t, row_index):
	num_time_steps = int(t_max / delta_t) + 1  # Adjusting to include t=0
	time_vec = np.arange(0, num_time_steps * delta_t, delta_t)
	fig, ax = plt.subplots(1, 1, figsize=(12, 6))
	
	ax.plot(time_vec, np.array(overall_current), label='Continuous Current',
	        alpha=0.5)  # Label every 100th pixel for clarity
	
	ax.set_title(f'Step Currents for Each Pixel in a {row_index} Row as a Function of Time')
	ax.set_xlabel('Time (ms)')
	ax.set_ylabel('Current (nA)')
	ax.grid()
	ax.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.15, 1))
	plt.tight_layout()
	plt.show()

def sum_row_currents(row_currents, pixel_duration, delta_t):
	overall_current = np.zeros(row_currents.shape[1])
	# Shift each pixel's current by its index in the row
	for i in range(row_currents.shape[0]):
		# Compute the time shift (in terms of indices) for this pixel
		start_idx = int(i * pixel_duration / delta_t)
		end_idx = start_idx + int(pixel_duration / delta_t)  # Each pixel's current spans the duration of t_max
		
		# Add the current to the overall vector with the appropriate shift
		overall_current[start_idx:end_idx] += row_currents[i, :int(pixel_duration / delta_t)]
	return overall_current


if __name__ == '__main__':
	flow()
 

# def simulate(params, inputs):
#     currents = jx.datapoint_to_step_currents(i_delay=1.0, i_dur=1.0, i_amp=inputs / 10, delta_t=0.025, t_max=10.0)
#
#     data_stimuli = None
#     data_stimuli = net.cell(0).branch(2).loc(1.0).data_stimulate(currents[0], data_stimuli=data_stimuli)
#     data_stimuli = net.cell(1).branch(2).loc(1.0).data_stimulate(currents[1], data_stimuli=data_stimuli)
#
#     return jx.integrate(net, params=params, data_stimuli=data_stimuli)

# batched_simulate = vmap(simulate, in_axes=(None, 0))