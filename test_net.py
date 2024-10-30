import numpy as np

from main import *
from load_data import load_dataset
from jax import vmap, value_and_grad, debug, grad
from sklearn.model_selection import train_test_split
import optax
from tensorflow.data import Dataset
from jax import lax


def image_row_to_current(row):
	overall_current_jnp = jnp.repeat(row * i_amp, int(pixel_duration / delta_t))
	return overall_current_jnp
	
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

visualize_net = False

i_amp = 0.005
delta_t = 0.025
pixel_duration = 1

digit_width = 100
batch_size = 1


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

net.delete_recordings()
net.cell(total_cells_num - 1).branch(0).loc(0.0).record()

net.delete_trainables()
net.make_trainable("radius")
parameters = net.get_parameters()
opt_params = parameters

# Visualize net
if visualize_net:
	net.compute_xyz()
	net.rotate(180)
	fig, ax = plt.subplots(1, 1)
	_ = net.vis(ax=ax, detail="full", layers=[layer0_size, layer1_size, layer2_size], layer_kwargs={"within_layer_offset": 15, "between_layer_offset": 25})
	plt.show()

# Currently returns 1 sample
X, Y, d = load_dataset()


def preprocess_labels(labels):
	result = jnp.roll(labels, digit_width) # Shift 100 pixels right
	result = jnp.repeat(result * i_amp, int(pixel_duration / delta_t), axis=1) # Convert to current
	return jnp.insert(result, 0, 0, axis=1) # Add t0 value

# The following two functions are for different length inputs (not efficient)
# def preprocess_label(labels):
# 	for i in range(len(labels)):
# 		labels[i] = image_row_to_current(np.roll(labels[i], digit_width))
# 	return labels

# def preprocess_label(label):
# 	label = image_row_to_current(jnp.roll(label, digit_width))
# 	return label

Y = preprocess_labels(Y)

X_train, X_test, Y_train, Y_test, d_train, d_test = train_test_split(X, Y, d, test_size=100, random_state=99)

train_dataset = Dataset.from_tensor_slices((X_train, Y_train))
train_dataloader = train_dataset.batch(batch_size)
test_dataset = Dataset.from_tensor_slices((X_test, Y_test))
test_dataloader = test_dataset.batch(batch_size)


def simulate(params, input):
	data_stimuli = None
	for row_index, row in enumerate(input):
		overall_current_jnp = image_row_to_current(row)
		data_stimuli = net.cell(row_index).branch(0).loc(0.0).data_stimulate(overall_current_jnp,
		                                                                     data_stimuli=data_stimuli)
		# if jnp.isnan(data_stimuli[0]).any():
		# 	print("NaN in data_stimuli after data_stimulate")
		# 	debug.breakpoint()
	return jx.integrate(net, params=params, data_stimuli=data_stimuli)


batched_simulate = vmap(simulate, in_axes=(None, 0))
## version for single sample (only for debugging)
# batched_simulate = simulate

def loss_fn(opt_params, input, label):
	recording = batched_simulate(opt_params, input)
	if jnp.isnan(recording).any():
		print("NaN in recording output")
		debug.breakpoint()  # Inspect variable values here
	result = jnp.mean(jnp.abs((recording - label)))
	# result = jnp.mean((recording - label) ** 2) # MSE version
	return result

grad_fn = jit(grad(loss_fn, argnums=0))
loss_and_grad_fn = jit(value_and_grad(loss_fn, argnums=0))

# Define the optimizer.
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Adjust the threshold as needed (start with 1.0 and fine-tune)
    optax.adam(learning_rate=0.0001)
)
opt_state = optimizer.init(opt_params)

for epoch in range(10):
	epoch_loss = 0.0
	for batch_index, batch in enumerate(train_dataloader):
		stimuli = batch[0].numpy()
		labels = batch[1].numpy()
		# For now, gradient is always Nan
		loss, gradient = loss_and_grad_fn(opt_params, stimuli, labels)
		
		# loss = loss_fn(opt_params, stimuli, labels)
		# gradient = grad_fn(opt_params, stimuli, labels)
		# debug.breakpoint()
		# if jnp.isnan(loss) or jnp.isnan(gradient[0]['radius']).any():
		# 	print("NaN in loss or gradient")
		# 	debug.breakpoint()
		
		# Optimizer step.
		updates, opt_state = optimizer.update(gradient, opt_state)
		opt_params = optax.apply_updates(opt_params, updates)
		epoch_loss += loss
		print(f"epoch {epoch}, batch {batch_index}, loss {epoch_loss}")

