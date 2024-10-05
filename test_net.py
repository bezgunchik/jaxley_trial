from main import *

cell = build_cell_type2()
net = jx.Network([cell for _ in range(10)])

layer0 = net.cell(range(2))
layer1 = net.cell(range(2, 9))
layer2 = net.cell([9])
fully_connect(layer0, layer1, IonotropicSynapse())
fully_connect(layer1, layer2, IonotropicSynapse())

net.compute_xyz()
net.rotate(180)
fig, ax = plt.subplots(1, 1)
_ = net.vis(ax=ax, detail="full", layers=[2, 7, 1], layer_kwargs={"within_layer_offset": 8, "between_layer_offset": 25})
plt.show()