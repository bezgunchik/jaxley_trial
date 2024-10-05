from utils import *
from flow_runner import FlowRunner

	
def build_cell_type1():
	comp = jx.Compartment()
	soma = jx.Branch(comp, nseg=1)
	cell = jx.Cell(soma, parents=[-1])
	return cell

def build_cell_type2():
	comp = jx.Compartment()
	soma = jx.Branch(comp, nseg=1)
	branch = jx.Branch(comp, nseg=1)
	branches = [soma] + [branch for i in range(4)]
	cell = jx.Cell(branches, parents=[-1, 0, 0, 0, 0])
	return cell

def build_cell_type3():
	comp = jx.Compartment()
	soma = jx.Branch(comp, nseg=1)
	branch = jx.Branch(comp, nseg=1)
	branches = [soma] + [branch for i in range(12)]
	cell = jx.Cell(branches, parents=[-1, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
	return cell

if __name__ == '__main__':
	cell = build_cell_type3()
	runner = FlowRunner(cell=cell)
	runner.run()
