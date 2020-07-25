import numpy
from matplotlib import pyplot as plt
import copy

# Assumption : Fuel tank runs out atleast once

MAX_FUEL = 20 #km
COST_BOUND = 100000 #h
END = [0,0]
MIN_COST_PATH = None
MIN_REFUEL_COST = 0 #h/km
VELOCITY = 1 #km/h

class cost_point:
	def __init__(self,fuel,cost,rate):
		self.fuel = fuel
		self.cost = cost
		self.refuel_cost = rate

	def print(self, caption = ''):
		print('.'+caption+'.')
		print('fuel : ',self.fuel)
		print('cost : ',self.cost)
		print('rate : ',self.refuel_cost)

	def data(self):
		return ((self.fuel, self.cost, self.refuel_cost))

def plot_cost(cost_points):
	fuels = [c.fuel for c in cost_points]
	costs = [c.cost for c in cost_points]
	rates = [c.refuel_cost for c in cost_points]
	next_costs = [c+1*r for (c,r) in zip(costs, rates)]
	next_fuels = [f+1 for f in fuels]
	plt.plot(fuels, costs, 'ko')
	for i in range(len(fuels)):
		plt.plot([fuels[i],next_fuels[i]],[costs[i],next_costs[i]])
	plt.plot()
	plt.show()

def update_min_cost(path):
	global COST_BOUND, MIN_COST_PATH
	if (not cost_too_high(path.cost)):
		COST_BOUND = path.cost[0].cost
		MIN_COST_PATH = path
		print('NEW GOOD PATH FOUND AT ',COST_BOUND,'!!!!!')
		path.plot()

def cost_too_high(cost_vector, dist_to_end = 0):
	# idx = 0
	global COST_BOUND
	# check if lower_bound(this cost) > achieved min cost
	lower_bound_cost = cost_vector[0].cost + dist_to_end/VELOCITY + dist_to_end*MIN_REFUEL_COST
	return(lower_bound_cost > COST_BOUND)

	# if (not COST_POINT_BOUND):
	# 	return False

	# for bound in COST_POINT_BOUND:
	# 	try:
	# 		while (bound.fuel > cost_vector[idx+1].fuel):
	# 			idx = idx + 1
	# 	except:
	# 		pass
	# 	if(abs(bound.fuel - cost_vector[idx].fuel) < 0.01):
	# 		if(bound.cost > cost_vector[idx].cost + cost_to_end):
	# 			return False
	# 	else:
	# 		b = (bound.cost - cost_vector[idx].cost + cost_to_end)/(bound.fuel - cost_vector[idx].fuel)
	# 		if(b > cost_vector[idx].refuel_cost):
	# 			return False

	# return True

class pathType:
	def __init__(self, node):
		self.points = [node.location]
		self.cost = [cost_point(0,0,0)]
		self.node = node
		self.cost_history = [{'cost':self.cost, 'refuel start':0, 'location':node.location, 'distance from prev': 0}]

	def set_cost(self,cost):
		self.cost = cost

	def add_point(self,p):
		self.points.append(p)

	def move_to(self,neighbor):
		distance = neighbor[1]
		(self.cost, refuel_start) = move(self.cost, distance, VELOCITY, neighbor[0].rate)
		self.cost_history.append({'cost':self.cost,'refuel start':refuel_start, 'location':neighbor[0].location, 'distance from prev':distance})
		self.points.append(neighbor[0].location)
		self.node = neighbor[0]

	def plot(self):
		print('Printing path...')
		for hist in self.cost_history:
			print('x:',hist['location'][0],'; y:',hist['location'][1])
			print('Start refuel for over: ',hist['refuel start'])
			plot_cost(hist['cost'])
		print('Cost :',self.cost[0].cost)
		
	def print_points(self):
		print('.',len(self.cost_history),'.')
		for hist in self.cost_history:
			print('(',hist['location'][0],', ',hist['location'][1],')')
		print('..')

	def is_pruneable(self):
		return cost_too_high(self.cost, self.node.distance_to_end)

	def get_instructions(self):
		instructions = []
		fuel_needed = 0
		for data in reversed(self.cost_history):
			refuel_amt = max(0, fuel_needed - data['refuel start'])
			instructions.insert(0, refuel_amt)
			fuel_needed = fuel_needed + data['distance from prev'] - refuel_amt
		pretty_print(instructions,self.cost_history)

def pretty_print(instructions, history):
	fuel = 0
	print('To reach the destination efficiently, follow these instructions ..........')
	for (i,h) in zip(instructions,history):
		print('Drive ',h['distance from prev'], ' to get to station')
		print('At node : ', h['location'][0],', ',h['location'][1])
		print('Fill fuel for ', i, ' km')
		fuel = fuel-h['distance from prev']+i
		print('Final fuel level is ', fuel)
		print('.....Moving on.....')

class node:
	def __init__(self, rate, x, y):
		self.rate = rate
		self.location = numpy.array([x, y])
		self.distance_to_end = numpy.linalg.norm(self.location-END)
		self.neighbors = []

	def add_neighbor(self, node):
		node_dist = numpy.linalg.norm(self.location - node.location)
		if (node_dist > MAX_FUEL):
			return False
		self.neighbors.append((node, node_dist))
		return True

	def print_location(self):
		print('(',self.location[0],', ',self.location[1],')')

def move(cost_initial, distance, velocity, refuel_cost):
	# % distance = km
	# % velocity = km/h
	# % refuel rate = h/km

	if(distance > MAX_FUEL):
		print('ERROR cannot reach neighbor!!!')

	distance_cost = distance/velocity
	
	# temporary variables
	first_element_flag = True
	possible_segment = cost_point(0,0,0)

	# outputs
	cost_final = []
	refuel_start = MAX_FUEL

	cost_shifted = [cost_point(cost.fuel-distance,cost.cost+distance_cost,cost.refuel_cost) for cost in cost_initial]	
	cost_point_temp = None
	while len(cost_shifted):
		if cost_shifted[0].fuel > 0:
			break
		cost_point_temp = cost_shifted.pop(0)

	cost_temp_at_zero = -cost_point_temp.fuel * cost_point_temp.refuel_cost + cost_point_temp.cost
	cost_shifted.insert(0,cost_point(0, cost_temp_at_zero, cost_point_temp.refuel_cost))

	for cost in cost_shifted:
		cost_final.append(cost)
		if(cost_final[-1].refuel_cost > refuel_cost):
			cost_final[-1].refuel_cost = refuel_cost
			refuel_start = min(refuel_start, cost_final[-1].fuel)
			break

	if(cost_final[-1].refuel_cost != refuel_cost):
		last_segment_fuel = MAX_FUEL-distance
		last_segment_cost = (last_segment_fuel-cost_final[-1].fuel)*cost_final[-1].refuel_cost + cost_final[-1].cost
		last_segment = cost_point(last_segment_fuel, last_segment_cost, refuel_cost)
		cost_final.append(last_segment)
		refuel_start = min(refuel_start, last_segment.fuel)

	return (cost_final,refuel_start)

# def move(cost_initial, distance, velocity, refuel_cost):
# 	# % distance = km
# 	# % velocity = km/h
# 	# % refuel rate = km/h

# 	if(distance > MAX_FUEL):
# 		print('ERROR cannot reach neighbor!!!')

# 	cost_final = []
# 	s = len(cost_initial)
# 	distance_cost = distance/velocity
# 	first_element_flag = True
# 	possible_segment = cost_point(0,0,0)
# 	refuel_start = MAX_FUEL


# 	for i in range(s):
# 		cost_initial[i].fuel -= distance
# 		cost_initial[i].cost += distance_cost
# 		if(first_element_flag):
# 			if(cost_initial[i].fuel > 0):
# 				cost_final.append(possible_segment)
# 				first_element_flag = False
# 			else:
# 				cost_at_zero = -cost_initial[i].fuel*cost_initial[i].refuel_cost + cost_initial[i].cost
# 				possible_segment = cost_point(0, cost_at_zero, cost_initial[i].refuel_cost)
# 				if (possible_segment.refuel_cost > refuel_cost):
# 					possible_segment.refuel_cost = refuel_cost
# 					refuel_start = 0
# 				continue
# 		cost_final.append(cost_initial[i])
# 		if(cost_final[-1].refuel_cost > refuel_cost):
# 			cost_final[-1].refuel_cost = refuel_cost
# 			refuel_start = min(refuel_start, cost_final[-1].fuel)
# 			break
# 	if(first_element_flag):
# 		cost_final.append(possible_segment)
# 	if(cost_final[-1].refuel_cost != refuel_cost):
# 		last_segment_fuel = MAX_FUEL-distance
# 		last_segment_cost = (MAX_FUEL-distance-cost_final[-1].fuel)*cost_final[-1].refuel_cost + cost_final[-1].cost
# 		last_segment = cost_point(last_segment_fuel, last_segment_cost, refuel_cost)
# 		cost_final.append(last_segment)
# 		refuel_start = min(refuel_start, last_segment.fuel)

# 	return (cost_final,refuel_start)

def test_move():
	cost = []
	cost.append(cost_point(0,0,0))
	cost.append(cost_point(5,0,2))
	cost.append(cost_point(15,20,3))
	[new_cost, refuel_start] = move(cost, 10, 1, 3)
	for c in new_cost:
		c.print()

def create_graph():
	graph = []

	# arg1 is refuel cost in km/h

	node_s = node(0, 0, 0)
	node_a = node(0.5, 0, 5)
	node_b = node(1.0, 10, 0)
	node_c = node(1.5, 10, 5)
	node_d = node(2.0, 20, 5)
	node_e = node(1000, 22, 5)

	global MIN_REFUEL_COST
	MIN_REFUEL_COST = 5

	node_s.add_neighbor(node_a)
	node_s.add_neighbor(node_b)

	node_a.add_neighbor(node_s)
	node_a.add_neighbor(node_c)

	node_b.add_neighbor(node_s)
	node_b.add_neighbor(node_c)
	node_b.add_neighbor(node_d)

	node_c.add_neighbor(node_a)
	node_c.add_neighbor(node_b)
	node_c.add_neighbor(node_e)

	node_d.add_neighbor(node_b)
	node_d.add_neighbor(node_e)


	graph.append(node_s) #Start
	graph.append(node_a)
	graph.append(node_b)
	graph.append(node_c)
	graph.append(node_d)
	graph.append(node_e) #End

	return graph

def BnB():
	graph = create_graph()
	start_node = graph[0]
	end_node = graph[-1]
	global END
	END = end_node.location

	paths = [pathType(start_node)]

	while (len(paths)):
		print('Paths in list : ',len(paths))
		path = paths[0]
		if (not path.is_pruneable()):
			for n in path.node.neighbors:
				new_path = copy.deepcopy(path)
				new_path.move_to(n)
				if (n[0] == end_node):
					update_min_cost(new_path)
				else:
					paths.append(new_path)

		paths.remove(path)
	MIN_COST_PATH.get_instructions()

def main():
	BnB()

if __name__ == "__main__":
	main()

		

