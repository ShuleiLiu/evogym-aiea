from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot
import os
import numpy as np

# create a EvoWorld object with environment data and visualize it

world = EvoWorld.from_json(os.path.join('world_data', 'simple_walker_env.json'))
# world.pretty_print()

# add a randomly sampled 5x5 robot to the world

robot_structure, robot_connections = sample_robot((5, 5))
world.add_from_array(
    name='robot', 
    structure=robot_structure, 
    x=3, 
    y=1, 
    connections=robot_connections)

world.pretty_print()

# create a simulation using our world object

sim = EvoSim(world)
sim.reset()

# a viewer object will allow us to visualize our simulation

viewer = EvoViewer(sim)
viewer.track_objects('robot', 'box')

# we put it all together in this loop in which we sample a random action for our simulation, step the simulation, and render it

# for i in range(500):
#
#     tmp_actions = np.random.uniform(low = 0.6, high = 1.6, size=(sim.get_dim_action_space('robot'),))
#
#     sim.set_action('robot', tmp_actions)
#     sim.step()
#     sim_time = sim.get_time()
#     agent_pos = sim.pos_at_time(sim_time)
#     box_pos = sim.object_pos_at_time(sim_time, 'box')
#     print('time:')
#     print(sim_time)
#     print('agent_position')
#     print(agent_pos)
#     print('box_position')
#     print(box_pos)
#     viewer.render('screen')

