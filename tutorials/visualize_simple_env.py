import gym
from evogym import sample_robot
from evogym import get_full_connectivity
import numpy as np
from PIL import Image
import imageio
from examples.ppo.envs import make_vec_envs

# import envs from the envs folder and register them
import envs

if __name__ == '__main__':

    # create a random robot
    # body, connections = sample_robot((5,5))
    # body = np.array([[0,0,2,0,2],[4,3,3,3,4],[4,0,1,0,0],[2,0,4,2,0],[4,4,4,4,0]])
    body = np.array([[0,2,4,4,0],[3,0,0,4,4],[2,3,4,4,3],[3,3,4,4,4],[0,0,0,1,2]])
    connections = get_full_connectivity(body)

    env = make_vec_envs('BridgeWalker-v0', body, 1, 1, None, None, device='cpu', allow_early_resets=False)
    env.reset()

    img_array = env.render(mode='rgb_array')

    print(img_array)

    # make the SimpleWalkingEnv using gym.make and with the robot information
    # env = gym.make('BridgeWalker-v0', body=body)
    # env.reset()
    # img = env.render(mode='img')
    # # img = env.render(
    # #     'human',
    # #     verbose=True,
    # #     hide_background=False,
    # #     hide_grid=True,
    # #     hide_edges=False,
    # #     hide_voxels=False)
    # imageio.save('tmp1.png', img)
    # # sum_reward = 0
    # # imgs = []
    # env.close()

    # step the environment for 500 iterations
    # for i in range(500):
    #     action = np.random.uniform(low=0.6, high=1.6, size=(env.action_space.shape[0]))
    #     ob, reward, done, info = env.step(action)
    #
    #     img = env.render(mode='img')
    #     imgs.append(img)
    #
    #     sum_reward = sum_reward + np.array(reward)
    #
    #     if done:
    #         env.reset()
    #
    # env.close()
    #
    # imageio.mimsave('tmp2.gif', imgs, duration=(1 / 50.0))
    #
    # print(reward)

    # action = env.action_space.sample()
