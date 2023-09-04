import random
import numpy as np
import datetime
from ga.GA import run_ga

if __name__ == "__main__":
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)

    # tasks = ['Walker-v0', 'BridgeWalker-v0', 'Carrier-v0', 'Pusher-v0', 'BeamToppler-v0']
    # max_evaluations = [100, 100, 100, 100, 100]
    # tcs = [500, 500, 500, 500, 1000]

    tasks = ['BidirectionalWalker-v0', 'Pusher-v1', 'Thrower-v0', 'Climber-v1', 'UpStepper-v0', 'ObstacleTraverser-v0', 'CaveCrawler-v0']
    max_evaluations = [150, 150, 150, 150, 150, 150, 150]
    tcs = [1000, 600, 300, 600, 600, 1000, 1000]

    for i in range(len(tasks)):

        start = datetime.datetime.now()

        run_ga(
            experiment_name='GA_' + tasks[i]+'_3',
            structure_shape = (5,5),
            pop_size=25,
            max_evaluations = max_evaluations[i],
            train_iters = tcs[i],
            num_cores = 15,
            env_name = tasks[i],
        )

        end = datetime.datetime.now()

        print(start)
        print('--------')
        print(end)