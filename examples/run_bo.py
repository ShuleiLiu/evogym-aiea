import random
import numpy as np
import datetime

from bo.run import run_bo

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    start = datetime.datetime.now()

    best_robot, best_fitness = run_bo(
        experiment_name='bo_brw',
        structure_shape=(5, 5),
        pop_size= 25,
        max_evaluations=250,
        train_iters=1000,
        num_cores=10,
    )
    end = datetime.datetime.now()

    print(start)
    print('----------')
    print(end)
    print('Best robot:')
    print(best_robot)
    print('Best fitness:', best_fitness)