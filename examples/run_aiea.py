from SAEA.Parameters import get_par
from SAEA.AIEA import run_aiea

if __name__ == "__main__":

    tasks = [
        'BridgeWalker-v0'
    ]

    for i in range(len(tasks)):

        env_name = tasks[i]
        max_eva, tc = get_par(env_name)

        run_aiea(
            pop_size=20,
            structure_shape=(5, 5),
            experiment_name='AIEA_' + env_name,
            max_evaluations=max_eva,
            train_iters=tc,
            num_cores=15,
            env_name=env_name
        )
