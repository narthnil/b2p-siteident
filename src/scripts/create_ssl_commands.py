import itertools
from os import path

if __name__ == "__main__":
    hyperparams = {
        "lambda-u": [0, 0.01, 0.1, 1.0],
        "ema-decay": [0.5, 0.75, 0.9],
        "T": [0.5, 0.75],
        "alpha": [0.25, 0.5, 0.75],
        "tile_size": [300, 1200],
        "model": ["resnet50"],
    }
    gpu_allocation = {}
    # devbox
    offset = len(gpu_allocation)
    num_gpus = 4
    num_jobs_per_gpu = 2
    for gpu in range(num_gpus):
        for job in range(num_jobs_per_gpu):
            gpu_allocation[offset + gpu * num_jobs_per_gpu + job] = gpu
    # geodeep
    offset = len(gpu_allocation)
    num_gpus = 8
    num_jobs_per_gpu = 3
    for gpu in range(num_gpus):
        for job in range(num_jobs_per_gpu):
            gpu_allocation[offset + gpu * num_jobs_per_gpu + job] = gpu
    # springroll
    offset = len(gpu_allocation)
    num_gpus = 4
    num_jobs_per_gpu = 3
    for gpu in range(num_gpus):
        for job in range(num_jobs_per_gpu):
            gpu_allocation[offset + gpu * num_jobs_per_gpu + job] = gpu
    print(gpu_allocation)

    CMD = (
        "python train_ssl.py --out {out} --model {model} --tile_size {tile} "
        "--manualSeed {seed} --use_several_test_samples "
        "--data_version {version} --lambda-u {lambdau} --ema-decay {ema} "
        "--alpha {alpha} --gpu {gpu} | tee logs/{out}.txt"
    )
    MODEL_NAME = "{model}_ema-{ema}_lmdu-{lmdu}_T-{T}_a-{a}_tile-{tile}"
    sorted_names = sorted(list(hyperparams.keys()))
    jobs = {job_name: [] for job_name in gpu_allocation}
    current_job = 0
    for data_version in ["v1", "v2"]:
        vfpath = "results/ssl-{}".format(data_version)
        combinations = list(itertools.product(
            *[hyperparams[k] for k in sorted_names]))
        # print("Number of combinations: {}".format(len(combinations)))
        for combination in combinations:
            # print(combination)
            for i, manualSeed in enumerate([1, 10, 42]):
                model_name = MODEL_NAME.format(
                    model=combination[sorted_names.index("model")],
                    ema=combination[sorted_names.index("ema-decay")],
                    lmdu=combination[sorted_names.index("lambda-u")],
                    T=combination[sorted_names.index("T")],
                    a=combination[sorted_names.index("alpha")],
                    tile=combination[sorted_names.index("tile_size")])
                version_name = "v{version}".format(version=i)
                out = path.join(vfpath, model_name, version_name)
                cmd = CMD.format(
                    out=out,
                    model=combination[sorted_names.index("model")],
                    tile=combination[sorted_names.index("tile_size")],
                    seed=manualSeed,
                    version=data_version,
                    lambdau=combination[sorted_names.index("lambda-u")],
                    ema=combination[sorted_names.index("ema-decay")],
                    alpha=combination[sorted_names.index("alpha")],
                    gpu=gpu_allocation[current_job]
                )
                jobs[current_job].append(cmd)
                current_job += 1
                current_job = current_job % len(jobs)
    for k, cmds in jobs.items():
        with open("scripts/ssl_job_{}_gpu_{}.sh".format(k, gpu_allocation[k]),
                  "w+") as f:
            f.write("\n".join(cmds))