from pettingzoo.mpe import simple_adversary_v1
import pettingzoo.tests.performance_benchmark as performance_benchmark


def main():
    # env = simple_adversary_v1.parallel_env()
    # performance_benchmark.performance_benchmark_parallel(env)
    env = simple_adversary_v1.env()
    performance_benchmark.performance_benchmark(env)


if __name__ == "__main__":
    main()
