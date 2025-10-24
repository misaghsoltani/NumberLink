"""Benchmark environment creation time for NumberLink single vs vector envs.

Compares creating a single `NumberLinkRGB-v0` environment vs a `NumberLinkRGBVectorEnv`.


Usage:
    python examples/bench_create_envs.py --num-envs 1 --iters 50

The script prints timing statistics for environment construction (mean, std, min, max).
"""

from __future__ import annotations

import argparse
import statistics
import sys  # Import sys
import time
from typing import TYPE_CHECKING

import gymnasium as gym

from numberlink import GeneratorConfig, NumberLinkRGBVectorEnv, RenderConfig, VariantConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import LiteralString


def timeit(func: Callable[[], object]) -> float:
    """Run callable once and return elapsed seconds using perf_counter."""
    t0: float = time.perf_counter()
    func()
    t1: float = time.perf_counter()
    return t1 - t0


def make_single_env(gen_cfg: GeneratorConfig, var_cfg: VariantConfig, render_cfg: RenderConfig) -> None:
    """Create a single env and close it to measure construction time only."""
    env = gym.make("NumberLinkRGB-v0", generator=gen_cfg, variant=var_cfg, render_config=render_cfg, render_mode=None)
    env.close()


def make_vector_env(num_envs: int, gen_cfg: GeneratorConfig, var_cfg: VariantConfig, render_cfg: RenderConfig) -> None:
    """Create a vector env instance (num_envs) and close it to measure construction time.

    The vector env class is constructed directly to avoid any gym registration or wrapper
    overhead so the benchmark measures the vectorized implementation cost itself.
    """
    vec = NumberLinkRGBVectorEnv(
        num_envs, generator=gen_cfg, variant=var_cfg, render_config=render_cfg, render_mode=None
    )
    vec.close()


def bench_one_case(
    name: str, gen_cfg: GeneratorConfig, var_cfg: VariantConfig, render_cfg: RenderConfig, num_envs: int, iters: int
) -> None:
    """Run `iters` measurements for single and vector env construction and print summary."""
    single_times: list[float] = []
    vector_times: list[float] = []

    # Header showing what these results belong to
    header: str = (
        f"Case: {name} | mode={gen_cfg.mode} colors={gen_cfg.colors} "
        f"size={gen_cfg.width}x{gen_cfg.height} | variant.must_fill={var_cfg.must_fill} "
        f"| num_envs={num_envs}"
    )
    sep: LiteralString = "=" * len(header)
    print(sep, flush=True)
    print(header, flush=True)
    print(sep, flush=True)

    # Per-iteration table header
    print("Per-iteration timings:", flush=True)
    print("+-----------+----------------------+-----------------------+", flush=True)
    print("| Iteration | Single env (seconds) | Vector env (seconds)  |", flush=True)
    print("+-----------+----------------------+-----------------------+", flush=True)

    for i in range(1, iters + 1):
        single_t: float = timeit(lambda: make_single_env(gen_cfg, var_cfg, render_cfg))
        single_times.append(single_t)

        vector_t: float = timeit(lambda: make_vector_env(num_envs, gen_cfg, var_cfg, render_cfg))
        vector_times.append(vector_t)

        # Row for this iteration
        print(f"| {i:9d} | {single_t:20.6f} | {vector_t:21.6f} |", flush=True)

    print("+-----------+----------------------+-----------------------+", flush=True)
    print("", flush=True)

    def summarize(arr: Iterable[float]) -> str:
        a: list[float] = list(arr)
        mean: float = statistics.mean(a)
        std: float = statistics.stdev(a) if len(a) >= 2 else 0.0
        return f"mean={mean:.6f}s std={std:.6f}s min={min(a):.6f}s max={max(a):.6f}s"

    # Summary table
    summary_single: str = summarize(single_times)
    summary_vector = summarize(vector_times)

    print("Summary:", flush=True)
    print("+----------------+-----------------------------------------------+", flush=True)
    print("| Type           | stats                                         |", flush=True)
    print("+----------------+-----------------------------------------------+", flush=True)
    print(f"| Single env     | {summary_single:45s} |", flush=True)
    print(f"| Vector env     | {summary_vector:45s} |", flush=True)
    print("+----------------+-----------------------------------------------+", flush=True)
    print("", flush=True)


def main() -> None:
    """CLI entrypoint for the benchmark script.

    Flags:
      --num-envs: number of parallel envs for the vector env
      --iters: how many times to repeat timing for statistics
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Benchmark NumberLink env creation")
    # Check if running in a Colab environment
    args: argparse.Namespace
    if "google.colab" in sys.modules:
        # Provide default values for args when in Colab
        args = argparse.Namespace(num_envs=1, iters=10)
    else:
        # Parse command line arguments when not in Colab
        parser.add_argument("--num-envs", type=int, default=1, help="Number of envs in the vector env (default: 1)")
        parser.add_argument(
            "--iters", type=int, default=10, help="Number of iterations to time each case (default: 10)"
        )
        args = parser.parse_args()

    # Common variant and render configs, matching run_human.py where applicable
    # Use must_fill=False for benchmarks so puzzles do not require full-board fill to be valid.
    var_cfg = VariantConfig(must_fill=False, allow_diagonal=False, cell_switching_mode=False, bridges_enabled=False)
    render_cfg = RenderConfig(gridline_color=(60, 60, 60), gridline_thickness=1, show_endpoint_numbers=True)

    cases: list[tuple[str, GeneratorConfig]] = [
        ("8x8_7colors", GeneratorConfig(mode="hamiltonian", colors=7, width=8, height=8, min_path_length=3)),
        ("10x10_10colors", GeneratorConfig(mode="hamiltonian", colors=10, width=10, height=10, min_path_length=3)),
        ("20x20_10colors", GeneratorConfig(mode="hamiltonian", colors=10, width=20, height=20, min_path_length=3)),
        ("20x20_20colors", GeneratorConfig(mode="hamiltonian", colors=20, width=20, height=20, min_path_length=3)),
        ("8x8_7colors", GeneratorConfig(mode="random_walk", colors=7, width=8, height=8, min_path_length=3)),
        ("10x10_10colors", GeneratorConfig(mode="random_walk", colors=10, width=10, height=10, min_path_length=3)),
        ("20x20_10colors", GeneratorConfig(mode="random_walk", colors=10, width=20, height=20, min_path_length=3)),
        ("20x20_20colors", GeneratorConfig(mode="random_walk", colors=20, width=20, height=20, min_path_length=3)),
    ]

    print("Starting environment creation benchmarks. Results will be printed as they become available.", flush=True)

    for name, gen_cfg in cases:
        bench_one_case(name, gen_cfg, var_cfg, render_cfg, args.num_envs, args.iters)


if __name__ == "__main__":
    main()
