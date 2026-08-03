from titrate.environments.cstr_env import CSTREnvironment
from titrate.evaluation.benchmark import run_benchmark, run_trial


def test_run_trial_produces_expected_row_count():
    env = CSTREnvironment()
    df = run_trial("random", env, budget=6, seed=0)
    assert len(df) == 6
    assert set(df["iteration"]) == {1, 2, 3, 4, 5, 6}


def test_run_benchmark_smoke_all_strategies_tiny_budget():
    env = CSTREnvironment()
    df = run_benchmark(
        env,
        budget=7,
        n_seeds=2,
        strategies=("random", "lhs", "bo_constrained"),
        bo_n_initial=3,
        verbose=False,
    )
    assert set(df["strategy"]) == {"random", "lhs", "bo_constrained"}
    assert set(df["seed"]) == {0, 1}
    for (strategy, seed), group in df.groupby(["strategy", "seed"]):
        assert len(group) == 7
