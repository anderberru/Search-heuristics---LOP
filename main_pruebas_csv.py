import os
import csv

from functions import *
from algorithms.ils import iterated_local_search
from algorithms.tabu2 import tabu_search_insert
from algorithms.genetic import genetic_algorithm



TEST_BATTERIES_BY_ALGORITHM = {
    "ILS": [
        {
            "name": "ils_suave",
            "params": {
                "perturbation_strength": 2,
                "max_iters": 100,
            },
        },
        {
            "name": "ils_fuerte",
            "params": {
                "perturbation_strength": 8,
                "max_iters": 100,
            },
        },
        {
            "name": "ils_fuerte_grande",
            "params": {
                "perturbation_strength": 40,
                "max_iters": 100,
            },
        },
    ],
    "TABU": [
        {
            "name": "tabu_basico",
            "params": {
                "max_iters": 100,
                "tenure": 3,
                "medium_term": False,
                "long_term": False,
            },
        },
        {
            "name": "tabu_memorias",
            "params": {
                "max_iters": 100,
                "tenure": 5,
                "medium_term": True,
                "long_term": True,
                "elite_size": 6,
                "intensify_after": 30,
                "long_lambda": 0.01,
            },
        },
        {
            "name": "tabu_memorias_grande",
            "params": {
                "max_iters": 100,
                "tenure": 10,
                "medium_term": True,
                "long_term": True,
                "elite_size": 12,
                "intensify_after": 50,
                "long_lambda": 0.01,
            },
        },
    ],
    "GA": [
        {
            "name": "ga_tournament_insert",
            "params": {
                "generations": 100,
                "parent_selection_method": "tournament",
                "tournament_size": 3,
                "crossover_method": "order_crossover",
                "mutation_method": "insert",
                "new_population_method": "elitist",
            },
        },
        {
            "name": "ga_proportional_swap",
            "params": {
                "generations": 100,
                "parent_selection_method": "proportional",
                "tournament_size": 2,
                "crossover_method": "order_crossover",
                "mutation_method": "swap",
                "new_population_method": "pure_gen_elite1",
            },
        },
        {
            "name": "ga_best",
            "params": {
                "generations": 100,
                "parent_selection_method": "tournament",
                "tournament_size": 3,
                "crossover_method": "order_crossover",
                "mutation_method": "insert",
                "new_population_method": "pure_gen_elite1",
            },
        },
    ],
}

CSV_FIELDNAMES = ["instance", "run", "n", "alg", "test", "initial_f", "best_f", "delta", "time", "valid_perm"]


def init_results_file(output_dir="data", filename="results_global.csv"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

    return output_path


def append_result(output_path, result_row):
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writerow(result_row)

    return output_path

def is_valid_permutation(perm):
    n = len(perm)
    return set(perm) == set(range(n))


instances_dir = "instances"
lop_instances = sorted(
    name for name in os.listdir(instances_dir)
    if os.path.isfile(os.path.join(instances_dir, name)) and not name.startswith(".")
)

seed = 0
RUNS_PER_CONFIG = 3
output_path = init_results_file()

for instance in lop_instances:
    instance_path = os.path.join(instances_dir, instance)
    W = load_matrix_from_file(instance_path)
    n = W.shape[0]
    for alg, tests in TEST_BATTERIES_BY_ALGORITHM.items():
        for test in tests:
            for run in range(1, RUNS_PER_CONFIG + 1):
                cur_seed = seed + run - 1
                rng = np.random.default_rng(cur_seed)
                sigma0 = rng.permutation(n)
                initial_f = objective_function(W, sigma0)
                population_size = 20
                population0 = [rng.permutation(n) for _ in range(population_size)]
                params = test["params"]

                if alg == "ILS":
                    best_sigma, best_f, elapsed_time = iterated_local_search(W, sigma=sigma0, **params)
                    print(f"ILS test '{test['name']}' run {run}/{RUNS_PER_CONFIG} completed in {elapsed_time:.4f} seconds.")
                elif alg == "TABU":
                    best_sigma, best_f, elapsed_time = tabu_search_insert(W, start_perm=sigma0, **params)
                    print(f"Tabu Search test '{test['name']}' run {run}/{RUNS_PER_CONFIG} completed in {elapsed_time:.4f} seconds.")
                elif alg == "GA":
                    initial_f = max(objective_function(W, ind) for ind in population0)
                    best_sigma, best_f, elapsed_time = genetic_algorithm(W, population0=population0, **params)
                    print(f"Genetic Algorithm test '{test['name']}' run {run}/{RUNS_PER_CONFIG} completed in {elapsed_time:.4f} seconds.")

                delta = best_f - initial_f
                valid_perm = is_valid_permutation(best_sigma)

                result_row = {
                    "instance": instance,
                    "run": run,
                    "n": n,
                    "alg": alg,
                    "test": test["name"],
                    "initial_f": initial_f,
                    "best_f": best_f,
                    "delta": delta,
                    "time": f"{elapsed_time:.6f}",
                    "valid_perm": valid_perm,
                }
                append_result(output_path, result_row)

    print(f"Results for {instance_path} written incrementally to {output_path}.\n")

print(f"Global results written to {output_path}\n")

from plyer import notification
notification.notify(message='Execution Finished!')

