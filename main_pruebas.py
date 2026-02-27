import os

from functions import *
from algorithms.ils import iterated_local_search
from algorithms.tabu2 import tabu_search_insert
from algorithms.genetic import genetic_algorithm



TEST_BATTERIES_BY_ALGORITHM = {
    "ILS": [
        {
            "name": "ils_suave",
            "params": {
                "perturbation_strength": 1,
                "max_iters": 100,
            },
        },
        {
            "name": "ils_fuerte",
            "params": {
                "perturbation_strength": 2,
                "max_iters": 100,
            },
        },
        {
            "name": "ils_fuerte_grande",
            "params": {
                "perturbation_strength": 10,
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
                "intensify_after": 100,
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

def format_results_text(instance_name, n, results):
    def fmt_value(v):
        return "-" if v is None else str(v)

    lines = []
    lines.append("=" * 100)
    lines.append(f"Instancia: {instance_name} (n={n})")
    lines.append("-" * 100)
    lines.append(f"{'Algoritmo':<8} {'Prueba':<24} {'f_ini':>12} {'f_best':>12} {'delta':>12} {'tiempo(s)':>10} {'valid_perm':>12}")
    lines.append("-" * 100)
    for r in results:
        lines.append(
            f"{r['alg']:<8} "
            f"{r['test']:<24} "
            f"{fmt_value(r['initial_f']):>12} "
            f"{r['best_f']:>12} "
            f"{fmt_value(r['delta']):>12} "
            f"{r['time']:>10.4f} "
            f"{str(r['valid_perm']):>10}"
        )
    return "\n".join(lines) + "\n"


def write_results_file(instance_path, n, results, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    instance_name = os.path.basename(instance_path)
    output_path = os.path.join(output_dir, f"{instance_name}.txt")
    report = format_results_text(instance_path, n, results)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return output_path

def is_valid_permutation(perm):
    n = len(perm)
    return set(perm) == set(range(n))


lop_instances = ["Cebe.lop.n10.1", 
                 "Cebe.lop.n30.4", 
                 "N-r100a2", 
                 "N-r250e0", 
                ]

seed = 0
rng = np.random.default_rng(seed)

for instance in lop_instances:
    instance_path = f"instances/{instance}"
    W = load_matrix_from_file(instance_path)
    n = W.shape[0]
    sigma0 = rng.permutation(n)
    initial_f = objective_function(W, sigma0)
    population_size = 20
    population0 = [np.random.permutation(n) for _ in range(population_size)]
    results = []
    for alg, tests in TEST_BATTERIES_BY_ALGORITHM.items():
        for test in tests:
            params = test["params"]
            if alg == "ILS":
                best_sigma, best_f, elapsed_time = iterated_local_search(W, sigma=sigma0, **params)
                print(f"ILS test '{test['name']}' completed in {elapsed_time:.4f} seconds.")
            elif alg == "TABU":
                best_sigma, best_f, elapsed_time = tabu_search_insert(W, start_perm=sigma0, **params)
                print(f"Tabu Search test '{test['name']}' completed in {elapsed_time:.4f} seconds.")
            elif alg == "GA":
                initial_f = max(objective_function(W, ind) for ind in population0)
                best_sigma, best_f, elapsed_time = genetic_algorithm(W, population0=population0, **params)
                print(f"Genetic Algorithm test '{test['name']}' completed in {elapsed_time:.4f} seconds.")

            delta = best_f - initial_f
            valid_perm = is_valid_permutation(best_sigma)

            results.append({
                "alg": alg,
                "test": test["name"],
                "initial_f": initial_f,
                "best_f": best_f,
                "delta": delta,
                "time": elapsed_time,
                "valid_perm": valid_perm,
            })

    output_path = write_results_file(instance_path, n, results)
    print(f"Results for {instance_path} written to {output_path}\n")

    from plyer import notification
    notification.notify(message='Execution Finished!')

