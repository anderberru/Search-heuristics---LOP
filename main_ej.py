import os

from functions import *
from algorithms.ils import iterated_local_search
from algorithms.tabu2 import tabu_search_insert
from algorithms.genetic import genetic_algorithm

# Matrices a probar (puedes comentar/descomentar según el tiempo disponible)
INSTANCE_PATHS = [
    "instances/Cebe.lop.n10.1",
    "instances/Cebe.lop.n30.4",
    # "instances/N-r100a2",
]


TEST_BATTERIES_BY_INSTANCE = {
    "instances/Cebe.lop.n10.1": {
        "ILS": [
            {"name": "ils_suave", "params": {"perturbation_strength": 1, "max_iters": 100}},
            {"name": "ils_fuerte", "params": {"perturbation_strength": 2, "max_iters": 100}},
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
                    "tenure": 4,
                    "medium_term": True,
                    "long_term": True,
                    "elite_size": 5,
                    "intensify_after": 10,
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
                    "tournament_size": 2,
                    "crossover_method": "order_crossover",
                    "mutation_method": "insert",
                    "new_population_method": "elitist",
                    "population_size": 8,
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
                    "population_size": 10,
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
                    "population_size": 20,
                },
            },
        ],
    },
    "instances/Cebe.lop.n30.4": {
        "ILS": [
            {"name": "ils_suave", "params": {"perturbation_strength": 1, "max_iters": 100}},
            {"name": "ils_fuerte", "params": {"perturbation_strength": 3, "max_iters": 100}},
        ],
        "TABU": [
            {
                "name": "tabu_basico",
                "params": {
                    "max_iters": 100,
                    "tenure": 4,
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
                    "population_size": 10,
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
                    "population_size": 15,
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
                    "population_size": 20,
                },
            },
        ],
    },
    "instances/N-r100a2": {
        "ILS": [
            {"name": "ils_suave", "params": {"perturbation_strength": 5, "max_iters": 100}},
            {"name": "ils_fuerte", "params": {"perturbation_strength": 10, "max_iters": 100}},
        ],
        "TABU": [
            {
                "name": "tabu_basico",
                "params": {
                    "max_iters": 100,
                    "tenure": 8,
                    "medium_term": False,
                    "long_term": False,
                },
            },
            {
                "name": "tabu_memorias",
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
                    "generations": 180,
                    "parent_selection_method": "tournament",
                    "tournament_size": 4,
                    "crossover_method": "order_crossover",
                    "mutation_method": "insert",
                    "new_population_method": "elitist",
                    "population_size": 30,
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
                    "population_size": 40,
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
                    "population_size": 20,
                },
            },
        ],
    },
}


def build_test_battery(instance_path):
    """
    Devuelve la batería de pruebas definida manualmente para cada matriz.
    """
    if instance_path not in TEST_BATTERIES_BY_INSTANCE:
        raise ValueError(
            f"No hay batería configurada para {instance_path}. "
            "Añade sus parámetros manualmente en TEST_BATTERIES_BY_INSTANCE."
        )
    return TEST_BATTERIES_BY_INSTANCE[instance_path]


def run_ils_tests(W, tests):
    n = W.shape[0]
    results = []
    for test in tests:
        best_sigma, best_f, elapsed = iterated_local_search(W, **test["params"])
        results.append(
            {
                "alg": "ILS",
                "test": test["name"],
                "initial_f": None,
                "best_f": int(best_f),
                "delta": None,
                "time": elapsed,
                "valid_perm": len(set(best_sigma)) == n,
            }
        )
    return results


def run_tabu_tests(W, tests):
    n = W.shape[0]
    results = []
    for test in tests:
        best_sigma, best_f, elapsed = tabu_search_insert(W, **test["params"])
        results.append(
            {
                "alg": "TABU",
                "test": test["name"],
                "initial_f": None,
                "best_f": int(best_f),
                "delta": None,
                "time": elapsed,
                "valid_perm": len(set(best_sigma)) == n,
            }
        )
    return results


def run_ga_tests(W, tests):
    n = W.shape[0]
    results = []
    for test in tests:
        best_sigma, best_f, elapsed = genetic_algorithm(W, **test["params"])
        results.append(
            {
                "alg": "GA",
                "test": test["name"],
                "initial_f": None,
                "best_f": int(best_f),
                "delta": None,
                "time": elapsed,
                "valid_perm": len(set(best_sigma)) == n,
            }
        )
    return results


def format_results_text(instance_name, n, results):
    def fmt_value(v):
        return "-" if v is None else str(v)

    lines = []
    lines.append("=" * 90)
    lines.append(f"Instancia: {instance_name} (n={n})")
    lines.append("-" * 90)
    lines.append(f"{'Algoritmo':<8} {'Prueba':<24} {'f_ini':>12} {'f_best':>12} {'delta':>12} {'tiempo(s)':>10} {'ok':>4}")
    lines.append("-" * 90)
    for r in results:
        lines.append(
            f"{r['alg']:<8} "
            f"{r['test']:<24} "
            f"{fmt_value(r['initial_f']):>12} "
            f"{r['best_f']:>12} "
            f"{fmt_value(r['delta']):>12} "
            f"{r['time']:>10.4f} "
            f"{str(r['valid_perm']):>4}"
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


def main():
    for path in INSTANCE_PATHS:
        W = load_matrix_from_file(path)
        n = W.shape[0]
        battery = build_test_battery(path)

        results = []
        results.extend(run_ils_tests(W, battery["ILS"]))
        results.extend(run_tabu_tests(W, battery["TABU"]))
        results.extend(run_ga_tests(W, battery["GA"]))

        output_path = write_results_file(path, n, results)
        print(f"Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
