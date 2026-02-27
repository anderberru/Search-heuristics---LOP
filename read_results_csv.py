import csv
from collections import defaultdict
from statistics import mean


CSV_PATH = "data/results_global.csv"


def _to_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "instance": row.get("instance", ""),
                    "run": _to_int(row.get("run", 0), 0),
                    "seed": _to_int(row.get("seed", 0), 0),
                    "n": _to_int(row.get("n", 0), 0),
                    "alg": row.get("alg", ""),
                    "test": row.get("test", ""),
                    "initial_f": _to_float(row.get("initial_f", 0), 0.0),
                    "best_f": _to_float(row.get("best_f", 0), 0.0),
                    "delta": _to_float(row.get("delta", 0), 0.0),
                    "time": _to_float(row.get("time", 0), 0.0),
                    "valid_perm": _to_bool(row.get("valid_perm", False)),
                }
            )
    return rows


def print_global_summary(rows):
    total = len(rows)
    valid = sum(1 for r in rows if r["valid_perm"])
    unique_instances = len({r["instance"] for r in rows})
    unique_cfgs = len({(r["alg"], r["test"]) for r in rows})
    avg_delta = mean(r["delta"] for r in rows) if rows else 0.0
    avg_time = mean(r["time"] for r in rows) if rows else 0.0

    print("=== GLOBAL SUMMARY ===")
    print(f"Rows: {total}")
    print(f"Instances: {unique_instances}")
    print(f"Algorithm/test configs: {unique_cfgs}")
    print(f"Valid permutations: {valid}/{total} ({(100 * valid / total) if total else 0:.2f}%)")
    print(f"Average delta: {avg_delta:.3f}")
    print(f"Average time (s): {avg_time:.6f}")
    print()


def print_config_ranking(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["alg"], r["test"])].append(r)

    ranking = []
    for (alg, test), items in grouped.items():
        ranking.append(
            {
                "alg": alg,
                "test": test,
                "runs": len(items),
                "avg_best_f": mean(x["best_f"] for x in items),
                "avg_delta": mean(x["delta"] for x in items),
                "avg_time": mean(x["time"] for x in items),
                "valid_rate": sum(1 for x in items if x["valid_perm"]) / len(items),
            }
        )

    ranking.sort(key=lambda x: (x["avg_best_f"], x["avg_delta"]), reverse=True)

    print("=== RANKING BY CONFIG (sorted by avg_best_f) ===")
    print(
        f"{'alg':<8} {'test':<24} {'runs':>5} {'avg_best_f':>12} "
        f"{'avg_delta':>12} {'avg_time(s)':>12} {'valid_%':>8}"
    )
    for x in ranking:
        print(
            f"{x['alg']:<8} {x['test']:<24} {x['runs']:>5} "
            f"{x['avg_best_f']:>12.3f} {x['avg_delta']:>12.3f} "
            f"{x['avg_time']:>12.6f} {100 * x['valid_rate']:>8.2f}"
        )
    print()


def print_best_config_per_instance(rows):
    by_instance_cfg = defaultdict(list)
    for r in rows:
        by_instance_cfg[(r["instance"], r["alg"], r["test"])].append(r)

    best_per_instance = {}
    for (instance, alg, test), items in by_instance_cfg.items():
        score = mean(x["best_f"] for x in items)
        current = best_per_instance.get(instance)
        if current is None or score > current["score"]:
            best_per_instance[instance] = {
                "alg": alg,
                "test": test,
                "score": score,
                "avg_delta": mean(x["delta"] for x in items),
                "avg_time": mean(x["time"] for x in items),
            }

    print("=== BEST CONFIG PER INSTANCE ===")
    print(f"{'instance':<20} {'alg':<8} {'test':<24} {'avg_best_f':>12} {'avg_delta':>12} {'avg_time(s)':>12}")
    for instance in sorted(best_per_instance):
        v = best_per_instance[instance]
        print(
            f"{instance:<20} {v['alg']:<8} {v['test']:<24} "
            f"{v['score']:>12.3f} {v['avg_delta']:>12.3f} {v['avg_time']:>12.6f}"
        )
    print()


def main():
    rows = load_rows(CSV_PATH)
    if not rows:
        print(f"No data found in {CSV_PATH}")
        return

    print_global_summary(rows)
    print_config_ranking(rows)
    print_best_config_per_instance(rows)


if __name__ == "__main__":
    main()
