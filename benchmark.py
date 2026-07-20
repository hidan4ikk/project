"""Главная метрика проекта: качество ПОИСКА с эвристикой и без неё.

Точность классификатора не измеряет то, ради чего эвристика затевалась:
многие правила G4ip обратимы, к доказательству ведут разные ходы с разной
ценой. Поэтому здесь сравнивается сам поиск, по одинаковому бюджету:

  * доля задач, решённых в бюджет (аналог таймаута, но воспроизводимый:
    бюджет — это максимум посещённых секвентов, а не время);
  * число посещённых секвентов;
  * число откатов (ходов, все посылки которых доказать не удалось);
  * парное сравнение на задачах, решённых обоими методами.

Два набора задач (свежие seed'ы, обучающие формулы исключены):
  1) случайные формулы — то же распределение, что при обучении;
     большинство недоказуемы, а там перебор полный и порядок ходов
     почти не влияет на цену;
  2) заведомо доказуемые нетривиальные: ~~F для случайных классических
     тавтологий F (доказуемы по теореме Гливенко) — именно на доказуемых
     задачах порядок перебора и должен экономить поиск.

Три метода:
  * фиксированный порядок (текущий решатель, baseline);
  * эвристика (ранжирование ходов моделью из heuristic.pkl);
  * обратный порядок — контроль чувствительности: если бенчмарк не видит
    разницы даже между нормальным и заведомо плохим порядком, он не измеряет
    ничего.

Запуск: python benchmark.py  (нужны branch_dataset.jsonl и heuristic.pkl)
"""

import argparse
import json
import os
import pickle
import random
import statistics
import sys
import time

import numpy as np

from solver import (BudgetExceeded, SearchStats, classical_tautology, fmt,
                    neg, prove, random_formula, sequent_features)
from ml import pair_vector, state_to_vector


def load_chooser(path="heuristic.pkl"):
    """Эвристика: переставляет применимые ходы по убыванию оценки модели."""
    if not os.path.exists(path):
        sys.exit(f"нет файла {path} — сначала выполните: "
                 f"python solver.py, затем python ml.py")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    def chooser(gamma, goal, moves):
        sv = state_to_vector(sequent_features(gamma, goal))
        Xc = np.array([pair_vector(sv, name) for name, _ in moves], dtype=float)
        scores = model.predict_proba(Xc)[:, 1]
        order = np.argsort(-scores, kind="stable")
        return [moves[i] for i in order]

    return chooser


def reverse_chooser(gamma, goal, moves):
    return list(reversed(moves))


def training_formulas(dataset="branch_dataset.jsonl"):
    """Формулы из обучающего датасета — исключаем их из оценки."""
    seen = set()
    if not os.path.exists(dataset):
        print(f"внимание: {dataset} не найден — обучающие формулы "
              f"не исключаются из тестовых наборов")
        return seen
    with open(dataset, encoding="utf8") as f:
        for line in f:
            seen.add(json.loads(line)["formula"])
    return seen


def suite_random(n, depth, seed, exclude):
    random.seed(seed)
    out, guard = [], 0
    while len(out) < n and guard < 500 * n:
        guard += 1
        f = random_formula(random.randint(1, depth))
        s = fmt(f)
        if s in exclude:            # ни обучающих формул, ни дублей
            continue
        exclude.add(s)
        out.append(f)
    if len(out) < n:
        print(f"внимание: набрано только {len(out)} случайных формул из {n}")
    return out


def suite_provable(n, depth, seed, exclude):
    """~~F для случайных классических тавтологий F: доказуемы по Гливенко."""
    random.seed(seed)
    out, guard = [], 0
    while len(out) < n and guard < 2000 * n:
        guard += 1
        f = random_formula(random.randint(3, depth))
        if not classical_tautology(f):
            continue
        g = neg(neg(f))
        s = fmt(g)
        if s in exclude:
            continue
        exclude.add(s)
        out.append(g)
    if len(out) < n:
        print(f"внимание: набрано только {len(out)} заведомо доказуемых задач из {n}")
    return out


def run_one(f, chooser, budget):
    stats = SearchStats()
    t0 = time.perf_counter()
    try:
        tree = prove([], f, chooser=chooser, stats=stats, budget=budget)
        status = "proved" if tree is not None else "refuted"
    except BudgetExceeded:
        status = "budget"
    return {"status": status, "nodes": stats.nodes,
            "backtracks": stats.backtracks, "time": time.perf_counter() - t0}


def summarize(name, res):
    n = len(res)
    proved = sum(r["status"] == "proved" for r in res)
    refuted = sum(r["status"] == "refuted" for r in res)
    over = sum(r["status"] == "budget" for r in res)
    nodes = [r["nodes"] for r in res]
    backs = [r["backtracks"] for r in res]
    print(f"  {name}:")
    print(f"    решено в бюджет: {n - over}/{n} ({(n - over) / n:.1%})  "
          f"[доказано {proved}, опровергнуто {refuted}, вне бюджета {over}]")
    print(f"    посещено секвентов: всего {sum(nodes)}, среднее "
          f"{statistics.mean(nodes):.1f}, медиана {statistics.median(nodes):.1f}")
    print(f"    откатов всего: {sum(backs)}; время: "
          f"{sum(r['time'] for r in res):.2f} c")


def paired(name_a, a, name_b, b):
    """Парное сравнение по числу секвентов на задачах, доказанных обоими."""
    idx = [i for i in range(len(a))
           if a[i]["status"] == "proved" and b[i]["status"] == "proved"]
    if not idx:
        print(f"  Нет задач, доказанных и «{name_a}», и «{name_b}».")
        return
    na = [a[i]["nodes"] for i in idx]
    nb = [b[i]["nodes"] for i in idx]
    wins = sum(y < x for x, y in zip(na, nb))
    ties = sum(y == x for x, y in zip(na, nb))
    print(f"  «{name_b}» против «{name_a}» (доказано обоими: {len(idx)}):")
    print(f"    секвентов всего: {sum(na)} -> {sum(nb)} (x{sum(nb) / sum(na):.2f});"
          f" медиана {statistics.median(na):.0f} -> {statistics.median(nb):.0f}")
    print(f"    лучше: {wins}, вничью: {ties}, хуже: {len(idx) - wins - ties}")


def run_suite(title, formulas, methods, budget):
    print(f"\n=== {title}: {len(formulas)} формул, "
          f"бюджет {budget} секвентов на формулу ===")
    results = {}
    for name, chooser in methods:
        results[name] = [run_one(f, chooser, budget) for f in formulas]
        summarize(name, results[name])
    base = methods[0][0]
    for name, _ in methods[1:]:
        paired(base, results[base], name, results[name])
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Бенчмарк поиска: эвристика vs фиксированный порядок")
    ap.add_argument("--n-random", type=int, default=400,
                    help="размер набора случайных формул")
    ap.add_argument("--n-provable", type=int, default=120,
                    help="размер набора заведомо доказуемых задач")
    ap.add_argument("--depth", type=int, default=5,
                    help="максимальная глубина случайных формул")
    ap.add_argument("--depth-provable", type=int, default=7,
                    help="максимальная глубина F в наборе ~~F")
    ap.add_argument("--seed", type=int, default=123,
                    help="seed тестовых наборов (не совпадает с seed датасета)")
    ap.add_argument("--budget", type=int, default=20000,
                    help="бюджет: максимум посещённых секвентов на формулу")
    ap.add_argument("--model", default="heuristic.pkl")
    args = ap.parse_args()

    exclude = training_formulas()
    print(f"исключено обучающих формул: {len(exclude)}")

    methods = [
        ("фиксированный порядок", None),
        ("эвристика", load_chooser(args.model)),
        ("обратный порядок (контроль)", reverse_chooser),
    ]

    rnd = suite_random(args.n_random, args.depth, args.seed, exclude)
    run_suite("Случайные формулы (распределение как при обучении)",
              rnd, methods, args.budget)

    prv = suite_provable(args.n_provable, args.depth_provable,
                         args.seed + 1, exclude)
    run_suite("Заведомо доказуемые нетривиальные (~~тавтологии, Гливенко)",
              prv, methods, args.budget)

    print("\nПримечания: время эвристики включает накладные расходы на вызов "
          "модели,\nобъём поиска сравнивается по числу посещённых секвентов. "
          "На недоказуемых\nформулах перебор полный, и порядок ходов почти не "
          "влияет на его размер —\nэффект эвристики виден на доказуемых задачах.")


if __name__ == "__main__":
    main()
