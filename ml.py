"""Обучение эвристики выбора хода для G4ip-решателя.

Постановка (ИСПРАВЛЕНО): вместо классификации «угадай правило по proof-строкам»
обучается модель ранжирования ходов. Каждая строка журнала — это пара
(состояние, ход) с исходом proof / dead_end, поэтому dead_end-строки (их
подавляющее большинство) больше не выбрасываются, а дают основной сигнал:
модель предсказывает P(ход приведёт к доказательству | состояние, ход).
На инференсе решатель оценивает этой моделью ТОЛЬКО применимые в секвенте
ходы и перебирает их в порядке убывания оценки — маскировка по доступным
ходам получается по построению.

Обучение и все метрики — только на строках с len(choices) > 1: там, где ход
единственный, выбора нет и учиться/мериться не на чем.

ВАЖНО: печатаемые здесь метрики — внутренняя диагностика обучения.
Главная метрика проекта — качество ПОИСКА (benchmark.py): посещённые
секвенты, откаты, доля задач, решённых в бюджет, с эвристикой и без.
"""

import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

DATASET = "branch_dataset.jsonl"
MODEL_OUT = "heuristic.pkl"

GOAL_TYPES = {"var": 0, "and": 1, "or": 2, "imp": 3, "bot": 4}

# Все правила, которые может вернуть collect_moves (порядок фиксирует one-hot).
MOVE_NAMES = ["R->", "R&", "R|1", "R|2", "L&", "L|", "L->atom", "L->&", "L->|", "L->->"]
MOVE_INDEX = {m: i for i, m in enumerate(MOVE_NAMES)}


def state_to_vector(state):
    return [
        state["gamma_size"],
        GOAL_TYPES[state["goal_type"]],
        state["formula_size"],
        state["max_depth"],
        state["num_vars"],
        state["num_and"],
        state["num_or"],
        state["num_imp"],
        state["num_bot"],
    ]


def pair_vector(state_vec, move_name):
    """Вектор признаков пары (состояние, кандидатный ход)."""
    onehot = [0.0] * len(MOVE_NAMES)
    onehot[MOVE_INDEX[move_name]] = 1.0
    return list(state_vec) + onehot


def load_rows(path=DATASET):
    rows = []
    with open(path, encoding="utf8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def branch_rows(rows):
    """Только строки в настоящих точках ветвления (есть из чего выбирать)."""
    return [r for r in rows if len(r["choices"]) > 1]


def build_xy(rows):
    X, y, groups = [], [], []
    for r in rows:
        X.append(pair_vector(state_to_vector(r["state"]), r["choice"]))
        y.append(1 if r["result"] == "proof" else 0)
        groups.append(r["formula"])
    return (np.array(X, dtype=float), np.array(y, dtype=int), np.array(groups))


def decision_nodes(rows):
    """Точки ветвления для оценки: len(choices) > 1 и известен «правильный» ход
    (в этой точке есть proof-строка — ход, попавший в найденное доказательство).
    Строки группируются по node_id, а не по значению признаков: одинаковые
    состояния из разных мест поиска не смешиваются.
    """
    by_node = defaultdict(list)
    for r in rows:
        by_node[r["node_id"]].append(r)
    nodes = []
    for node_rows in by_node.values():
        proof = [r for r in node_rows if r["result"] == "proof"]
        if proof and len(node_rows[0]["choices"]) > 1:
            p = proof[0]
            nodes.append({
                "state": p["state"],
                "choices": p["choices"],
                "correct": p["choice"],
                "correct_id": p["choice_id"],
                "formula": p["formula"],
            })
    return nodes


def predict_move(model, state, choices):
    """Ранжирование: оценка модели для каждого ДОСТУПНОГО хода, берём argmax.
    Модель физически не может предложить неприменимое правило."""
    sv = state_to_vector(state)
    Xc = np.array([pair_vector(sv, m) for m in choices], dtype=float)
    scores = model.predict_proba(Xc)[:, 1]
    return choices[int(np.argmax(scores))]


def make_model():
    return HistGradientBoostingClassifier(random_state=0)


def evaluate(rows, n_splits=5, test_size=0.2, seed=42):
    """Диагностика по нескольким сплитам (группировка по формулам)."""
    X, y, groups = build_xy(rows)
    nodes = decision_nodes(rows)

    print(f"строк в точках ветвления: {len(rows)} "
          f"(proof: {int(y.sum())}, dead_end: {int((1 - y).sum())})")
    print(f"уникальных формул: {len(set(groups))}, "
          f"точек ветвления с известным ответом: {len(nodes)}")

    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size,
                                 random_state=seed)
    acc_model, acc_major, acc_first = [], [], []
    for split_no, (tr, te) in enumerate(splitter.split(X, y, groups), 1):
        train_forms = set(groups[tr])
        test_forms = set(groups[te])
        train_nodes = [nd for nd in nodes if nd["formula"] in train_forms]
        test_nodes = [nd for nd in nodes if nd["formula"] in test_forms]
        if not train_nodes or not test_nodes:
            continue

        model = make_model()
        model.fit(X[tr], y[tr])

        majority = Counter(nd["correct"] for nd in train_nodes).most_common(1)[0][0]
        hits_m = np.mean([predict_move(model, nd["state"], nd["choices"]) == nd["correct"]
                          for nd in test_nodes])
        hits_maj = np.mean([nd["correct"] == majority for nd in test_nodes])
        hits_first = np.mean([nd["correct_id"] == 0 for nd in test_nodes])
        acc_model.append(hits_m)
        acc_major.append(hits_maj)
        acc_first.append(hits_first)
        print(f"  сплит {split_no}: формул train/test {len(train_forms)}/{len(test_forms)}, "
              f"точек в тесте {len(test_nodes)}, top-1 модели {hits_m:.3f}, "
              f"baseline «всегда {majority}» {hits_maj:.3f}, "
              f"baseline «текущий порядок» {hits_first:.3f}")

    def ms(a):
        return f"{np.mean(a):.3f} ± {np.std(a):.3f}"

    if not acc_model:
        print("Ни одного пригодного сплита (слишком мало данных) — "
              "диагностика пропущена.")
        return
    print("\nДиагностика (top-1 по применимым ходам, только точки с выбором,",
          f"среднее по {len(acc_model)} сплитам):")
    print(f"  модель:                      {ms(acc_model)}")
    print(f"  baseline (частый ход):       {ms(acc_major)}")
    print(f"  baseline (текущий порядок):  {ms(acc_first)}")
    print("Напоминание: это диагностика обучения; главный результат — benchmark.py.")
    print("Учтите: метка = ход, попавший в доказательство ПРИ ТЕКУЩЕМ порядке")
    print("перебора, поэтому baseline «текущий порядок» здесь завышен по")
    print("построению (см. п.5 ревью) — офлайн-метрика не сравнивает цену поиска.")


def train_final(rows, out=MODEL_OUT):
    X, y, _ = build_xy(rows)
    model = make_model()
    model.fit(X, y)
    bundle = {
        "model": model,
        "moves": MOVE_NAMES,
        "features": "state_to_vector(state) + one-hot хода (см. ml.pair_vector)",
    }
    # ИСПРАВЛЕНО: модель и словарь ходов сохраняются одним файлом и один раз
    # (раньше encoder сохранялся дважды).
    with open(out, "wb") as f:
        pickle.dump(bundle, f)
    print(f"эвристика сохранена в {out}")


def main():
    if not os.path.exists(DATASET):
        sys.exit(f"нет файла {DATASET} — сначала сгенерируйте датасет: python solver.py")
    rows = load_rows()
    print(f"строк в датасете всего: {len(rows)}")
    rows = branch_rows(rows)
    evaluate(rows)
    train_final(rows)


if __name__ == "__main__":
    main()
