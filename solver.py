"""G4ip (LJT) — решатель для интуиционистской логики высказываний.

Формулы представляются кортежами:
    ('var', 'p')      — переменная
    ('bot',)          — ложь (false)
    ('and', A, B), ('or', A, B), ('imp', A, B)
Отрицание ~A — это сокращение для ('imp', A, ('bot',)).

Пайплайн проекта:
    python solver.py     — сгенерировать датасет точек ветвления (branch_dataset.jsonl)
    python ml.py         — обучить эвристику ранжирования ходов (heuristic.pkl)
    python benchmark.py  — главная метрика: поиск с эвристикой и без
    python tests.py      — тесты корректности решателя и признаков
"""

import argparse
import itertools
import json
import random

# ---------------------------------------------------------------------------
# Печать и разбор формул
# ---------------------------------------------------------------------------

def fmt(f):
    if f[0] == 'var': return f[1]
    if f[0] == 'bot': return 'false'
    if f[0] == 'imp' and f[2] == ('bot',): return '~' + fmt(f[1])
    op = {'and': '&', 'or': '|', 'imp': '->'}[f[0]]
    return '(' + fmt(f[1]) + op + fmt(f[2]) + ')'

def tokenize(s):
    tokens = []
    i = 0
    while i < len(s):
        if s[i] == ' ':
            i += 1
            continue
        elif s[i] == '-':
            if i + 1 < len(s) and s[i + 1] == '>':
                tokens.append('->')
                i += 2
                continue
            else:
                raise ValueError('ожидалось "->" на позиции ' + str(i))
        elif s[i] in '&|~()':
            tokens.append(s[i])
            i += 1
            continue
        elif s[i].isalpha():
            j = i
            while j < len(s) and (s[j].isalpha() or s[j].isdigit() or s[j] == '_'):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            raise ValueError(f"неожиданный символ: {s[i]}")
    return tokens

def parse(s):
    tokens = tokenize(s)
    if len(tokens) == 0:
        raise ValueError('пустая формула')
    f, i = parse_imp(tokens, 0)
    if i != len(tokens):
        raise ValueError('лишний токен "' + tokens[i] + '" (позиция ' + str(i) + ')')
    return f

def parse_imp(tokens, i):
    left, i = parse_or(tokens, i)
    if i < len(tokens) and tokens[i] == '->':
        right, i = parse_imp(tokens, i + 1)
        return ('imp', left, right), i
    return left, i

def parse_or(tokens, i):
    left, i = parse_and(tokens, i)
    while i < len(tokens) and tokens[i] == '|':
        right, i = parse_and(tokens, i + 1)
        left = ('or', left, right)
    return left, i

def parse_and(tokens, i):
    left, i = parse_unary(tokens, i)
    while i < len(tokens) and tokens[i] == '&':
        right, i = parse_unary(tokens, i + 1)
        left = ('and', left, right)
    return left, i

def parse_unary(tokens, i):
    if i < len(tokens) and tokens[i] == '~':
        f, i = parse_unary(tokens, i + 1)
        return ('imp', f, ('bot',)), i
    return parse_atom(tokens, i)

def parse_atom(tokens, i):
    if i >= len(tokens):
        raise ValueError('неожиданный конец формулы')
    t = tokens[i]
    if t == '(':
        f, i = parse_imp(tokens, i + 1)
        if i >= len(tokens) or tokens[i] != ')':
            raise ValueError('нет закрывающей скобки (токен ' + str(i) + ')')
        return f, i + 1
    if t == 'false':
        return ('bot',), i + 1
    if t[0].isalpha():
        return ('var', t), i + 1
    raise ValueError('ожидалась формула, а не "' + t + '" (токен ' + str(i) + ')')

# ---------------------------------------------------------------------------
# Классическая семантика (таблицы истинности) — независимый эталон для тестов
# и генерации заведомо доказуемых задач (по теореме Гливенко: F классически
# общезначима <=> ~~F доказуема интуиционистски).
# ---------------------------------------------------------------------------

def variables(f):
    if f[0] == 'var':
        return {f[1]}
    if f[0] == 'bot':
        return set()
    return variables(f[1]) | variables(f[2])

def eval_classical(f, env):
    if f[0] == 'var':
        return env[f[1]]
    if f[0] == 'bot':
        return False
    a = eval_classical(f[1], env)
    if f[0] == 'and':
        return a and eval_classical(f[2], env)
    if f[0] == 'or':
        return a or eval_classical(f[2], env)
    if f[0] == 'imp':
        return (not a) or eval_classical(f[2], env)
    raise ValueError(f)

def classical_tautology(f):
    vs = sorted(variables(f))
    return all(eval_classical(f, dict(zip(vs, bits)))
               for bits in itertools.product([False, True], repeat=len(vs)))

def neg(f):
    """~f как сокращение: ('imp', f, ('bot',))."""
    return ('imp', f, ('bot',))

# ---------------------------------------------------------------------------
# Признаки для ML
# ---------------------------------------------------------------------------

def formula_features(f):
    if f[0] == 'var':
        return {"size": 1, "depth": 1, "vars": 1, "and": 0, "or": 0, "imp": 0, "bot": 0}
    if f[0] == 'bot':
        return {"size": 1, "depth": 1, "vars": 0, "and": 0, "or": 0, "imp": 0, "bot": 1}
    a = formula_features(f[1])
    b = formula_features(f[2])
    # ИСПРАВЛЕНО (баг №1): счётчики связок теперь суммируют ОБА поддерева.
    # Раньше правое поддерево (b) терялось: для (p&q)&(r&s) выходило and=2 вместо 3.
    return {
        "size": 1 + a["size"] + b["size"],
        "depth": 1 + max(a["depth"], b["depth"]),
        "vars": a["vars"] + b["vars"],
        "and": a["and"] + b["and"] + (f[0] == "and"),
        "or":  a["or"]  + b["or"]  + (f[0] == "or"),
        "imp": a["imp"] + b["imp"] + (f[0] == "imp"),
        "bot": a["bot"] + b["bot"],
    }

def sequent_features(gamma, goal):
    fs = [formula_features(x) for x in gamma + [goal]]
    return {
        "gamma_size": len(gamma),
        "goal_type": goal[0],
        "formula_size": sum(x["size"] for x in fs),
        "max_depth": max(x["depth"] for x in fs),
        "num_vars": sum(x["vars"] for x in fs),
        "num_and": sum(x["and"] for x in fs),
        "num_or": sum(x["or"] for x in fs),
        "num_imp": sum(x["imp"] for x in fs),
        "num_bot": sum(x["bot"] for x in fs),
    }

# ---------------------------------------------------------------------------
# Журнал точек ветвления (для датасета)
# ---------------------------------------------------------------------------

BRANCH_LOG = []
NODE_COUNTER = 0  # глобально уникальный id точки ветвления (не сбрасывается clear_log)

def clear_log():
    BRANCH_LOG.clear()

def save_log(path):
    with open(path, "w", encoding="utf8") as f:
        for x in BRANCH_LOG:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------------
# Поиск вывода
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    """Поисковый бюджет (максимум посещённых секвентов) исчерпан."""

class SearchStats:
    """Статистика поиска: посещённые секвенты и откаты (проваленные ходы)."""
    __slots__ = ("nodes", "backtracks")

    def __init__(self):
        self.nodes = 0
        self.backtracks = 0

def prove(gamma, goal, chooser=None, stats=None, budget=None, log=False):
    """Ищет вывод секвента gamma |- goal. Возвращает дерево вывода или None.

    chooser — необязательная эвристика: chooser(gamma, goal, moves) -> moves,
              переставляет список ходов (порядок перебора); маскировка по
              применимым ходам получается автоматически, т.к. эвристика видит
              только ходы из collect_moves.
    stats   — SearchStats; считает посещённые секвенты (nodes) и откаты
              (backtracks = ходы, все посылки которых доказать не удалось).
    budget  — максимум посещённых секвентов; при превышении — BudgetExceeded.
    log     — писать точки ветвления в BRANCH_LOG (только для сбора датасета).
    """
    if stats is not None:
        stats.nodes += 1
        if budget is not None and stats.nodes > budget:
            raise BudgetExceeded()

    if goal[0] == 'var' and goal in gamma:
        return ('Ax', gamma, goal, [])
    if ('bot',) in gamma:
        return ('L_bot', gamma, goal, [])

    moves = collect_moves(gamma, goal)
    if chooser is not None and len(moves) > 1:
        moves = chooser(gamma, goal, moves)

    node_id = None
    if log:
        global NODE_COUNTER
        NODE_COUNTER += 1
        node_id = NODE_COUNTER

    for k, (name, premises) in enumerate(moves):
        if log:
            entry = {
                "node_id": node_id,          # id точки ветвления: строки одной
                "state": sequent_features(gamma, goal),  # точки имеют общий id
                "choices": [m[0] for m in moves],
                "choice": name,
                "choice_id": k,
                "premises": len(premises),
            }
        children = []
        ok = True
        for g2, c2 in premises:
            t = prove(g2, c2, chooser, stats, budget, log)
            if t is None:
                ok = False
                break
            children.append(t)

        if ok:
            if log:
                entry["result"] = "proof"
                entry["selected"] = name
                BRANCH_LOG.append(entry)
            return (name, gamma, goal, children)
        else:
            if stats is not None:
                stats.backtracks += 1
            if log:
                entry["result"] = "dead_end"
                BRANCH_LOG.append(entry)

    return None

def collect_moves(gamma, goal):
    moves = []
    if goal[0] == 'imp':
        moves.append(('R->', [(gamma + [goal[1]], goal[2])]))
    elif goal[0] == 'and':
        moves.append(('R&', [(gamma, goal[1]), (gamma, goal[2])]))
    elif goal[0] == 'or':
        moves.append(('R|1', [(gamma, goal[1])]))
        moves.append(('R|2', [(gamma, goal[2])]))
    for i in range(len(gamma)):
        f = gamma[i]
        rest = gamma[:i] + gamma[i + 1:]
        if f[0] == 'and':
            moves.append(('L&', [(rest + [f[1], f[2]], goal)]))
        elif f[0] == 'or':
            moves.append(('L|', [(rest + [f[1]], goal), (rest + [f[2]], goal)]))
        elif f[0] == 'imp':
            x = f[1]
            c = f[2]

            if x[0] == 'var':
                if x in rest:
                    moves.append(('L->atom', [(rest + [c], goal)]))

            # x[0] == 'bot': хода нет намеренно. Гипотеза false -> C — тавтология
            # и не несёт информации (Γ, false->C |- G равносильно Γ |- G),
            # поэтому её можно просто игнорировать; полнота сохраняется, а
            # лишний ход только раздувал бы перебор.

            elif x[0] == 'and':
                curried = ('imp', x[1], ('imp', x[2], c))
                moves.append(('L->&', [(rest + [curried], goal)]))

            elif x[0] == 'or':
                left = ('imp', x[1], c)
                right = ('imp', x[2], c)
                moves.append(('L->|', [(rest + [left, right], goal)]))

            elif x[0] == 'imp':
                premise1 = (rest + [('imp', x[2], c)], ('imp', x[1], x[2]))
                premise2 = (rest + [c], goal)
                moves.append(('L->->', [premise1, premise2]))
    return moves

def print_tree(tree, depth):
    name, gamma, goal, children = tree
    print('  ' * depth + fmt_sequent(gamma, goal) + '   [' + name + ']')
    for child in children:
        print_tree(child, depth + 1)

def fmt_sequent(gamma, goal):
    if len(gamma) == 0:
        return '|- ' + fmt(goal)
    s = fmt(gamma[0])
    for i in range(1, len(gamma)):
        s = s + ', ' + fmt(gamma[i])
    return s + ' |- ' + fmt(goal)

# ---------------------------------------------------------------------------
# Генерация случайных формул и датасета
# ---------------------------------------------------------------------------

def random_formula(depth, vars=("p", "q", "r", "s"), p_leaf=0.35, p_bot=0.08, p_neg=0.15):
    """Случайная формула с глубиной вложенности не больше depth.

    ИСПРАВЛЕНО (баг №2): генератор теперь порождает и false, и отрицания.
      p_leaf — вероятность досрочно остановиться на листе;
      p_bot  — вероятность того, что лист — это false (bot);
      p_neg  — вероятность породить отрицание ~A, то есть ('imp', A, ('bot',)).
    Благодаря этому в датасет попадают правило L_bot, ненулевой признак num_bot
    и интуиционистская специфика (~~p -> p, p | ~p, закон Пирса и т.п.).
    """
    if depth <= 0 or random.random() < p_leaf:
        if random.random() < p_bot:
            return ('bot',)
        return ('var', random.choice(vars))
    if random.random() < p_neg:
        return ('imp', random_formula(depth - 1, vars, p_leaf, p_bot, p_neg), ('bot',))
    op = random.choice(["and", "or", "imp"])
    return (op,
            random_formula(depth - 1, vars, p_leaf, p_bot, p_neg),
            random_formula(depth - 1, vars, p_leaf, p_bot, p_neg))

def run_dataset(n=5000, depth=5, out="branch_dataset.jsonl", seed=1, budget=20000):
    """Генерирует n случайных формул, прогоняет решатель и пишет журнал
    точек ветвления в out (jsonl).

    ИСПРАВЛЕНО: seed фиксируется (воспроизводимость), n увеличено на порядок,
    на формулу выделяется бюджет поиска budget (патологически тяжёлые формулы
    не подвешивают генерацию; уже записанные строки при этом корректны).
    """
    random.seed(seed)
    formulas = [random_formula(random.randint(1, depth)) for _ in range(n)]
    all_logs = []
    n_proved = n_refuted = n_budget = 0
    for f in formulas:
        clear_log()
        stats = SearchStats()
        try:
            tree = prove([], f, stats=stats, budget=budget, log=True)
            if tree is not None:
                n_proved += 1
            else:
                n_refuted += 1
        except BudgetExceeded:
            n_budget += 1
        for row in BRANCH_LOG:
            row["formula"] = fmt(f)
            all_logs.append(row)
    with open(out, "w", encoding="utf8") as fp:
        for row in all_logs:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"формул: {n} (доказуемых {n_proved}, недоказуемых {n_refuted}, "
          f"вышли за бюджет {n_budget})")
    return len(all_logs)

def main_interactive():
    while True:
        try:
            s = input('Введите формулу (пустая строка — выход): ')
        except EOFError:
            break
        s = s.strip()
        if s == '' or s == 'exit':
            break
        try:
            f = parse(s)
        except ValueError as e:
            print('Ошибка разбора: ' + str(e))
            continue
        tree = prove([], f)
        if tree is None:
            print('Недоказуема.')
        else:
            print('Доказуема. Дерево вывода:')
            print_tree(tree, 0)
        print()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Генерация датасета точек ветвления")
    ap.add_argument("--n", type=int, default=5000, help="число формул")
    ap.add_argument("--depth", type=int, default=5, help="максимальная глубина формул")
    ap.add_argument("--seed", type=int, default=1, help="seed генератора")
    ap.add_argument("--out", default="branch_dataset.jsonl", help="файл датасета")
    ap.add_argument("--budget", type=int, default=20000,
                    help="бюджет поиска (секвентов) на одну формулу")
    ap.add_argument("--interactive", action="store_true",
                    help="интерактивный режим: ввод формул с клавиатуры")
    args = ap.parse_args()
    if args.interactive:
        main_interactive()
    else:
        count = run_dataset(n=args.n, depth=args.depth, out=args.out,
                            seed=args.seed, budget=args.budget)
        print("saved records:", count)
