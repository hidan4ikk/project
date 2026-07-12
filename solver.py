import json
import random
from collections import Counter

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
            if i+1 < len(s) and s[i+1] == '>':
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
    return(tokens)

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
        right, i = parse_imp(tokens, i+1)
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

BRANCH_LOG = []

def formula_features(f):
    if f[0] == 'var':
        return {"size":1, "depth":1, "vars":1, "and":0, "or":0, "imp":0, "bot":0}
    if f[0] == 'bot':
        return {"size":1, "depth":1, "vars":0, "and":0, "or":0, "imp":0, "bot":1}
    a = formula_features(f[1])
    b = formula_features(f[2])
    return {
        "size":1+a["size"]+b["size"],
        "depth":1+max(a["depth"], b["depth"]),
        "vars":a["vars"]+b["vars"],
        "and":a["and"]+(f[0]=="and"),
        "or":a["or"]+(f[0]=="or"),
        "imp":a["imp"]+(f[0]=="imp"),
        "bot":a["bot"]+b["bot"]
    }

def sequent_features(gamma, goal):
    fs = [formula_features(x) for x in gamma+[goal]]
    return {
        "gamma_size": len(gamma),
        "goal_type": goal[0],
        "formula_size": sum(x["size"] for x in fs),
        "max_depth": max(x["depth"] for x in fs),
        "num_vars": sum(x["vars"] for x in fs),
        "num_and": sum(x["and"] for x in fs),
        "num_or": sum(x["or"] for x in fs),
        "num_imp": sum(x["imp"] for x in fs),
        "num_bot": sum(x["bot"] for x in fs)
    }

def clear_log():
    BRANCH_LOG.clear()

def save_log(path):
    with open(path, "w", encoding="utf8") as f:
        for x in BRANCH_LOG:
            f.write(json.dumps(x, ensure_ascii=False)+"\n")

def prove(gamma, goal):
    if goal[0] == 'var' and goal in gamma:
        return ('Ax', gamma, goal, [])
    if ('bot',) in gamma:
        return ('L_bot', gamma, goal, [])

    moves = collect_moves(gamma, goal)
    for k, (name, premises) in enumerate(moves):
        entry = {
            "state": sequent_features(gamma, goal),
            "choices": [m[0] for m in moves],
            "choice": name,
            "choice_id": k,
            "premises": len(premises)
        }
        children = []
        ok = True
        for g2, c2 in premises:
            t = prove(g2, c2)
            if t is None:
                ok = False
                break
            children.append(t)

        if ok:
            entry["result"] = "proof"
            entry["selected"] = name
            BRANCH_LOG.append(entry)
            return (name, gamma, goal, children)
        else:
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

def main():
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

def random_formula(depth, vars=("p","q","r","s")):
    if depth <= 0 or random.random() < 0.35:
        return ('var', random.choice(vars))
    op=random.choice(["and","or","imp"])
    return (op, random_formula(depth-1, vars), random_formula(depth-1, vars))

def run_dataset(n=500, depth=5, out="branch_dataset.jsonl"):
    formulas=[random_formula(random.randint(1, depth)) for _ in range(n)]
    all_logs=[]
    for f in formulas:
        clear_log()
        prove([], f)
        for row in BRANCH_LOG:
            row["formula"] = fmt(f)
            all_logs.append(row)
    with open(out,"w",encoding="utf8") as fp:
        for row in all_logs:
            fp.write(json.dumps(row, ensure_ascii=False)+"\n")
    return len(all_logs)

if name == "main":
    count=run_dataset()
    print("saved records:", count)