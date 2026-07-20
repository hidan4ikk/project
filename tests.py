"""Тесты корректности решателя, признаков, генератора и журнала.

Запуск: python tests.py   (или python -m unittest tests -v)

Ключевые проверки:
  * известные интуиционистские теоремы доказываются, известные
    НЕ-теоремы (LEM, снятие двойного отрицания, закон Пирса, ...) — нет;
  * корректность: всё, что доказал решатель, — классическая тавтология;
  * полнота (кросс-проверка по теореме Гливенко независимым методом —
    таблицами истинности): F классически общезначима <=> ~~F доказуема
    интуиционистски;
  * счётчики связок в formula_features совпадают с независимым пересчётом
    (в т.ч. примеры из ревью: (p&q)&(r&s) и (p->q)|(r->s));
  * генератор порождает false и отрицания, все типы узлов, соблюдает глубину;
  * журнал точек ветвления имеет корректную схему (node_id, choices, ...).
"""

import json
import os
import random
import tempfile
import unittest

from solver import (BRANCH_LOG, BudgetExceeded, SearchStats,
                    classical_tautology, clear_log, collect_moves, fmt,
                    formula_features, neg, parse, prove, random_formula,
                    run_dataset, sequent_features)

# Классический эталон (таблицы истинности) реализован в solver.py и
# используется здесь как независимая от решателя проверка.

def provable(f, budget=None):
    stats = SearchStats()
    return prove([], f, stats=stats, budget=budget) is not None

def count_op(f, op):
    """Независимый пересчёт числа узлов данного типа."""
    if f[0] in ('var', 'bot'):
        return int(f[0] == op)
    return int(f[0] == op) + count_op(f[1], op) + count_op(f[2], op)

# ---------------------------------------------------------------------------

class TestParser(unittest.TestCase):
    def test_roundtrip_random(self):
        random.seed(100)
        for _ in range(300):
            f = random_formula(random.randint(1, 5))
            self.assertEqual(parse(fmt(f)), f, fmt(f))

    def test_precedence_and_negation(self):
        self.assertEqual(parse('p -> q -> r'),
                         ('imp', ('var', 'p'), ('imp', ('var', 'q'), ('var', 'r'))))
        self.assertEqual(parse('p & q | r'),
                         ('or', ('and', ('var', 'p'), ('var', 'q')), ('var', 'r')))
        self.assertEqual(parse('~p'), ('imp', ('var', 'p'), ('bot',)))
        self.assertEqual(parse('~~p'), neg(neg(('var', 'p'))))
        self.assertEqual(parse('false'), ('bot',))

    def test_errors(self):
        for bad in ['', '(p', 'p )', 'p -', '->', 'p q']:
            with self.assertRaises(ValueError, msg=bad):
                parse(bad)


class TestKnownTheorems(unittest.TestCase):
    PROVABLE = [
        'p -> p',
        'p -> (q -> p)',
        '(p -> (q -> r)) -> ((p -> q) -> (p -> r))',
        '(p & q) -> p',
        '(p & q) -> q',
        'p -> (p | q)',
        'q -> (p | q)',
        'false -> p',                      # ex falso, работает L_bot
        '~(p & ~p)',
        '~~(p | ~p)',                      # дв. отрицание искл. третьего — теорема
        '(p -> q) -> (~q -> ~p)',
        '~~~p -> ~p',
        'p -> ~~p',
        '((p | q) -> r) -> ((p -> r) & (q -> r))',
        '(~p | q) -> (p -> q)',
        '~(p | q) -> (~p & ~q)',           # интуиционистское направление де Моргана
        '(~p & ~q) -> ~(p | q)',
        '(p -> q) -> ((q -> r) -> (p -> r))',
        '((p & q) -> r) -> (p -> (q -> r))',
        '(p -> (q -> r)) -> ((p & q) -> r)',
    ]
    # Классические, но НЕ интуиционистские (или вовсе не тавтологии):
    UNPROVABLE = [
        'p',
        'false',
        'p | ~p',                          # исключённое третье
        '~~p -> p',                        # снятие двойного отрицания
        '((p -> q) -> p) -> p',            # закон Пирса
        '~(p & q) -> (~p | ~q)',           # неинтуиционистское направление де Моргана
        '(p -> q) | (q -> p)',
        '(p -> q) -> (~p | q)',
        'p -> q',
        '(p | q) -> p',
    ]

    def test_provable(self):
        for s in self.PROVABLE:
            self.assertTrue(provable(parse(s)), 'должна доказываться: ' + s)

    def test_unprovable(self):
        for s in self.UNPROVABLE:
            self.assertFalse(provable(parse(s)), 'НЕ должна доказываться: ' + s)

    def test_l_bot_rule_fires(self):
        tree = prove([], parse('false -> p'))
        self.assertIsNotNone(tree)
        self.assertEqual(tree[0], 'R->')
        self.assertEqual(tree[3][0][0], 'L_bot')

    def test_axiom(self):
        self.assertIsNotNone(prove([('var', 'p')], ('var', 'p')))
        self.assertIsNone(prove([('var', 'p')], ('var', 'q')))


class TestSemantics(unittest.TestCase):
    """Кросс-проверка решателя независимым методом (таблицы истинности)."""

    def test_soundness_random(self):
        # Всё интуиционистски доказуемое обязано быть классической тавтологией.
        random.seed(200)
        checked = 0
        for _ in range(400):
            f = random_formula(random.randint(1, 5))
            try:
                if provable(f, budget=200000):
                    self.assertTrue(classical_tautology(f),
                                    'решатель доказал не-тавтологию: ' + fmt(f))
                checked += 1
            except BudgetExceeded:
                pass
        self.assertGreater(checked, 350)

    def test_glivenko_random(self):
        # Теорема Гливенко: F классически общезначима <=> ~~F доказуема
        # интуиционистски. Проверяет и корректность, и полноту решателя.
        random.seed(300)
        checked = 0
        for _ in range(300):
            f = random_formula(random.randint(1, 4))
            try:
                got = provable(neg(neg(f)), budget=200000)
            except BudgetExceeded:
                continue
            self.assertEqual(classical_tautology(f), got,
                             'расхождение с Гливенко на: ' + fmt(f))
            checked += 1
        self.assertGreater(checked, 250)


class TestFormulaFeatures(unittest.TestCase):
    def test_review_examples(self):
        # Примеры из ревью (раньше терялось правое поддерево):
        self.assertEqual(formula_features(parse('(p&q)&(r&s)'))['and'], 3)
        f = formula_features(parse('(p->q)|(r->s)'))
        self.assertEqual(f['imp'], 2)
        self.assertEqual(f['or'], 1)

    def test_matches_independent_recount(self):
        random.seed(400)
        for _ in range(300):
            f = random_formula(random.randint(1, 5))
            feats = formula_features(f)
            for op, key in [('and', 'and'), ('or', 'or'), ('imp', 'imp'),
                            ('bot', 'bot'), ('var', 'vars')]:
                self.assertEqual(feats[key], count_op(f, op),
                                 f'{key} для {fmt(f)}')
            total = sum(count_op(f, op) for op in ('and', 'or', 'imp', 'bot', 'var'))
            self.assertEqual(feats['size'], total, fmt(f))

    def test_sequent_features_num_bot(self):
        st = sequent_features([parse('~p')], parse('false'))
        self.assertEqual(st['num_bot'], 2)
        self.assertEqual(st['goal_type'], 'bot')


class TestGenerator(unittest.TestCase):
    def test_produces_bot_and_negation(self):
        # По ревью: раньше на 2000 формул было ноль с false. Теперь есть и
        # false, и отрицания ~A (= A -> false), и все бинарные связки.
        random.seed(500)
        fs = [random_formula(random.randint(1, 5)) for _ in range(2000)]
        n_bot = sum(count_op(f, 'bot') > 0 for f in fs)
        n_neg = sum(self._has_neg(f) for f in fs)
        self.assertGreater(n_bot, 50, 'false почти не порождается')
        self.assertGreater(n_neg, 50, 'отрицания почти не порождаются')
        for op in ('and', 'or', 'imp'):
            self.assertTrue(any(count_op(f, op) > 0 for f in fs), op)

    def _has_neg(self, f):
        if f[0] in ('var', 'bot'):
            return False
        if f[0] == 'imp' and f[2] == ('bot',):
            return True
        return self._has_neg(f[1]) or self._has_neg(f[2])

    def test_depth_bound(self):
        random.seed(600)
        for _ in range(500):
            d = random.randint(1, 6)
            f = random_formula(d)
            self.assertLessEqual(formula_features(f)['depth'], d + 1)


class TestSearchInfrastructure(unittest.TestCase):
    def test_stats_counts(self):
        stats = SearchStats()
        prove([], parse('p | ~p'), stats=stats)
        self.assertGreater(stats.nodes, 1)
        self.assertGreaterEqual(stats.backtracks, 1)

    def test_budget_exceeded(self):
        stats = SearchStats()
        with self.assertRaises(BudgetExceeded):
            prove([], parse('((p->q)->p)->p'), stats=stats, budget=2)

    def test_chooser_reorders_but_result_same(self):
        # Любая перестановка ходов не меняет доказуемость (меняет только цену).
        def reverse_chooser(gamma, goal, moves):
            return list(reversed(moves))
        for s in ['~~(p | ~p)', 'p | ~p', '(p -> q) -> (~q -> ~p)']:
            f = parse(s)
            self.assertEqual(prove([], f) is not None,
                             prove([], f, chooser=reverse_chooser) is not None, s)


class TestDatasetLogging(unittest.TestCase):
    def test_schema_and_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'ds.jsonl')
            n_rows = run_dataset(n=120, depth=4, out=out, seed=7, budget=5000)
            self.assertGreater(n_rows, 0)
            with open(out, encoding='utf8') as fp:
                rows = [json.loads(line) for line in fp]
        self.assertEqual(len(rows), n_rows)
        for r in rows:
            self.assertIn('node_id', r)
            self.assertIn(r['result'], ('proof', 'dead_end'))
            self.assertIn(r['choice'], r['choices'])
            self.assertEqual(r['choice'], r['choices'][r['choice_id']])
            self.assertGreaterEqual(len(r['choices']), 1)
        results = {r['result'] for r in rows}
        self.assertEqual(results, {'proof', 'dead_end'})
        # После исправления генератора признак num_bot больше не всегда 0:
        self.assertTrue(any(r['state']['num_bot'] > 0 for r in rows))
        # Есть настоящие точки ветвления:
        self.assertTrue(any(len(r['choices']) > 1 for r in rows))
        # Строки одной точки ветвления согласованы:
        by_node = {}
        for r in rows:
            by_node.setdefault(r['node_id'], []).append(r)
        for node_rows in by_node.values():
            self.assertEqual(len({tuple(r['choices']) for r in node_rows}), 1)
            self.assertLessEqual(sum(r['result'] == 'proof' for r in node_rows), 1)

    def test_log_disabled_by_default(self):
        clear_log()
        prove([], parse('~~(p | ~p)'))
        self.assertEqual(len(BRANCH_LOG), 0)


class TestMlConsistency(unittest.TestCase):
    def test_move_names_cover_collect_moves(self):
        try:
            from ml import MOVE_NAMES
        except ImportError:
            self.skipTest('ml.py требует numpy/sklearn')
        random.seed(700)
        seen = set()
        for _ in range(300):
            f = random_formula(random.randint(1, 5))
            g = random_formula(random.randint(1, 3))
            for name, _ in collect_moves([f], g):
                seen.add(name)
        self.assertTrue(seen.issubset(set(MOVE_NAMES)),
                        seen - set(MOVE_NAMES))


if __name__ == '__main__':
    unittest.main(verbosity=2)
