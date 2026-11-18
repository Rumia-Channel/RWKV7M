"""Label-free public-card prediction and sealed elementary/MIDTERM assessment.

Every bank is prepared from its world's windows before any question is read.
Only a previous hard prediction chooses a card row. Gold values and source
positions are used after the public program, never by that program or its bank.
"""
import argparse
import csv
from dataclasses import dataclass
from hashlib import file_digest
import json
from operator import index
from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from rwkv7m.io.role_binding import load_policy
from rwkv7m.model.episodic_address import (
    BITS, SLOTS, _address, empty_state, normalize, project, write_window,
)
from rwkv7m.model.episodic_lexical import lexical_features, word_counts
from rwkv7m.model.role_binding import ARMS, marked_write, read_lanes, route

SEEDS = (42, 43, 44)
UPDATES = '960'
COUNT_ARMS = ('bag', 'marked-count')
MARKED_ARMS = ('marked-count', 'marked-rwkv')
BASE = ('arm', 'seed', 'curriculum', 'weight_sha', 'head_sha', 'form', 'surface', 'live', 'gap')
IDENTITY = ('draw', 'family', 'input_sha', 'question')
RAW_FIELDS = (*BASE, *IDENTITY, 'world', 'step', 'worlds', 'reads', 'query_index',
              'query_tokens', 'active', 'gold', 'hard', 'support', 'source',
              'read_address', 'first_gold', 'final_gold', 'first_hard', 'final_hard',
              'swapped_first', 'swapped_hard', 'early_stop_hard', 'oracle_terminal',
              'oracle_source', 'actual_reads', 'step_bank_reads', 'bank_reads',
              'one_step_bank_reads')
FAMILY_FIELDS = (*BASE, *IDENTITY, 'worlds', 'reads', 'answer_success',
                 'source_success', 'joint_success', 'all_steps_answer_success',
                 'swapped_success', 'early_stop_success', 'oracle_terminal_success')
SCORE_FIELDS = (*BASE, 'worlds', 'reads', 'families', 'family_questions',
                'world_questions', 'answer_count', 'answer_rate', 'source_count',
                'source_rate', 'joint_count', 'joint_rate', 'final_correct',
                'first_correct', 'first_fail', 'later_fail', 'all_steps_correct',
                'all_steps_answer_count', 'swapped_count', 'early_stop_count',
                'oracle_terminal_count', 'oracle_terminal_correct', 'actual_reads',
                'bank_reads', 'one_step_bank_reads', 'programmed_bank_reads')
STEP_FIELDS = (*BASE, 'worlds', 'reads', 'step', 'world_questions', 'correct',
               'source_correct', 'joint_correct', 'active_reads', 'bank_reads')
GATE_FIELDS = ('arm', 'form', 'surface', 'live', 'seed', 'conditions',
               'answer_gate', 'source_gate', 'joint_gate')
EFFECT_FIELDS = ('form', 'surface', 'live', 'gap', 'treatment', 'reference',
                 'metric', 'delta_pp', 'lower95_pp', 'upper95_pp', 'resamples',
                 'draws', 'families_per_draw', 'seeds', 'questions_per_family')
STATUS_FIELDS = (*BASE, 'status', 'reason')
OUTPUT_FILES = ('readouts.csv', 'families.csv', 'scores.csv', 'steps.csv',
                'gates.csv', 'paired-effects.csv', 'status.csv')


def digest(path):
    with Path(path).open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


@dataclass(frozen=True)
class Policy:
    params: dict
    mean: np.ndarray
    scale: np.ndarray
    meta: dict
    router: object
    head_sha: str

    @property
    def arm(self):
        return self.meta['arm']

    @property
    def neural(self):
        return self.arm not in COUNT_ARMS


def load_registered_policy(path, *, arm=None, seed=None, weight_sha=None, curriculum=None):
    params, mean, scale, meta, router = load_policy(path)
    if (meta['updates'] != UPDATES or int(meta['seed']) not in SEEDS or
            (arm is not None and meta['arm'] != arm) or
            (seed is not None and meta['seed'] != str(seed)) or
            (weight_sha is not None and meta['weight_sha'] != weight_sha) or
            (curriculum is not None and meta['curriculum'] != curriculum)):
        raise ValueError(f'wrong registered 960-update role-binding fit: {path}')
    return Policy(params, mean, scale, meta, router, digest(path))


def _indices(values, shape, name, upper=None):
    values = np.asarray(values)
    if (values.shape != shape or values.dtype.kind not in 'iu' or
            np.any(values < 0) or np.any(values > np.iinfo(np.int32).max) or
            (upper is not None and np.any(values >= upper))):
        raise ValueError(f'expected nonnegative integer {name}{shape}' +
                         ('' if upper is None else f' below {upper}'))
    return values.astype(np.int32, copy=False)


def select_followup(cards, predictions):
    """Class-indexed public lookup; -1 is absorbing, never Python's last row."""
    cards, predictions = np.asarray(cards), np.asarray(predictions)
    if predictions.ndim != 2 or predictions.dtype.kind not in 'iu':
        raise ValueError('expected integer predictions [world, question]')
    cards = _indices(cards, (len(predictions), 8), 'cards')
    if np.any((predictions < -1) | (predictions >= 8)):
        raise ValueError('hard predictions must be -1 or an object class 0..7')
    active = predictions >= 0
    selected = np.full(predictions.shape, -1, np.int32)
    lanes, questions = np.nonzero(active)
    selected[lanes, questions] = cards[lanes, predictions[lanes, questions]]
    return selected, active


class Readout(NamedTuple):
    answers: np.ndarray
    sources: np.ndarray
    addresses: np.ndarray


@dataclass(frozen=True)
class ProgramTrace:
    predictions: np.ndarray  # [world, question, step]
    sources: np.ndarray
    addresses: np.ndarray
    queries: np.ndarray
    active: np.ndarray

    @property
    def first(self):
        return self.predictions[..., 0]

    @property
    def final(self):
        return self.predictions[..., -1]


def run_program(reader, queries, cards, reads):
    """Run prepare(W) -> root -> predicted-value card lookup -> query, without labels.

    reader.read receives only the selected public question IDs and active lanes.
    It must not mutate the prepared world. Inactive paths never issue a bank read.
    """
    try:
        steps = index(reads)
    except TypeError as error:
        raise ValueError('reads must be a positive integer') from error
    if isinstance(reads, (bool, np.bool_)) or steps < 1:
        raise ValueError('reads must be a positive integer')
    queries = np.asarray(queries)
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError('expected two starting questions per world')
    queries = _indices(queries, queries.shape, 'queries')
    cards = np.asarray(cards)
    card_width = 0 if steps == 1 and cards.shape == (len(queries), 0) else 8
    cards = _indices(cards, (len(queries), card_width), 'cards')
    shape = (*queries.shape, steps)
    predictions, sources, addresses, chosen = (np.full(shape, -1, np.int32) for _ in range(4))
    activity = np.zeros(shape, bool)
    current, active = queries, np.ones(queries.shape, bool)
    for step in range(steps):
        chosen[..., step] = np.where(active, current, -1)
        activity[..., step] = active
        if active.any():
            result = reader.read(current, active)
            for name, values in zip(('answers', 'sources', 'addresses'), result, strict=True):
                if np.asarray(values).shape != queries.shape:
                    raise ValueError(f'reader returned malformed {name}')
            hard = np.asarray(result.answers)
            if hard.dtype.kind not in 'iu' or np.any((hard[active] < -1) | (hard[active] >= 8)):
                raise ValueError('reader returned an invalid object class')
            predictions[..., step] = np.where(active, hard, -1)
            sources[..., step] = np.where(active, result.sources, -1)
            addresses[..., step] = np.where(active, result.addresses, -1)
        if step + 1 < steps:
            current, active = select_followup(cards, predictions[..., step])
    return ProgramTrace(predictions, sources, addresses, chosen, activity)


class PreparedPolicy:
    """One immutable world bank per lane, plus an external scoring-only owner trace."""

    def __init__(self, policy, tokens, blocks, *, hidden=None, ordinary=None,
                 ordinary_queries=None, track_sources=True):
        self.policy = policy
        tokens = np.asarray(tokens)
        if tokens.ndim != 2 or not len(tokens):
            raise ValueError('expected a nonempty public token table')
        self.tokens = _indices(tokens, (len(tokens), 8), 'feature_tokens', 128)
        blocks = np.asarray(blocks)
        if blocks.ndim != 2 or not blocks.shape[1]:
            raise ValueError('expected nonempty world windows [world, window]')
        blocks = _indices(blocks, blocks.shape, 'blocks', len(tokens))
        self.lanes = len(blocks)
        self.state = None
        self.owners = None  # Never a member of the model's State, or a model input.
        if policy.arm == 'ordinary':
            keys = np.asarray(ordinary_queries)
            if keys.ndim != 2 or len(keys) != self.lanes:
                raise ValueError('expected ordinary public query keys [world, query]')
            self.ordinary_queries = _indices(keys, keys.shape, 'ordinary_queries', len(tokens))
            ordinary = np.asarray(ordinary)
            if (ordinary.shape != (*keys.shape, 256) or ordinary.dtype != np.float32 or
                    not np.isfinite(ordinary).all()):
                raise ValueError('expected actual continuous ordinary RWKV features')
            self.ordinary = normalize(ordinary, policy.mean, policy.scale)
            return
        if policy.neural:
            hidden = np.asarray(hidden)
            if hidden.shape != (len(tokens), 256):
                raise ValueError('expected local frozen RWKV features [utterance,256]')
            features = lexical_features(hidden, tokens)
        else:
            features = word_counts(tokens)
        self.features = normalize(features, policy.mean, policy.scale)
        self.state = empty_state(self.lanes)
        if track_sources: self.owners = np.full((self.lanes, SLOTS), -1, np.int32)
        marked = policy.arm in MARKED_ARMS
        routes = route(policy.router, tokens)[blocks] if marked else None
        present = np.ones(self.lanes, bool) if marked else None
        for step in range(blocks.shape[1]):
            local = self.features[blocks[:, step]]
            if marked:
                self.state, addresses, enabled = marked_write(
                    policy.params, self.state, local, present, routes[:, step])
            else:
                self.state, addresses, enabled = write_window(policy.params, self.state, local)
            if self.owners is not None:
                addresses, enabled = np.asarray(addresses), np.asarray(enabled)
                lanes = np.flatnonzero(enabled)
                self.owners[lanes, addresses[lanes]] = step

    def read(self, query_ids, active):
        query_ids, active = np.asarray(query_ids), np.asarray(active)
        if (query_ids.shape != (self.lanes, 2) or query_ids.dtype.kind not in 'iu' or
                active.shape != query_ids.shape or active.dtype != bool):
            raise ValueError('expected two integer queries and activity bits per world')
        if np.any((query_ids[active] < 0) | (query_ids[active] >= len(self.tokens))):
            raise ValueError('public query index out of bounds')
        answers, sources, addresses = (np.full(query_ids.shape, -1, np.int32) for _ in range(3))
        lanes, questions = np.nonzero(active)
        if not len(lanes):
            return Readout(answers, sources, addresses)
        selected = query_ids[lanes, questions]
        policy = self.policy
        if policy.arm == 'ordinary':
            matches = self.ordinary_queries[lanes] == selected[:, None]
            if not matches.any(axis=1).all():
                raise ValueError('selected public question has no continuous prefix feature')
            # Gather only selected encodings. Never project all ten branch answers.
            x = self.ordinary[lanes, matches.argmax(axis=1)]
            answers[lanes, questions] = np.asarray(project(policy.params, x).argmax(axis=-1))
        else:
            if policy.arm in MARKED_ARMS:
                addr = route(policy.router, self.tokens[selected])
            else:
                addr = _address(project(policy.params, self.features[selected])[:, BITS:2*BITS])
            result, address = map(np.asarray, read_lanes(self.state, lanes, addr))
            answers[lanes, questions], addresses[lanes, questions] = result, address
            if self.owners is not None:
                safe = (address >= 0) & (address < SLOTS)
                sources[lanes[safe], questions[safe]] = self.owners[lanes[safe], address[safe]]
        return Readout(answers, sources, addresses)


def prepare(policy, tokens, blocks, **kwargs):
    """Prepare from public windows/features only; no labels or support arguments."""
    return PreparedPolicy(policy, tokens, blocks, **kwargs)


def _public_layout(data):
    blocks = np.asarray(data['blocks'])
    if blocks.ndim != 3 or blocks.shape[1] not in (2, 4):
        raise ValueError('expected paired or four-world public windows')
    families, worlds, windows = blocks.shape
    queries = _indices(data['queries'], (families, 2), 'queries', len(data['feature_tokens']))
    reads = data.get('reads', 1)
    cards = data.get('cards', np.empty((families, 0), np.int32))
    return families, worlds, windows, queries, np.asarray(cards), reads


def _prepare_cell(policy, data):
    families, worlds, windows, queries, cards, reads = _public_layout(data)
    keys = np.repeat(np.concatenate((queries, cards), axis=1), worlds, axis=0)
    kwargs = {}
    if policy.arm == 'ordinary':
        kwargs = dict(ordinary=np.asarray(data['ordinary']).reshape(families * worlds, keys.shape[1], 256),
                      ordinary_queries=keys)
    elif policy.neural:
        kwargs = dict(hidden=data['features'])
    reader = prepare(policy, data['feature_tokens'], data['blocks'].reshape(families * worlds, windows), **kwargs)
    return reader, np.repeat(queries, worlds, axis=0), np.repeat(cards, worlds, axis=0), reads


def quartet_flags(predictions, gold, sources=None, support=None):
    """Pair/quartet success is an AND within each fixed family and question."""
    predictions, gold = np.asarray(predictions), np.asarray(gold)
    if predictions.shape != gold.shape or predictions.ndim != 3:
        raise ValueError('expected final predictions/gold [family,world,question]')
    answer = (predictions == gold).all(axis=1)
    if sources is None:
        return answer, None, None
    sources, support = np.asarray(sources), np.asarray(support)
    if (sources.shape != support.shape or sources.ndim != 4 or sources.shape[:3] != gold.shape):
        raise ValueError('expected source/support [family,world,question,step]')
    source = (sources == support).all(axis=(1, 3))
    return answer, source, answer & source


def _labels(data, families, worlds, reads):
    gold = _indices(data['answers'], (families, worlds, 2), 'answers', 8)
    shape = (families, worlds, 2, reads)
    values = np.asarray(data.get('values', gold[..., None]))
    support = np.asarray(data['support'])
    if reads == 1 and support.ndim == 3:
        support = support[..., None]
    values = _indices(values, shape, 'values', 8)
    support = _indices(support, shape, 'support', data['blocks'].shape[-1])
    if not np.array_equal(values[..., -1], gold):
        raise ValueError('terminal gold does not match the last sealed step')
    return gold, values, support


def evaluate_cell(policy, data, base):
    """Return actual paths first; then sealed labels and separately named controls."""
    families, worlds, _, _, _, _ = _public_layout(data)
    reader, queries, cards, reads = _prepare_cell(policy, data)
    trace = run_program(reader, queries, cards, reads)
    swapped = run_program(reader, queries[:, ::-1], cards, reads)
    # Gold is first accessed after both actual public programs are complete.
    gold, values, support = _labels(data, families, worlds, reads)
    if reads > 1:
        oracle_query, oracle_active = select_followup(cards, values[..., -2].reshape(-1, 2))
        oracle = reader.read(oracle_query, oracle_active)
    else:
        oracle = Readout(trace.final, trace.sources[..., -1], trace.addresses[..., -1])
    shape = (families, worlds, 2, reads)
    hard, sources, active = (x.reshape(shape) for x in (trace.predictions, trace.sources, trace.active))
    final = hard[..., -1]
    answer, source, joint = quartet_flags(final, gold,
        None if policy.arm == 'ordinary' else sources, None if policy.arm == 'ordinary' else support)
    step_hit = hard == values
    swapped_final = swapped.final.reshape(gold.shape)
    first = hard[..., 0]
    oracle_final = oracle.answers.reshape(gold.shape)
    family_flags = dict(answer_success=answer, source_success=source, joint_success=joint,
                        all_steps_answer_success=step_hit.all(axis=(1, 3)),
                        swapped_success=(swapped_final == gold).all(axis=1),
                        early_stop_success=(first == gold).all(axis=1),
                        oracle_terminal_success=(oracle_final == gold).all(axis=1))
    nq = families * 2
    count = lambda name: '' if family_flags[name] is None else int(family_flags[name].sum())
    rate = lambda name: '' if family_flags[name] is None else float(family_flags[name].mean())
    bank = policy.arm != 'ordinary'
    summary = dict(base, worlds=worlds, reads=reads, families=families,
                   family_questions=nq, world_questions=gold.size,
                   answer_count=count('answer_success'), answer_rate=rate('answer_success'),
                   source_count=count('source_success'), source_rate=rate('source_success'),
                   joint_count=count('joint_success'), joint_rate=rate('joint_success'),
                   final_correct=int((final == gold).sum()), first_correct=int(step_hit[..., 0].sum()),
                   first_fail=int((~step_hit[..., 0]).sum()),
                   later_fail=int((step_hit[..., 0] & ~step_hit.all(axis=-1)).sum()),
                   all_steps_correct=int(step_hit.all(axis=-1).sum()),
                   all_steps_answer_count=count('all_steps_answer_success'),
                   swapped_count=count('swapped_success'), early_stop_count=count('early_stop_success'),
                   oracle_terminal_count=count('oracle_terminal_success'),
                   oracle_terminal_correct=int((oracle_final == gold).sum()), actual_reads=int(active.sum()),
                   bank_reads=int(active.sum()) if bank else '',
                   one_step_bank_reads=gold.size if bank else '',
                   programmed_bank_reads=gold.size * reads if bank else '')
    steps = []
    for step in range(reads):
        located = sources[..., step] == support[..., step]
        steps.append(dict(base, worlds=worlds, reads=reads, step=step,
                          world_questions=gold.size, correct=int(step_hit[..., step].sum()),
                          source_correct=int(located.sum()) if bank else '',
                          joint_correct=int((located & step_hit[..., step]).sum()) if bank else '',
                          active_reads=int(active[..., step].sum()),
                          bank_reads=int(active[..., step].sum()) if bank else ''))
    family_rows, raw_rows = [], []
    for family in range(families):
        identity = dict(draw=int(data['draw'][family]), family=int(data['family'][family]),
                        input_sha=str(data['input_sha'][family]))
        for question in range(2):
            record = dict(base, **identity, question=question, worlds=worlds, reads=reads)
            family_rows.append(record | {key: '' if flags is None else int(flags[family, question])
                                         for key, flags in family_flags.items()})
            for world in range(worlds):
                lane = family * worlds + world
                suffix = dict(first_gold=int(values[family, world, question, 0]),
                              final_gold=int(gold[family, world, question]),
                              first_hard=int(first[family, world, question]),
                              final_hard=int(final[family, world, question]),
                              swapped_first=int(swapped.first[lane, question]),
                              swapped_hard=int(swapped.final[lane, question]),
                              early_stop_hard=int(first[family, world, question]),
                              oracle_terminal=int(oracle.answers[lane, question]),
                              oracle_source=int(oracle.sources[lane, question]) if bank else '',
                              actual_reads=int(trace.active[lane, question].sum()),
                              bank_reads=int(trace.active[lane, question].sum()) if bank else '',
                              one_step_bank_reads=1 if bank else '')
                for step in range(reads):
                    query = int(trace.queries[lane, question, step])
                    raw_rows.append(dict(record, world=world, step=step, query_index=query,
                        query_tokens='' if query < 0 else ' '.join(map(str, reader.tokens[query])),
                        active=int(trace.active[lane, question, step]),
                        gold=int(values[family, world, question, step]),
                        hard=int(trace.predictions[lane, question, step]),
                        support=int(support[family, world, question, step]),
                        source=int(trace.sources[lane, question, step]) if bank else '',
                        read_address=int(trace.addresses[lane, question, step]) if bank else '',
                        step_bank_reads=int(trace.active[lane, question, step]) if bank else '', **suffix))
    return raw_rows, family_rows, summary, steps


def _writer(stream, fields):
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    return writer


def _save_csv(path, fields, rows):
    with Path(path).open('x', encoding='utf8', newline='') as stream:
        _writer(stream, fields).writerows(rows)


def gate_rows(summaries):
    groups = {}
    for row in summaries:
        key = tuple(row[key] for key in ('arm', 'form', 'surface', 'live'))
        groups.setdefault(key, []).append(row)
    for (arm, form, surface, live), rows in groups.items():
        for seed in (*SEEDS, 'all'):
            selected = [row for row in rows if seed == 'all' or row['seed'] == seed]
            # A missing seed/gap is not a passing gate.
            expected = {(s, gap) for s in (SEEDS if seed == 'all' else (seed,)) for gap in (0, 128)}
            complete = (len(selected) == len(expected) and
                        {(row['seed'], row['gap']) for row in selected} == expected)
            def passes(metric):
                return int(complete and all(10 * row[metric] >= 9 * row['family_questions'] for row in selected))
            yield dict(arm=arm, form=form, surface=surface, live=live, seed=seed,
                       conditions=len(selected), answer_gate=passes('answer_count'),
                       source_gate='' if arm == 'ordinary' else passes('source_count'),
                       joint_gate='' if arm == 'ordinary' else passes('joint_count'))


def paired_effect_rows(family_rows, *, resamples=10000):
    """Paired draw -> family bootstrap, with all three weight seeds kept together."""
    rng = np.random.default_rng(766919)
    draws = rng.integers(5, size=(resamples, 5, 1))
    families = rng.integers(8, size=(resamples, 5, 8))
    groups = {}
    for row in family_rows:
        if row['arm'] in ('marked-rwkv', 'learned', 'marked-count'):
            key = tuple(row[key] for key in ('form', 'surface', 'live', 'gap'))
            groups.setdefault(key, {})[(row['arm'], row['seed'], row['draw'], row['family'], row['question'])] = row
    for condition, records in groups.items():
        draw_ids = sorted({key[2] for key in records})
        if len(draw_ids) != 5:
            raise ValueError('paired bootstrap requires exactly five independent draws')
        for reference in ('learned', 'marked-count'):
            available = {key[0] for key in records}
            if 'marked-rwkv' not in available or reference not in available:
                continue
            complete = all((arm, seed, draw, family, question) in records
                           for arm in ('marked-rwkv', reference) for seed in SEEDS
                           for draw in draw_ids for family in range(8) for question in range(2))
            if not complete:
                if condition[1] == 'seen':
                    continue
                raise ValueError('incomplete paired arm/seed/draw/family/question outcomes')
            for metric in ('answer_success', 'source_success', 'joint_success'):
                differences = np.empty((3, 5, 8, 2), np.float64)
                for si, seed in enumerate(SEEDS):
                    for di, draw in enumerate(draw_ids):
                        for family in range(8):
                            for question in range(2):
                                key = (seed, draw, family, question)
                                a = records[('marked-rwkv', *key)][metric]
                                b = records[(reference, *key)][metric]
                                if a == '' or b == '':
                                    raise ValueError('missing paired source or joint outcome')
                                differences[si, di, family, question] = a - b
                paired = differences.mean(axis=(0, 3)) * 100.
                lower, upper = np.quantile(paired[draws, families].mean(axis=(1, 2)), (.025, .975))
                yield dict(zip(('form', 'surface', 'live', 'gap'), condition, strict=True),
                           treatment='marked-rwkv', reference=reference, metric=metric,
                           delta_pp=float(paired.mean()), lower95_pp=float(lower), upper95_pp=float(upper),
                           resamples=resamples, draws=5, families_per_draw=8, seeds=3, questions_per_family=2)


def assess(cache, heads, output, *, basic_control=False):
    from benchmark_address_screening import checkpoints
    from role_binding_data import TEST_CELLS
    from role_binding_mixed_features import exam_cell
    from role_binding_observations import load_cell, raw_cell

    cache, heads, output = Path(cache), Path(heads), Path(output)
    registry = {seed: (path, sha) for seed, path, sha in checkpoints()}
    arms = MARKED_ARMS if basic_control else ARMS
    curriculum = 'basic-only' if basic_control else 'mixed'
    fitted = {}
    for seed in SEEDS:
        for arm in arms:
            sha = 'none' if arm in COUNT_ARMS else registry[seed][1]
            fitted[seed, arm] = load_registered_policy(heads / f'{arm}-s{seed}.safetensors',
                                                      arm=arm, seed=seed, weight_sha=sha,
                                                      curriculum=curriculum)
    output.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        if (output / name).exists():
            raise FileExistsError(f'refusing to replace assessment output: {output / name}')
    summaries, all_families, all_steps, statuses, identities = [], [], [], [], {}
    with (output / 'readouts.csv').open('x', encoding='utf8', newline='') as raw_stream, \
            (output / 'families.csv').open('x', encoding='utf8', newline='') as family_stream:
        raw_writer, family_writer = _writer(raw_stream, RAW_FIELDS), _writer(family_stream, FAMILY_FIELDS)
        for seed in SEEDS:
            sha = registry[seed][1]
            conditions = [] if basic_control else [
                ('elementary', surface, live, gap)
                for surface in ('held', 'shift', 'seen') for live, gap in TEST_CELLS]
            conditions += [('midterm', surface, None, gap)
                           for surface in ('midterm', 'familiar') for gap in (0, 128)]
            for form, surface, live, gap in conditions:
                if form == 'elementary':
                    raw = raw_cell('test', surface, live, gap)
                    neural = load_cell(cache, 'test', seed, sha, surface, live, gap)
                else:
                    raw = exam_cell(cache, 'midterm', seed, 'none', surface, gap, neural=False)
                    neural = exam_cell(cache, 'midterm', seed, sha, surface, gap, neural=True)
                    live = raw['blocks'].shape[-1] - gap
                for arm in arms:
                    policy = fitted[seed, arm]
                    base = dict(arm=arm, seed=seed, curriculum=curriculum, weight_sha=policy.meta['weight_sha'],
                                head_sha=policy.head_sha, form=form, surface=surface, live=live, gap=gap)
                    data = neural if policy.neural else raw
                    manifest = tuple(zip(map(int, data['draw']), map(int, data['family']),
                                         map(str, data['input_sha']), strict=True))
                    key = (form, surface, live, gap)
                    if identities.setdefault(key, manifest) != manifest:
                        raise ValueError('public family identities differ across arms or weights')
                    raw_rows, family_rows, summary, step_rows = evaluate_cell(policy, data, base)
                    raw_writer.writerows(raw_rows)
                    family_writer.writerows(family_rows)
                    summaries.append(summary)
                    all_families.extend(family_rows)
                    all_steps.extend(step_rows)
                    statuses.append(dict(base, status='scored', reason=''))
                raw_stream.flush()
                family_stream.flush()
    _save_csv(output / 'scores.csv', SCORE_FIELDS, summaries)
    _save_csv(output / 'steps.csv', STEP_FIELDS, all_steps)
    _save_csv(output / 'gates.csv', GATE_FIELDS, gate_rows(summaries))
    _save_csv(output / 'paired-effects.csv', EFFECT_FIELDS, paired_effect_rows(all_families))
    _save_csv(output / 'status.csv', STATUS_FIELDS, statuses)
    return summaries


def parse_public(data):
    """Validate the exact raw JSON boundary, including arbitrary public card order."""
    from role_binding_data import OBJECT_TOKENS

    if not isinstance(data, dict) or set(data) != {'blocks', 'queries', 'reads', 'cards'}:
        raise ValueError('only public blocks, queries, reads, and cards are accepted')
    reads = data['reads']
    if isinstance(reads, bool) or not isinstance(reads, int) or reads < 1:
        raise ValueError('reads must be a positive integer')
    blocks = np.asarray(data['blocks'])
    if blocks.ndim != 2 or not len(blocks):
        raise ValueError('expected nonempty raw blocks [window,8]')
    blocks = _indices(blocks, (len(blocks), 8), 'blocks', 128)
    queries = _indices(data['queries'], (2, 8), 'queries', 128)
    cards = data['cards']
    if not isinstance(cards, list) or len(cards) not in ((0, 8) if reads == 1 else (8,)):
        raise ValueError('cards must contain eight object-token rows (or be empty for reads=1)')
    ordered = np.empty((len(cards), 8), np.int32)
    seen = set()
    for card in cards:
        if not isinstance(card, dict) or set(card) != {'on', 'ask'}:
            raise ValueError('each card row must contain only on and ask')
        on = card['on']
        if isinstance(on, bool) or not isinstance(on, int) or on not in OBJECT_TOKENS or on in seen:
            raise ValueError('card on values must be each actual public object token exactly once')
        seen.add(on)
        ordered[OBJECT_TOKENS.index(on)] = _indices(card['ask'], (8,), 'card ask', 128)
    return blocks, queries, ordered, reads


def predict_data(policy, data, *, encoder=None):
    """Shared streaming implementation for raw CLI prediction; accepts no supervision."""
    blocks, queries, cards, reads = parse_public(data)
    tokens, ids = np.unique(np.concatenate((blocks, queries, cards)), axis=0, return_inverse=True)
    ib, iq, ic = np.split(ids.astype(np.int32), (len(blocks), len(blocks) + 2))
    kwargs = {}
    if policy.neural:
        if encoder is None:
            from benchmark_address_screening import checkpoints
            from episodic_scene_features import WindowEncoder
            _, path, sha = next(item for item in checkpoints() if item[0] == int(policy.meta['seed']))
            if sha != policy.meta['weight_sha']:
                raise ValueError('wrong SHA-authenticated frozen RWKV for this head')
            encoder = WindowEncoder(path)
        if (encoder.weight_seed != int(policy.meta['seed']) or
                encoder.weight_sha != policy.meta['weight_sha']):
            raise ValueError('encoder identity does not match the portable head')
        if policy.arm == 'ordinary':
            public_queries = np.concatenate((queries, cards))
            surfaces = len(public_queries) // 2
            q = public_queries.reshape(surfaces, 1, 2, 8)
            continuous = encoder.ordinary_many(blocks[None], q).transpose(1, 0, 2, 3).reshape(1, -1, 256)
            kwargs = dict(ordinary=continuous, ordinary_queries=np.concatenate((iq, ic))[None])
        else:
            kwargs = dict(hidden=encoder.local(tokens))
    reader = prepare(policy, tokens, ib[None], track_sources=False, **kwargs)
    trace = run_program(reader, iq[None], ic[None], reads)
    paths = []
    for question in range(2):
        path = []
        for step in range(reads):
            query = int(trace.queries[0, question, step])
            path.append(dict(step=step, ask=None if query < 0 else tokens[query].tolist(),
                             prediction=int(trace.predictions[0, question, step]),
                             address=None if policy.arm == 'ordinary' else int(trace.addresses[0, question, step]),
                             active=bool(trace.active[0, question, step])))
        paths.append(path)
    bank = policy.arm != 'ordinary'
    return dict(answers=trace.final[0].tolist(), paths=paths, reads=reads,
                actual_reads=trace.active[0].sum(axis=-1).tolist(),
                bank_reads=trace.active[0].sum(axis=-1).tolist() if bank else None,
                one_step_bank_reads=[1, 1] if bank else None, head_sha=policy.head_sha,
                curriculum=policy.meta['curriculum'])


def predict(head, source):
    policy = load_registered_policy(head)
    if source == '-':
        data = json.load(sys.stdin)
    else:
        with Path(source).open(encoding='utf8') as stream:
            data = json.load(stream)
    result = predict_data(policy, data)
    print(json.dumps(result, allow_nan=False))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    assessment = sub.add_parser('assess')
    assessment.add_argument('cache', type=Path)
    assessment.add_argument('heads', type=Path)
    assessment.add_argument('output', type=Path)
    assessment.add_argument('--basic-control', action='store_true',
                            help='score basic-only marked heads on the sealed midterm forms only')
    prediction = sub.add_parser('predict')
    prediction.add_argument('head', type=Path)
    prediction.add_argument('input', help='public JSON file, or - for stdin')
    args = parser.parse_args(argv)
    if args.mode == 'assess':
        assess(args.cache, args.heads, args.output, basic_control=args.basic_control)
    else:
        predict(args.head, args.input)


if __name__ == '__main__':
    main()
