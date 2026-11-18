"""Score blind, fixed-width binding replay readouts without loading a model.

Mixed subsets come from changed truths against the original near case, not the
mixed-near recency ring. The far public recency mask must agree with those four
rewritten keys. Prediction -1 denotes nonfinite logits; incomplete vectors fail
rather than disappearing from a denominator. All token IDs 32..47 are values.
"""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
from pathlib import Path

from score_binding_state import _answers, _csv, _programs, key_hits


ARMS = ('guard', 'replay', 'fresh')
WEIGHT_SEEDS = (42, 43, 44)
FAMILIES = 16
CONDITIONS = (('near', 0), ('overwrite', 32), ('change', 1024),
              ('filler', 1024), ('mixed', 0), ('mixed', 1024))
DEPTH = 24
ERROR_KINDS = ('latest', 'own_obsolete', 'another_queried', 'distractor_value', 'nonvalue')
COLUMNS = ('weight_seed', 'task', 'gap', 'data_seed', 'family', 'keys', 'recent', 'hit',
           *(f'{arm}{world}' for arm in ('truth', *ARMS, 'tail') for world in (0, 1)),
           'input_sha256')


def observations(path):
    expected = {(seed, family, task, gap) for seed in WEIGHT_SEEDS
                for family in range(FAMILIES) for task, gap in CONDITIONS}
    data = {}; inputs = {}
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(COLUMNS) <= set(reader.fieldnames or ()):
            raise ValueError('missing replay readout columns')
        for line in reader:
            seed, family, gap, draw = (int(line[name]) for name in
                                      ('weight_seed', 'family', 'gap', 'data_seed'))
            task = line['task']; key = seed, family, task, gap
            if key not in expected or key in data:
                raise ValueError(f'unexpected or duplicate observation {key}')
            n = 2 if task == 'overwrite' else 8

            def integers(field):
                result = tuple(map(int, (line[field] or '').split()))
                if len(result) != n:
                    raise ValueError(f'expected {n} entries in {field} for {key}')
                return result

            keys = integers('keys')
            if len(set(keys)) != n or any(not 16 <= k <= 31 for k in keys):
                raise ValueError(f'invalid query keys for {key}')
            arms = (*ARMS, 'tail') if task == 'mixed' else ARMS
            values = {arm: tuple(integers(f'{arm}{w}') for w in (0, 1))
                      for arm in ('truth', *arms)}
            if any(not 32 <= value <= 47 for world in values['truth'] for value in world):
                raise ValueError(f'invalid truth for {key}')
            for mask in ('recent', 'hit'):
                values[mask] = integers(mask)
                if any(flag not in (0, 1) for flag in values[mask]):
                    raise ValueError(f'invalid {mask} mask for {key}')
            if task != 'mixed' and any((line[f'tail{w}'] or '').strip() for w in (0, 1)):
                raise ValueError(f'tail is only a mixed diagnostic: {key}')
            signature = (draw, line['input_sha256'], keys, values['truth'],
                         values['recent'], values['hit'])
            if inputs.setdefault((family, task, gap), signature) != signature:
                raise ValueError(f'weight seeds received different inputs: {key}')
            data[key] = draw, keys, values
    if set(data) != expected:
        raise ValueError('incomplete fixed replay panel: need seeds 42/43/44, 16 families and six cases')
    return data


def _matched_near(data, seed, family, task, draw, keys, values):
    if task == 'overwrite':
        return None
    near_draw, near_keys, near = data[seed, family, 'mixed' if task == 'mixed' else 'near', 0]
    if (draw, keys, values['truth']) != (near_draw, near_keys, near['truth']):
        raise ValueError('matched near/far case changed operands or questions')
    return near


def _rewritten_keys(data):
    groups = {}
    for seed, family, task, _ in data:
        if task != 'mixed' or (seed, family) in groups:
            continue
        draw, keys, original = data[seed, family, 'near', 0]
        close_draw, close_keys, close = data[seed, family, 'mixed', 0]
        far_draw, far_keys, far = data[seed, family, 'mixed', 1024]
        if (draw, keys) != (close_draw, close_keys) or (draw, keys) != (far_draw, far_keys):
            raise ValueError('mixed cases do not match the original near queries')
        if close['truth'] != far['truth']:
            raise ValueError('mixed near/far truths differ')
        changed = tuple(tuple(latest != old for latest, old in zip(world, base, strict=True))
                        for world, base in zip(close['truth'], original['truth'], strict=True))
        if changed[0] != changed[1] or sum(changed[0]) != 4:
            raise ValueError('mixed cases must genuinely rewrite the same four keys in both worlds')
        if tuple(far['recent']) != changed[0]:
            raise ValueError('mixed-far public recent flags disagree with the four rewritten keys')
        groups[seed, family] = changed[0]
    return groups


def prediction_kind(prediction, latest, obsolete, queried):
    """Classify one world; own stale wins over another queried key's latest value."""
    if prediction == latest:
        return 'latest'
    if prediction == obsolete:
        return 'own_obsolete'
    if prediction in queried:
        return 'another_queried'
    return 'distractor_value' if prediction in range(32, 48) else 'nonvalue'


def summarize(data, *, equations=True):
    retrieval = defaultdict(Counter); arithmetic = defaultdict(Counter); errors = defaultdict(Counter)
    rewritten = _rewritten_keys(data)
    for (seed, family, task, gap), (draw, keys, values) in data.items():
        near = _matched_near(data, seed, family, task, draw, keys, values)
        arms = (*ARMS, 'tail') if task == 'mixed' else ARMS
        if 'calibrated' in values:
            arms = (*arms, 'calibrated')
        subsets = {'all': tuple(range(len(keys)))}
        if task == 'mixed':
            mask = rewritten[seed, family]
            subsets.update(rewritten=tuple(i for i, flag in enumerate(mask) if flag),
                           old=tuple(i for i, flag in enumerate(mask) if not flag))
        for arm in arms:
            pred = values[arm]; truth = values['truth']; hits = key_hits(pred, truth)
            close_hits = key_hits(near[arm], truth) if near is not None else None
            for subset, indices in subsets.items():
                r = retrieval[seed, task, gap, arm, subset]
                r.update(families=1, key_pairs=len(indices), both_worlds=sum(hits[i] for i in indices),
                         questions=2*len(indices),
                         correct=sum(pred[w][i] == truth[w][i] for w in (0, 1) for i in indices),
                         nonfinite=sum(pred[w][i] == -1 for w in (0, 1) for i in indices),
                         different_from_guard=sum(pred[w][i] != values['guard'][w][i]
                                                  for w in (0, 1) for i in indices),
                         suffix_hits=sum(values['hit'][i] for i in indices),
                         paired_key_pairs=len(indices) if near is not None else 0)
                if near is not None:
                    r['near_and_far'] += sum(hits[i] and close_hits[i] for i in indices)
            if task == 'mixed':
                original = data[seed, family, 'near', 0][2]['truth']
                c = errors[seed, gap, arm]
                c.update(families=1, key_pairs=len(subsets['rewritten']),
                         questions=2*len(subsets['rewritten']), nonfinite=0,
                         **dict.fromkeys(ERROR_KINDS, 0))
                for w in (0, 1):
                    queried = set(truth[w])
                    for i in subsets['rewritten']:
                        c[prediction_kind(pred[w][i], truth[w][i], original[w][i], queried)] += 1
                        c['nonfinite'] += pred[w][i] == -1
        if not equations or near is None:
            continue  # The true-overwrite control has only two queries, not an equation group.
        for program in _programs(draw, family, keys, DEPTH):
            gold = _answers(program, keys, values['truth'])
            for arm in arms:
                got = _answers(program, keys, values[arm]); close = _answers(program, keys, near[arm])
                a_hit = all(p[0] == g[0] for p, g in zip(got, gold, strict=True))
                acd_hit = all(p[1] == g[1] for p, g in zip(got, gold, strict=True))
                a_close = all(p[0] == g[0] for p, g in zip(close, gold, strict=True))
                acd_close = all(p[1] == g[1] for p, g in zip(close, gold, strict=True))
                arithmetic[seed, task, gap, DEPTH, arm].update(
                    program_pairs=1, a_both_worlds=int(a_hit), acd_both_worlds=int(acd_hit),
                    both_outputs_both_worlds=int(got == gold),
                    a_near_and_far=int(a_hit and a_close), acd_near_and_far=int(acd_hit and acd_close),
                    near_and_far=int(got == gold and close == gold),
                    both_targets_changed=int(all(a != b for a, b in zip(*gold, strict=True))),
                    invalid_worlds=sum(None in world for world in got))
    return retrieval, arithmetic, errors


def gate_results(retrieval, candidate_arm='replay', *, uniform_gain=False):
    def delta(seed, task, gap, subset='all', field='both_worlds'):
        denominator = 'paired_key_pairs' if field == 'near_and_far' else 'key_pairs'
        candidate = retrieval[seed, task, gap, candidate_arm, subset]
        guard = retrieval[seed, task, gap, 'guard', subset]
        return Fraction(candidate[field], candidate[denominator]) - Fraction(guard[field], guard[denominator])

    gates = {}
    for seed in WEIGHT_SEEDS:
        recent = delta(seed, 'mixed', 1024, 'rewritten')
        old_changed = retrieval[seed, 'mixed', 1024, candidate_arm, 'old']['different_from_guard']
        near = delta(seed, 'near', 0); overwrite = delta(seed, 'overwrite', 32)
        primary = delta(seed, 'change', 1024, field='near_and_far')
        nonfinite = sum(row['nonfinite'] for (s, _, _, arm, subset), row in retrieval.items()
                        if s == seed and arm == candidate_arm and subset == 'all')
        floor = Fraction(1, 10) if uniform_gain or seed == 44 else Fraction(-1, 50)
        passed = (recent >= floor and old_changed == 0 and min(near, overwrite, primary) >= Fraction(-1, 50)
                  and nonfinite == 0)
        gates[seed] = dict(recent_delta=recent, old_changed=old_changed, near_delta=near,
                           overwrite_delta=overwrite, primary_delta=primary, nonfinite=nonfinite, passed=passed)
    return gates


def calibration_sources(development, evaluation):
    """Require the same checkpoints and wholly disjoint development/evaluation draws."""
    if {key[0] for key in development} != {key[0] for key in evaluation}:
        raise ValueError('development/evaluation checkpoint sets differ')
    source = tuple(sorted({draw for draw, _, _ in development.values()}))
    heldout = tuple(sorted({draw for draw, _, _ in evaluation.values()}))
    overlap = set(source) & set(heldout)
    if overlap:
        raise ValueError(f'development/evaluation data seeds overlap: {sorted(overlap)}')
    return source, heldout


def calibrated_data(evaluation, flags):
    """Select an entire checkpoint's replay or guard; never use a row's labels or hits."""
    if set(flags) != {key[0] for key in evaluation}:
        raise ValueError('calibration flags do not match evaluation checkpoints')
    return {key: (draw, keys, {**values, 'calibrated': values['replay' if flags[key[0]] else 'guard']})
            for key, (draw, keys, values) in evaluation.items()}


def _gate_csv_row(gate):
    return {name: float(value) if isinstance(value, Fraction) else value for name, value in gate.items()}


def report(path, directory, *, equations=True, calibrate_with=None):
    development = None; development_gates = {}; flags = {}
    if calibrate_with is not None:
        # Freeze every flag before opening any evaluation predictions.
        development = observations(calibrate_with)
        development_retrieval, _, _ = summarize(development, equations=False)
        development_gates = gate_results(development_retrieval, uniform_gain=True)
        flags = {seed: gate['passed'] for seed, gate in development_gates.items()}
    evaluation = observations(path)
    source = ()
    heldout = tuple(sorted({draw for draw, _, _ in evaluation.values()}))
    if development is not None:
        source, heldout = calibration_sources(development, evaluation)
        evaluation = calibrated_data(evaluation, flags)
    retrieval, arithmetic, errors = summarize(evaluation, equations=equations)
    gates = {'replay': gate_results(retrieval)}
    if development is not None:
        gates['calibrated'] = gate_results(retrieval, 'calibrated')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    _csv(directory/'retrieval.csv', ('weight_seed', 'task', 'gap', 'arm', 'subset'),
         ('families', 'key_pairs', 'both_worlds', 'questions', 'correct', 'nonfinite',
          'paired_key_pairs', 'near_and_far', 'different_from_guard', 'suffix_hits'), retrieval)
    _csv(directory/'equations.csv', ('weight_seed', 'task', 'gap', 'depth', 'arm'),
         ('program_pairs', 'a_both_worlds', 'acd_both_worlds', 'both_outputs_both_worlds',
          'a_near_and_far', 'acd_near_and_far', 'near_and_far', 'both_targets_changed', 'invalid_worlds'), arithmetic)
    _csv(directory/'mixed-recent-errors.csv', ('weight_seed', 'gap', 'arm'),
         ('families', 'key_pairs', 'questions', *ERROR_KINDS, 'nonfinite'), errors)
    source_text = ' '.join(map(str, source)); heldout_text = ' '.join(map(str, heldout))
    gate_fields = ('recent_delta', 'old_changed', 'near_delta', 'overwrite_delta',
                   'primary_delta', 'nonfinite', 'passed')
    gate_rows = {(arm, seed): dict(_gate_csv_row(gate), evaluation_data_seeds=heldout_text,
                                 development_data_seeds=source_text,
                                 calibration_enabled=flags[seed] if arm == 'calibrated' else '')
                 for arm, models in gates.items() for seed, gate in models.items()}
    _csv(directory/'gates.csv', ('arm', 'weight_seed'),
         ('evaluation_data_seeds', 'development_data_seeds', 'calibration_enabled', *gate_fields), gate_rows)
    print(f'Evaluation data seeds: {heldout_text}')
    if development is not None:
        flag_rows = {(seed,): dict(_gate_csv_row(gate), enabled=flags[seed],
                                  development_data_seeds=source_text, evaluation_data_seeds=heldout_text)
                     for seed, gate in development_gates.items()}
        _csv(directory/'calibration.csv', ('weight_seed',),
             ('enabled', 'development_data_seeds', 'evaluation_data_seeds', *gate_fields), flag_rows)
        print(f'Calibration development data seeds: {source_text}')
        for seed, gate in development_gates.items():
            print(f'calibration parent {seed}: frozen enable {flags[seed]}; '
                  f'development recent BOTH-world delta {float(gate["recent_delta"]):+.4f}')
    for arm, models in gates.items():
        for seed, g in models.items():
            print(f'{arm} parent {seed}: mixed-far recent BOTH-world delta {float(g["recent_delta"]):+.4f}; '
                  f'old prediction changes {g["old_changed"]}; near delta {float(g["near_delta"]):+.4f}; '
                  f'true overwrite delta {float(g["overwrite_delta"]):+.4f}; '
                  f'primary near+change1024 delta {float(g["primary_delta"]):+.4f}; '
                  f'nonfinite {g["nonfinite"]}; gate {g["passed"]}')
        print(f'Overall predeclared heldout gate ({arm}): {all(g["passed"] for g in models.values())}')
    return all(g['passed'] for g in gates['replay'].values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('readouts', type=Path)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--skip-equations', action='store_true',
                        help='write an empty equations table instead of running the depth-24 calculator')
    parser.add_argument('--calibrate-with', type=Path, metavar='PUBLISHED_RAW_CSV',
                        help='freeze per-checkpoint enable flags from disjoint development readouts')
    args = parser.parse_args()
    report(args.readouts, args.directory, equations=not args.skip_equations, calibrate_with=args.calibrate_with)
