"""Read the late-write fine-tunes on the public binding panel without any replay.

Three arms answer the same six public cases: the frozen parent, a checkpoint
fine-tuned on counterfactual late writes, and one fine-tuned with the frozen
last-64-token teacher logits. Every arm reads the whole visible history from a
zero state: no replay suffix, no cache, no mask and no label at inference. The
public ground truth is read only after the predictions exist, and only to score
them.

The rewritten subset of the mixed cases is the set of keys whose value really
changed against the matched near case, in both counterfactual worlds. It is not
a recency window, so a stale world, a near miss or a merely recent key never
earns credit.
"""
import argparse
import csv
import json
import struct
from collections import Counter, defaultdict
from hashlib import sha256
from math import prod
from pathlib import Path

import numpy as np

from score_binding_state import _answers, _programs, key_hits


ROOT = Path(__file__).resolve().parents[1]
GUARDED = ROOT / 'experiments/binding-state/guarded'
DEFAULT_PARENTS = ROOT / 'experiments/retention/parents'
ARMS = ('parent', 'ce', 'distill')
WEIGHT_SEEDS = (42, 43, 44)
CASES = (('near', 0), ('mixed', 0), ('mixed', 1024),
         ('change', 1024), ('filler', 1024), ('overwrite', 32))
DIRECT_SEED = 731967
OVERWRITE_SEED = 731968
FAMILIES = 32
BATCH = 8
DEPTH = 24
CAPACITY = 16
PARAMETERS = 504960
KEY_TOKENS = range(16, 32)
VALUE_TOKENS = range(32, 48)
KEY_COUNTS = {('overwrite', 32): 2}
READOUT_COLUMNS = ('arm', 'weight_seed', 'families', 'task', 'gap', 'data_seed', 'family',
                   'checkpoint_sha256', 'input_sha256', 'keys', 'truth0', 'truth1',
                   'pred0', 'pred1', 'rewritten0', 'rewritten1')
MANIFEST_COLUMNS = ('arm', 'weight_seed', 'path', 'checkpoint_sha256', 'parameters',
                    'binding_arm', 'declared_weight_seed', 'parent_weights_sha256')
RETRIEVAL_KEYS = ('weight_seed', 'arm', 'task', 'gap', 'subset')
RETRIEVAL_METRICS = ('both_worlds', 'key_pairs', 'nonfinite', 'near_and_far',
                      'families', 'questions', 'correct', 'paired_key_pairs')
EQUATION_KEYS = ('weight_seed', 'arm', 'task', 'gap', 'depth')
EQUATION_METRICS = ('program_pairs', 'a_both_worlds', 'acd_both_worlds',
                     'both_outputs_both_worlds', 'a_near_and_far', 'acd_near_and_far',
                     'near_and_far', 'both_targets_changed', 'invalid_worlds')


def _digest(history, queries):
    """Public input fingerprint: the visible history and the questions, as int32."""
    digest = sha256(np.asarray(history).astype('<i4', copy=False).tobytes())
    digest.update(np.asarray(queries).astype('<i4', copy=False).tobytes())
    return digest.hexdigest()


def _pack(values):
    return ' '.join(map(str, map(int, values)))


def _header(path):
    """Return the safetensors metadata and parameter count without loading tensors."""
    with Path(path).open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
    metadata = {str(key): str(value) for key, value in (header.get('__metadata__') or {}).items()}
    parameters = sum(prod(tensor['shape']) for name, tensor in header.items() if name != '__metadata__')
    return metadata, parameters


def _published_parents():
    """The published digest of every frozen parent checkpoint."""
    with (GUARDED/'weights.csv').open(encoding='utf-8', newline='') as stream:
        return {int(row['weight_seed']): row['checkpoint_sha256'] for row in csv.DictReader(stream)}


def checkpoint_path(arm, seed, checkpoints, parents):
    """Frozen parents keep their published name; trained arms use <arm>-s<seed>."""
    if arm == 'parent':
        return Path(parents)/f'long-s{seed}.safetensors'
    return Path(checkpoints)/f'{arm}-s{seed}.safetensors'


def resolve_checkpoint(arm, seed, checkpoints, parents, published):
    """Check one checkpoint's identity, size, screening and parent lineage."""
    path = checkpoint_path(arm, seed, checkpoints, parents)
    if not path.is_file():
        raise FileNotFoundError(f'missing {arm} checkpoint for seed {seed}: {path}')
    digest = sha256(path.read_bytes()).hexdigest()
    metadata, parameters = _header(path)
    if parameters != PARAMETERS:
        raise ValueError(f'{path} holds {parameters} parameters, not {PARAMETERS}')
    if metadata.get('use_screening', 'false').lower() == 'true':
        raise ValueError(f'{path} uses screening; the trial requires an ordinary RWKV')
    if metadata.get('vocab_size', '128') != '128':
        raise ValueError(f'{path} does not use the 128-token public vocabulary')
    if arm == 'parent':
        if published.get(seed) != digest:
            raise ValueError(f'{path} is not the published frozen parent for seed {seed}')
    else:
        parent = Path(parents)/f'long-s{seed}.safetensors'
        parent_digest = sha256(parent.read_bytes()).hexdigest()
        if published.get(seed) != parent_digest:
            raise ValueError(f'{parent} is not the published frozen parent for seed {seed}')
        if metadata.get('binding_arm') != arm:
            raise ValueError(f'{path} declares arm {metadata.get("binding_arm")!r}, not {arm!r}')
        try:
            declared_seed = int(metadata.get('weight_seed', ''))
        except ValueError:
            declared_seed = None
        if declared_seed != seed:
            raise ValueError(f'{path} declares weight seed {metadata.get("weight_seed")!r}, not {seed}')
        if metadata.get('parent_weights_sha256') != parent_digest:
            raise ValueError(f'{path} was not trained from the published parent of seed {seed}')
    return path, digest, metadata, parameters


def public_cases(families=FAMILIES):
    """The six fixed public cases: histories, questions and separate ground truth."""
    from benchmark_binding_state import panel
    cases = {}
    for task, gap, data_seed, history, queries, truth in panel(families, DIRECT_SEED,
                                                                OVERWRITE_SEED, guard_study=True):
        if (task, gap) in CASES:
            if (task, gap) in cases:
                raise ValueError(f'the public panel repeats the case {task} {gap}')
            cases[task, gap] = (data_seed, history, queries, truth)
    missing = set(CASES) - set(cases)
    if missing:
        raise ValueError(f'the public panel does not provide {sorted(missing)}')
    return cases


def rewritten_flags(truth, near_truth):
    """The keys whose value really changed against the matched near case, per world."""
    return tuple(tuple(int(value != base) for value, base in zip(world, other, strict=True))
                 for world, other in zip(truth, near_truth, strict=True))


def _visible_writes(tokens, store_token):
    """Complete public STORE/key/value events, in order."""
    events = []
    for position in range(len(tokens) - 2):
        marker, key, value = (int(token) for token in tokens[position:position+3])
        if marker == store_token and key in KEY_TOKENS and value in VALUE_TOKENS:
            events.append((key, value))
    return events


def _check_mixed(history, keys, flags, store_token):
    """The mixed case must really rewrite the flagged keys in the visible tail."""
    count = sum(flags[0])
    if count < 1:
        raise ValueError('the mixed case rewrites no key')
    flagged = {key for key, flag in zip(keys, flags[0], strict=True) if flag}
    for world in (0, 1):
        events = _visible_writes(history[world], store_token)
        if len(events) < count + 1:
            raise ValueError('the mixed case has no counterfactual writes to check')
        latest = events[-count:]
        before = {}
        for key, value in events[:-count]:
            before[key] = value
        for key, value in latest:
            if before.get(key) == value:
                raise ValueError('a mixed write did not change the stored value')
            before[key] = value
        if {key for key, _ in latest} != flagged:
            raise ValueError('the rewritten keys do not match the visible counterfactual writes')


def _row(arm, seed, families, task, gap, data_seed, family, digest, history, queries,
         truth, prediction, near, store_token):
    """One raw readout row with the rewritten flags taken from the changed truth."""
    keys = tuple(map(int, queries[0, :, 1]))
    if tuple(map(int, queries[1, :, 1])) != keys:
        raise ValueError('the counterfactual worlds must share their question keys')
    if task == 'overwrite':
        flags = tuple(tuple(0 for _ in world) for world in truth)
    elif task == 'near':
        flags = tuple(tuple(0 for _ in world) for world in truth)
    else:
        if family not in near:
            raise ValueError('the matched near case must be read before its far case')
        flags = rewritten_flags(truth, near[family])
        if flags[0] != flags[1]:
            raise ValueError('the two worlds must rewrite the same keys')
        if task == 'mixed':
            _check_mixed(history, keys, flags, store_token)
    return (arm, seed, families, task, gap, data_seed, family, digest,
            _digest(history, queries), _pack(keys), _pack(truth[0]), _pack(truth[1]),
            _pack(prediction[0]), _pack(prediction[1]), _pack(flags[0]), _pack(flags[1]))


def measure(checkpoints, output, *, families=FAMILIES, batch=BATCH, seeds=WEIGHT_SEEDS,
            arms=ARMS, parents=DEFAULT_PARENTS):
    """Read every arm over the public panel and write the raw per-family readouts."""
    from binding_state_memory import SnapshotReader
    from retrieval_frontier import STORE
    if families < 1 or batch < 1:
        raise ValueError('families and batch must be positive')
    cases = public_cases(families)
    published = _published_parents()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    with (output/'readouts.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(READOUT_COLUMNS)
        for arm in arms:
            for seed in seeds:
                path, digest, metadata, parameters = resolve_checkpoint(arm, seed, checkpoints,
                                                                        parents, published)
                manifest.append((arm, seed, str(path), digest, parameters,
                                 metadata.get('binding_arm', ''), metadata.get('weight_seed', ''),
                                 metadata.get('parent_weights_sha256', '')))
                reader = SnapshotReader(path, capacity=CAPACITY)
                near = {}
                for task, gap in CASES:
                    data_seed, history, queries, truth = cases[task, gap]
                    for first in range(0, families, batch):
                        stop = first + batch
                        hx, qx, yx = history[first:stop], queries[first:stop], truth[first:stop]
                        predictions = reader.predict_fresh(hx, qx)
                        for index in range(len(hx)):
                            family = first + index
                            writer.writerow(_row(arm, seed, families, task, gap, data_seed, family,
                                                 digest, hx[index], qx[index], yx[index],
                                                 predictions[index], near, STORE))
                            if task == 'near':
                                near[family] = yx[index]
                print(f'{arm} seed {seed}: {families} families over {len(CASES)} cases', flush=True)
    with (output/'checkpoints.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(MANIFEST_COLUMNS)
        writer.writerows(manifest)


def observations(path):
    """Load the raw readouts and enforce the fixed panel's completeness."""
    data = {}
    signatures = {}
    families = {}
    with Path(path).open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(READOUT_COLUMNS) <= set(reader.fieldnames or ()):
            raise ValueError('missing distill readout columns')
        for line in reader:
            arm = line['arm']
            seed = int(line['weight_seed'])
            count = int(line['families'])
            task, gap = line['task'], int(line['gap'])
            draw, family = int(line['data_seed']), int(line['family'])
            key = arm, seed, family, task, gap
            if arm not in ARMS or (task, gap) not in CASES or key in data:
                raise ValueError(f'unexpected or duplicate observation {key}')
            if not 0 <= family < count or families.setdefault((arm, seed), count) != count:
                raise ValueError(f'inconsistent family count for {key}')
            number = KEY_COUNTS.get((task, gap), 8)

            def integers(field):
                result = tuple(map(int, (line[field] or '').split()))
                if len(result) != number:
                    raise ValueError(f'expected {number} entries in {field} for {key}')
                return result

            keys = integers('keys')
            if len(set(keys)) != len(keys) or any(key not in KEY_TOKENS for key in keys):
                raise ValueError(f'invalid question keys for {key}')
            truth = (integers('truth0'), integers('truth1'))
            if any(value not in VALUE_TOKENS for world in truth for value in world):
                raise ValueError(f'invalid truth for {key}')
            prediction = (integers('pred0'), integers('pred1'))
            if any(not -1 <= value < 128 for world in prediction for value in world):
                raise ValueError(f'invalid prediction for {key}')
            flags = (integers('rewritten0'), integers('rewritten1'))
            if any(flag not in (0, 1) for world in flags for flag in world):
                raise ValueError(f'invalid rewritten flag for {key}')
            signature = (draw, line['input_sha256'], keys, truth)
            if draw != (OVERWRITE_SEED if task == 'overwrite' else DIRECT_SEED):
                raise ValueError(f'unregistered evaluation seed: {key}')
            if signatures.setdefault((family, task, gap), signature) != signature:
                raise ValueError(f'arms received different inputs: {key}')
            data[key] = dict(draw=draw, keys=keys, truth=truth, pred=prediction, flags=flags,
                             checkpoint=line['checkpoint_sha256'])
    for (arm, seed), count in families.items():
        expected = {(family, task, gap) for family in range(count) for task, gap in CASES}
        seen = {key[2:] for key in data if key[0] == arm and key[1] == seed}
        if seen != expected:
            raise ValueError(f'incomplete panel for {arm} seed {seed}')
    _validate_flags(data)
    return data


def _validate_flags(data):
    """The recorded rewritten flags must be the keys that really changed."""
    for (arm, seed, family, task, gap), row in data.items():
        if task == 'overwrite':
            if any(flag for world in row['flags'] for flag in world):
                raise ValueError(f'the overwrite control has no late rewrite: {(arm, seed, family)}')
            continue
        near = data[arm, seed, family, 'near', 0]
        flags = rewritten_flags(row['truth'], near['truth'])
        if flags != row['flags']:
            raise ValueError(f'the rewritten flags disagree with the changed truth: {(arm, seed, family, task, gap)}')
        if flags[0] != flags[1]:
            raise ValueError(f'the two worlds must rewrite the same keys: {(arm, seed, family, task, gap)}')


def _matched_near(data, arm, seed, family, task, gap, keys, truth):
    """The matched near row: same family, same questions, same answers."""
    if task == 'overwrite':
        return None
    near_task, near_gap = ('mixed', 0) if task == 'mixed' else ('near', 0)
    near = data[arm, seed, family, near_task, near_gap]
    if near['keys'] != keys:
        raise ValueError('the matched near case changed its questions')
    if near['truth'] != truth:
        raise ValueError('the matched near and far cases changed their answers')
    return near


def summarize(data):
    """Both-world retrieval totals and the depth-24 equation outcomes, per arm."""
    retrieval = defaultdict(Counter)
    arithmetic = defaultdict(Counter)
    for (arm, seed, family, task, gap), row in data.items():
        keys, truth, pred, flags = row['keys'], row['truth'], row['pred'], row['flags']
        near = _matched_near(data, arm, seed, family, task, gap, keys, truth)
        subsets = {'all': tuple(range(len(keys)))}
        if task == 'mixed':
            subsets.update(rewritten=tuple(i for i, flag in enumerate(flags[0]) if flag),
                           old=tuple(i for i, flag in enumerate(flags[0]) if not flag))
        hits = key_hits(pred, truth)
        close_hits = key_hits(near['pred'], truth) if near is not None else None
        for subset, indices in subsets.items():
            counter = retrieval[seed, arm, task, gap, subset]
            counter.update(families=1, key_pairs=len(indices), questions=2*len(indices),
                           both_worlds=sum(hits[i] for i in indices),
                           correct=sum(pred[w][i] == truth[w][i] for w in (0, 1) for i in indices),
                           nonfinite=sum(pred[w][i] == -1 for w in (0, 1) for i in indices),
                           paired_key_pairs=len(indices) if near is not None else 0)
            if near is not None:
                counter['near_and_far'] += sum(hits[i] and close_hits[i] for i in indices)
        if near is None:
            continue  # The true-overwrite control has two operands, not an equation group.
        for program in _programs(row['draw'], family, keys, DEPTH):
            gold = _answers(program, keys, truth)
            got = _answers(program, keys, pred)
            close = _answers(program, keys, near['pred'])
            a_hit = all(p[0] == g[0] for p, g in zip(got, gold, strict=True))
            acd_hit = all(p[1] == g[1] for p, g in zip(got, gold, strict=True))
            a_close = all(p[0] == g[0] for p, g in zip(close, gold, strict=True))
            acd_close = all(p[1] == g[1] for p, g in zip(close, gold, strict=True))
            arithmetic[seed, arm, task, gap, DEPTH].update(
                program_pairs=1, a_both_worlds=int(a_hit), acd_both_worlds=int(acd_hit),
                both_outputs_both_worlds=int(got == gold),
                a_near_and_far=int(a_hit and a_close), acd_near_and_far=int(acd_hit and acd_close),
                near_and_far=int(got == gold and close == gold),
                both_targets_changed=int(all(a != b for a, b in zip(*gold, strict=True))),
                invalid_worlds=sum(None in world for world in got))
    return retrieval, arithmetic


def _write_csv(path, key_columns, metrics, counts):
    """Write one summary table with every metric present in every row."""
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow((*key_columns, *metrics))
        for key in sorted(counts):
            row = counts[key]
            writer.writerow((*key, *(row.get(metric, 0) for metric in metrics)))


def score(readouts, directory):
    """Re-score saved readout rows without a GPU, JAX or checkpoint."""
    data = observations(readouts)
    retrieval, arithmetic = summarize(data)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_csv(directory/'retrieval.csv', RETRIEVAL_KEYS, RETRIEVAL_METRICS, retrieval)
    _write_csv(directory/'equations.csv', EQUATION_KEYS, EQUATION_METRICS, arithmetic)
    for seed, arm, task, gap, subset in sorted(retrieval):
        if subset != 'all':
            continue
        row = retrieval[seed, arm, task, gap, subset]
        print(f'{arm} seed {seed} {task} {gap}: both-world {row["both_worlds"]}/{row["key_pairs"]}, '
              f'near+far {row["near_and_far"]}/{row["paired_key_pairs"]}, nonfinite {row["nonfinite"]}',
              flush=True)
    return retrieval, arithmetic


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoints', type=Path,
                        help='directory with the trained <arm>-s<seed> checkpoints')
    parser.add_argument('output', type=Path, help='new directory for the readouts and their scores')
    parser.add_argument('--families', type=int, default=FAMILIES, help='public history pairs')
    parser.add_argument('--batch', type=int, default=BATCH, help='histories per forward pass')
    parser.add_argument('--seed', type=int, nargs='+', default=list(WEIGHT_SEEDS),
                        help='weight seeds of the checkpoints')
    parser.add_argument('--arm', nargs='+', choices=ARMS, default=list(ARMS),
                        help='arms to read')
    parser.add_argument('--parents', type=Path, default=DEFAULT_PARENTS,
                        help='directory with the frozen long-s<seed> parents')
    parser.add_argument('--score-only', action='store_true',
                        help='re-score an existing readouts.csv without a GPU')
    args = parser.parse_args(argv)
    if args.families < 1 or args.batch < 1:
        parser.error('families and batch must be positive')
    if not args.seed or len(set(args.seed)) != len(args.seed):
        parser.error('weight seeds must be unique')
    if args.score_only:
        score(args.output/'readouts.csv', args.output)
        return
    measure(args.checkpoints, args.output, families=args.families, batch=args.batch,
            seeds=tuple(args.seed), arms=tuple(args.arm), parents=args.parents)
    score(args.output/'readouts.csv', args.output)


if __name__ == '__main__':
    main()
