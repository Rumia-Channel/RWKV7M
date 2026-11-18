"""Supplement to §22.17: two histories differ only in two middle COPY destinations."""
import argparse
import csv
from pathlib import Path

import numpy as np

from addressed_binding_tokens import COPY
from benchmark_address_screening import RAW, checkpoints, feature_readers, record
from score_binding_state import _answers, _programs
from versioned_binding_panel import CHAINS, literal_reference, versioned_cases

SEED, DEPTH, GAP = 732019, 24, 1024


def middle_swap_case(families=32):
    d, g, base, queries, original, live = next(versioned_cases(SEED, families, (DEPTH,), (GAP,)))
    # Start both counterfactuals with *identical* input and original literal values.
    history = np.repeat(base[:, :1], 2, axis=1)
    starts = np.flatnonzero(history[0, 0] == COPY)
    if len(starts) != d * CHAINS or not np.all(history[:, :, starts] == COPY):
        raise ValueError('unaligned copy schedule')
    neutral = history[:, 0, starts[0] + 5].copy()
    history[:, :, starts + 5] = neutral[:, None, None]
    questions = np.repeat(queries[:, :1], 2, axis=1)
    # Swap the addresses of the first two writes in stage 12. Sources and all
    # later bytes, including the 1,024 intervening tokens, remain unchanged.
    pair = starts[(d // 2 - 1) * CHAINS:][:2] + 1
    history[:, 1, pair] = history[:, 0, pair[::-1]]
    # Independent labels: follow the two changed wires (names only) to their
    # terminal slots. All other protected values come from the original lineage.
    truth0 = np.where(live, original[:, 0], neutral[:, None])
    truth = np.repeat(truth0[:, None], 2, axis=1)
    for f in range(families):
        endpoints = base[f, 0, pair]
        for stage in range(d // 2, d):
            pos = starts[stage*CHAINS:(stage+1)*CHAINS]
            move = dict(zip(base[f, 0, pos+2], base[f, 0, pos+1], strict=True))
            endpoints = [move[key] for key in endpoints]
        key_order = questions[f, 0, :, 1].tolist()
        left, right = (key_order.index(key) for key in endpoints)
        truth[f, 1, [left, right]] = truth0[f, [right, left]]
    check = literal_reference(history.reshape(-1, history.shape[-1]), questions[..., 1].reshape(-1, questions.shape[2])).reshape(original.shape)
    if not np.array_equal(truth, check):
        raise ValueError('wire transposition and independent literal interpreter disagree')
    changed = truth[:, 0] != truth[:, 1]
    if (not np.all((history[:, 0] != history[:, 1]).sum(1) == 2)
            or not np.all(changed.sum(1) == 2) or np.any(changed & ~live)):
        raise ValueError('the middle swap must change precisely two surviving operands')
    return ('middle_swap', d, g, SEED, history, questions, truth, live), check


def measure(case, predictions):
    _, d, _, seed, _, q, truth, live = case
    both = (predictions == truth).all(axis=1)
    changed = truth[:, 0] != truth[:, 1]
    correct = different = 0
    for f in range(len(truth)):
        keys = tuple(map(int, q[f, 0, :, 1]))
        operands = tuple(key for key, keep in zip(keys, live[f], strict=True) if keep)
        program, = _programs(seed, f, operands, d)
        def answers(values):
            return _answers(program, keys, tuple(tuple(map(int, world)) for world in values))
        expected = answers(truth[f])
        different += int(expected[0] != expected[1])
        correct += int(answers(predictions[f]) == expected)
    return int(both[changed].sum()), int(changed.sum()), int(both[live].sum()), int(live.sum()), correct, len(truth), different


def run(directory):
    case, literal = middle_swap_case(); h, q = case[4:6]
    # A privileged negative control: even perfect answers for world 0 cannot stand in for world 1.
    no_swap = np.repeat(literal[:, :1], 2, axis=1)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'readouts.csv').open('x', encoding='utf8', newline='') as raw, (directory / 'scores.csv').open('x', encoding='utf8', newline='') as summary:
        outputs = csv.writer(raw, lineterminator='\n'); outputs.writerow(RAW)
        scores = csv.writer(summary, lineterminator='\n')
        scores.writerow(('seed','arm','changed_both','changed_pairs','live_both','live_pairs','both_outputs_both_worlds','program_pairs','different_gold_program_pairs'))
        for seed, path, digest in checkpoints():
            _, readers = feature_readers(path)
            predictions = {arm: reader.predict(h, q) for arm, reader in readers.items()}
            predictions.update(literal=literal, ignore_middle=no_swap)
            for arm, pred in predictions.items():
                record(outputs, case, seed, arm, digest, pred)
                scores.writerow((seed, arm, *measure(case, pred)))
            print(f'seed {seed}: neutral-poison middle-swap counterfactual complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    run(parser.parse_args().directory)
