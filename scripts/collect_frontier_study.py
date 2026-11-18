"""Publish auditable tables from the native study outputs and two external score panels."""

import argparse
import csv
from hashlib import file_digest
import json
from math import prod
from pathlib import Path

from safetensors import safe_open

from analyze_retrieval_frontier import analyze, audit_schedules


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def collect(study, parents, trained, out):
    out.mkdir(parents=True, exist_ok=True)
    raw = []
    for source in (parents, trained):
        with source.open(encoding='utf-8', newline='') as f:
            raw.extend(csv.DictReader(f))
    conditions = {(kind, gap) for kind in ('filler', 'other-writes') for gap in (0, 64, 128, 256, 1024)} | {('overwrite', 32)}
    groups = {(r['arm'], int(r['weight_seed']), r['task'], int(r['gap']), int(r['family'])) for r in raw}
    expected = {(arm, s, task, gap, f) for arm in ('parent', 'fixed', 'adaptive', 'yoked') for s in (42, 43, 44)
                for task, gap in conditions for f in range(32)}
    if len(raw) != len(groups) or groups != expected:
        raise ValueError('incomplete or duplicated independent evaluation')
    costs, intervals, updates = [], [], []
    for arm in ('adaptive', 'fixed', 'yoked'):
        for seed in (42, 43, 44):
            base = study / f'{arm}-s{seed}'
            if (base / 'latest').read_text().strip() != 'step-096': raise ValueError(f'incomplete: {base}')
            boundary = base / 'step-096'
            record = json.loads((boundary / 'study.json').read_text()); protocol = record['protocol']
            if (record['step'], len(record['updates']), len(record['schedule']), protocol['arm'], protocol['weight_seed']) != (96, 96, 96, arm, seed):
                raise ValueError(f'wrong training budget or source: {base}')
            for step, (cell, stats) in enumerate(zip(record['schedule'], record['updates'], strict=True)):
                gap, writes = (32, 0) if cell is None else cell
                updates.append(dict(arm=arm, weight_seed=seed, step=step, anchor=int(cell is None), gap=gap, writes=writes, **stats))
            cals = record['calibrations']
            if [c['step'] for c in cals] != [0, 24, 48, 72, 96]: raise ValueError(f'missing calibration: {base}')
            for c in cals:
                for condition, hits in c['hits'].items():
                    gap, writes = map(int, condition.split(','))
                    intervals.append(dict(arm=arm, weight_seed=seed, step=c['step'], gap=gap, writes=writes, anchor=0,
                                          all_four_correct=hits, families=protocol['cal_pairs']))
                intervals.append(dict(arm=arm, weight_seed=seed, step=c['step'], gap=32, writes=0, anchor=1,
                                      all_four_correct=c['anchor_hits'], families=protocol['cal_pairs']))
            weights = boundary / 'model.safetensors'
            with weights.open('rb') as file: sha = file_digest(file, 'sha256').hexdigest()
            with safe_open(weights, framework='numpy') as tensors:
                count = sum(prod(tensors.get_slice(k).get_shape()) for k in tensors.keys())
                if tensors.metadata()['parent_weights_sha256'] != protocol['parent_sha256'] or count != 504960:
                    raise ValueError(f'noncomparable exported model: {base}')
            costs.append(dict(arm=arm, weight_seed=seed, donor=protocol['donor'], data_seed=protocol['data_seed'], cal_seed=protocol['cal_seed'],
                              params=count, updates=len(record['updates']), train_tokens=sum(u['tokens'] for u in record['updates']),
                              train_answers=sum(u['answers'] for u in record['updates']), calibration_families=len(cals)*10*protocol['cal_pairs'],
                              train_seconds=record['train_seconds'], calibration_seconds=record['calibration_seconds'],
                              promotions=record['policy']['promotions'] if record['policy'] is not None else '',
                              parent_sha256=protocol['parent_sha256'], weights_sha256=sha))
    if any((r['train_tokens'], r['train_answers']) != (196608, 768) for r in costs):
        raise ValueError('actual token/answer budgets do not match')
    families = out / 'frontier-families.csv'; schedule = out / 'frontier-updates.csv'
    write_csv(families, raw); write_csv(schedule, updates)
    result = analyze(families); result['schedule_audit'] = audit_schedules(schedule)
    write_csv(out / 'frontier-calibrations.csv', intervals); write_csv(out / 'frontier-costs.csv', costs)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'parents', 'trained', 'output'): p.add_argument(name, type=Path)
    a = p.parse_args(); collect(a.study, a.parents, a.trained, a.output)
