"""Assess saved §22.19 heads, or predict from a label-free visible JSON episode."""
import argparse
import csv
from hashlib import file_digest
import json
import os
from pathlib import Path
import re
import sys
from typing import NamedTuple

# Both entry points are deliberately real, batch-capable CPU inference paths.
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np

from addressed_binding_tokens import COUNT, VALUE_BASE
from rwkv7m.cli.prepare_retrieval import ANSWER_MARKER_TOKEN, QUERY_TOKEN
from rwkv7m.io.iterative_reader import WEIGHT_SEEDS, load_controller
from rwkv7m.model.iterative_reader import hard_read, init_controller

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "experiments/binding-state/distill"
TRAINED_ARMS = {"ordinary": ("ordinary", True), "isolated": ("isolated", True),
                "reset": ("isolated", False)}
ARM_NAMES = (*TRAINED_ARMS, "masked", "cap1", "untrained")
METRICS = ("both_worlds", "key_pairs", "all_seven", "families", "read_total",
           "individual_questions", "timeouts")
CONFIG_FIELDS = ("weight_seed", "arm", "checkpoint_sha256", "head", "head_sha256",
                 "frontend", "feedback", "trained_steps", "trained_cap", "cap")
RAW_FIELDS = (*CONFIG_FIELDS, "task", "even_stride", "odd_stride", "draw", "family", "gap",
              "input_sha256", "query0", "query1", "demo_src0", "demo_src1", "demo_dst0", "demo_dst1",
              "answer0", "answer1", "pred0", "pred1", "reads0", "reads1", "timeouts0", "timeouts1",
              "head0", "head1", "rounds")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class _Head(NamedTuple):
    params: dict
    metadata: dict
    name: str
    sha: str


class _Arm(NamedTuple):
    name: str
    frontend: str
    feedback: bool
    cap: int
    mask_demo: bool
    head: _Head


def _sha(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _registered_checkpoint(seed, expected_sha=None):
    if seed not in WEIGHT_SEEDS:
        raise ValueError(f"unregistered frozen weight seed: {seed}")
    with (WEIGHTS / "panel/checkpoints.csv").open(encoding="utf8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["arm"] == "ce" and int(row["weight_seed"]) == seed]
    if len(rows) != 1 or not _SHA256.fullmatch(rows[0]["checkpoint_sha256"]):
        raise ValueError(f"missing or ambiguous frozen checkpoint registration: {seed}")
    registered = rows[0]["checkpoint_sha256"]
    if expected_sha is not None and expected_sha != registered:
        raise ValueError(f"head/cache is associated with a different frozen checkpoint: {seed}")
    path = WEIGHTS / f"ce-s{seed}.safetensors"
    if _sha(path) != registered:
        raise ValueError(f"changed registered frozen checkpoint: {path}")
    return path, registered


def _load_test_cache(directory, seed, digest):
    # Import supervision only in the assessment path, never the JSON predictor.
    from iterative_read_tasks import SPLITS

    path = Path(directory) / f"test-s{seed}.npz"
    keys = {"ordinary", "isolated", "truth", "query", "demo", "draw", "family", "gap",
            "digest", "weight_sha", "weight_seed"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != keys:
            raise ValueError(f"unexpected test-cache fields: {path}")
        data = {name: archive[name] for name in archive.files}
    n = len(data["draw"]) if data["draw"].ndim == 1 else 0
    shapes = {"ordinary": (n, 2, COUNT), "isolated": (n, 2, COUNT), "truth": (n, 2, COUNT),
              "query": (n, 2, 7), "demo": (n, 2, 3), "draw": (n,), "family": (n,), "gap": (n,)}
    if n == 0:
        raise ValueError(f"empty or malformed registered test cache: {path}")
    for name, shape in shapes.items():
        if data[name].dtype != np.dtype(np.int32) or data[name].shape != shape:
            raise ValueError(f"bad test-cache int32 shape: {name}")
    for name in ("ordinary", "isolated", "truth", "query", "demo"):
        limit = COUNT if name in ("ordinary", "isolated") else COUNT - 1
        if np.any((data[name] < 0) | (data[name] > limit)):
            raise ValueError(f"out-of-domain test-cache stations: {name}")
    if not np.all(np.sort(data["truth"], axis=-1) == np.arange(COUNT)):
        raise ValueError("test-cache truth must contain generator permutations")
    if (data["digest"].shape != (n,) or data["digest"].dtype.kind != "U"
            or any(not _SHA256.fullmatch(value) for value in data["digest"])):
        raise ValueError("test-cache input digests must be Unicode SHA256 values")
    if (data["weight_sha"].shape != () or data["weight_sha"].dtype.kind != "U"
            or data["weight_sha"].item() != digest
            or data["weight_seed"].shape != () or data["weight_seed"].dtype != np.dtype(np.int32)
            or data["weight_seed"].item() != seed):
        raise ValueError("test-cache frozen checkpoint association mismatch")
    draws, families, gaps = SPLITS["test"]
    expected = {(draw, family, gap) for draw in draws for family in range(families) for gap in gaps}
    ids = list(zip(map(int, data["draw"]), map(int, data["family"]), map(int, data["gap"]), strict=True))
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError("cache is not exactly the registered test split")
    return data


def _load_arms(directory, seed, digest):
    trained = {}
    for name, (frontend, feedback) in TRAINED_ARMS.items():
        path = Path(directory) / f"{name}-s{seed}.safetensors"
        params, metadata = load_controller(path)
        expected = {"frontend": frontend, "feedback": str(feedback).lower(),
                    "weight_seed": str(seed), "weight_sha": digest, "cap": "6", "steps": "640"}
        if any(metadata[key] != value for key, value in expected.items()):
            raise ValueError(f"saved head does not match its trained arm: {path}")
        trained[name] = _Head(params, metadata, path.name, _sha(path))
    arms = [_Arm(name, frontend, feedback, 6, False, trained[name])
            for name, (frontend, feedback) in TRAINED_ARMS.items()]
    isolated = trained["isolated"]
    arms.extend((_Arm("masked", "isolated", True, 6, True, isolated),
                 _Arm("cap1", "isolated", True, 1, False, isolated)))
    reference = _Head(init_controller(seed), {"steps": "0", "cap": "6"}, f"init_controller({seed})", "")
    arms.append(_Arm("untrained", "isolated", True, 6, False, reference))
    return arms


def _config(seed, digest, arm):
    return dict(zip(CONFIG_FIELDS, (seed, arm.name, digest, arm.head.name, arm.head.sha,
                                   arm.frontend, str(arm.feedback).lower(), arm.head.metadata["steps"],
                                   arm.head.metadata["cap"], arm.cap), strict=True))


def _pack(values):
    return " ".join(str(int(value)) for value in values)


def _vocabulary(values):
    # Preserve abstentions/timeouts exactly; never fill them from the oracle.
    return np.where(values >= 0, values + VALUE_BASE, values)


def _metrics(prediction, answer, reads, timed_out):
    hits = prediction == answer
    paired = hits.all(axis=1)
    return {"both_worlds": int(paired.sum()), "key_pairs": int(paired.size),
            "all_seven": int(paired.all(axis=-1).sum()), "families": len(paired),
            "read_total": int(reads.sum()), "individual_questions": int(hits.size),
            "timeouts": int(timed_out.sum())}


def _write_gates(directory, scores, tasks, gaps):
    lookup = {(row["weight_seed"], row["arm"], row["task"], row["gap"]): row for row in scores}
    fields = ["weight_seed", "arm", "checkpoint_sha256"]
    for task in tasks:
        fields.extend(f"{task}_g{gap}_percent" for gap in gaps)
        fields.append(f"{task}_gate")
    fields.extend(("new_rule_gate", "anchor_gate", "gate"))
    with (directory / "gates.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for seed in WEIGHT_SEEDS:
            for arm in ARM_NAMES:
                first = lookup[seed, arm, "next", gaps[0]]
                row = {"weight_seed": seed, "arm": arm, "checkpoint_sha256": first["checkpoint_sha256"]}
                passed = {}
                for task in tasks:
                    checks = []
                    for gap in gaps:
                        score = lookup[seed, arm, task, gap]
                        row[f"{task}_g{gap}_percent"] = 100 * score["both_worlds"] / score["key_pairs"]
                        checks.append(10 * score["both_worlds"] >= 9 * score["key_pairs"])
                    passed[task] = all(checks)
                    row[f"{task}_gate"] = int(passed[task])
                row["new_rule_gate"] = int(all(passed[task] for task in ("fourth", "previous", "conditional")))
                row["anchor_gate"] = int(passed["next"])
                # The (+4,-1) task has its own reported gate, not a substituted core task.
                row["gate"] = int(bool(row["new_rule_gate"] and row["anchor_gate"]))
                writer.writerow(row)


def assess(cache, heads, output):
    """Evaluate saved heads on the complete registered test split, without training."""
    from iterative_read_tasks import SPLITS, TEST_RULES, route_labels

    output = Path(output)
    with jax.default_device(jax.devices("cpu")[0]):
        registered = []
        reference_inputs = None
        for seed in WEIGHT_SEEDS:
            _, digest = _registered_checkpoint(seed)
            data = _load_test_cache(cache, seed, digest)
            comparable = {name: data[name] for name in ("truth", "query", "demo", "draw", "family", "gap", "digest")}
            if reference_inputs is not None and any(not np.array_equal(value, reference_inputs[name])
                                                    for name, value in comparable.items()):
                raise ValueError("weight-seed caches do not contain the same registered test episodes")
            reference_inputs = comparable
            registered.append((seed, digest, data, _load_arms(heads, seed, digest)))
        output.mkdir(parents=True, exist_ok=True)
        scores = []
        with (output / "readouts.csv").open("w", encoding="utf8", newline="") as stream:
            writer = csv.DictWriter(stream, RAW_FIELDS)
            writer.writeheader()
            for seed, digest, data, arms in registered:
                n = len(data["draw"])
                lanes = n * 2
                demo_src = data["demo"]
                query = data["query"]
                for arm in arms:
                    config = _config(seed, digest, arm)
                    edges = jnp.asarray(data[arm.frontend].reshape(lanes, COUNT))
                    for task, pair in TEST_RULES.items():
                        # Only visible demonstration answers use the hidden rule before inference.
                        demo_dst = (np.full_like(demo_src, COUNT) if arm.mask_demo else
                                    route_labels(data["truth"], demo_src, pair))
                        result = hard_read(arm.head.params, edges, jnp.asarray(demo_src.reshape(lanes, 3)),
                                           jnp.asarray(demo_dst.reshape(lanes, 3)),
                                           jnp.asarray(query.reshape(lanes, 7)), cap=arm.cap, feedback=arm.feedback)
                        prediction = np.asarray(result.answer).reshape(n, 2, 7)
                        reads = np.asarray(result.reads).reshape(n, 2, 7)
                        timed_out = np.asarray(result.timed_out).reshape(n, 2, 7)
                        selected = np.asarray(result.head).reshape(n, 2, 7)
                        # The test oracle is separate and is never passed to hard_read.
                        answer = route_labels(data["truth"], query, pair)
                        vocab_pred, vocab_answer = _vocabulary(prediction), _vocabulary(answer)
                        for family in range(n):
                            row = {**config, "task": task, "even_stride": pair[0], "odd_stride": pair[1],
                                   "draw": int(data["draw"][family]), "family": int(data["family"][family]),
                                   "gap": int(data["gap"][family]), "input_sha256": data["digest"][family],
                                   "rounds": int(np.asarray(result.rounds))}
                            for world in range(2):
                                for name, values in (("query", query), ("demo_src", demo_src), ("demo_dst", demo_dst),
                                                     ("answer", vocab_answer), ("pred", vocab_pred), ("reads", reads),
                                                     ("timeouts", timed_out), ("head", selected)):
                                    row[f"{name}{world}"] = _pack(values[family, world])
                            writer.writerow(row)
                        for gap in SPLITS["test"][2]:
                            take = data["gap"] == gap
                            scores.append({**config, "task": task, "gap": gap,
                                           **_metrics(prediction[take], answer[take], reads[take], timed_out[take])})
                stream.flush()
        with (output / "scores.csv").open("w", encoding="utf8", newline="") as stream:
            writer = csv.DictWriter(stream, (*CONFIG_FIELDS, "task", "gap", *METRICS))
            writer.writeheader()
            writer.writerows(scores)
        _write_gates(output, scores, TEST_RULES, SPLITS["test"][2])


def _tokens(value, name, ndim):
    result = np.asarray(value)
    if result.ndim != ndim or result.dtype.kind not in "iu" or np.any((result < 0) | (result >= 128)):
        raise ValueError(f"{name} must be a rank-{ndim} array of integer vocabulary IDs in [0,128)")
    return result.astype(np.int32, copy=False)


def _visible_episode(payload):
    if not isinstance(payload, dict) or set(payload) != {"history", "examples", "queries"}:
        raise ValueError("input must contain only history, examples and queries; no task or target labels")
    history = _tokens(payload["history"], "history", 1)
    examples = _tokens(payload["examples"], "examples", 2)
    queries = _tokens(payload["queries"], "queries", 2)
    if history.size == 0 or examples.shape != (3, 4) or queries.shape[0] == 0 or queries.shape[1] != 3:
        raise ValueError("expected nonempty history, three [3,name,4,value] examples and Q [3,name,4] queries")
    for name, rows in (("examples", examples), ("queries", queries)):
        if np.any(rows[:, 0] != QUERY_TOKEN) or np.any(rows[:, 2] != ANSWER_MARKER_TOKEN):
            raise ValueError(f"malformed visible {name} grammar")
    if np.any((examples[:, 3] < VALUE_BASE) | (examples[:, 3] >= VALUE_BASE + COUNT)):
        raise ValueError("example answers must be station-value vocabulary IDs")
    return history[None, :], examples[None, :, :], queries[None, :, :]


def predict(head, source):
    """Read no task name, rule, hidden graph, or query target; print measured outputs."""
    from addressed_rwkv_reader import AddressedReader, FrozenBindingCodec
    from iterative_read_frontend import observe, resolve_names

    params, metadata = load_controller(head)
    checkpoint, _ = _registered_checkpoint(int(metadata["weight_seed"]), metadata["weight_sha"])
    if source == "-":
        payload = json.load(sys.stdin)
    else:
        with Path(source).open(encoding="utf8") as stream:
            payload = json.load(stream)
    history, examples, queries = _visible_episode(payload)
    with jax.default_device(jax.devices("cpu")[0]):
        codec = FrozenBindingCodec(checkpoint)
        isolated = AddressedReader(codec, "isolated")
        frontend = metadata["frontend"]
        edges = observe(codec, isolated, history, frontends=(frontend,))[frontend]
        demo_src = resolve_names(history, examples[..., 1])
        query = resolve_names(history, queries[..., 1])
        result = hard_read(params, jnp.asarray(edges), jnp.asarray(demo_src),
                           jnp.asarray(examples[..., 3] - VALUE_BASE), jnp.asarray(query),
                           cap=int(metadata["cap"]), feedback=metadata["feedback"] == "true")
        response = {"predictions": _vocabulary(np.asarray(result.answer)[0]).tolist(),
                    "steps": np.asarray(result.reads)[0].tolist(),
                    "timeout": np.asarray(result.timed_out)[0].tolist(),
                    "head": np.asarray(result.head)[0].tolist()}
    print(json.dumps(response, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assessment = commands.add_parser("assess", help="score saved heads on registered cached test episodes")
    assessment.add_argument("cache", type=Path)
    assessment.add_argument("heads", type=Path)
    assessment.add_argument("output", type=Path)
    prediction = commands.add_parser("predict", help="predict a visible JSON episode (use '-' for stdin)")
    prediction.add_argument("head", type=Path)
    prediction.add_argument("input")
    args = parser.parse_args()
    if args.command == "assess":
        assess(args.cache, args.heads, args.output)
    else:
        predict(args.head, args.input)


if __name__ == "__main__":
    main()
