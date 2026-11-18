import csv
from pathlib import Path
import sys

import numpy as np
import pytest
from safetensors import safe_open
from safetensors.numpy import save_file

from rwkv7m.io.iterative_reader import FORMAT_NAME, FORMAT_VERSION, load_controller, save_controller
from rwkv7m.model.iterative_reader import init_controller

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from probe_iterative_reader import ARM_NAMES, WEIGHT_SEEDS, _metrics, _visible_episode, _write_gates
from iterative_read_tasks import TEST_RULES
from rwkv7m.cli.prepare_retrieval import STORE_TOKEN


@pytest.fixture(scope="module")
def params():
    return {name: np.asarray(value) for name, value in init_controller(0).items()}


@pytest.fixture
def artifact(tmp_path, params):
    path = tmp_path / "nested" / "isolated-s42.safetensors"
    save_controller(path, params, frontend="isolated", feedback=True, weight_seed=42,
                    weight_sha="a" * 64, steps=17)
    return path


def test_controller_artifact_roundtrip_is_portable_and_explicit(artifact, params):
    restored, metadata = load_controller(artifact)
    assert metadata == {"format": FORMAT_NAME, "format_version": FORMAT_VERSION, "dtype": "float32",
                        "frontend": "isolated", "feedback": "true", "weight_seed": "42",
                        "weight_sha": "a" * 64, "steps": "17", "cap": "6"}
    assert all(isinstance(value, str) for value in metadata.values())
    assert restored.keys() == params.keys()
    with safe_open(str(artifact), framework="np") as external:
        assert set(external.keys()) == set(params)
        for name, value in params.items():
            assert restored[name].dtype == np.float32
            assert restored[name].shape == value.shape
            np.testing.assert_array_equal(restored[name], value)
            np.testing.assert_array_equal(external.get_tensor(name), value)


def test_reset_metadata_survives_without_frontend_relabeling(tmp_path, params):
    path = tmp_path / "reset-s43.safetensors"
    save_controller(path, params, frontend="isolated", feedback=False, weight_seed=43,
                    weight_sha="b" * 64, steps=31, cap=1)
    _, metadata = load_controller(path)
    assert metadata["frontend"] == "isolated"
    assert metadata["feedback"] == "false"
    assert metadata["weight_seed"] == "43"
    assert metadata["cap"] == "1"


@pytest.mark.parametrize("damage", ["missing", "extra", "shape", "dtype", "nan", "infinity"])
def test_mismatched_or_corrupt_tensors_are_rejected_on_save_and_load(tmp_path, artifact, params, damage):
    tensors = dict(params)
    name = next(iter(tensors))
    if damage == "missing":
        del tensors[name]
    elif damage == "extra":
        tensors["unexpected_tensor"] = np.zeros(1, np.float32)
    elif damage == "shape":
        tensors[name] = np.zeros(tensors[name].shape + (1,), np.float32)
    elif damage == "dtype":
        tensors[name] = tensors[name].astype(np.float64)
    else:
        tensors[name] = np.full_like(tensors[name], np.nan if damage == "nan" else np.inf)
    with pytest.raises(ValueError):
        save_controller(tmp_path / "rejected.safetensors", tensors, frontend="isolated", feedback=True,
                        weight_seed=42, weight_sha="a" * 64, steps=17)
    with safe_open(str(artifact), framework="np") as handle:
        metadata = handle.metadata()
    damaged = tmp_path / "damaged.safetensors"
    save_file(tensors, str(damaged), metadata=metadata)
    with pytest.raises(ValueError):
        load_controller(damaged)


@pytest.mark.parametrize(("field", "value"), [
    ("format", "rwkv7m"), ("format_version", "2"), ("dtype", "bfloat16"),
    ("frontend", "oracle"), ("feedback", "yes"), ("weight_seed", "45"),
    ("weight_seed", "042"), ("weight_sha", "not-a-digest"), ("steps", "-1"),
    ("cap", "0"), ("cap", "6.0"), ("frontend", None),
])
def test_invalid_or_missing_artifact_metadata_is_rejected(tmp_path, artifact, params, field, value):
    with safe_open(str(artifact), framework="np") as handle:
        metadata = handle.metadata()
    if value is None:
        del metadata[field]
    else:
        metadata[field] = value
    path = tmp_path / "wrong-metadata.safetensors"
    save_file(params, str(path), metadata=metadata)
    with pytest.raises(ValueError):
        load_controller(path)


@pytest.mark.parametrize("options", [
    {"frontend": "reset"}, {"feedback": "false"}, {"weight_seed": 45},
    {"weight_sha": "0" * 63}, {"steps": -1}, {"cap": 0}, {"cap": True},
])
def test_save_rejects_invalid_association_and_execution_metadata(tmp_path, params, options):
    kwargs = dict(frontend="ordinary", feedback=True, weight_seed=42, weight_sha="a" * 64, steps=17)
    kwargs.update(options)
    with pytest.raises(ValueError):
        save_controller(tmp_path / "bad.safetensors", params, **kwargs)


def test_non_safetensors_file_is_rejected(tmp_path):
    path = tmp_path / "not-a-head.safetensors"
    path.write_bytes(b"not a safetensors artifact")
    with pytest.raises(ValueError):
        load_controller(path)


def test_scores_count_paired_worlds_and_preserve_measured_timeouts():
    answer = np.broadcast_to(np.arange(7, dtype=np.int32), (2, 2, 7)).copy()
    prediction = answer.copy()
    prediction[0, 0, 0] = -2
    prediction[1, 1, 1] = -2
    timed_out = prediction == -2
    reads = np.where(timed_out, 1, 3)
    assert _metrics(prediction, answer, reads, timed_out) == {
        "both_worlds": 12, "key_pairs": 14, "all_seven": 0, "families": 2,
        "read_total": 80, "individual_questions": 28, "timeouts": 2,
    }
    assert prediction[0, 0, 0] == -2


def _episode(q):
    history = [token for i in range(16) for token in (6, 16 + i, 32 + i, 64 + i)]
    history += [STORE_TOKEN, 64, 33]
    return {"history": history, "examples": [[3, 64 + i, 4, 33 + i] for i in range(3)],
            "queries": [[3, 64 + i % 16, 4] for i in range(q)]}


def test_prediction_input_is_label_free_batch_one_with_arbitrary_query_count():
    history, examples, queries = _visible_episode(_episode(23))
    assert history.shape == (1, 67)
    assert examples.shape == (1, 3, 4)
    assert queries.shape == (1, 23, 3)
    assert history.dtype == examples.dtype == queries.dtype == np.int32
    for forbidden in ("task", "target", "truth", "labels"):
        payload = _episode(2)
        payload[forbidden] = "next"
        with pytest.raises(ValueError):
            _visible_episode(payload)


@pytest.mark.parametrize("field", ["history", "examples", "queries"])
def test_prediction_input_disallows_boolean_vocabulary_ids(field):
    payload = _episode(2)
    payload[field] = np.asarray(payload[field], dtype=np.bool_).tolist()
    with pytest.raises(ValueError):
        _visible_episode(payload)


def test_gate_requires_each_gap_and_reports_mixed_rule_separately(tmp_path):
    scores = []
    for seed in WEIGHT_SEEDS:
        for arm in ARM_NAMES:
            for task in TEST_RULES:
                for gap in (0, 1024):
                    hits = 0 if task == "mixed_novel" else 90
                    if (seed, arm, task, gap) == (42, "ordinary", "fourth", 1024):
                        hits = 89
                    scores.append({"weight_seed": seed, "arm": arm, "task": task, "gap": gap,
                                   "checkpoint_sha256": "a" * 64, "both_worlds": hits, "key_pairs": 100})
    _write_gates(tmp_path, scores, TEST_RULES, (0, 1024))
    with (tmp_path / "gates.csv").open(encoding="utf8", newline="") as stream:
        rows = {(int(row["weight_seed"]), row["arm"]): row for row in csv.DictReader(stream)}
    assert len(rows) == len(WEIGHT_SEEDS) * len(ARM_NAMES)
    assert rows[42, "ordinary"]["fourth_gate"] == "0"
    assert rows[42, "ordinary"]["anchor_gate"] == "1"
    assert rows[42, "ordinary"]["gate"] == "0"
    assert rows[42, "isolated"]["gate"] == "1"
    assert rows[42, "isolated"]["mixed_novel_gate"] == "0"
