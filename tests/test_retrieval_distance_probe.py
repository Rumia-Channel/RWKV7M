from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from rwkv7m.cli.prepare_retrieval import ANSWER_MARKER_TOKEN, QUERY_TOKEN, STORE_TOKEN


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return import_module('probe_retrieval_distance')


@pytest.mark.parametrize('kind', ['filler', 'other-writes'])
def test_counterfactual_labels_match_latest_visible_writes_at_claimed_distances(probe, kind):
    prefix, block, queries, labels = probe.histories(37, 3, kind)
    for i, worlds in enumerate(prefix.reshape(3, 2, -1)):
        np.testing.assert_array_equal(np.sort(worlds[0]), np.sort(worlds[1]))
        np.testing.assert_array_equal(queries[i, 0], queries[i, 1])
        assert np.all(labels[i, 0] != labels[i, 1])
        for w, history in enumerate(worlds):
            intervening = block[2 * i + w]
            if kind == 'other-writes':
                positions = np.flatnonzero(intervening == STORE_TOKEN)
                assert len(positions) == 8
                assert set(intervening[positions + 1]).isdisjoint(queries[i, w, :, 1])
                assert set(intervening[positions + 2]).isdisjoint(labels[i, w])
            visible = np.concatenate([history, intervening])
            writes = {visible[p + 1]: p + 2 for p in np.flatnonzero(visible == STORE_TOKEN)}
            for j, (question, label) in enumerate(zip(queries[i, w], labels[i, w], strict=True)):
                marker, key, answer_marker = question
                assert (marker, answer_marker) == (QUERY_TOKEN, ANSWER_MARKER_TOKEN)
                assert visible[writes[key]] == label
                assert len(visible) - writes[key] - 1 == len(intervening) + 56 - 8 * j


def test_nonfinite_answer_cannot_be_a_correct_class_or_a_counterfactual_flip(probe):
    labels = np.array([[[0, 0, 0], [1, 1, 1]]])
    scores = np.array([[[[0., -2.], [np.nan, -1.], [-2., 0.]],
                        [[-2., 0.], [-2., 0.], [-2., 0.]]]])
    row = probe.score_answers(scores, labels)
    assert (row['correct'], row['questions']) == (4, 6)
    assert (row['both_worlds_correct'], row['key_pairs']) == (1, 3)
    assert (row['prediction_changes'], row['finite_key_pairs']) == (1, 2)
    assert row['nonfinite_questions'] == 1
    assert row['predictions'][0][0][1] == -1
    assert row['mean_nll'] is None
