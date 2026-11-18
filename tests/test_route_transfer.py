import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from route_transfer_panel import (COUNT, DATA_SEEDS, KEY_BASE, MAX_STRIDE, NO_RULE, RULES,
                                  VALUE_BASE, infer_walk, make_panel, normalize_sources, questions, walk_tables)
from versioned_binding_panel import literal_reference


def test_examples_identify_tasks_without_revealing_the_test_line():
    short = make_panel(DATA_SEEDS[0], 0, families=3)
    long = make_panel(DATA_SEEDS[0], 1024, families=3)
    np.testing.assert_array_equal(short.gold, long.gold)
    np.testing.assert_array_equal(short.queries, long.queries)
    np.testing.assert_array_equal(short.examples, long.examples)
    for f in range(len(short.queries)):
        query = short.queries[f,0,:,1] - KEY_BASE
        for d in short.examples[:,f]:
            np.testing.assert_array_equal(d[0], d[1])
            assert not np.isin(d[...,1]-KEY_BASE, query).any()
            assert not np.isin(d[...,3]-VALUE_BASE, query).any()
            np.testing.assert_array_equal(d[0,:,:3], short.examples[0,f,0,:,:3])
        # A single guessed answer cannot pass both worlds for any of the rules.
        assert np.all(short.gold[:,f,0] != short.gold[:,f,1])
        for w0, w1 in (long.known[f], long.alias[f]):
            np.testing.assert_array_equal(np.bincount(w0, minlength=128), np.bincount(w1, minlength=128))


def test_episode_specific_aliases_and_public_examples_are_both_needed():
    p = make_panel(DATA_SEEDS[1], 0, families=3)
    canonical = normalize_sources(p.alias)
    np.testing.assert_array_equal(canonical, p.known)
    lanes = canonical.shape[0]*2
    query = p.queries.reshape(lanes, -1, 3)
    all_keys = np.tile(np.arange(KEY_BASE, KEY_BASE+COUNT, dtype=np.int32), (lanes,1))
    raw_alias_reads = literal_reference(p.alias.reshape(lanes,-1), all_keys)
    assert np.all(raw_alias_reads == -2)
    walks = walk_tables(literal_reference(canonical.reshape(lanes,-1), all_keys))
    for r, rule in enumerate(RULES):
        pred, selected = infer_walk(walks, p.examples[r].reshape(lanes,-1,4), query)
        if rule == 'conditional':
            assert np.all(selected == NO_RULE) and np.all(pred == -2)
        else:
            assert np.all(selected == {'next':1, 'fourth':4, 'previous':-1}[rule])
            np.testing.assert_array_equal(pred.reshape(p.gold[r].shape), p.gold[r])
    # Same graph bytes; swapping alias declarations must change which source gets an edge.
    altered = p.alias.copy(); legend = altered[...,:4*COUNT].reshape(3,2,COUNT,4)
    legend[:,:,[0,1],3] = legend[:,:,[1,0],3]
    altered_edges = literal_reference(normalize_sources(altered).reshape(lanes,-1), all_keys)
    assert np.all((altered_edges != walks[MAX_STRIDE+1]+VALUE_BASE).sum(axis=1) == 2)


def test_multiple_predecessors_cause_abstention_not_arbitrary_choice():
    edges = (VALUE_BASE + (np.arange(COUNT)+1) % COUNT)[None,:]
    edges[0,0] = VALUE_BASE+2  # sources 0 and 1 now both point to node 2
    examples = np.concatenate((questions([[8,9,10]]), np.array([[[39],[40],[41]]],np.int32)), axis=-1)
    pred, stride = infer_walk(walk_tables(edges), examples, questions([[2,7]]))
    np.testing.assert_array_equal(stride, [-1])
    np.testing.assert_array_equal(pred, [[-2,VALUE_BASE+6]])
