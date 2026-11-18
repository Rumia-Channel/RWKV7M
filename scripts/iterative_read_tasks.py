"""Supervision and split definitions for §22.19. Never import into the reader."""
import numpy as np

SPANS = (1, 2, 3, -2)
TRAIN_PAIRS = tuple((e, o) for e in SPANS for o in SPANS if (e, o) != (1, 2))
TEST_RULES = {'next': (1, 1), 'fourth': (4, 4), 'previous': (-1, -1),
              'conditional': (1, 2), 'mixed_novel': (4, -1)}
# Same physical training-graph/demonstration splits for every controller arm.
SPLITS = {
    'train': ((741101, 741103, 741107, 741109), 16, (0, 128)),
    'dev': ((742101, 742103), 8, (0, 128)),
    'test': ((743101, 743103, 743107, 743109, 743113), 8, (0, 1024)),
}


def route_labels(truth_edges, starts, pair):
    """Apply a hidden labeling rule to *generator* permutations, never model predictions."""
    edges = np.asarray(truth_edges, np.int32)
    nodes = np.asarray(starts, np.int32)
    pair = np.broadcast_to(np.asarray(pair, np.int32), edges.shape[:-1] + (2,))
    stride = np.take_along_axis(pair, nodes % 2, axis=-1)
    inverse = np.argsort(edges, axis=-1)
    cur = nodes
    for i in range(int(np.abs(stride).max(initial=0))):
        front = np.take_along_axis(edges, cur, axis=-1)
        back = np.take_along_axis(inverse, cur, axis=-1)
        cur = np.where(stride > i, front, np.where(stride < -i, back, cur))
    return cur.astype(np.int32, copy=False)
