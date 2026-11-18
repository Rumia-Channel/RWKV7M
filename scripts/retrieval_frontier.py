"""Retrieval-frontier dataset and adaptive cell-selection controller.

This module is the data and feedback half of the retrieval-frontier study.  It
owns the token-level dataset, the schedule helpers that decide *which* cell an
arm trains on, and the feedback controller that promotes a cell once it is
reliably solved.  It never touches the model or the trainer: a study runner
asks it for a batch of sequences, reports aggregate calibration feedback, and
reads back a JSON-serializable policy state that can be checkpointed.

Dataset
-------
``make_examples(seed, index, pairs, *, gap, writes, anchor=False)`` returns

* ``ids``: ``int32`` array of shape ``(pairs, 2 worlds, 2 questions, 256)``
* ``labels``: ``int32`` array of shape ``(pairs, 2, 2)``

Every row is exactly 256 answer-hidden input tokens.  The row ends with
``(QUERY_TOKEN, key, ANSWER_MARKER_TOKEN)`` and the label is the token the
model has to predict at that final position, i.e. the token *after* the answer
marker.  :func:`to_next_token_batch` converts a batch into the repository's
``(input_ids, target_ids, mask)`` triple.

Token layout of a regular (non-anchor) row, with ``D = gap`` and ``W =
writes``::

    [ filler + (8 - W) refresh stores ][ 16-key prefix ][ gap region ][ query ]
    |<------- pre_len = 190 - D ----->||<-- 63 -->|<-- D -->|<-- 3 -->|

* The **16-key prefix** holds 16 ``(STORE, key, value)`` triples on a stride of
  4, exactly like the prior direct-prefix probe ``probe_retrieval_distance``.
  The eight odd slots are *queryable*; two of them are queried by the two
  questions of the family.  A queried value sits ``0..56`` tokens before the
  end of the prefix, so the distance from a queried value to the query is
  ``gap + (0..56)`` tokens, the same distance convention as the probe.
* The **gap region** is exactly ``gap`` tokens wide and holds ``writes``
  *refresh* stores of unrelated keys; the other ``8 - writes`` refresh stores
  are moved *before* the shared 16-key prefix.  The total is always 8 refresh
  stores, so every cell writes the same 24 records over the same 256 tokens
  and the per-key write histories are identical across cells.  A refresh store
  re-stores an unrelated key with that key's own value, so no label ever
  appears in a refresh store.
* The **query** is the last three tokens; nothing follows the scored query.

Both worlds of a family share every key, every query and every unigram count.
They differ only in the values stored at the two queried slots, so the answer
to *both* questions changes between worlds while the token multiset of the row
stays identical.  The facts of family ``f`` depend only on ``(seed, index, f)``:
changing ``gap`` or ``writes`` moves tokens around but never changes the
sampled keys, values or questions.

Anchor rows
-----------
``anchor=True`` (with ``gap=32`` and ``writes=0``) yields a short row holding
four keys that are each written twice: a stale value followed by a different
latest value.  The queried key must be answered with its *latest* value, the
stale value never equals the latest one, and the four latest values sit
``32..44`` tokens before the query while the stale values sit further back.
Anchor rows keep the same fixed length and the same answer accounting as
regular rows.

Schedule
--------
Updates come in blocks of four.  ``step % 4 == 3`` is the **anchor update**
(``gap=32``, ``writes=0``, anchor rows): a quarter of the updates protect the
short overwrite skill that the promotion rule guards.  Every other step is a
**cell update** and carries the ``(gap, writes)`` cell of that arm:

* the fixed arm cycles the nine cells, so each cell is visited exactly eight
  times over 96 steps;
* the adaptive arm picks the cell with the policy described below;
* the yoked arm replays the adaptive schedule of another seed.

All arms use the same number of updates, the same anchor positions and the
same number of examples, so they differ only in which cell they train.

Adaptive policy (observable)
----------------------------
* **Candidates**: one cell per writes regime -- its *first* gap in ``GAPS``
  whose cell is not mastered yet ("one first-unmastered gap per regime").  A
  regime whose whole ladder is mastered keeps its hardest cell as candidate.
* **Weakness**: candidates are ordered by the Wilson lower bound of their
  most recent measurement, weakest first.  A cell that was never measured, or
  that was not measured in the most recent calibration, counts as weakest, so
  the policy re-measures what it does not know.
* **Coverage**: a regime whose selection count is more than ``coverage_slack``
  above the least-served candidate regime is dropped from the candidate set,
  so all three writes regimes keep a positive share of the cell updates.
* **Promotion**: the *active* cell of a regime is promoted after
  ``consecutive`` successive calibrations whose one-sided 95% Wilson lower
  bound is at least ``mastery_threshold`` (0.90).  A farther gap of the same
  regime is never promoted before the nearer one, and a calibration that does
  not measure the cell breaks its streak.  Promotion is irreversible; the
  anchor guard can block a new promotion, not undo an old one.
* **Freeze**: when the short anchor accuracy falls 1 pp or more below the
  initial baseline the calibration is frozen: nothing is promoted and every
  promotion streak is cleared, so promotion has to be earned again.

The controller only ever sees aggregate feedback -- per cell the number of
families correct on all four questions and the number evaluated, plus the
fraction of anchor families that were correct.  It never sees labels,
per-question scores or the final score of a run.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from numbers import Integral

import numpy as np

from rwkv7m.cli.prepare_retrieval import (
    ANSWER_MARKER_TOKEN,
    FILLER_TOKEN,
    QUERY_TOKEN,
    STORE_TOKEN,
)

#: Token ids of the synthetic retrieval vocabulary.
FILLER = FILLER_TOKEN
STORE = STORE_TOKEN
QUERY = QUERY_TOKEN
ANSWER_MARKER = ANSWER_MARKER_TOKEN

#: Key tokens of the synthetic vocabulary (16 distinct keys).
KEY_TOKENS = tuple(range(16, 32))
#: Value tokens of the synthetic vocabulary (16 distinct values).
VALUE_TOKENS = tuple(range(32, 48))

#: Input tokens per row.  The answer is hidden, so the label is the target of
#: the final position.
ROW_TOKENS = 256
#: Tokens of the shared prefix region (store triples on a stride of 4).
PREFIX_TOKENS = 63
#: Tokens of the scored query block ``(QUERY, key, ANSWER_MARKER)``.
QUERY_TOKENS = 3
#: Keys stored in the shared prefix of a regular row.
PREFIX_KEYS = 16
#: Refresh stores per regular row, split between the gap region and the
#: region in front of the prefix.
REFRESH_TOTAL = 8
#: Prefix slots whose values may be queried.  Slot ``k`` holds a value that is
#: ``60 - 4k`` tokens before the end of the prefix, i.e. up to 56 tokens older
#: than the stated post-prefix gap.
QUERYABLE_SLOTS = (1, 3, 5, 7, 9, 11, 13, 15)
#: Number of keys of an anchor row and the number of writes per anchor key.
ANCHOR_KEYS = 4
ANCHOR_WRITES_PER_KEY = 2
#: Prefix offset of the anchor block: it is translated so that the last value
#: of the block ends the prefix region, which puts the latest value of every
#: anchor key ``gap..gap+12`` tokens before the query.
ANCHOR_BLOCK_OFFSET = 32

#: Gaps (post-prefix distance in tokens) of the study grid.
GAPS = (32, 64, 128)
#: Refresh stores inside the gap region of the study grid.
WRITES = (0, 4, 8)
#: The nine cells of the study grid, gap major: ``(32,0), (32,4), ...``.
CELLS = tuple((gap, writes) for gap in GAPS for writes in WRITES)
_CELL_SET = frozenset(CELLS)

#: Updates per schedule block.
BLOCK = 4
#: Offset of the anchor update inside a block.
ANCHOR_OFFSET = 3
#: Index base of calibration data windows; disjoint from every training index.
CALIBRATION_INDEX_BASE = 1_000_000

#: One-sided 95% normal quantile used by the Wilson lower bound.
Z_ONE_SIDED_95 = 1.6448536269514722
#: Lower bound a cell has to reach before it counts as mastered.
MASTERY_THRESHOLD = 0.90
#: Successive qualifying calibrations required for a promotion.
CONSECUTIVE_CALIBRATIONS = 2
#: How far the short anchor accuracy may fall below the initial baseline.
ANCHOR_TOLERANCE = 0.01
#: How far a regime may lead the least served regime before it is skipped.
COVERAGE_SLACK = 1


def _require_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def wilson_lower_bound(correct, total, z=Z_ONE_SIDED_95):
    """One-sided 95% Wilson score lower bound of a binomial rate.

    ``correct`` successes out of ``total`` trials give the lower end of the
    Wilson score interval at the one-sided 95% level.  ``total == 0`` yields
    ``0.0`` (nothing observed, nothing proven).
    """
    correct = _require_int(correct, "correct")
    total = _require_int(total, "total")
    if correct < 0 or total < 0 or correct > total:
        raise ValueError("correct and total must satisfy 0 <= correct <= total")
    if total == 0:
        return 0.0
    rate = correct / total
    denominator = 1.0 + z * z / total
    centre = (rate + z * z / (2.0 * total)) / denominator
    spread = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return max(0.0, centre - spread / denominator)


def _regular_family(key_rng, value_rng, query_rng):
    """Sample the facts of a regular family: 16 keys, 16 values, 2 queries."""
    keys = key_rng.permutation(np.asarray(KEY_TOKENS))
    values = value_rng.permutation(np.asarray(VALUE_TOKENS))
    slots = np.sort(query_rng.choice(np.asarray(QUERYABLE_SLOTS), size=2, replace=False))
    return keys, values, slots


def _anchor_family(key_rng, value_rng, query_rng):
    """Sample the facts of an anchor family: 4 keys, 8 values, 2 queries."""
    keys = key_rng.choice(np.asarray(KEY_TOKENS), size=ANCHOR_KEYS, replace=False)
    values = key_rng.choice(
        np.asarray(VALUE_TOKENS), size=ANCHOR_KEYS * ANCHOR_WRITES_PER_KEY, replace=False
    )
    slots = np.sort(query_rng.choice(ANCHOR_KEYS, size=2, replace=False))
    return keys, values, slots


def _regular_rows(keys, values, slots, gap, writes):
    """Build the four rows (2 worlds x 2 questions) of a regular family."""
    pre_len = ROW_TOKENS - PREFIX_TOKENS - gap - QUERY_TOKENS
    gap_base = pre_len + PREFIX_TOKENS
    # Refresh stores re-store unrelated keys with their own value, so a label
    # never appears in a refresh store.
    refresh = [(int(keys[2 * j]), int(values[2 * j])) for j in range(REFRESH_TOTAL)]

    rows = np.empty((2, 2, ROW_TOKENS), np.int32)
    labels = np.empty((2, 2), np.int32)
    for world in (0, 1):
        slot_values = values.copy()
        if world == 1:
            # Counterfactual world: the values of the two queried slots are
            # swapped, so both answers change while every unigram count stays.
            first, second = int(slots[0]), int(slots[1])
            slot_values[first], slot_values[second] = values[second], values[first]
        row = np.full(ROW_TOKENS, FILLER, np.int32)
        for index in range(PREFIX_KEYS):
            base = pre_len + 4 * index
            row[base] = STORE
            row[base + 1] = keys[index]
            row[base + 2] = slot_values[index]
        if writes:
            stride = gap // writes
            for j in range(writes):
                base = gap_base + j * stride
                row[base] = STORE
                row[base + 1] = refresh[j][0]
                row[base + 2] = refresh[j][1]
        remaining = REFRESH_TOTAL - writes
        if remaining:
            stride = pre_len // remaining
            for j in range(remaining):
                base = j * stride
                row[base] = STORE
                row[base + 1] = refresh[writes + j][0]
                row[base + 2] = refresh[writes + j][1]
        for question in (0, 1):
            row_out = row.copy()
            row_out[ROW_TOKENS - 3] = QUERY
            row_out[ROW_TOKENS - 2] = keys[int(slots[question])]
            row_out[ROW_TOKENS - 1] = ANSWER_MARKER
            rows[world, question] = row_out
            labels[world, question] = slot_values[int(slots[question])]
    return rows, labels


def _anchor_rows(keys, values, slots, gap):
    """Build the four rows of an anchor family (four keys, two writes each).

    Key ``i`` is written twice: a stale value on prefix offset
    ``ANCHOR_BLOCK_OFFSET + 4i`` and a latest value ``16`` tokens later.  The
    block ends the prefix region, so the latest value of every anchor key sits
    ``gap..gap+12`` tokens before the query and no stale value ever equals a
    latest value.
    """
    pre_len = ROW_TOKENS - PREFIX_TOKENS - gap - QUERY_TOKENS
    stale = values[:ANCHOR_KEYS]
    latest = values[ANCHOR_KEYS:]

    rows = np.empty((2, 2, ROW_TOKENS), np.int32)
    labels = np.empty((2, 2), np.int32)
    for world in (0, 1):
        latest_world = latest.copy()
        if world == 1:
            first, second = int(slots[0]), int(slots[1])
            latest_world[first], latest_world[second] = latest[second], latest[first]
        row = np.full(ROW_TOKENS, FILLER, np.int32)
        for index in range(ANCHOR_KEYS):
            base = pre_len + ANCHOR_BLOCK_OFFSET + 4 * index
            row[base] = STORE
            row[base + 1] = keys[index]
            row[base + 2] = stale[index]
        for index in range(ANCHOR_KEYS):
            base = pre_len + ANCHOR_BLOCK_OFFSET + 16 + 4 * index
            row[base] = STORE
            row[base + 1] = keys[index]
            row[base + 2] = latest_world[index]
        for question in (0, 1):
            row_out = row.copy()
            row_out[ROW_TOKENS - 3] = QUERY
            row_out[ROW_TOKENS - 2] = keys[int(slots[question])]
            row_out[ROW_TOKENS - 1] = ANSWER_MARKER
            rows[world, question] = row_out
            labels[world, question] = latest_world[int(slots[question])]
    return rows, labels


def make_examples(seed, index, pairs, *, gap, writes, anchor=False):
    """Sample a batch of frontier histories.

    Parameters
    ----------
    seed, index:
        Identify the data window.  The facts of family ``f`` depend only on
        ``(seed, index, f)``, never on ``gap``/``writes``, so all arms of a
        study see the same facts and questions at the same step.
    pairs:
        Number of independent history families in the batch.
    gap:
        Width of the post-prefix gap region in tokens (``GAPS``).
    writes:
        Number of refresh stores inside the gap region (``WRITES``).
    anchor:
        Produce anchor rows (four keys with a stale and a latest value).  Only
        valid together with ``gap=32`` and ``writes=0``.

    Returns
    -------
    (ids, labels):
        ``ids`` is ``int32`` with shape ``(pairs, 2, 2, 256)`` and ``labels``
        is ``int32`` with shape ``(pairs, 2, 2)``.
    """
    seed = _require_int(seed, "seed")
    index = _require_int(index, "index")
    pairs = _require_int(pairs, "pairs")
    gap = _require_int(gap, "gap")
    writes = _require_int(writes, "writes")
    if seed < 0 or index < 0:
        raise ValueError("seed and index must not be negative")
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if gap not in GAPS:
        raise ValueError(f"gap must be one of {GAPS}")
    if writes not in WRITES:
        raise ValueError(f"writes must be one of {WRITES}")
    if anchor and (gap, writes) != (GAPS[0], WRITES[0]):
        raise ValueError("anchor rows use gap=32 and writes=0")

    ids = np.empty((pairs, 2, 2, ROW_TOKENS), np.int32)
    labels = np.empty((pairs, 2, 2), np.int32)
    # The anchor template has its own stream, so anchor families never share
    # facts with the regular families of the same step.
    root = np.random.SeedSequence([seed, index, 1 if anchor else 0])
    children = root.spawn(3 * pairs)
    for family in range(pairs):
        key_rng = np.random.default_rng(children[3 * family])
        value_rng = np.random.default_rng(children[3 * family + 1])
        query_rng = np.random.default_rng(children[3 * family + 2])
        if anchor:
            keys, values, slots = _anchor_family(key_rng, value_rng, query_rng)
            rows, family_labels = _anchor_rows(keys, values, slots, gap)
        else:
            keys, values, slots = _regular_family(key_rng, value_rng, query_rng)
            rows, family_labels = _regular_rows(keys, values, slots, gap, writes)
        ids[family] = rows
        labels[family] = family_labels
    return ids, labels


def to_next_token_batch(ids, labels):
    """Convert examples into the repository's next-token training triple.

    ``input_ids`` keeps the 256 answer-hidden tokens of every row.  The
    ``target_ids`` hold the shifted row with the answer at the final position,
    and ``mask`` selects exactly that position, which is where the model has to
    predict the hidden answer.
    """
    ids = np.asarray(ids)
    labels = np.asarray(labels)
    if ids.shape[:-1] != labels.shape:
        raise ValueError("labels must match the leading shape of ids")
    if ids.shape[-1] != ROW_TOKENS:
        raise ValueError(f"ids must have {ROW_TOKENS} tokens per row")
    target_ids = np.empty(ids.shape, np.int32)
    target_ids[..., :-1] = ids[..., 1:]
    target_ids[..., -1] = labels
    mask = (ids == ANSWER_MARKER).astype(np.float32)
    return {"input_ids": ids, "target_ids": target_ids, "mask": mask}


def calibration_index(round_index):
    """Index of a calibration window, disjoint from every training index."""
    return CALIBRATION_INDEX_BASE + _require_int(round_index, "round_index")


def is_anchor_step(step):
    """True when the step is the anchor update of its block."""
    return _require_int(step, "step") % BLOCK == ANCHOR_OFFSET


def cell_update_index(step):
    """Index of a step among the cell updates (anchor steps excluded)."""
    step = _require_int(step, "step")
    if step < 0:
        raise ValueError("step must not be negative")
    return step - step // BLOCK


def fixed_cell(step):
    """Cell of the fixed arm: round robin over the nine cells.

    Every cell update of the study carries the next cell of the rotation, so
    over 96 steps each of the nine cells is visited exactly eight times.  The
    anchor steps return ``None``, exactly like :meth:`Frontier.cell`.
    """
    step = _require_int(step, "step")
    if step < 0:
        raise ValueError("step must not be negative")
    if step % BLOCK == ANCHOR_OFFSET:
        return None
    return CELLS[cell_update_index(step) % len(CELLS)]


def replay_cell(schedule, step):
    """Cell of the yoked arm: a recorded per-update schedule.

    ``schedule`` maps step numbers to cells, as produced by
    :attr:`Frontier.schedule`.  Anchor steps return ``None``.  A schedule that
    does not cover the step is rejected instead of silently padded.
    """
    step = _require_int(step, "step")
    if step < 0:
        raise ValueError("step must not be negative")
    if step % BLOCK == ANCHOR_OFFSET:
        return None
    if not isinstance(schedule, Mapping):
        raise ValueError("schedule must be a mapping of step -> cell")
    if step not in schedule:
        raise ValueError(f"yoked schedule does not cover step {step}")
    cell = tuple(int(part) for part in schedule[step])
    if cell not in _CELL_SET:
        raise ValueError(f"yoked schedule holds an unknown cell {cell!r}")
    return cell


def yoked_schedule(schedules, seed, seeds):
    """Schedule of the yoked arm: the adaptive schedule of the next seed.

    ``schedules`` maps a seed to its adaptive schedule; the yoked arm of
    ``seed`` replays the schedule of the next seed in ``seeds``, cycling at the
    end.  A seed never replays its own schedule.
    """
    seeds = tuple(_require_int(value, "seed") for value in seeds)
    seed = _require_int(seed, "seed")
    if seed not in seeds:
        raise ValueError(f"seed {seed} is not part of {seeds}")
    position = seeds.index(seed)
    donor = seeds[(position + 1) % len(seeds)]
    if donor not in schedules:
        raise ValueError(f"the adaptive schedule of seed {donor} is missing")
    schedule = schedules[donor]
    if not isinstance(schedule, Mapping) or not schedule:
        raise ValueError(f"the adaptive schedule of seed {donor} is empty")
    return schedule


class Frontier:
    """Adaptive cell selection for the retrieval-frontier study.

    The controller only ever sees aggregate feedback: per cell the number of
    history families that were correct on all four questions and the number of
    families that were evaluated, plus the fraction of anchor families that
    were correct.  It never sees labels, per-question scores or the final
    score of a run, so it cannot optimize for the evaluation set.
    """

    VERSION = 1

    def __init__(
        self,
        *,
        threshold=MASTERY_THRESHOLD,
        consecutive=CONSECUTIVE_CALIBRATIONS,
        anchor_tolerance=ANCHOR_TOLERANCE,
        coverage_slack=COVERAGE_SLACK,
        baseline_anchor=None,
    ):
        self.threshold = float(threshold)
        self.consecutive = _require_int(consecutive, "consecutive")
        self.anchor_tolerance = float(anchor_tolerance)
        self.coverage_slack = _require_int(coverage_slack, "coverage_slack")
        if self.consecutive < 1:
            raise ValueError("consecutive must be positive")
        if self.coverage_slack < 0:
            raise ValueError("coverage_slack must not be negative")
        self.mastered = set()
        self.streak = {cell: 0 for cell in CELLS}
        self.latest = {}
        self.selections = {writes: 0 for writes in WRITES}
        self.baseline_anchor = None if baseline_anchor is None else float(baseline_anchor)
        self.last_anchor = None
        self.frozen = False
        self.observations = 0
        self.promotions = 0
        self.schedule = {}

    # -- feedback ---------------------------------------------------------
    def observe(self, counts, anchor_accuracy):
        """Record one calibration.

        Parameters
        ----------
        counts:
            Mapping ``(gap, writes) -> (correct families, evaluated families)``
            where a family counts as correct only when all four questions
            (two worlds times two questions) were answered correctly.  A cell
            that is absent from the mapping was not measured and its promotion
            streak is broken.
        anchor_accuracy:
            Fraction of anchor families that were correct on all questions.

        Returns a summary dictionary of the policy state.
        """
        if not isinstance(counts, Mapping):
            raise ValueError("counts must be a mapping of cell -> (correct, total)")
        anchor = float(anchor_accuracy)
        if not math.isfinite(anchor) or not 0.0 <= anchor <= 1.0:
            raise ValueError("anchor_accuracy must be a fraction in [0, 1]")
        parsed = {}
        for cell, value in counts.items():
            key = _parse_cell(cell)
            if key not in _CELL_SET:
                raise ValueError(f"unknown cell {cell!r}")
            correct, total = value
            correct = _require_int(correct, "correct")
            total = _require_int(total, "total")
            if correct < 0 or total < 0 or correct > total:
                raise ValueError(f"invalid counts for cell {key}: {value!r}")
            parsed[key] = (correct, total)

        if self.baseline_anchor is None:
            self.baseline_anchor = anchor
        guard_ok = (self.baseline_anchor - anchor) < self.anchor_tolerance
        self.frozen = not guard_ok
        self.last_anchor = anchor
        self.observations += 1

        for cell, measured in parsed.items():
            self.latest[cell] = (measured[0], measured[1], self.observations)
        # The active gap of every regime is frozen while this calibration is
        # scored, so a promotion can only open the next gap at the following
        # calibration.
        active_gaps = {writes: self.active_gap(writes) for writes in WRITES}
        if not guard_ok:
            # Frozen calibration: the short anchor skill regressed, so nothing
            # may be promoted and the promotion evidence collected so far is
            # dropped.
            for cell in self.streak:
                self.streak[cell] = 0
            return self.summary()
        for cell in CELLS:
            measured = parsed.get(cell)
            if measured is None:
                # Not measured in this calibration: the chain of successive
                # qualifying calibrations is broken.
                self.streak[cell] = 0
                continue
            qualifies = (
                active_gaps.get(cell[1]) == cell[0]
                and wilson_lower_bound(*measured) >= self.threshold
            )
            self.streak[cell] = self.streak[cell] + 1 if qualifies else 0
        for cell in CELLS:
            if self.streak[cell] >= self.consecutive and cell not in self.mastered:
                self.mastered.add(cell)
                self.promotions += 1
        return self.summary()

    # -- policy -----------------------------------------------------------
    def active_gap(self, writes):
        """First gap of the writes regime whose cell is not mastered yet."""
        for gap in GAPS:
            if (gap, writes) not in self.mastered:
                return gap
        return None

    def _weakness(self, cell):
        """Wilson lower bound of the most recent measurement; ``-1`` if stale."""
        measured = self.latest.get(cell)
        if measured is None or measured[2] != self.observations:
            return -1.0
        correct, total, _ = measured
        if total == 0:
            return -1.0
        return wilson_lower_bound(correct, total)

    def candidates(self):
        """The candidate cells: one per writes regime."""
        candidates = []
        for writes in WRITES:
            gap = self.active_gap(writes)
            candidates.append((gap if gap is not None else GAPS[-1], writes))
        return candidates

    def _rank(self, cell):
        """Sort key of a candidate: weakest first, then coverage, then order."""
        return (
            self._weakness(cell),
            self.selections[cell[1]],
            cell[1],
            cell[0],
        )

    def _select(self):
        candidates = self.candidates()
        least_served = min(self.selections[cell[1]] for cell in candidates)
        eligible = [
            cell
            for cell in candidates
            if self.selections[cell[1]] <= least_served + self.coverage_slack
        ]
        return min(eligible, key=self._rank)

    def cell(self, step):
        """Cell of the adaptive arm for ``step``.

        Returns the chosen ``(gap, writes)`` cell on every cell update and
        ``None`` on an anchor step.  The decision is recorded in
        :attr:`schedule`, so a checkpointed run replays identically and the
        yoked arm can reuse the trace.
        """
        step = _require_int(step, "step")
        if step < 0:
            raise ValueError("step must not be negative")
        if step % BLOCK == ANCHOR_OFFSET:
            return None
        if step in self.schedule:
            return self.schedule[step]
        choice = self._select()
        self.schedule[step] = choice
        self.selections[choice[1]] += 1
        return choice

    @property
    def trace(self):
        """Cells chosen so far, in step order."""
        return [self.schedule[step] for step in sorted(self.schedule)]

    # -- reporting --------------------------------------------------------
    def summary(self):
        """Snapshot of the policy state."""
        return {
            "mastered": sorted(self.mastered),
            "frozen": self.frozen,
            "baseline_anchor": self.baseline_anchor,
            "last_anchor": self.last_anchor,
            "observations": self.observations,
            "promotions": self.promotions,
            "selections": dict(self.selections),
            "streaks": {f"{gap},{writes}": self.streak[(gap, writes)] for gap, writes in CELLS},
            "lower_bounds": {
                f"{gap},{writes}": self._weakness((gap, writes)) for gap, writes in CELLS
            },
            "trace": [list(cell) for cell in self.trace],
        }

    # -- checkpointing ----------------------------------------------------
    def state_dict(self):
        """JSON-serializable policy state."""
        return {
            "version": self.VERSION,
            "threshold": self.threshold,
            "consecutive": self.consecutive,
            "anchor_tolerance": self.anchor_tolerance,
            "coverage_slack": self.coverage_slack,
            "streak": {f"{gap},{writes}": self.streak[(gap, writes)] for gap, writes in CELLS},
            "mastered": [list(cell) for cell in sorted(self.mastered)],
            "latest": {
                f"{gap},{writes}": list(self.latest[(gap, writes)])
                for gap, writes in CELLS
                if (gap, writes) in self.latest
            },
            "selections": {str(writes): self.selections[writes] for writes in WRITES},
            "baseline_anchor": self.baseline_anchor,
            "last_anchor": self.last_anchor,
            "frozen": self.frozen,
            "observations": self.observations,
            "promotions": self.promotions,
            "schedule": {str(step): list(cell) for step, cell in sorted(self.schedule.items())},
        }

    @classmethod
    def from_state_dict(cls, data):
        """Restore a controller from :meth:`state_dict`.

        Raises ``ValueError`` when the state is malformed, so a corrupted
        checkpoint fails loudly instead of silently resetting the policy.
        """
        if not isinstance(data, Mapping):
            raise ValueError("frontier state must be a mapping")
        version = data.get("version")
        if version != cls.VERSION:
            raise ValueError(f"unsupported frontier state version {version!r}")
        for key in ("threshold", "consecutive", "anchor_tolerance", "coverage_slack"):
            if key not in data:
                raise ValueError(f"frontier state is missing {key!r}")

        controller = cls(
            threshold=float(data["threshold"]),
            consecutive=_require_int(data["consecutive"], "consecutive"),
            anchor_tolerance=float(data["anchor_tolerance"]),
            coverage_slack=_require_int(data["coverage_slack"], "coverage_slack"),
        )
        streak = data.get("streak", {})
        if not isinstance(streak, Mapping):
            raise ValueError("streak must be a mapping")
        for key, value in streak.items():
            cell = _parse_cell(key)
            if cell not in _CELL_SET:
                raise ValueError(f"unknown cell in streak: {key!r}")
            count = _require_int(value, "streak")
            if count < 0:
                raise ValueError("streaks must not be negative")
            controller.streak[cell] = count
        mastered = data.get("mastered", [])
        if not isinstance(mastered, (list, tuple)):
            raise ValueError("mastered must be a list")
        for cell in mastered:
            key = _parse_cell(cell)
            if key not in _CELL_SET:
                raise ValueError(f"unknown cell in mastered set: {cell!r}")
            controller.mastered.add(key)
        latest = data.get("latest", {})
        if not isinstance(latest, Mapping):
            raise ValueError("latest must be a mapping")
        for key, value in latest.items():
            cell = _parse_cell(key)
            if cell not in _CELL_SET:
                raise ValueError(f"unknown cell in latest: {key!r}")
            correct, total, seen = (_require_int(part, "latest") for part in value)
            if not 0 <= correct <= total:
                raise ValueError(f"invalid measurement for cell {cell}")
            controller.latest[cell] = (correct, total, seen)
        selections = data.get("selections", {})
        if not isinstance(selections, Mapping):
            raise ValueError("selections must be a mapping")
        for key, value in selections.items():
            writes = int(key)
            if writes not in WRITES:
                raise ValueError(f"unknown writes regime {key!r}")
            count = _require_int(value, "selections")
            if count < 0:
                raise ValueError("selections must not be negative")
            controller.selections[writes] = count
        baseline = data.get("baseline_anchor")
        if baseline is not None:
            baseline = float(baseline)
            if not 0.0 <= baseline <= 1.0:
                raise ValueError("baseline_anchor must be a fraction in [0, 1]")
        controller.baseline_anchor = baseline
        last_anchor = data.get("last_anchor")
        if last_anchor is not None:
            last_anchor = float(last_anchor)
            if not 0.0 <= last_anchor <= 1.0:
                raise ValueError("last_anchor must be a fraction in [0, 1]")
        controller.last_anchor = last_anchor
        controller.frozen = bool(data.get("frozen", False))
        controller.observations = _require_int(data.get("observations", 0), "observations")
        controller.promotions = _require_int(data.get("promotions", 0), "promotions")
        schedule = data.get("schedule", {})
        if not isinstance(schedule, Mapping):
            raise ValueError("schedule must be a mapping")
        for key, cell in schedule.items():
            try:
                step = int(key)
            except (TypeError, ValueError) as error:
                raise ValueError(f"malformed schedule step {key!r}") from error
            if step < 0:
                raise ValueError("schedule steps must not be negative")
            parsed = _parse_cell(cell)
            if parsed not in _CELL_SET:
                raise ValueError(f"unknown cell in schedule: {cell!r}")
            controller.schedule[step] = parsed
        return controller


def _parse_cell(value):
    """Parse a ``(gap, writes)`` cell given as a tuple or as ``"gap,writes"``."""
    if isinstance(value, str):
        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError(f"malformed cell {value!r}")
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError as error:
            raise ValueError(f"malformed cell {value!r}") from error
    try:
        gap, writes = value
    except (TypeError, ValueError) as error:
        raise ValueError(f"malformed cell {value!r}") from error
    return (int(gap), int(writes))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=96)
    args = parser.parse_args(argv)

    ids, labels = make_examples(
        args.seed, args.index, args.pairs, gap=GAPS[0], writes=WRITES[0]
    )
    visits = {}
    for step in range(args.steps):
        cell = fixed_cell(step)
        if cell is not None:
            visits[cell] = visits.get(cell, 0) + 1
    print(
        json.dumps(
            {
                "ids_shape": list(ids.shape),
                "labels_shape": list(labels.shape),
                "dtype": str(ids.dtype),
                "anchor_updates": sum(1 for s in range(args.steps) if is_anchor_step(s)),
                "cell_updates": sum(1 for s in range(args.steps) if not is_anchor_step(s)),
                "fixed_visits": {f"{gap},{writes}": count for (gap, writes), count in sorted(visits.items())},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
