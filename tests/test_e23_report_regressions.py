import pytest

from scripts.e23_report import acquisition_pass, cluster_point, pooled_cluster_sums, summarize


def row(identifier, correct, *, group="g", pair=None, variant="base", rule_in="options"):
    return dict(id=identifier, group_id=group, correct=correct, gold_idx=0,
                pair=pair, variant=variant, rule_in=rule_in)


def test_pair_results_keep_both_rule_placements():
    rows = [row("a", 1, pair="p", variant="pair_hi"),
            row("b", 1, pair="p", variant="pair_lo"),
            row("c", 1, pair="p", variant="pair_hi", rule_in="question"),
            row("d", 0, pair="p", variant="pair_lo", rule_in="question")]
    result = summarize(rows)
    assert result["pair_both"] == 0.5
    assert result["pair_format_count"] == 2
    assert result["pair_both_by_rule_in"]["options"]["rate"] == 1.0
    assert result["pair_both_by_rule_in"]["question"]["rate"] == 0.0
    assert summarize(rows[::-1]) == result


def test_every_seed_contributes_instead_of_last_seed_overwriting():
    control = [row("a", 0)]
    candidates = [[row("a", 1)], [row("a", 1)], [row("a", 0)]]
    pairs = [(control, candidate) for candidate in candidates]
    assert cluster_point(pooled_cluster_sums(pairs)) == pytest.approx(2 / 3)
    assert pooled_cluster_sums(pairs) == pooled_cluster_sums(pairs[::-1])


def test_unequal_cluster_sizes_preserve_question_accuracy():
    controls = [row(str(i), 0, group="small" if i == 0 else "large") for i in range(4)]
    candidates = [r | {"correct": int(i == 0)} for i, r in enumerate(controls)]
    assert cluster_point(pooled_cluster_sums([(controls, candidates)])) == 0.25


def test_pairing_rejects_missing_rows():
    with pytest.raises(ValueError):
        pooled_cluster_sums([([row("a", 1), row("b", 0)], [row("a", 1)])])


def test_accuracy_gain_alone_does_not_satisfy_acquisition_rule():
    assert not acquisition_pass(0.20, [0.0, 0.0, 0.0])
    assert not acquisition_pass(0.10, [0.5, 0.5, 0.5])
    assert acquisition_pass(0.20, [0.5, 0.5, 0.0])


def test_incomplete_pair_is_not_silently_dropped():
    with pytest.raises(ValueError, match="Incomplete"):
        summarize([row("a", 1, pair="p", variant="pair_hi")])
