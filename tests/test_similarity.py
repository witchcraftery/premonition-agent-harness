from foresight_harness.models import Branch, MatchGrade
from foresight_harness.similarity import grade_branch_match, normalized_tokens, semantic_overlap


def test_normalized_tokens_removes_noise_words():
    assert normalized_tokens("Customer asks for THE refund, please!") == {
        "customer",
        "asks",
        "refund",
    }


def test_semantic_overlap_scores_related_text():
    score = semantic_overlap(
        "customer asks for refund on damaged delivery",
        "customer wants damaged item refund",
    )

    assert score >= 0.45


def test_grade_exact_intent_wins():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks for refund",
        intent="refund_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks whether refund is available",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.EXACT_INTENT
    assert graded.match_score == 1.0


def test_grade_unsafe_takes_precedence_over_exact_intent():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks for guaranteed refund",
        intent="refund_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks whether refund is available",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.UNSAFE
    assert graded.match_score == 0.0


def test_grade_semantic_equivalent_threshold():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks refund damaged delivery",
        intent="unknown",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks damaged delivery refund",
        expected_intent="refund_request",
    )

    assert graded.match_grade == MatchGrade.SEMANTIC_EQUIVALENT


def test_grade_useful_partial_threshold():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks refund damaged item",
        intent="unknown",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks duplicate charge refund timing",
        expected_intent="billing_refund_timing",
    )

    assert graded.match_grade == MatchGrade.USEFUL_PARTIAL


def test_grade_miss_for_unrelated_branch():
    branch = Branch(
        branch_id="br-1",
        predicted_event="customer asks for supervisor escalation",
        intent="escalation_request",
        probability=0.6,
        rank=1,
    )

    graded = grade_branch_match(
        branch,
        actual_next_event="customer asks about duplicate charge refund timing",
        expected_intent="billing_refund_timing",
    )

    assert graded.match_grade == MatchGrade.MISS
