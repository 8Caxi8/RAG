from .models import AnsweredQuestion, MinimalSearchResults, MinimalSource

DEFAULT_OVERLAP_THRESHOLD = 0.05


def _iou(retrieved: MinimalSource, correct: MinimalSource) -> float:
    """Intersection over Union (IoU) between two character ranges.

    Args:
        retrieved: A source returned by the retrieval system.
        correct: A ground-truth source to compare against.

    Returns:
        0.0 if the sources are in different files or don't overlap at
        all; otherwise the length of the overlapping range divided by
        the length of the *union* of the two ranges (1.0 means the two
        ranges are identical). Note this is symmetric and penalizes a
        retrieved chunk that is much larger than the correct span, even
        if it fully contains it — unlike a plain "coverage of correct"
        ratio, which would give 1.0 regardless of how much extra
        content surrounds the correct span.
    """
    if retrieved.file_path != correct.file_path:
        return 0.0

    overlap_start = max(retrieved.first_character_index,
                        correct.first_character_index)
    overlap_end = min(retrieved.last_character_index,
                      correct.last_character_index)
    intersection = max(0, overlap_end - overlap_start)

    retrieved_len = (retrieved.last_character_index -
                     retrieved.first_character_index)
    correct_len = correct.last_character_index - correct.first_character_index
    union = retrieved_len + correct_len - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def _is_found(
    correct: MinimalSource,
    retrieved_sources: list[MinimalSource],
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> bool:
    """Whether any retrieved source overlaps `correct` by >= threshold."""
    return any(_iou(r, correct) >= threshold for r in retrieved_sources)


def recall_at_k(
    retrieved_sources: list[MinimalSource],
    correct_sources: list[MinimalSource],
    k: int,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> float:
    """Compute recall@k for a single question.

    Args:
        retrieved_sources: Sources returned by the retrieval system,
            ranked best-first (only the first `k` are considered).
        correct_sources: Ground-truth sources for the question.
        k: Number of top retrieved sources to consider.
        threshold: Minimum overlap fraction to count a source as found.

    Returns:
        The fraction of `correct_sources` found among the top-k
        `retrieved_sources`. Returns 1.0 if there are no ground-truth
        sources (nothing to miss).
    """
    if not correct_sources:
        return 1.0

    top_k = retrieved_sources[:k]
    found = sum(1 for c in correct_sources if _is_found(c, top_k, threshold))
    return found / len(correct_sources)


def evaluate_dataset(
    student_results: list[MinimalSearchResults],
    ground_truth: dict[str, AnsweredQuestion],
    k_values: list[int],
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> dict[int, float]:
    """Average recall@k over a whole dataset, for several values of k.

    Args:
        student_results: The retrieval system's results, one per question.
        ground_truth: Mapping from question_id to its ground-truth
            AnsweredQuestion (with the correct sources).
        k_values: The k values to compute recall for (e.g. [1, 3, 5, 10]).
        threshold: Minimum overlap fraction to count a source as found.

    Returns:
        A mapping from k to the mean recall@k across all matched
        questions. Questions whose question_id has no matching ground
        truth entry are skipped (and reported separately by the caller
        if needed).
    """
    per_k_scores: dict[int, list[float]] = {k: [] for k in k_values}

    for result in student_results:
        truth = ground_truth.get(result.question_id)
        if truth is None:
            continue
        for k in k_values:
            per_k_scores[k].append(
                recall_at_k(result.retrieved_sources,
                            truth.sources,
                            k,
                            threshold)
            )

    return {
        k: (sum(scores) / len(scores) if scores else 0.0)
        for k, scores in per_k_scores.items()
    }
