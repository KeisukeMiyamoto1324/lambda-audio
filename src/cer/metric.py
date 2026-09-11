from dataclasses import dataclass

from src.data.text import normalize_transcription


@dataclass(frozen=True)
class CerCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_characters: int = 0
    samples: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    def __add__(self, other: "CerCounts") -> "CerCounts":
        return CerCounts(
            substitutions=self.substitutions + other.substitutions,
            deletions=self.deletions + other.deletions,
            insertions=self.insertions + other.insertions,
            reference_characters=self.reference_characters + other.reference_characters,
            samples=self.samples + other.samples,
        )


def compute_cer_counts(references: list[str], hypotheses: list[str]) -> CerCounts:
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length.")

    counts = CerCounts()
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        normalized_reference = normalize_transcription(reference)
        normalized_hypothesis = normalize_transcription(hypothesis)
        substitutions, deletions, insertions = _edit_counts(
            normalized_reference,
            normalized_hypothesis,
        )
        counts += CerCounts(
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            reference_characters=len(normalized_reference),
            samples=1,
        )
    return counts


def _edit_counts(reference: str, hypothesis: str) -> tuple[int, int, int]:
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distances = [[0] * columns for _ in range(rows)]
    for row in range(1, rows):
        distances[row][0] = row
    for column in range(1, columns):
        distances[0][column] = column

    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                distances[row][column] = distances[row - 1][column - 1]
                continue
            distances[row][column] = 1 + min(
                distances[row - 1][column - 1],
                distances[row - 1][column],
                distances[row][column - 1],
            )

    substitutions = 0
    deletions = 0
    insertions = 0
    row = len(reference)
    column = len(hypothesis)
    while row > 0 or column > 0:
        if row > 0 and column > 0 and reference[row - 1] == hypothesis[column - 1]:
            row -= 1
            column -= 1
            continue
        if row > 0 and column > 0 and distances[row][column] == distances[row - 1][column - 1] + 1:
            substitutions += 1
            row -= 1
            column -= 1
            continue
        if row > 0 and distances[row][column] == distances[row - 1][column] + 1:
            deletions += 1
            row -= 1
            continue
        insertions += 1
        column -= 1

    return substitutions, deletions, insertions
