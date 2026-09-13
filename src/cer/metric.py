class CerMetric:
    def __init__(self) -> None:
        self.errors = 0
        self.reference_characters = 0

    def update(self, reference: str, prediction: str) -> None:
        # ---------------------------------------------------------
        # Keep one edit-distance row to limit memory use.
        # ---------------------------------------------------------
        previous = list(range(len(prediction) + 1))
        for reference_index, reference_character in enumerate(reference, start=1):
            current = [reference_index]
            for prediction_index, prediction_character in enumerate(prediction, start=1):
                substitution = previous[prediction_index - 1] + (reference_character != prediction_character)
                deletion = previous[prediction_index] + 1
                insertion = current[prediction_index - 1] + 1
                current.append(min(substitution, deletion, insertion))
            previous = current

        self.errors += previous[-1]
        self.reference_characters += len(reference)

    @property
    def value(self) -> float:
        if self.reference_characters == 0:
            raise ValueError("CER requires at least one reference character.")
        return self.errors / self.reference_characters
