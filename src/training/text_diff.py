from difflib import SequenceMatcher

from rich.text import Text

REFERENCE_DIFFERENCE_STYLE = "bold red"
PREDICTION_DIFFERENCE_STYLE = "bold yellow"


def create_text_diff(reference: str, prediction: str) -> tuple[Text, Text]:
    reference_text = Text()
    prediction_text = Text()
    matcher = SequenceMatcher(None, reference, prediction, autojunk=False)

    for tag, reference_start, reference_end, prediction_start, prediction_end in (
        matcher.get_opcodes()
    ):
        style_changed = tag != "equal"
        reference_text.append(
            reference[reference_start:reference_end],
            style=REFERENCE_DIFFERENCE_STYLE if style_changed else None,
        )
        prediction_text.append(
            prediction[prediction_start:prediction_end],
            style=PREDICTION_DIFFERENCE_STYLE if style_changed else None,
        )

    return reference_text, prediction_text
