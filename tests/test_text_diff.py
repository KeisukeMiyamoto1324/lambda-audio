from src.training.text_diff import (
    PREDICTION_DIFFERENCE_STYLE,
    REFERENCE_DIFFERENCE_STYLE,
    create_text_diff,
)


def test_create_text_diff_highlights_replacement() -> None:
    reference, prediction = create_text_diff("今日は晴れ", "今日は雨")

    assert reference.plain == "今日は晴れ"
    assert prediction.plain == "今日は雨"
    assert [(span.start, span.end, span.style) for span in reference.spans] == [
        (3, 5, REFERENCE_DIFFERENCE_STYLE)
    ]
    assert [(span.start, span.end, span.style) for span in prediction.spans] == [
        (3, 4, PREDICTION_DIFFERENCE_STYLE)
    ]


def test_create_text_diff_highlights_insertion_and_deletion() -> None:
    deleted_reference, deleted_prediction = create_text_diff("音声認識", "音声")
    inserted_reference, inserted_prediction = create_text_diff("音声", "音声認識")

    assert deleted_reference.spans[0].style == REFERENCE_DIFFERENCE_STYLE
    assert deleted_prediction.spans == []
    assert inserted_reference.spans == []
    assert inserted_prediction.spans[0].style == PREDICTION_DIFFERENCE_STYLE


def test_create_text_diff_leaves_matching_text_unstyled() -> None:
    reference, prediction = create_text_diff("同じ文章", "同じ文章")

    assert reference.plain == "同じ文章"
    assert prediction.plain == "同じ文章"
    assert reference.spans == []
    assert prediction.spans == []
