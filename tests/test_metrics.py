import pytest

from defectbench.metrics import (
    _norm,
    cer,
    classification_report,
    field_accuracy,
    json_validity_rate,
    parse_json_output,
)


def test_classification_report_perfect():
    report = classification_report([True, False], [True, False], ["bottle", "bottle"])
    assert report.accuracy == 1.0
    assert report.f1 == 1.0
    assert report.per_category["bottle"]["f1"] == 1.0


def test_classification_report_mixed():
    y_true = [True, True, False, False, True]
    y_pred = [True, False, False, True, True]
    report = classification_report(y_true, y_pred, ["screw"] * 5)
    assert report.tp == 2
    assert report.fp == 1
    assert report.tn == 1
    assert report.fn == 1
    assert report.precision == pytest.approx(2 / 3)
    assert report.recall == pytest.approx(2 / 3)
    assert report.accuracy == pytest.approx(3 / 5)


def test_classification_report_empty_raises():
    with pytest.raises(ValueError):
        classification_report([], [], [])


def test_classification_report_length_mismatch_raises():
    with pytest.raises(ValueError):
        classification_report([True], [True, False], ["a", "b"])


def test_cer_identical():
    assert cer("broken_large", "broken_large") == 0.0


def test_cer_partial():
    # one substitution in 12 chars
    assert cer("broken_large", "broken_largeX") == pytest.approx(1 / 12)


def test_cer_empty_reference():
    assert cer("", "abc") == 1.0
    assert cer("", "") == 0.0


def test_parse_json_output_plain():
    parsed = parse_json_output('{"defect_type": "good", "confidence": 0.9}')
    assert parsed == {"defect_type": "good", "confidence": 0.9}


def test_parse_json_output_fenced():
    raw = "```json\n{\"defect_type\": \"scratch\"}\n```"
    assert parse_json_output(raw) == {"defect_type": "scratch"}


def test_parse_json_output_with_prose():
    raw = 'Here is the result: {"defect_type": "crack"} hope that helps'
    assert parse_json_output(raw) == {"defect_type": "crack"}


def test_parse_json_output_invalid():
    assert parse_json_output("no json here") is None
    assert parse_json_output("") is None
    assert parse_json_output("[1, 2, 3]") is None


def test_json_validity_rate():
    raws = ['{"a": 1}', "garbage", '```json\n{"b": 2}\n```']
    assert json_validity_rate(raws) == pytest.approx(2 / 3)


def test_field_accuracy():
    parsed = [
        {"defect_type": "broken_large"},
        {"defect_type": "good"},
        None,
        {"defect_type": "scratch"},
    ]
    expected = [
        {"defect_type": "broken_large"},
        {"defect_type": "good"},
        {"defect_type": "broken_small"},
        {"defect_type": "crack"},
    ]
    # 2 hits out of 4 scored fields (None does not count as a hit)
    assert field_accuracy(parsed, expected, "defect_type") == pytest.approx(0.5)


def test_norm():
    assert _norm(" Broken_Large ") == "broken_large"
