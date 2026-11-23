from ready_eval.prompt import parse_output, to_label_id

LABELS = ["activate_my_card", "age_limit", "card_arrival", "exchange_rate"]


def test_label_exact_normalized_fuzzy():
    assert to_label_id("card_arrival", LABELS) == (2, "exact")
    assert to_label_id("Card Arrival", LABELS) == (2, "normalized")
    assert to_label_id("card-arrival ", LABELS) == (2, "normalized")
    assert to_label_id("exchange_rates", LABELS)[0] == 3  # 近似一致
    assert to_label_id("refund_not_showing_up", LABELS) == (-1, "none")


def test_parse_ok():
    p = parse_output('{"predicted_label": "age_limit", "confidence": 0.8}', LABELS)
    assert (p.status, p.predicted_id, p.confidence) == ("ok", 1, 0.8)


def test_parse_percent_and_clip():
    p = parse_output('{"predicted_label": "age_limit", "confidence": 85}', LABELS)
    assert p.confidence == 0.85 and p.note == "confidence_percent_rescaled"
    p = parse_output('{"predicted_label": "age_limit", "confidence": -0.2}', LABELS)
    assert p.confidence == 0.0 and p.note == "confidence_clipped"


def test_parse_failures_are_recorded():
    assert parse_output(None, LABELS).status == "invalid_output"
    assert parse_output("not json", LABELS).status == "invalid_output"
    assert parse_output('{"predicted_label": "age_limit"}', LABELS).status == "invalid_output"
    assert parse_output('{"predicted_label": "age_limit", "confidence": "high"}', LABELS).status == "invalid_output"
    p = parse_output('{"predicted_label": "unknown_thing", "confidence": 0.9}', LABELS)
    assert p.status == "invalid_label" and p.predicted_id == -1
