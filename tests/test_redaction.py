from eightysix.redaction import redact


def test_phone_formats_are_redacted():
    for raw in ("call 555-867-5309", "call (555) 867-5309", "call 555.867.5309",
                "call +1 555 867 5309"):
        assert "[PHONE]" in redact(raw)
        assert "5309" not in redact(raw)


def test_email_is_redacted():
    assert redact("reach me at sarah.k+vip@example.co.uk") == "reach me at [EMAIL]"


def test_clean_text_passes_through_untouched():
    q = "do we have enough fresh mozzarella for 12 pies on the 14th?"
    assert redact(q) == q
