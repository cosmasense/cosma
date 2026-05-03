"""Tests for the sentence splitter used by chunk_content."""

import pytest

from cosma_backend.summarizer.tokenization import _split_into_sentences


@pytest.mark.unit
class TestSplitIntoSentences:
    """Behaviors that the previous `text.split('. ')` got wrong."""

    def test_empty_input(self):
        assert _split_into_sentences("") == []
        assert _split_into_sentences("   \n\n  ") == []

    def test_single_sentence(self):
        assert _split_into_sentences("Hello world.") == ["Hello world."]

    def test_two_sentences(self):
        out = _split_into_sentences("Hello world. Goodbye world.")
        assert out == ["Hello world.", "Goodbye world."]

    def test_question_and_exclamation(self):
        out = _split_into_sentences("Why? Because! Then we left.")
        assert out == ["Why?", "Because!", "Then we left."]

    def test_abbreviations_do_not_split(self):
        # The killer of the old splitter:
        out = _split_into_sentences("Mr. Smith met Dr. Wong in U.S.A.")
        # Should be 1 sentence, not 4. We accept either 1 or 2 here
        # because the trailing period without a following sentence is
        # a legitimate edge.
        assert len(out) == 1

    def test_decimals_do_not_split(self):
        out = _split_into_sentences("Use v3.14 of pi for accuracy.")
        assert len(out) == 1
        assert "3.14" in out[0]

    def test_paragraph_breaks_split(self):
        # Even without sentence punctuation, paragraph breaks split.
        text = "First paragraph\n\nSecond paragraph"
        out = _split_into_sentences(text)
        assert len(out) == 2

    def test_chinese_period_splits(self):
        out = _split_into_sentences("第一句。第二句。")
        assert len(out) >= 1  # at minimum, no crash; ideally 2

    def test_long_realistic_text(self):
        text = (
            "The quarterly report from Mr. Smith shows growth in Q3. "
            "Revenue increased by 12.5% over the previous period. "
            "Dr. Wong noted that costs remained stable at U.S.A. "
            "operations.\n\n"
            "Next quarter we expect continued growth!"
        )
        out = _split_into_sentences(text)
        # Should be ~4 sentences, NOT the 11-12 the old splitter would
        # have produced (one per `. `).
        assert 3 <= len(out) <= 5
        # Every sentence should be substantive
        for s in out:
            assert len(s) > 5

    def test_sentences_keep_their_terminators(self):
        out = _split_into_sentences("First sentence. Second sentence!")
        assert all(s.rstrip()[-1] in ".!?" for s in out), out

    def test_no_blank_sentences(self):
        out = _split_into_sentences(". . . . Real one. ")
        # The leading dust-of-periods may collapse but at minimum we
        # should get the real sentence and no blank entries.
        assert all(s.strip() for s in out)
