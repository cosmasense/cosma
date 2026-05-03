"""Tests for hardware-aware tuning helpers."""

import pytest

from cosma_backend.utils.hardware import choose_n_ctx_for_ram, resolved_n_ctx


@pytest.mark.unit
class TestChooseNctxForRam:
    """The helper must scale with RAM, clamp pathological inputs, and
    align cleanly. Concrete brackets for common Mac configs:"""

    def test_8gb_mac(self):
        assert choose_n_ctx_for_ram(8) == 4096

    def test_16gb_mac(self):
        assert choose_n_ctx_for_ram(16) == 16384

    def test_32gb_mac_uses_more_context(self):
        assert choose_n_ctx_for_ram(32) > choose_n_ctx_for_ram(16)
        assert choose_n_ctx_for_ram(32) == 32768

    def test_64gb_mac_uses_more_context(self):
        assert choose_n_ctx_for_ram(64) > choose_n_ctx_for_ram(32)
        assert choose_n_ctx_for_ram(64) == 65536

    def test_huge_ram_capped_at_ceiling(self):
        assert choose_n_ctx_for_ram(192) == 131072
        assert choose_n_ctx_for_ram(512) == 131072

    def test_tiny_ram_clamped_to_floor(self):
        assert choose_n_ctx_for_ram(2) == 4096
        assert choose_n_ctx_for_ram(0.5) == 4096

    def test_alignment(self):
        for ram in (4, 8, 12, 16, 24, 32, 48, 64, 128):
            assert choose_n_ctx_for_ram(ram) % 1024 == 0, (
                f"unaligned for {ram} GiB"
            )


@pytest.mark.unit
class TestResolvedNctx:
    def test_explicit_wins(self):
        assert resolved_n_ctx(8192) == 8192
        assert resolved_n_ctx(65536) == 65536

    def test_zero_triggers_detection(self):
        # Any value should be at least the floor.
        v = resolved_n_ctx(0)
        assert v >= 4096
        assert v % 1024 == 0
