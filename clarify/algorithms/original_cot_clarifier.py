"""SDK entry point for the frozen original CoT prompt (547/770 historical run)."""

from clarify.algorithms.contrast_question_clarifier import FrozenCoTClarifier


class OriginalCoTClarifier(FrozenCoTClarifier):
    """Original targeted question and frozen four-step final-generation prompt."""
