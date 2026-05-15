"""Verify the new session-persistence settings exist on Settings with
the documented defaults. Catches accidental rename/removal regressions.
"""
from docai.config import settings


def test_enable_session_persistence_default_true() -> None:
    assert settings.ENABLE_SESSION_PERSISTENCE is True


def test_enable_session_summarization_default_false() -> None:
    assert settings.ENABLE_SESSION_SUMMARIZATION is False


def test_session_window_messages_default_20() -> None:
    assert settings.SESSION_WINDOW_MESSAGES == 20


def test_summarization_cadence_default_10() -> None:
    assert settings.SUMMARIZATION_CADENCE_MESSAGES == 10


def test_summarization_model_defaults_empty() -> None:
    assert settings.SUMMARIZATION_OLLAMA_MODEL == ""


def test_summary_max_tokens_default_300() -> None:
    assert settings.SUMMARY_MAX_TOKENS == 300


def test_session_user_id_cookie_default() -> None:
    assert settings.SESSION_USER_ID_COOKIE == "docai_uid"
