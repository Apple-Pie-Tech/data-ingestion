import pytest

from app.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _no_dotenv_leak(monkeypatch):
    """Keep the developer's real .env out of the test run.

    Settings.model_config points env_file at ".env" (relative to cwd), so
    running pytest from the project directory picks up the real, gitignored
    .env and makes test results depend on whatever is in it (e.g.
    AUDIO_STORAGE_ENABLED=true). Environment variables set explicitly via
    monkeypatch.setenv still take effect (they outrank the dotenv source
    anyway), so this only removes the file, not real Settings validation.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    # get_settings() is lru_cache'd and app.main calls it at import time, i.e.
    # before this fixture runs -- so the cached instance was still built from
    # the real .env and is what the route dependencies resolve. Drop it so
    # request-time settings honour the isolation above too.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
