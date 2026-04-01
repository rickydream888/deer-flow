"""Tests for GitHubCopilotChatModel and supporting helpers."""

import json
import time
from unittest import mock

import httpx
import pytest

from deerflow.models.github_copilot_provider import (
    COPILOT_ENV_VARS,
    DEFAULT_COPILOT_BASE_URL,
    DEFAULT_COPILOT_INTEGRATION_ID,
    DEFAULT_EDITOR_VERSION,
    DEFAULT_USER_AGENT,
    GitHubCopilotChatModel,
    _CopilotTokenManager,
    _exchange_github_token,
    _load_github_token_from_env,
    _load_github_token_from_hosts_json,
    _parse_copilot_expires_at,
    _poll_for_access_token,
    _request_device_code,
    _save_github_token_to_hosts_json,
    github_copilot_login,
    load_github_copilot_github_token,
)

# ---------------------------------------------------------------------------
# _parse_copilot_expires_at
# ---------------------------------------------------------------------------


def test_parse_expires_at_valid_iso():
    raw = "2030-01-01T00:00:00Z"
    ts = _parse_copilot_expires_at(raw)
    assert ts > time.time()


def test_parse_expires_at_none_returns_future():
    ts = _parse_copilot_expires_at(None)
    assert ts > time.time()


def test_parse_expires_at_invalid_string_returns_future():
    ts = _parse_copilot_expires_at("not-a-date")
    assert ts > time.time()


def test_parse_expires_at_empty_string_returns_future():
    ts = _parse_copilot_expires_at("")
    assert ts > time.time()


# ---------------------------------------------------------------------------
# _load_github_token_from_env
# ---------------------------------------------------------------------------


def _clear_copilot_env(monkeypatch) -> None:
    for var in COPILOT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_load_github_token_from_copilot_env(monkeypatch):
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_copilot")
    assert _load_github_token_from_env() == "gho_copilot"


def test_load_github_token_from_gh_token(monkeypatch):
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("GH_TOKEN", "gho_gh")
    assert _load_github_token_from_env() == "gho_gh"


def test_load_github_token_from_github_token(monkeypatch):
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "gho_github")
    assert _load_github_token_from_env() == "gho_github"


def test_load_github_token_priority(monkeypatch):
    """COPILOT_GITHUB_TOKEN takes precedence over GH_TOKEN and GITHUB_TOKEN."""
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "first")
    monkeypatch.setenv("GH_TOKEN", "second")
    monkeypatch.setenv("GITHUB_TOKEN", "third")
    assert _load_github_token_from_env() == "first"


def test_load_github_token_returns_empty_when_none(monkeypatch):
    _clear_copilot_env(monkeypatch)
    assert _load_github_token_from_env() == ""


def test_load_github_token_strips_whitespace(monkeypatch):
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "  gho_space  ")
    assert _load_github_token_from_env() == "gho_space"


# ---------------------------------------------------------------------------
# _load_github_token_from_hosts_json
# ---------------------------------------------------------------------------


def test_load_github_token_from_hosts_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hosts_dir = tmp_path / ".config" / "github-copilot"
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "hosts.json").write_text(json.dumps({"github.com": {"oauth_token": "gho_from_file"}}))
    assert _load_github_token_from_hosts_json() == "gho_from_file"


def test_load_github_token_from_hosts_json_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert _load_github_token_from_hosts_json() == ""


def test_load_github_token_from_hosts_json_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hosts_dir = tmp_path / ".config" / "github-copilot"
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "hosts.json").write_text("not-json")
    assert _load_github_token_from_hosts_json() == ""


def test_load_github_token_from_hosts_json_no_token_field(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hosts_dir = tmp_path / ".config" / "github-copilot"
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "hosts.json").write_text(json.dumps({"github.com": {"user": "alice"}}))
    assert _load_github_token_from_hosts_json() == ""


# ---------------------------------------------------------------------------
# load_github_copilot_github_token (combined)
# ---------------------------------------------------------------------------


def test_combined_loader_prefers_env(tmp_path, monkeypatch):
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("GH_TOKEN", "gho_env")
    hosts_dir = tmp_path / ".config" / "github-copilot"
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "hosts.json").write_text(json.dumps({"github.com": {"oauth_token": "gho_file"}}))
    assert load_github_copilot_github_token() == "gho_env"


def test_combined_loader_falls_back_to_file(tmp_path, monkeypatch):
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    hosts_dir = tmp_path / ".config" / "github-copilot"
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "hosts.json").write_text(json.dumps({"github.com": {"oauth_token": "gho_file"}}))
    assert load_github_copilot_github_token() == "gho_file"


def test_combined_loader_returns_empty_when_none(monkeypatch):
    _clear_copilot_env(monkeypatch)
    # Ensure no hosts.json exists by redirecting HOME to /dev/null (or tmp_path)
    with mock.patch(
        "deerflow.models.github_copilot_provider._load_github_token_from_hosts_json",
        return_value="",
    ):
        assert load_github_copilot_github_token() == ""


# ---------------------------------------------------------------------------
# _exchange_github_token
# ---------------------------------------------------------------------------


def _mock_token_response(
    token: str = "cop_token",
    base_url: str = DEFAULT_COPILOT_BASE_URL,
    expires_at: str = "2099-01-01T00:00:00Z",
) -> dict:
    return {
        "token": token,
        "expires_at": expires_at,
        "endpoints": {"api": base_url},
    }


def test_exchange_github_token_success():
    resp_data = _mock_token_response()
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = resp_data
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp
        token, base_url, expires_at = _exchange_github_token("gho_test", DEFAULT_EDITOR_VERSION)

    assert token == "cop_token"
    assert base_url == DEFAULT_COPILOT_BASE_URL
    assert expires_at > time.time()


def test_exchange_github_token_empty_token_raises():
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {"token": ""}
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp
        with pytest.raises(ValueError, match="empty token"):
            _exchange_github_token("gho_test", DEFAULT_EDITOR_VERSION)


def test_exchange_github_token_custom_base_url():
    custom_url = "https://enterprise.githubcopilot.com"
    resp_data = _mock_token_response(base_url=custom_url)
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = resp_data
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp
        _, base_url, _ = _exchange_github_token("gho_test", DEFAULT_EDITOR_VERSION)

    assert base_url == custom_url


def test_exchange_github_token_trailing_slash_stripped():
    """base_url returned by the exchange should have no trailing slash."""
    resp_data = _mock_token_response(base_url="https://api.githubcopilot.com/")
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = resp_data
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp
        _, base_url, _ = _exchange_github_token("gho_test", DEFAULT_EDITOR_VERSION)

    assert not base_url.endswith("/")


# ---------------------------------------------------------------------------
# _CopilotTokenManager
# ---------------------------------------------------------------------------


def _make_manager(
    token: str = "cop_tok",
    base_url: str = DEFAULT_COPILOT_BASE_URL,
    expires_at: str = "2099-01-01T00:00:00Z",
) -> _CopilotTokenManager:
    resp_data = _mock_token_response(token=token, base_url=base_url, expires_at=expires_at)
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = resp_data
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp
        mgr = _CopilotTokenManager("gho_test", DEFAULT_EDITOR_VERSION)
        mgr.refresh()
    return mgr


def test_token_manager_get_token_returns_token():
    mgr = _make_manager(token="my_tok")
    assert mgr.get_token() == "my_tok"


def test_token_manager_base_url():
    mgr = _make_manager(base_url="https://custom.example.com")
    assert mgr.base_url == "https://custom.example.com"


def test_token_manager_does_not_refresh_valid_token():
    mgr = _make_manager(token="fresh")
    with mock.patch.object(mgr, "refresh") as mock_refresh:
        _ = mgr.get_token()
    mock_refresh.assert_not_called()


def test_token_manager_refreshes_expired_token():
    """Manager must call refresh() when the stored token is expired."""
    mgr = _make_manager(expires_at="2000-01-01T00:00:00Z")
    # Manually expire the token
    mgr._expires_at = time.time() - 1000

    new_resp_data = _mock_token_response(token="new_tok")
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = new_resp_data
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.get.return_value = mock_resp
        token = mgr.get_token()

    assert token == "new_tok"


# ---------------------------------------------------------------------------
# GitHubCopilotChatModel constructor
# ---------------------------------------------------------------------------


def _patch_exchange(monkeypatch, token="cop_tok", base_url=DEFAULT_COPILOT_BASE_URL):
    """Patch _exchange_github_token so no real network call is made."""
    expires_at = time.time() + 1800
    monkeypatch.setattr(
        "deerflow.models.github_copilot_provider._exchange_github_token",
        lambda *_a, **_k: (token, base_url, expires_at),
    )


def _make_copilot_model(monkeypatch, github_token: str = "gho_test", **extra_kwargs):
    """Construct a GitHubCopilotChatModel with network calls patched out."""
    _patch_exchange(monkeypatch)
    return GitHubCopilotChatModel(
        model="gpt-4o",
        github_token=github_token,
        **extra_kwargs,
    )


def test_model_construction_sets_correct_base_url(monkeypatch):
    model = _make_copilot_model(monkeypatch)
    assert model.openai_api_base == DEFAULT_COPILOT_BASE_URL


def test_model_construction_sets_copilot_headers(monkeypatch):
    model = _make_copilot_model(monkeypatch)
    headers = model.default_headers or {}
    assert headers.get("Editor-Version") == DEFAULT_EDITOR_VERSION
    assert headers.get("Copilot-Integration-Id") == DEFAULT_COPILOT_INTEGRATION_ID
    assert headers.get("openai-intent") == "conversation-panel"
    assert headers.get("User-Agent") == DEFAULT_USER_AGENT


def test_model_construction_uses_custom_http_client(monkeypatch):
    model = _make_copilot_model(monkeypatch)
    # http_client should be a custom httpx.Client (not None)
    assert model.http_client is not None
    assert isinstance(model.http_client, httpx.Client)


def test_model_construction_uses_custom_async_http_client(monkeypatch):
    model = _make_copilot_model(monkeypatch)
    assert model.http_async_client is not None
    assert isinstance(model.http_async_client, httpx.AsyncClient)


def test_model_raises_without_github_token(monkeypatch, tmp_path):
    """Construction must fail when no GitHub token is available."""
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ValueError, match="GitHub Copilot"):
        GitHubCopilotChatModel(model="gpt-4o")


def test_model_explicit_github_token_overrides_env(monkeypatch):
    """Explicit github_token field takes priority over env vars."""
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("GH_TOKEN", "gho_env")
    _patch_exchange(monkeypatch, token="cop_explicit")
    model = GitHubCopilotChatModel(model="gpt-4o", github_token="gho_explicit")
    # The model was constructed successfully with the explicit token
    assert model.github_token == "gho_explicit"


def test_model_loads_token_from_env(monkeypatch):
    """If github_token field is absent, env var is used."""
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_from_env")
    _patch_exchange(monkeypatch)
    model = GitHubCopilotChatModel(model="gpt-4o")
    assert model.github_token == "gho_from_env"


def test_model_llm_type(monkeypatch):
    model = _make_copilot_model(monkeypatch)
    assert model._llm_type == "github-copilot"


def test_model_custom_editor_version(monkeypatch):
    _patch_exchange(monkeypatch)
    model = GitHubCopilotChatModel(
        model="gpt-4o",
        github_token="gho_test",
        editor_version="neovim/0.10.0",
    )
    assert model.default_headers.get("Editor-Version") == "neovim/0.10.0"


# ---------------------------------------------------------------------------
# _request_device_code
# ---------------------------------------------------------------------------


def test_request_device_code_success():
    response_data = {
        "device_code": "dev123",
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "expires_in": 900,
        "interval": 5,
    }
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp
        result = _request_device_code("read:user")

    assert result["device_code"] == "dev123"
    assert result["user_code"] == "ABCD-1234"
    assert result["verification_uri"] == "https://github.com/login/device"


def test_request_device_code_missing_fields_raises():
    response_data = {"device_code": "dev123"}  # missing user_code, verification_uri
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp
        with pytest.raises(ValueError, match="missing fields"):
            _request_device_code("read:user")


# ---------------------------------------------------------------------------
# _poll_for_access_token
# ---------------------------------------------------------------------------


def test_poll_for_access_token_success():
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {"access_token": "gho_new_token", "token_type": "bearer"}
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp
        token = _poll_for_access_token("dev123", 1.0, time.time() + 900)

    assert token == "gho_new_token"


def test_poll_for_access_token_authorization_pending_then_success():
    """Should retry on authorization_pending and succeed on the next poll."""
    pending_resp = mock.MagicMock(spec=httpx.Response)
    pending_resp.json.return_value = {"error": "authorization_pending"}
    pending_resp.raise_for_status = mock.MagicMock()

    success_resp = mock.MagicMock(spec=httpx.Response)
    success_resp.json.return_value = {"access_token": "gho_ok"}
    success_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.post.side_effect = [pending_resp, success_resp]
        with mock.patch("time.sleep"):
            token = _poll_for_access_token("dev123", 1.0, time.time() + 900)

    assert token == "gho_ok"


def test_poll_for_access_token_slow_down_then_success():
    slow_resp = mock.MagicMock(spec=httpx.Response)
    slow_resp.json.return_value = {"error": "slow_down"}
    slow_resp.raise_for_status = mock.MagicMock()

    success_resp = mock.MagicMock(spec=httpx.Response)
    success_resp.json.return_value = {"access_token": "gho_ok"}
    success_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.post.side_effect = [slow_resp, success_resp]
        with mock.patch("time.sleep"):
            token = _poll_for_access_token("dev123", 1.0, time.time() + 900)

    assert token == "gho_ok"


def test_poll_for_access_token_expired_token_raises():
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {"error": "expired_token"}
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp
        with pytest.raises(ValueError, match="expired"):
            _poll_for_access_token("dev123", 1.0, time.time() + 900)


def test_poll_for_access_token_access_denied_raises():
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {"error": "access_denied"}
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp
        with pytest.raises(ValueError, match="cancelled"):
            _poll_for_access_token("dev123", 1.0, time.time() + 900)


def test_poll_for_access_token_unknown_error_raises():
    mock_resp = mock.MagicMock(spec=httpx.Response)
    mock_resp.json.return_value = {"error": "some_weird_error"}
    mock_resp.raise_for_status = mock.MagicMock()

    with mock.patch("httpx.Client") as mock_client_cls:
        ctx = mock_client_cls.return_value.__enter__.return_value
        ctx.post.return_value = mock_resp
        with pytest.raises(ValueError, match="some_weird_error"):
            _poll_for_access_token("dev123", 1.0, time.time() + 900)


def test_poll_for_access_token_already_expired_raises():
    """Raises immediately when expires_at is already in the past."""
    with mock.patch("httpx.Client"):
        with pytest.raises(ValueError, match="expired"):
            _poll_for_access_token("dev123", 1.0, time.time() - 1)


# ---------------------------------------------------------------------------
# _save_github_token_to_hosts_json
# ---------------------------------------------------------------------------


def test_save_github_token_creates_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _save_github_token_to_hosts_json("gho_saved")
    hosts_path = tmp_path / ".config" / "github-copilot" / "hosts.json"
    assert hosts_path.exists()
    data = json.loads(hosts_path.read_text())
    assert data["github.com"]["oauth_token"] == "gho_saved"


def test_save_github_token_overwrites_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hosts_dir = tmp_path / ".config" / "github-copilot"
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "hosts.json").write_text(json.dumps({"github.com": {"oauth_token": "gho_old"}}))

    _save_github_token_to_hosts_json("gho_new")
    data = json.loads((hosts_dir / "hosts.json").read_text())
    assert data["github.com"]["oauth_token"] == "gho_new"


def test_save_github_token_preserves_other_hosts(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hosts_dir = tmp_path / ".config" / "github-copilot"
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "hosts.json").write_text(json.dumps({"other.example.com": {"oauth_token": "tok"}}))

    _save_github_token_to_hosts_json("gho_new")
    data = json.loads((hosts_dir / "hosts.json").read_text())
    assert data["other.example.com"]["oauth_token"] == "tok"
    assert data["github.com"]["oauth_token"] == "gho_new"


def test_save_github_token_handles_corrupted_existing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    hosts_dir = tmp_path / ".config" / "github-copilot"
    hosts_dir.mkdir(parents=True)
    (hosts_dir / "hosts.json").write_text("not valid json {{")

    # Should not raise; corrupted file is treated as empty.
    _save_github_token_to_hosts_json("gho_recovered")
    data = json.loads((hosts_dir / "hosts.json").read_text())
    assert data["github.com"]["oauth_token"] == "gho_recovered"


# ---------------------------------------------------------------------------
# github_copilot_login (integration of device code flow)
# ---------------------------------------------------------------------------


def test_github_copilot_login_success(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    device_response = {
        "device_code": "dev123",
        "user_code": "ABCD-1234",
        "verification_uri": "https://github.com/login/device",
        "expires_in": 900,
        "interval": 5,
    }

    monkeypatch.setattr(
        "deerflow.models.github_copilot_provider._request_device_code",
        lambda *_a, **_kw: device_response,
    )
    monkeypatch.setattr(
        "deerflow.models.github_copilot_provider._poll_for_access_token",
        lambda *_a, **_kw: "gho_login_token",
    )

    result = github_copilot_login()

    assert result == "gho_login_token"
    hosts_path = tmp_path / ".config" / "github-copilot" / "hosts.json"
    data = json.loads(hosts_path.read_text())
    assert data["github.com"]["oauth_token"] == "gho_login_token"


def test_github_copilot_login_propagates_poll_error(monkeypatch):
    monkeypatch.setattr(
        "deerflow.models.github_copilot_provider._request_device_code",
        lambda *_a, **_kw: {
            "device_code": "dev123",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "expires_in": 900,
            "interval": 5,
        },
    )

    def _raise_cancelled(*_a, **_kw):
        raise ValueError("GitHub login cancelled by user")

    monkeypatch.setattr(
        "deerflow.models.github_copilot_provider._poll_for_access_token",
        _raise_cancelled,
    )

    with pytest.raises(ValueError, match="cancelled"):
        github_copilot_login()
