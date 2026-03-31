"""GitHub Copilot provider for DeerFlow.

Implements a LangChain-compatible chat model that:
1. Loads a GitHub token from environment variables or credential files
2. Exchanges the GitHub token for a short-lived Copilot API token
3. Calls the GitHub Copilot OpenAI-compatible chat completions API

GitHub token lookup order:
  - COPILOT_GITHUB_TOKEN env var
  - GH_TOKEN env var
  - GITHUB_TOKEN env var
  - ~/.config/github-copilot/hosts.json (GitHub CLI or VS Code credential store)

Config example:
    - name: github-copilot-gpt-4o
      use: deerflow.models.github_copilot_provider:GitHubCopilotChatModel
      model: gpt-4o
      max_tokens: 4096
      supports_vision: true
"""

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from pydantic import model_validator

logger = logging.getLogger(__name__)

COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
DEFAULT_COPILOT_BASE_URL = "https://api.githubcopilot.com"
DEFAULT_EDITOR_VERSION = "vscode/1.96.2"
DEFAULT_USER_AGENT = "GitHubCopilotChat/0.26.7"
DEFAULT_COPILOT_INTEGRATION_ID = "vscode-chat"

COPILOT_ENV_VARS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")

# Refresh the Copilot token this many seconds before it actually expires.
_TOKEN_EXPIRY_BUFFER_SECS = 60


# ---------------------------------------------------------------------------
# Token exchange helpers
# ---------------------------------------------------------------------------


def _parse_copilot_expires_at(raw: str | None) -> float:
    """Convert an ISO 8601 expiry string to a Unix timestamp.

    Returns ``time.time() + 1800`` (30 minutes) if the string is missing or
    cannot be parsed, matching the typical Copilot token lifetime.
    """
    if not raw:
        return time.time() + 1800
    try:
        # Strip trailing Z / UTC offset so fromisoformat works on Python 3.10
        clean = raw.rstrip("Z").split("+")[0]
        dt = datetime.fromisoformat(clean).replace(tzinfo=UTC)
        return dt.timestamp()
    except (ValueError, TypeError):
        return time.time() + 1800


def _exchange_github_token(github_token: str, editor_version: str) -> tuple[str, str, float]:
    """Exchange a GitHub PAT/OAuth token for a short-lived Copilot API token.

    Returns ``(copilot_token, base_url, expires_at_unix_timestamp)``.
    Raises :class:`ValueError` if the exchange fails or the response is empty.
    """
    headers = {
        "Authorization": f"token {github_token}",
        "Editor-Version": editor_version,
        "Editor-Plugin-Version": "copilot/1.156.0",
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/json",
        "X-Github-Api-Version": "2025-04-01",
    }
    with httpx.Client(timeout=30) as client:
        resp = client.get(COPILOT_TOKEN_URL, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    token = data.get("token", "")
    if not token:
        raise ValueError(f"GitHub Copilot token exchange returned an empty token. Response keys: {list(data.keys())}")

    # The API returns the preferred endpoint for the caller's region.
    base_url = data.get("endpoints", {}).get("api", DEFAULT_COPILOT_BASE_URL).rstrip("/")
    expires_at = _parse_copilot_expires_at(data.get("expires_at"))

    logger.info("Copilot API token acquired (base_url=%s)", base_url)
    logger.debug("Copilot API token expires at %s", datetime.fromtimestamp(expires_at, tz=UTC).isoformat())
    return token, base_url, expires_at


# ---------------------------------------------------------------------------
# GitHub token loader
# ---------------------------------------------------------------------------


def _load_github_token_from_env() -> str:
    """Return the first non-empty GitHub token found in the environment."""
    for var in COPILOT_ENV_VARS:
        token = os.getenv(var, "").strip()
        if token:
            logger.debug("Loaded GitHub token from %s", var)
            return token
    return ""


def _load_github_token_from_hosts_json() -> str:
    """Return a GitHub token from the GitHub CLI / VS Code Copilot credential store.

    GitHub Copilot (VS Code extension) and the GitHub CLI both write OAuth
    tokens to ``~/.config/github-copilot/hosts.json`` in the format::

        {"github.com": {"oauth_token": "gho_..."}}
    """
    home = os.getenv("HOME", str(Path.home()))
    candidate_paths = [
        Path(home) / ".config" / "github-copilot" / "hosts.json",
        # Windows path – harmless on Linux/macOS since it won't exist
        Path(home) / "AppData" / "Local" / "github-copilot" / "hosts.json",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            token = data.get("github.com", {}).get("oauth_token", "").strip()
            if token:
                logger.info("Loaded GitHub token from %s", path)
                return token
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Failed to read %s: %s", path, exc)
    return ""


def load_github_copilot_github_token() -> str:
    """Load a GitHub token suitable for Copilot, trying env vars then files."""
    return _load_github_token_from_env() or _load_github_token_from_hosts_json()


# ---------------------------------------------------------------------------
# Thread-safe token manager
# ---------------------------------------------------------------------------


class _CopilotTokenManager:
    """Manages the lifecycle of a short-lived GitHub Copilot API token.

    Stores one Copilot token per ``(github_token, editor_version)`` pair and
    automatically exchanges it for a fresh one when it is about to expire.
    Thread-safe via a reentrant lock.
    """

    def __init__(self, github_token: str, editor_version: str) -> None:
        self._github_token = github_token
        self._editor_version = editor_version
        self._copilot_token = ""
        self._base_url = DEFAULT_COPILOT_BASE_URL
        self._expires_at: float = 0.0
        self._lock = threading.RLock()

    def _is_valid(self) -> bool:
        return bool(self._copilot_token) and time.time() < self._expires_at - _TOKEN_EXPIRY_BUFFER_SECS

    def refresh(self) -> None:
        """Force a token exchange regardless of current validity."""
        with self._lock:
            logger.info("Exchanging GitHub token for Copilot API token …")
            token, base_url, expires_at = _exchange_github_token(self._github_token, self._editor_version)
            self._copilot_token = token
            self._base_url = base_url
            self._expires_at = expires_at

    def get_token(self) -> str:
        """Return a valid Copilot token, refreshing if necessary."""
        if not self._is_valid():
            self.refresh()
        return self._copilot_token

    @property
    def base_url(self) -> str:
        """The Copilot API base URL (populated after the first token exchange)."""
        if not self._copilot_token:
            self.refresh()
        return self._base_url


# ---------------------------------------------------------------------------
# Custom httpx transports for automatic token injection
# ---------------------------------------------------------------------------


class _CopilotSyncTransport(httpx.BaseTransport):
    """Sync httpx transport that injects a fresh Copilot Bearer token on every request.

    The token is refreshed transparently whenever it is about to expire so
    that long-running sessions remain authenticated without recreating the
    LangChain/OpenAI client.
    """

    def __init__(self, manager: _CopilotTokenManager) -> None:
        self._manager = manager
        self._inner = httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        # Replace whatever Authorization header the openai SDK set with a fresh token.
        request.headers["authorization"] = f"Bearer {self._manager.get_token()}"
        return self._inner.handle_request(request)

    def close(self) -> None:
        self._inner.close()


class _CopilotAsyncTransport(httpx.AsyncBaseTransport):
    """Async httpx transport that injects a fresh Copilot Bearer token on every request.

    Token exchange is synchronous; since it is a fast network call and only
    needed when the token is near expiry (~every 30 minutes), the brief event-
    loop pause is acceptable.
    """

    def __init__(self, manager: _CopilotTokenManager) -> None:
        self._manager = manager
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        request.headers["authorization"] = f"Bearer {self._manager.get_token()}"
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------


class GitHubCopilotChatModel(ChatOpenAI):
    """LangChain chat model backed by GitHub Copilot's OpenAI-compatible API.

    Handles the full token lifecycle automatically:
    - Loads a GitHub token from env vars or credential files
    - Exchanges it for a short-lived Copilot API token on first use
    - Refreshes the Copilot token transparently without recreating the client
    - Injects all headers required by the Copilot API

    Config example::

        - name: github-copilot-gpt-4o
          use: deerflow.models.github_copilot_provider:GitHubCopilotChatModel
          model: gpt-4o
          max_tokens: 4096
          supports_vision: true

        - name: github-copilot-claude-sonnet
          use: deerflow.models.github_copilot_provider:GitHubCopilotChatModel
          model: claude-sonnet-4.5
          max_tokens: 8192
          supports_vision: true
    """

    # Custom fields recognised by this class.
    # ``github_token`` is optional — the model falls back to env vars and the
    # credential file if the field is empty or not provided in config.
    github_token: str = ""
    editor_version: str = DEFAULT_EDITOR_VERSION

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "github-copilot"

    # ------------------------------------------------------------------
    # Pydantic ``mode="before"`` validator
    # Runs *before* ChatOpenAI's own validators so the OpenAI client is
    # created with the correct Copilot token and base URL.
    # ------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def _setup_copilot_auth(cls, data: Any) -> Any:
        """Exchange a GitHub token for a Copilot API token and configure ChatOpenAI."""
        if not isinstance(data, dict):
            return data

        # ---- resolve GitHub token ----
        github_token = str(data.get("github_token") or "").strip()
        if not github_token:
            github_token = load_github_copilot_github_token()
        if not github_token:
            raise ValueError("GitHub Copilot: no GitHub token found. Set the COPILOT_GITHUB_TOKEN, GH_TOKEN, or GITHUB_TOKEN environment variable, or authenticate with `gh auth login`.")
        data["github_token"] = github_token

        editor_version = str(data.get("editor_version") or DEFAULT_EDITOR_VERSION)
        data["editor_version"] = editor_version

        # ---- token manager + initial exchange ----
        manager = _CopilotTokenManager(github_token, editor_version)
        manager.refresh()  # Eagerly exchange so base_url is available immediately.

        # ---- configure ChatOpenAI fields ----
        # openai_api_key is set to the current Copilot token; the transport will
        # override the Authorization header with a fresh token on every request.
        data["openai_api_key"] = manager.get_token()
        data["openai_api_base"] = manager.base_url

        copilot_headers: dict[str, str] = {
            "Editor-Version": editor_version,
            "Copilot-Integration-Id": DEFAULT_COPILOT_INTEGRATION_ID,
            "openai-intent": "conversation-panel",
            "User-Agent": DEFAULT_USER_AGENT,
        }
        existing_headers: dict = data.get("default_headers") or {}
        data["default_headers"] = {**copilot_headers, **existing_headers}

        # ---- custom httpx clients with auto-refreshing transport ----
        timeout = float(data.get("request_timeout", 600.0))
        data["http_client"] = httpx.Client(
            transport=_CopilotSyncTransport(manager),
            timeout=timeout,
        )
        data["http_async_client"] = httpx.AsyncClient(
            transport=_CopilotAsyncTransport(manager),
            timeout=timeout,
        )

        return data
