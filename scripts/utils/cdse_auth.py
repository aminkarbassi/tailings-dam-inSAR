"""
CDSE Authentication Utilities
==============================
Handles OAuth2 token acquisition and refresh for the Copernicus Data Space
Ecosystem (CDSE) API.

Credentials are read from environment variables:
    CDSE_USER      — your CDSE account email
    CDSE_PASSWORD  — your CDSE account password

Register a free account at: https://dataspace.copernicus.eu/
"""

import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


class CDSESession:
    """Authenticated requests session with automatic token refresh."""

    def __init__(self, user: str = None, password: str = None):
        self.user = user or os.environ.get("CDSE_USER")
        self.password = password or os.environ.get("CDSE_PASSWORD")

        if not self.user or not self.password:
            raise EnvironmentError(
                "CDSE credentials not found.\n"
                "Set environment variables CDSE_USER and CDSE_PASSWORD, or pass them explicitly.\n"
                "Register at: https://dataspace.copernicus.eu/"
            )

        self._token = None
        self._token_expiry = 0
        self.session = requests.Session()
        self._refresh_token()

    def _refresh_token(self):
        """Acquire a new OAuth2 access token."""
        response = requests.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "password",
                "username": self.user,
                "password": self.password,
                "client_id": "cdse-public",
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        self._token = data["access_token"]
        # Expire 60 seconds early to avoid edge cases
        self._token_expiry = time.time() + data.get("expires_in", 600) - 60
        self.session.headers.update({"Authorization": f"Bearer {self._token}"})
        logger.debug("CDSE token refreshed, expires in %d s", data.get("expires_in", 600))

    def get(self, url: str, **kwargs) -> requests.Response:
        """GET with automatic token refresh."""
        if time.time() >= self._token_expiry:
            self._refresh_token()
        return self.session.get(url, **kwargs)

    def get_token(self) -> str:
        """Return a valid access token, refreshing if necessary."""
        if time.time() >= self._token_expiry:
            self._refresh_token()
        return self._token


def get_session(user: str = None, password: str = None) -> CDSESession:
    """Convenience factory — returns an authenticated CDSESession."""
    return CDSESession(user=user, password=password)
