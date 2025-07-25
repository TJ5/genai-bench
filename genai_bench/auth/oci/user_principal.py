from typing import Any, Dict, Optional

import oci

from genai_bench.auth.auth_provider import AuthProvider


class OCIUserPrincipalAuth(AuthProvider):
    """OCI Authentication Provider using user principal."""

    def __init__(
        self,
        config_path: Optional[str] = oci.config.DEFAULT_LOCATION,
        profile: Optional[str] = oci.config.DEFAULT_PROFILE,
    ):
        """Initialize OCI User Principal Auth Provider.

        Args:
            config_path (Optional[str]): Path to OCI config file,
                and defaults to '~/.oci/config'.
            profile (Optional[str]): Profile to use from config file.
                Defaults to 'DEFAULT' if not specified.
        """
        self.config_path = config_path
        self.profile = profile
        self._config = None
        self._signer = None

    def get_config(self) -> Dict[str, Any]:
        """Get OCI configuration.

        Returns:
            Dict[str, Any]: OCI configuration dictionary

        Raises:
            oci.exceptions.ConfigFileError: If config file cannot be read
            oci.exceptions.InvalidConfig: If config is invalid
        """
        if self._config is None:
            # Use the specified profile when loading config
            config = oci.config.from_file(self.config_path, self.profile)
            # Validate config before caching
            oci.config.validate_config(config)
            self._config = config
        return self._config or {}

    def get_credentials(self) -> oci.signer.AbstractBaseSigner:
        """Get OCI signer for authentication.

        Returns:
            oci.signer.AbstractBaseSigner: OCI signer

        Raises:
            oci.exceptions.ConfigFileError: If config file cannot be read
            oci.exceptions.InvalidConfig: If config is invalid
        """
        if self._signer is None:
            config = self.get_config()
            self._signer = oci.signer.Signer.from_config(config)
        return self._signer
