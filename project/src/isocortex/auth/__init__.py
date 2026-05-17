"""
IsoCortex — Authentication & User Management
==============================================

Public API:
  - get_user_manager()  → Global UserManager singleton
  - UserManager         → User CRUD, auth, API key management
  - create_access_token() / decode_access_token() → JWT helpers
  - hash_password() / verify_password() → bcrypt helpers
  - generate_api_key() / hash_api_key() → API key helpers
"""

from isocortex.auth.auth import (
    UserManager,
    create_access_token,
    decode_access_token,
    generate_api_key,
    get_user_manager,
    hash_api_key,
    hash_password,
    verify_password,
)

__all__ = [
    "UserManager",
    "create_access_token",
    "decode_access_token",
    "generate_api_key",
    "get_user_manager",
    "hash_api_key",
    "hash_password",
    "verify_password",
]
