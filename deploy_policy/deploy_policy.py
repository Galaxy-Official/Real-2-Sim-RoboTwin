"""
Compatibility wrapper for Replay_Policy.

Keep the package entrypoint stable while delegating the actual implementation
to deploy_policy_v4.py.
"""

try:
    from .deploy_policy_v4 import *  # type: ignore # noqa: F401,F403
except ImportError:
    from deploy_policy_v4 import *  # noqa: F401,F403
