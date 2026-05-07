#!/usr/bin/env python3
"""
Shared utilities for the dashboard package.
"""

import os
import yaml
import bcrypt
from datetime import datetime
from flask import url_for

# ---------------------------------------------------------------------------
# Helpers for system health
# ---------------------------------------------------------------------------


def _fmt_age(dt):
    """Format a datetime as a precise relative age string (e.g. '2s ago', '3m ago')."""
    if dt is None:
        return "never"
    secs = int((datetime.now() - dt).total_seconds())
    if secs < 1:
        return "just now"
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        h = secs // 3600
        m = (secs % 3600) // 60
        return f"{h}h {m}m ago" if m else f"{h}h ago"
    d = secs // 86400
    h = (secs % 86400) // 3600
    return f"{d}d {h}h ago" if h else f"{d}d ago"


def _fmt_uptime(dt):
    """Format pipeline start time as '3d 14h 22m'."""
    if dt is None:
        return "unknown"
    total = int((datetime.now() - dt).total_seconds())
    d = total // 86400
    h = (total % 86400) // 3600
    m = (total % 3600) // 60
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    return " ".join(parts) if parts else "<1m"


def _resolve_snapshot_urls(snapshot_path):
    """Convert a filesystem snapshot path to thumbnail/full URLs."""
    if not snapshot_path:
        return None, None
    p = snapshot_path.replace("\\", "/")
    parts = p.split("/")
    if len(parts) < 2:
        return None, None
    tier = parts[-2]
    filename = parts[-1]
    valid = {"confirmed_45plus", "uncertain_35to45", "weak_25to35", "motion_no_detection"}
    if tier not in valid or not filename.endswith(".jpg"):
        return None, None
    try:
        return (
            url_for("snapshot_thumb", tier=tier, filename=filename),
            url_for("snapshot_full", tier=tier, filename=filename),
        )
    except Exception:
        return None, None


_VALID_TIERS = {"confirmed_45plus", "uncertain_35to45", "weak_25to35", "motion_no_detection"}


def _safe_filename(tier, filename):
    """Validate tier and filename for path traversal attacks."""
    if tier not in _VALID_TIERS:
        return False
    if not filename.endswith(".jpg"):
        return False
    if ".." in filename:
        return False
    if filename.startswith("."):
        return False
    if "/" in filename or "\\" in filename or "\x00" in filename:
        return False
    return True


def _check_password(password, hash_str):
    """Check a plaintext password against a bcrypt hash."""
    if not hash_str:
        return False
    try:
        return bcrypt.checkpw(password.encode(), hash_str.encode())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _ensure_config_hashed(config_path, dashboard_config):
    """Hash plaintext passwords and generate secret_key if missing."""
    changed = False
    for key in ("viewer_password", "admin_password"):
        pw = dashboard_config.get(key)
        if isinstance(pw, str) and pw and not pw.startswith("$2b$"):
            dashboard_config[key] = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
            changed = True
    if not dashboard_config.get("secret_key"):
        import secrets

        dashboard_config["secret_key"] = secrets.token_hex(32)
        changed = True
    if changed and os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                full_config = yaml.safe_load(f)
            if "dashboard" not in full_config:
                full_config["dashboard"] = {}
            for k, v in dashboard_config.items():
                full_config["dashboard"][k] = v
            with open(config_path, "w") as f:
                yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
            os.chmod(config_path, 0o600)
        except Exception:
            pass


def _save_dashboard_config(config_path, dashboard_config):
    """Persist the dashboard section back to config.yaml."""
    if not os.path.isfile(config_path):
        return
    try:
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
        if not isinstance(full_config, dict):
            full_config = {}
        full_config["dashboard"] = dashboard_config
        with open(config_path, "w") as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)
        os.chmod(config_path, 0o600)
    except Exception:
        pass


def _pct(part, total):
    """Calculate percentage. Returns 0.0 if total is 0."""
    return (part / total * 100) if total > 0 else 0.0
