#!/usr/bin/env python3
"""
Auth decorators, login/logout routes, and password helpers.
"""

import time
import bcrypt
from functools import wraps
from flask import session, request, redirect, url_for, render_template, flash
from urllib.parse import urlparse

from .helpers import _check_password, _save_dashboard_config


def _safe_next(next_url: str) -> str:
    """
    Validate a redirect target from the 'next' parameter.
    Only allow relative paths within this application (no external redirects).
    Returns url_for('home') if the value is missing or unsafe.
    """
    if not next_url:
        return url_for("home")
    parsed = urlparse(next_url)
    # Reject anything with a scheme or netloc (external URL or protocol-relative //)
    if parsed.scheme or parsed.netloc:
        return url_for("home")
    # Must start with / but not // (protocol-relative)
    if not next_url.startswith("/") or next_url.startswith("//"):
        return url_for("home")
    return next_url


def require_viewer(dashboard_config):
    """Decorator factory that returns a decorator requiring viewer+ role.
    
    Usage:
        @app.route('/')
        @require_viewer(dashboard_config)
        def home():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            public = dashboard_config.get("public_access", False)
            if public and request.endpoint in ("home", "alerts", "snapshot_thumb", "snapshot_full"):
                return f(*args, **kwargs)
            if session.get("role") in ("viewer", "admin"):
                return f(*args, **kwargs)
            return redirect(url_for("login", next=request.path))
        return decorated
    return decorator


def require_admin(f):
    """Decorator requiring admin role."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") == "admin":
            return f(*args, **kwargs)
        return redirect(url_for("login", next=request.path))
    return decorated


def handle_login(request, dashboard_config, config_path):
    """Handle login form submission. Returns redirect or None (to render template)."""
    if session.get("role"):
        return redirect(_safe_next(request.args.get("next", "")))

    if request.method == "POST":
        password = request.form.get("password", "")
        viewer_hash = dashboard_config.get("viewer_password", "") or ""
        admin_hash = dashboard_config.get("admin_password", "") or ""

        # First-login setup: if a hash is empty, accept any password and store it
        if not admin_hash and password:
            new_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            dashboard_config["admin_password"] = new_hash
            _save_dashboard_config(config_path, dashboard_config)
            session.clear()
            session.permanent = True
            session["role"] = "admin"
            return redirect(_safe_next(request.args.get("next", "")))

        if not viewer_hash and password:
            new_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            dashboard_config["viewer_password"] = new_hash
            _save_dashboard_config(config_path, dashboard_config)
            session.clear()
            session.permanent = True
            session["role"] = "viewer"
            return redirect(_safe_next(request.args.get("next", "")))

        if _check_password(password, viewer_hash):
            session.clear()
            session.permanent = True
            session["role"] = "viewer"
            return redirect(_safe_next(request.args.get("next", "")))
        if _check_password(password, admin_hash):
            session.clear()
            session.permanent = True
            session["role"] = "admin"
            return redirect(_safe_next(request.args.get("next", "")))

        time.sleep(1)
        flash("Incorrect password.")

    return None  # Caller should render login template


def handle_logout(dashboard_config):
    """Handle logout. Returns redirect."""
    session.clear()
    if dashboard_config.get("public_access", False):
        return redirect(url_for("home"))
    return redirect(url_for("login"))
