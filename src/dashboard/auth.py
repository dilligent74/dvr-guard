#!/usr/bin/env python3
"""
Auth decorators, login/logout routes, and password helpers.
"""

import time
import bcrypt
from functools import wraps
from flask import session, request, redirect, url_for, render_template, flash

from .helpers import _check_password, _save_dashboard_config


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
            public = dashboard_config.get("public_access", True)
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
        return redirect(request.args.get("next") or url_for("home"))
    
    if request.method == "POST":
        password = request.form.get("password", "")
        viewer_hash = dashboard_config.get("viewer_password", "") or ""
        admin_hash = dashboard_config.get("admin_password", "") or ""

        # First-login setup: if a hash is empty, accept any password and store it
        if not admin_hash and password:
            new_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            dashboard_config["admin_password"] = new_hash
            _save_dashboard_config(config_path, dashboard_config)
            session.permanent = True
            session["role"] = "admin"
            return redirect(request.args.get("next") or url_for("home"))

        if not viewer_hash and password:
            new_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            dashboard_config["viewer_password"] = new_hash
            _save_dashboard_config(config_path, dashboard_config)
            session.permanent = True
            session["role"] = "viewer"
            return redirect(request.args.get("next") or url_for("home"))

        if _check_password(password, viewer_hash):
            session.permanent = True
            session["role"] = "viewer"
            return redirect(request.args.get("next") or url_for("home"))
        if _check_password(password, admin_hash):
            session.permanent = True
            session["role"] = "admin"
            return redirect(request.args.get("next") or url_for("home"))

        time.sleep(1)
        flash("Incorrect password.")

    return None  # Caller should render login template


def handle_logout(dashboard_config):
    """Handle logout. Returns redirect."""
    session.clear()
    if dashboard_config.get("public_access", True):
        return redirect(url_for("home"))
    return redirect(url_for("login"))
