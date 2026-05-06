#!/usr/bin/env python3
"""
DVR Guard Dashboard — Mobile-first Flask interface.

App factory and route definitions.
"""

import os
import sys
import cv2
import time
import yaml
import threading
from datetime import datetime, timedelta
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, jsonify, make_response, flash,
)

from .helpers import (
    _fmt_age, _fmt_uptime, _resolve_snapshot_urls,
    _safe_filename, _check_password,
    _ensure_config_hashed, _save_dashboard_config,
)
from .snapshots import _parse_snapshot_filename, _get_folder_files, _walk_all_snapshots
from .stats import _cluster_sequences, _classify_sequences, _compute_stats, _format_stats_for_template
from .auth import require_viewer, require_admin, handle_login, handle_logout

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")


def _rel_time(dt):
    """Format datetime as relative time (e.g. '5 min ago')."""
    if dt is None:
        return "never"
    delta = datetime.now() - dt
    secs = delta.total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs / 60)} min ago"
    if secs < 86400:
        return f"{int(secs / 3600)}h ago"
    return dt.strftime("%d %b %H:%M")


_app_start_time = datetime.now()


def create_app(shared_state, dashboard_config):
    """Create Flask app instance."""
    global _app_start_time
    _app_start_time = datetime.now()

    _ensure_config_hashed(_CONFIG_PATH, dashboard_config)

    app = Flask(
        __name__,
        template_folder=os.path.join(_PROJECT_ROOT, "templates"),
        static_folder=os.path.join(_PROJECT_ROOT, "static"),
    )
    app.secret_key = dashboard_config["secret_key"]
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

    snapshots_dir = os.path.join(
        _PROJECT_ROOT, dashboard_config.get("snapshots_dir", "snapshots")
    )
    thumb_cache_dir = os.path.join(snapshots_dir, ".thumbs")
    os.makedirs(thumb_cache_dir, exist_ok=True)

    # Stats cache -----------------------------------------------------------
    _stats_cache = {"result": None, "computed_at": None, "lock": threading.Lock()}
    _stats_computing = {"active": False, "lock": threading.Lock()}
    _STATS_CACHE_TTL = 300

    @app.context_processor
    def inject_globals():
        return {
            "public_access": dashboard_config.get("public_access", True),
        }

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.route("/login", methods=["GET", "POST"])
    def login():
        result = handle_login(request, dashboard_config, _CONFIG_PATH)
        if result is not None:
            return result
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        return handle_logout(dashboard_config)

    @app.route("/")
    @require_viewer(dashboard_config)
    def home():
        now = datetime.now()

        # Load camera configs
        try:
            with open(_CONFIG_PATH, "r") as f:
                full_config = yaml.safe_load(f)
            cam_configs = full_config.get("cameras", [])
        except Exception:
            cam_configs = []

        # Read shared state
        stream_status = shared_state.get_stream_status()
        yolo_last = shared_state.get_yolo_last_inference()

        # Build camera list with LIVE/OFFLINE status
        cameras = []
        all_live = True
        error_reasons = []

        for cam in cam_configs:
            name = cam.get("name", f"Camera {cam.get('id', '?')}")
            last_ts = stream_status.get(name)
            if last_ts is None:
                live = False
                age_sec = None
            else:
                age_sec = (now - last_ts).total_seconds()
                live = age_sec <= 10
            if not live:
                all_live = False
                if age_sec is not None:
                    error_reasons.append(f"{name} stream offline · last frame {int(age_sec)}s ago")
                else:
                    error_reasons.append(f"{name} stream offline · no frames yet")
            cameras.append({"name": name, "live": live})

        # Inference thread status
        inference_stalled = False
        if yolo_last is None:
            inference_stalled = True
        else:
            yolo_age = (now - yolo_last).total_seconds()
            if yolo_age > 30:
                inference_stalled = True

        if inference_stalled:
            all_live = False
            if yolo_last is None:
                error_reasons.append("Inference stalled · never started")
            else:
                yolo_age = int((now - yolo_last).total_seconds())
                error_reasons.append(f"Inference stalled · last inference {yolo_age}s ago")

        system_status = "RUNNING" if all_live else "ERROR"
        system_reason = " · ".join(error_reasons) if error_reasons else None

        # Last alert
        last_alert = None
        detections = shared_state.get_recent_detections(limit=1)
        if detections:
            d = detections[-1]
            thumb_url, full_url = _resolve_snapshot_urls(d.snapshot_path)
            last_alert = {
                "camera_name": d.camera_name,
                "confidence": d.confidence,
                "rel_time": _rel_time(d.timestamp),
                "thumb_url": thumb_url,
                "full_url": full_url,
            }

        response = make_response(
            render_template(
                "home.html",
                system_status=system_status,
                system_reason=system_reason,
                cameras=cameras,
                last_alert=last_alert,
            )
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/alerts")
    @require_viewer(dashboard_config)
    def alerts():
        recent_count = dashboard_config.get("recent_alerts_count", 10)
        files = _get_folder_files(snapshots_dir, "confirmed_45plus")[:recent_count]

        alert_list = []
        for f in files:
            rec = _parse_snapshot_filename(f, "confirmed_45plus")
            if rec is None:
                continue
            alert_list.append(
                {
                    "filename": f,
                    "camera_name": rec["camera_name"],
                    "timestamp_str": rec["timestamp"].strftime("%a %d %b, %H:%M:%S"),
                    "confidence": rec["confidence"],
                }
            )

        response = make_response(render_template("alerts.html", alerts=alert_list))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/snapshots")
    @require_admin
    def snapshots():
        tiers = ["confirmed_45plus", "uncertain_35to45", "weak_25to35", "motion_no_detection"]
        tier = request.args.get("tier", "confirmed_45plus")
        if tier not in tiers:
            tier = "confirmed_45plus"
        camera = request.args.get("camera", "")
        date = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        page = int(request.args.get("page", 1))
        if page < 1:
            page = 1

        files = _get_folder_files(snapshots_dir, tier)
        parsed = [_parse_snapshot_filename(f, tier) for f in files]
        parsed = [p for p in parsed if p]

        if camera:
            parsed = [p for p in parsed if p["camera_name"] == camera]
        if date:
            date_str = date.replace("-", "")
            parsed = [p for p in parsed if date_str in p["filename"]]

        per_page = 24
        total = len(parsed)
        start = (page - 1) * per_page
        end = start + per_page
        page_items = parsed[start:end]
        has_next = total > end

        all_cameras = set()
        for t in tiers:
            for f in _get_folder_files(snapshots_dir, t):
                rec = _parse_snapshot_filename(f, t)
                if rec:
                    all_cameras.add(rec["camera_name"])

        snap_list = []
        for p in page_items:
            snap_list.append(
                {
                    "filename": p["filename"],
                    "time_str": p["timestamp"].strftime("%H:%M:%S"),
                    "confidence": p["confidence"],
                }
            )

        response = make_response(
            render_template(
                "snapshots.html",
                tiers=tiers,
                tier=tier,
                camera=camera,
                date=date,
                page=page,
                has_next=has_next,
                cameras=sorted(all_cameras),
                snapshots=snap_list,
            )
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/stats")
    @require_admin
    def stats():
        now = datetime.now()

        # System health data
        stream_status = shared_state.get_stream_status()
        yolo_last = shared_state.get_yolo_last_inference()
        pipeline_start = shared_state.get_pipeline_start_time()

        # Uptime
        uptime = _fmt_uptime(pipeline_start) if pipeline_start else "unknown"

        # Inference thread
        if yolo_last:
            yolo_age = (now - yolo_last).total_seconds()
            if yolo_age > 30:
                inference_status = "STALLED"
            else:
                inference_status = "RUNNING"
            inference_age = _fmt_age(yolo_last)
        else:
            inference_status = "STALLED"
            inference_age = "never"

        # Queue depths (placeholders - no queues currently wired)
        inf_queue = getattr(shared_state, "inference_queue", None)
        inf_queue_depth = inf_queue.qsize() if inf_queue else 0
        notif_queue = getattr(shared_state, "notifier_queue", None)
        notif_queue_depth = notif_queue.qsize() if notif_queue else 0

        # Camera streams health
        try:
            with open(_CONFIG_PATH, "r") as f:
                full_config = yaml.safe_load(f)
            cam_configs = full_config.get("cameras", [])
        except Exception:
            cam_configs = []

        cameras_health = []
        for cam in cam_configs:
            name = cam.get("name", f"Camera {cam.get('id', '?')}")
            last_ts = stream_status.get(name)
            if last_ts:
                age = (now - last_ts).total_seconds()
                live = age <= 10
                age_str = f"last frame {int(age)}s ago" if age >= 1 else "last frame just now"
            else:
                live = False
                age_str = "no frames yet"
            cameras_health.append({
                "name": name,
                "live": live,
                "age_str": age_str,
            })

        return render_template(
            "stats.html",
            uptime=uptime,
            inference_status=inference_status,
            inference_age=inference_age,
            inference_queue=inf_queue_depth,
            notifier_queue=notif_queue_depth,
            cameras=cameras_health,
        )

    @app.route("/stats/results")
    @require_admin
    def stats_results():
        force = request.args.get("force") == "true"
        gap_threshold = float(request.args.get("gap_threshold", 3.0))

        with _stats_cache["lock"]:
            cached = _stats_cache["result"]
            cached_at = _stats_cache["computed_at"]
            is_fresh = (
                cached is not None
                and cached_at is not None
                and (datetime.now() - cached_at).total_seconds() < _STATS_CACHE_TTL
            )

        if not force and is_fresh:
            age = str(datetime.now() - cached_at).split(".")[0]
            html = render_template("stats_results.html", stats=cached)
            return jsonify({"status": "done", "html": html, "age": age})

        with _stats_computing["lock"]:
            if _stats_computing["active"]:
                return jsonify({"status": "computing"})
            _stats_computing["active"] = True

        if force:
            with _stats_cache["lock"]:
                _stats_cache["result"] = None
                _stats_cache["computed_at"] = None

        def compute():
            try:
                records = _walk_all_snapshots(snapshots_dir)
                if len(records) < 10:
                    result = {
                        "per_camera": [],
                        "global_stats": {
                            "dropout_rate": 0.0,
                            "dropout_explanation": "Not enough data for meaningful analysis (<10 frames)",
                            "detected_not_alerted": 0,
                            "avg_peak_confidence": 0.0,
                            "recommendation": "Not enough data for analysis",
                        },
                    }
                else:
                    sequences = _cluster_sequences(records, gap_threshold)
                    enriched = _classify_sequences(sequences)
                    raw = _compute_stats(enriched)
                    result = _format_stats_for_template(raw, gap_threshold)
                with _stats_cache["lock"]:
                    _stats_cache["result"] = result
                    _stats_cache["computed_at"] = datetime.now()
            except Exception as e:
                with _stats_cache["lock"]:
                    _stats_cache["result"] = {
                        "per_camera": [],
                        "global_stats": {
                            "dropout_rate": 0.0,
                            "dropout_explanation": f"Error during computation: {e}",
                            "detected_not_alerted": 0,
                            "avg_peak_confidence": 0.0,
                            "recommendation": "Stats computation failed",
                        },
                    }
                    _stats_cache["computed_at"] = datetime.now()
            finally:
                with _stats_computing["lock"]:
                    _stats_computing["active"] = False

        t = threading.Thread(target=compute, name="StatsCompute", daemon=True)
        t.start()
        return jsonify({"status": "computing"})

    @app.route("/settings")
    @require_admin
    def settings():
        try:
            with open(_CONFIG_PATH, "r") as f:
                full_config = yaml.safe_load(f)
        except Exception:
            full_config = {}

        def mask(value):
            if isinstance(value, dict):
                new = {}
                for k, v in value.items():
                    if k in ("token", "chat_id", "admin_password", "viewer_password"):
                        new[k] = "••••••••"
                    elif k == "rtsp" and isinstance(v, str) and len(v) > 8:
                        new[k] = "••••••" + v[-8:]
                    else:
                        new[k] = mask(v)
                return new
            elif isinstance(value, list):
                return [mask(item) for item in value]
            return value

        masked = mask(full_config)
        response = make_response(render_template("settings.html", config=masked))
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/snapshot/thumb/<tier>/<filename>")
    @require_viewer(dashboard_config)
    def snapshot_thumb(tier, filename):
        if not _safe_filename(tier, filename):
            return "Not found", 404
        src_path = os.path.join(snapshots_dir, tier, filename)
        thumb_dir = os.path.join(thumb_cache_dir, tier)
        os.makedirs(thumb_dir, exist_ok=True)
        thumb_path = os.path.join(thumb_dir, filename)

        if not os.path.exists(thumb_path):
            if not os.path.exists(src_path):
                return "Not found", 404
            img = cv2.imread(src_path)
            if img is None:
                return "Invalid image", 400
            thumb = cv2.resize(img, (240, 135))
            cv2.imwrite(thumb_path, thumb)

        response = send_file(thumb_path, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @app.route("/snapshot/full/<tier>/<filename>")
    @require_viewer(dashboard_config)
    def snapshot_full(tier, filename):
        if not _safe_filename(tier, filename):
            return "Not found", 404
        path = os.path.join(snapshots_dir, tier, filename)
        if not os.path.exists(path):
            return "Not found", 404
        response = send_file(path, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    return app


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with open(_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    dashboard_config = config.get("dashboard", {})

    from ..state import SharedState

    state = SharedState()
    app = create_app(state, dashboard_config)
    port = dashboard_config.get("port", 8080)
    debug = "--debug" in sys.argv
    app.run(host="0.0.0.0", port=port, debug=debug)
