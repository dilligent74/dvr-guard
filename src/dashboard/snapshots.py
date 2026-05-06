#!/usr/bin/env python3
"""
Snapshot parsing, folder utils, and thumbnail serving helpers.
"""

import os
import time
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# Folder listing cache
# ---------------------------------------------------------------------------
_folder_cache = {}
_FOLDER_CACHE_TTL = 30.0


def _get_folder_files(snapshots_dir, tier):
    """Return cached, sorted file list for a tier."""
    now = time.monotonic()
    key = (snapshots_dir, tier)
    if key in _folder_cache:
        ts, files = _folder_cache[key]
        if now - ts < _FOLDER_CACHE_TTL:
            return files
    tier_dir = os.path.join(snapshots_dir, tier)
    if not os.path.isdir(tier_dir):
        files = []
    else:
        files = sorted(os.listdir(tier_dir), reverse=True)
    _folder_cache[key] = (now, files)
    return files


# ---------------------------------------------------------------------------
# Snapshot filename parsing (matches analyse_snapshots.py)
# ---------------------------------------------------------------------------


def _parse_snapshot_filename(filename, tier=None):
    """Parse a snapshot filename. Returns dict or None.

    Expected format: YYYYMMDD-HH_MM_SS-{camera}-c{NNN}.jpg
    Example: 20250506-14_30_25-Față-ST-c087.jpg
    """
    if not filename.endswith(".jpg"):
        return None
    name = filename[:-4]

    # Modern format: YYYYMMDD-HH_MM_SS-{camera}-c{NNN}.jpg
    if len(name) >= 8 and name[:8].isdigit():
        parts = name.split("-")
        if len(parts) >= 3:
            date_str = parts[0]
            time_str = parts[1]
            if len(date_str) == 8 and date_str.isdigit():
                if len(time_str) == 8 and time_str[2] == "_" and time_str[5] == "_":
                    hh = time_str[0:2]
                    mm = time_str[3:5]
                    ss = time_str[6:8]
                    if hh.isdigit() and mm.isdigit() and ss.isdigit():
                        remaining = parts[2:]
                        conf = None
                        if remaining and remaining[-1].startswith("c"):
                            conf_str = remaining[-1][1:]
                            if conf_str.isdigit():
                                conf = int(conf_str) / 100.0
                                camera_name = "-".join(remaining[:-1])
                            else:
                                camera_name = "-".join(remaining)
                        else:
                            camera_name = "-".join(remaining)
                        try:
                            timestamp = datetime(
                                int(date_str[:4]),
                                int(date_str[4:6]),
                                int(date_str[6:8]),
                                int(hh),
                                int(mm),
                                int(ss),
                            )
                        except ValueError:
                            return None
                        return {
                            "camera_name": camera_name,
                            "timestamp": timestamp,
                            "confidence": conf,
                            "filename": filename,
                            "tier": tier,
                        }

    return None


def _walk_all_snapshots(snapshots_dir):
    """Walk all tier folders and return parsed records."""
    tiers = ["confirmed_45plus", "uncertain_35to45", "weak_25to35", "motion_no_detection"]
    records = []
    for tier in tiers:
        for filename in _get_folder_files(snapshots_dir, tier):
            if not filename.endswith(".jpg"):
                continue
            rec = _parse_snapshot_filename(filename, tier)
            if rec:
                records.append(rec)
    return records
