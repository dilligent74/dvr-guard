#!/usr/bin/env python3
"""
Stats computation functions (adapted from analyse_snapshots.py).
"""

from collections import defaultdict
from .helpers import _pct


def _cluster_sequences(records, gap_threshold):
    """Cluster records into sequences by camera, with gap threshold."""
    by_camera = defaultdict(list)
    for record in records:
        by_camera[record["camera_name"]].append(record)
    for camera_name in by_camera:
        by_camera[camera_name].sort(key=lambda r: r["timestamp"])
    sequences_by_camera = {}
    for camera_name, cam_records in by_camera.items():
        sequences = []
        current_seq = []
        for record in cam_records:
            if not current_seq:
                current_seq.append(record)
            else:
                gap = (record["timestamp"] - current_seq[-1]["timestamp"]).total_seconds()
                if gap > gap_threshold:
                    sequences.append(current_seq)
                    current_seq = [record]
                else:
                    current_seq.append(record)
        if current_seq:
            sequences.append(current_seq)
        sequences_by_camera[camera_name] = sequences
    return sequences_by_camera


def _classify_sequences(sequences_by_camera):
    """Classify sequences as confirmed, subthreshold_only, or miss."""
    confirmed_tiers = {"confirmed_45plus"}
    subthreshold_tiers = {"uncertain_35to45", "weak_25to35"}
    enriched = {}
    for camera_name, sequences in sequences_by_camera.items():
        enriched[camera_name] = []
        for seq in sequences:
            tiers_in_seq = {r["tier"] for r in seq}
            has_confirmed = bool(tiers_in_seq & confirmed_tiers)
            has_subthreshold = bool(tiers_in_seq & subthreshold_tiers)
            if has_confirmed:
                seq_type = "confirmed_sequence"
            elif has_subthreshold:
                seq_type = "subthreshold_only"
            else:
                seq_type = "miss_sequence"
            classified_frames = []
            for record in seq:
                tier = record["tier"]
                if seq_type == "confirmed_sequence" and tier != "confirmed_45plus":
                    frame_type = "dropout"
                elif seq_type == "miss_sequence" and tier == "motion_no_detection":
                    frame_type = "genuine_miss"
                elif seq_type == "subthreshold_only" and tier in subthreshold_tiers:
                    frame_type = "subthreshold_detection"
                else:
                    frame_type = "other"
                classified_frames.append({**record, "frame_type": frame_type})
            enriched[camera_name].append(
                {
                    "type": seq_type,
                    "frames": classified_frames,
                    "start_time": seq[0]["timestamp"],
                    "end_time": seq[-1]["timestamp"],
                    "duration": (seq[-1]["timestamp"] - seq[0]["timestamp"]).total_seconds(),
                }
            )
    return enriched


def _compute_stats(enriched_sequences):
    """Compute aggregate statistics from enriched sequences."""
    stats = {
        "total_frames": 0,
        "total_sequences": 0,
        "tier_counts": defaultdict(int),
        "tier_confidences": defaultdict(list),
        "seq_type_counts": defaultdict(int),
        "seq_type_cameras": defaultdict(set),
        "dropout_counts": defaultdict(int),
        "total_dropout_frames": 0,
        "genuine_miss_sequences": 0,
        "genuine_miss_frames": 0,
        "genuine_miss_by_camera": defaultdict(int),
        "subthreshold_sequences": 0,
        "subthreshold_frames": 0,
        "subthreshold_peak_confs": [],
        "camera_summary": defaultdict(lambda: defaultdict(int)),
    }
    for camera_name, sequences in enriched_sequences.items():
        for seq in sequences:
            stats["total_sequences"] += 1
            stats["seq_type_counts"][seq["type"]] += 1
            stats["seq_type_cameras"][seq["type"]].add(camera_name)
            if seq["type"] == "miss_sequence":
                stats["genuine_miss_sequences"] += 1
                stats["genuine_miss_frames"] += len(seq["frames"])
                stats["genuine_miss_by_camera"][camera_name] += 1
            elif seq["type"] == "subthreshold_only":
                stats["subthreshold_sequences"] += 1
                stats["subthreshold_frames"] += len(seq["frames"])
                confs = [f["confidence"] for f in seq["frames"] if f["confidence"] is not None]
                if confs:
                    stats["subthreshold_peak_confs"].append(max(confs))
            for frame in seq["frames"]:
                stats["total_frames"] += 1
                stats["tier_counts"][frame["tier"]] += 1
                if frame["confidence"] is not None:
                    stats["tier_confidences"][frame["tier"]].append(frame["confidence"])
                frame_type = frame["frame_type"]
                if frame_type == "dropout":
                    stats["total_dropout_frames"] += 1
                    if frame["tier"] == "uncertain_35to45":
                        stats["dropout_counts"]["uncertain"] += 1
                    elif frame["tier"] == "weak_25to35":
                        stats["dropout_counts"]["weak"] += 1
                    elif frame["tier"] == "motion_no_detection":
                        stats["dropout_counts"]["motion"] += 1
            seq_key = {
                "confirmed_sequence": "confirmed",
                "miss_sequence": "miss",
                "subthreshold_only": "subthreshold",
            }.get(seq["type"])
            if seq_key:
                stats["camera_summary"][camera_name][seq_key] += 1
            stats["camera_summary"][camera_name]["total"] += 1

    stats["tier_conf_stats"] = {}
    for tier in ["confirmed_45plus", "uncertain_35to45", "weak_25to35"]:
        confs = stats["tier_confidences"].get(tier, [])
        if confs:
            stats["tier_conf_stats"][tier] = {
                "min": min(confs),
                "max": max(confs),
                "avg": sum(confs) / len(confs),
                "count": len(confs),
            }
        else:
            stats["tier_conf_stats"][tier] = None
    return stats


def _format_stats_for_template(raw_stats, gap_threshold):
    """Format raw stats into template-friendly structure."""
    per_camera = []
    any_warned = False
    for cam_name in sorted(raw_stats["camera_summary"]):
        s = raw_stats["camera_summary"][cam_name]
        cam_miss = s.get("miss", 0)
        cam_confirmed = s.get("confirmed", 0)
        cam_total = s.get("total", 0)
        cam_miss_rate = _pct(cam_miss, cam_total) / 100.0

        if cam_confirmed == 0:
            verdict = "skip"
            verdict_text = "skip — no confirmed detections"
        elif cam_miss > 0 and cam_confirmed > 0 and (cam_miss / cam_confirmed) > 10:
            verdict = "traffic"
            verdict_text = "traffic — busy scene"
        elif cam_miss_rate > 0.10 and cam_miss >= 5:
            verdict = "warn"
            verdict_text = "[WARN] high miss rate"
            any_warned = True
        else:
            verdict = "ok"
            verdict_text = "[OK]"

        per_camera.append(
            {
                "name": cam_name,
                "confirmed": s.get("confirmed", 0),
                "detected_only": s.get("subthreshold", 0),
                "missed": cam_miss,
                "miss_rate": cam_miss_rate,
                "verdict": verdict,
                "verdict_text": verdict_text,
            }
        )

    uncertain_total = raw_stats["tier_counts"]["uncertain_35to45"]
    weak_total = raw_stats["tier_counts"]["weak_25to35"]
    subthreshold_total = uncertain_total + weak_total
    uw_dropout = raw_stats["dropout_counts"]["uncertain"] + raw_stats["dropout_counts"]["weak"]
    dropout_rate = _pct(uw_dropout, subthreshold_total) / 100.0

    if dropout_rate > 0.60:
        dropout_explanation = (
            "Most sub-threshold frames are harmless dropouts inside confirmed events"
        )
    else:
        dropout_explanation = (
            "Many sub-threshold frames are NOT inside confirmed events — review detected-only events"
        )

    ss = raw_stats["subthreshold_sequences"]
    if raw_stats["subthreshold_peak_confs"]:
        avg_peak = sum(raw_stats["subthreshold_peak_confs"]) / len(raw_stats["subthreshold_peak_confs"])
    else:
        avg_peak = 0.0

    if any_warned:
        recommendation = (
            "One or more cameras have a high miss rate. Review motion gate or confidence threshold."
        )
    elif avg_peak > 0.35 and ss > 0:
        recommendation = (
            f"{ss} events detected but not alerted (avg peak {avg_peak:.2f}). "
            "Consider testing threshold 0.35."
        )
    else:
        recommendation = "System operating normally"

    return {
        "per_camera": per_camera,
        "global_stats": {
            "dropout_rate": dropout_rate,
            "dropout_explanation": dropout_explanation,
            "detected_not_alerted": ss,
            "avg_peak_confidence": avg_peak,
            "recommendation": recommendation,
        },
    }
