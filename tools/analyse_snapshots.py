#!/usr/bin/env python3
"""
Standalone snapshot analysis script for DVR Guard.

Analyses tiered snapshot folders to understand detection patterns:
- Dropouts (weak frames within confirmed sequences)
- Genuine misses (motion fired but YOLO found nothing)
- Subthreshold detections (detected but never confirmed)

Read-only: uses only stdlib (os, pathlib, datetime, argparse, collections).
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse DVR Guard snapshot tiers to understand detection patterns."
    )
    parser.add_argument(
        "--snapshots-dir",
        required=True,
        help="Root folder containing the four tier subfolders"
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=3.0,
        help="Max seconds between frames to still be considered the same person/event (default: 3.0)"
    )
    parser.add_argument(
        "--camera",
        default=None,
        help="Restrict analysis to one camera name (default: all cameras)"
    )
    return parser.parse_args()


def parse_filename(filename, tier):
    if not filename.endswith('.jpg'):
        return None

    name = filename[:-4]

    if len(name) >= 8 and name[:8].isdigit():
        parts = name.split('-')
        if len(parts) >= 3:
            date_str = parts[0]
            time_str = parts[1]

            if len(date_str) != 8 or not date_str.isdigit():
                return None
            if len(time_str) != 8 or time_str[2] != '_' or time_str[5] != '_':
                return None

            hh = time_str[0:2]
            mm = time_str[3:5]
            ss = time_str[6:8]

            if not (hh.isdigit() and mm.isdigit() and ss.isdigit()):
                return None

            remaining_parts = parts[2:]
            conf = None
            if remaining_parts[-1].startswith('c'):
                conf_str = remaining_parts[-1][1:]
                if not conf_str.isdigit():
                    return None
                conf = int(conf_str) / 100.0
                camera_parts = remaining_parts[:-1]
            else:
                camera_parts = remaining_parts

            camera_name = '-'.join(camera_parts)
            hour, minute, second, milliseconds = int(hh), int(mm), int(ss), 0
    else:
        parts = name.split('_')
        conf = None
        if parts[-1].startswith('c'):
            conf_str = parts[-1][1:]
            if not conf_str.isdigit():
                return None
            conf = int(conf_str) / 100.0
            if len(parts) < 5:
                return None
            date_str = parts[-4]
            time_part = parts[-3]
            ms_str = parts[-2]
            camera_parts = parts[0:-4]
        else:
            if len(parts) < 4:
                return None
            date_str = parts[-3]
            time_part = parts[-2]
            ms_str = parts[-1]
            camera_parts = parts[0:-3]
        camera_name = '_'.join(camera_parts)

        if len(date_str) != 8 or not date_str.isdigit():
            return None
        if len(time_part) != 6 or not time_part.isdigit():
            return None
        if len(ms_str) != 3 or not ms_str.isdigit():
            return None

        hour = int(time_part[:2])
        minute = int(time_part[2:4])
        second = int(time_part[4:6])
        milliseconds = int(ms_str)

    if len(date_str) != 8 or not date_str.isdigit():
        return None

    try:
        timestamp = datetime(
            int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]),
            hour, minute, second, milliseconds * 1000
        )
    except ValueError:
        return None

    return {
        'camera_id': camera_name,
        'timestamp': timestamp,
        'tier': tier,
        'confidence': conf,
        'filename': filename,
    }


def walk_snapshots(snapshots_dir, camera_filter=None):
    tiers = ['confirmed_45plus', 'uncertain_35to45', 'weak_25to35', 'motion_no_detection']

    tier_descriptions = {
        'confirmed_45plus':    'High-confidence detections (>=0.45) — Telegram alerts were sent for these',
        'uncertain_35to45':    'Medium-confidence detections (0.35–0.44) — detected but below alert threshold',
        'weak_25to35':         'Low-confidence detections (0.25–0.34) — model was unsure',
        'motion_no_detection': 'Motion triggered but YOLO found nobody — possible missed persons or noise',
    }

    records = []
    folder_counts = defaultdict(int)
    parse_errors = []

    print("\nReading snapshot folders:", file=sys.stderr)
    for tier in tiers:
        tier_path = Path(snapshots_dir) / tier

        if not tier_path.exists():
            print(f"  [MISSING] {tier}/  — folder not found, skipping", file=sys.stderr)
            continue
        if not tier_path.is_dir():
            print(f"  [SKIP] {tier} is not a directory", file=sys.stderr)
            continue

        file_count = 0
        for filename in os.listdir(tier_path):
            if not filename.endswith('.jpg'):
                continue
            record = parse_filename(filename, tier)
            if record is None:
                parse_errors.append((filename, tier))
                continue
            if camera_filter and record['camera_id'] != camera_filter:
                continue
            records.append(record)
            file_count += 1

        folder_counts[tier] = file_count
        desc = tier_descriptions[tier]
        print(f"  {file_count:>5} frames  [{tier}]  {desc}", file=sys.stderr)

    if parse_errors:
        print("\n  ERROR: Could not parse the following filenames.", file=sys.stderr)
        print("  Expected: {YYYYMMDD}-{HH_MM_SS}-{camera}-c{conf}.jpg", file=sys.stderr)
        print("  Example:  20260501-22_13_25-Curte-Dreapta-c087.jpg", file=sys.stderr)
        print("  Legacy:   {camera}_{YYYYMMDD}_{HHMMSS}_{ms}_c{conf}.jpg\n", file=sys.stderr)
        for filename, tier in parse_errors[:5]:
            print(f"    {filename}  (in folder: {tier})", file=sys.stderr)
        if len(parse_errors) > 5:
            print(f"    ... and {len(parse_errors) - 5} more unparseable files", file=sys.stderr)
        print("\n  Update parse_filename() if the format has changed.", file=sys.stderr)
        sys.exit(1)

    return records, folder_counts


def cluster_sequences(records, gap_threshold):
    by_camera = defaultdict(list)
    for record in records:
        by_camera[record['camera_id']].append(record)
    for camera_id in by_camera:
        by_camera[camera_id].sort(key=lambda r: r['timestamp'])

    sequences_by_camera = {}
    for camera_id, cam_records in by_camera.items():
        sequences = []
        current_seq = []
        for record in cam_records:
            if not current_seq:
                current_seq.append(record)
            else:
                gap = (record['timestamp'] - current_seq[-1]['timestamp']).total_seconds()
                if gap > gap_threshold:
                    sequences.append(current_seq)
                    current_seq = [record]
                else:
                    current_seq.append(record)
        if current_seq:
            sequences.append(current_seq)
        sequences_by_camera[camera_id] = sequences

    return sequences_by_camera


def classify_sequences(sequences_by_camera):
    confirmed_tiers   = {'confirmed_45plus'}
    subthreshold_tiers = {'uncertain_35to45', 'weak_25to35'}

    enriched = {}
    for camera_id, sequences in sequences_by_camera.items():
        enriched[camera_id] = []
        for seq in sequences:
            tiers_in_seq = set(r['tier'] for r in seq)
            has_confirmed     = bool(tiers_in_seq & confirmed_tiers)
            has_subthreshold  = bool(tiers_in_seq & subthreshold_tiers)

            if has_confirmed:
                seq_type = 'confirmed_sequence'
            elif has_subthreshold:
                seq_type = 'subthreshold_only'
            else:
                seq_type = 'miss_sequence'

            classified_frames = []
            for record in seq:
                tier = record['tier']
                if seq_type == 'confirmed_sequence' and tier != 'confirmed_45plus':
                    frame_type = 'dropout'
                elif seq_type == 'miss_sequence' and tier == 'motion_no_detection':
                    frame_type = 'genuine_miss'
                elif seq_type == 'subthreshold_only' and tier in subthreshold_tiers:
                    frame_type = 'subthreshold_detection'
                else:
                    frame_type = 'other'
                classified_frames.append({**record, 'frame_type': frame_type})

            enriched[camera_id].append({
                'type': seq_type,
                'frames': classified_frames,
                'start_time': seq[0]['timestamp'],
                'end_time':   seq[-1]['timestamp'],
                'duration':   (seq[-1]['timestamp'] - seq[0]['timestamp']).total_seconds(),
            })

    return enriched


def compute_stats(enriched_sequences):
    stats = {
        'total_frames': 0,
        'total_sequences': 0,
        'tier_counts': defaultdict(int),
        'tier_confidences': defaultdict(list),
        'seq_type_counts': defaultdict(int),
        'seq_type_cameras': defaultdict(set),
        'dropout_counts': defaultdict(int),
        'total_dropout_frames': 0,
        'genuine_miss_sequences': 0,
        'genuine_miss_frames': 0,
        'genuine_miss_by_camera': defaultdict(int),
        'subthreshold_sequences': 0,
        'subthreshold_frames': 0,
        'subthreshold_peak_confs': [],
        'camera_summary': defaultdict(lambda: defaultdict(int)),
    }

    for camera_id, sequences in enriched_sequences.items():
        for seq in sequences:
            stats['total_sequences'] += 1
            stats['seq_type_counts'][seq['type']] += 1
            stats['seq_type_cameras'][seq['type']].add(camera_id)

            if seq['type'] == 'miss_sequence':
                stats['genuine_miss_sequences'] += 1
                stats['genuine_miss_frames'] += len(seq['frames'])
                stats['genuine_miss_by_camera'][camera_id] += 1
            elif seq['type'] == 'subthreshold_only':
                stats['subthreshold_sequences'] += 1
                stats['subthreshold_frames'] += len(seq['frames'])
                confs = [f['confidence'] for f in seq['frames'] if f['confidence'] is not None]
                if confs:
                    stats['subthreshold_peak_confs'].append(max(confs))

            for frame in seq['frames']:
                stats['total_frames'] += 1
                stats['tier_counts'][frame['tier']] += 1
                if frame['confidence'] is not None:
                    stats['tier_confidences'][frame['tier']].append(frame['confidence'])

                frame_type = frame['frame_type']
                if frame_type == 'dropout':
                    stats['total_dropout_frames'] += 1
                    if frame['tier'] == 'uncertain_35to45':
                        stats['dropout_counts']['uncertain'] += 1
                    elif frame['tier'] == 'weak_25to35':
                        stats['dropout_counts']['weak'] += 1
                    elif frame['tier'] == 'motion_no_detection':
                        stats['dropout_counts']['motion'] += 1

            seq_key = {'confirmed_sequence': 'confirmed',
                       'miss_sequence': 'miss',
                       'subthreshold_only': 'subthreshold'}.get(seq['type'])
            if seq_key:
                stats['camera_summary'][camera_id][seq_key] += 1
            # always increment total so we can compute per-camera miss rate
            stats['camera_summary'][camera_id]['total'] += 1

    stats['tier_conf_stats'] = {}
    for tier in ['confirmed_45plus', 'uncertain_35to45', 'weak_25to35']:
        confs = stats['tier_confidences'].get(tier, [])
        if confs:
            stats['tier_conf_stats'][tier] = {
                'min': min(confs), 'max': max(confs),
                'avg': sum(confs) / len(confs), 'count': len(confs)
            }
        else:
            stats['tier_conf_stats'][tier] = None

    return stats


def pct(part, total):
    return (part / total * 100) if total > 0 else 0


def print_report(stats, gap_threshold, folder_counts):
    W = 65
    print("\n" + "=" * W)
    print("  DVR Guard — Detection Quality Report")
    print("=" * W)

    # ── Overview ──────────────────────────────────────────────────────────
    print("""
OVERVIEW
  A 'sequence' is a group of frames from the same camera where no gap
  between consecutive frames exceeds the threshold below. Each sequence
  represents one person-sighting event (or one block of noise).
""")
    print(f"  Frames analysed  : {stats['total_frames']}")
    print(f"  Events (sequences): {stats['total_sequences']}")
    print(f"  Sequence gap      : {gap_threshold:.1f}s  "
          f"(frames further apart than this start a new event)")

    # ── What each folder contains ─────────────────────────────────────────
    print("""
FRAMES PER CONFIDENCE TIER
  These are raw counts of saved frames, split by detection confidence.
  'confirmed' frames triggered Telegram alerts. The others were saved
  silently for analysis only.
""")
    tier_order = ['confirmed_45plus', 'uncertain_35to45', 'weak_25to35', 'motion_no_detection']
    tier_labels = {
        'confirmed_45plus':    'Confirmed  (>=0.45)  — Telegram alert sent',
        'uncertain_35to45':    'Uncertain  (0.35–0.44) — detected, no alert',
        'weak_25to35':         'Weak       (0.25–0.34) — low confidence, no alert',
        'motion_no_detection': 'No detection           — motion fired, YOLO found nobody',
    }
    for tier in tier_order:
        count = stats['tier_counts'][tier]
        label = tier_labels[tier]
        print(f"  {count:>5} frames  {label}")

    # ── Confidence distribution ───────────────────────────────────────────
    print("""
CONFIDENCE DISTRIBUTION (where actual scores were saved)
  Shows min/avg/max confidence within each tier. Useful to see how
  close uncertain/weak frames are to the alert threshold of 0.45.
""")
    for tier in ['confirmed_45plus', 'uncertain_35to45', 'weak_25to35']:
        cs = stats['tier_conf_stats'].get(tier)
        if cs:
            bar_avg = int((cs['avg'] - 0.25) / 0.75 * 20)
            bar_avg = max(0, min(20, bar_avg))
            bar = '[' + '#' * bar_avg + '-' * (20 - bar_avg) + ']'
            print(f"  {tier}")
            print(f"    min={cs['min']:.2f}  avg={cs['avg']:.2f}  max={cs['max']:.2f}  "
                  f"n={cs['count']}  {bar}")
        else:
            print(f"  {tier}: no confidence data in filenames")

    # ── Event classification ──────────────────────────────────────────────
    print("""
EVENT CLASSIFICATION
  Each event (sequence) is classified by whether it contained a
  confirmed detection, a sub-threshold detection only, or nothing at all.

  confirmed  = at least one frame reached 0.45+ → alert was sent
  detected   = person seen at weak/uncertain level but never hit 0.45
  missed     = motion triggered but YOLO returned zero detections
               (could be a real miss OR a false motion trigger)
""")
    confirmed_seq  = stats['seq_type_counts']['confirmed_sequence']
    sub_seq        = stats['seq_type_counts']['subthreshold_only']
    miss_seq       = stats['seq_type_counts']['miss_sequence']
    total_seq      = stats['total_sequences']
    confirmed_cams = len(stats['seq_type_cameras']['confirmed_sequence'])

    print(f"  {confirmed_seq:>4} confirmed events  "
          f"({pct(confirmed_seq, total_seq):.1f}% of all events, "
          f"across {confirmed_cams} camera(s))")
    print(f"  {sub_seq:>4} detected-only events  "
          f"({pct(sub_seq, total_seq):.1f}%)  — person seen below alert threshold")
    print(f"  {miss_seq:>4} missed events  "
          f"({pct(miss_seq, total_seq):.1f}%)  — motion fired, nothing detected")

    # ── Dropout analysis ──────────────────────────────────────────────────
    print("""
DROPOUT ANALYSIS
  A 'dropout' is a weak/uncertain/no-detection frame that sits INSIDE
  a confirmed event. This means the person WAS caught on surrounding
  frames — the alert was sent. The dropout is just YOLO having a bad
  frame, not a real miss. High dropout % here is normal and harmless.
""")
    uncertain_total = stats['tier_counts']['uncertain_35to45']
    weak_total      = stats['tier_counts']['weak_25to35']
    motion_total    = stats['tier_counts']['motion_no_detection']

    ud = stats['dropout_counts']['uncertain']
    wd = stats['dropout_counts']['weak']
    md = stats['dropout_counts']['motion']

    print(f"  Uncertain frames that are dropouts : "
          f"{ud}/{uncertain_total}  ({pct(ud, uncertain_total):.1f}%)")
    print(f"  Weak frames that are dropouts      : "
          f"{wd}/{weak_total}  ({pct(wd, weak_total):.1f}%)")
    print(f"  Motion-miss frames that are dropouts: "
          f"{md}/{motion_total}  ({pct(md, motion_total):.1f}%)")

    # ── Genuine misses ────────────────────────────────────────────────────
    print("""
GENUINE MISSES
  Events where motion triggered but YOLO found nothing at any confidence
  level — AND there are no confirmed frames nearby to explain it as a
  dropout. These are the cases worth manually reviewing.
  Could be: a real person YOLO missed, wind/IR noise, a moving object.
""")
    gm = stats['genuine_miss_sequences']
    gmf = stats['genuine_miss_frames']
    print(f"  Isolated miss events : {gm}  ({gmf} total frames)")
    if stats['genuine_miss_by_camera']:
        print("  Per camera:")
        for cam_id in sorted(stats['genuine_miss_by_camera']):
            count = stats['genuine_miss_by_camera'][cam_id]
            print(f"    {cam_id}: {count} miss events")
    else:
        print("  No isolated miss events found — good.")

    # ── Sub-threshold events ──────────────────────────────────────────────
    print("""
DETECTED-BUT-NOT-ALERTED EVENTS
  Events where the person was detected at weak or uncertain confidence
  but never hit 0.45 so no Telegram alert was sent. Review these to
  decide whether to lower the alert threshold.
""")
    ss  = stats['subthreshold_sequences']
    ssf = stats['subthreshold_frames']
    print(f"  Events detected below threshold : {ss}  ({ssf} frames)")
    if stats['subthreshold_peak_confs']:
        avg_peak = sum(stats['subthreshold_peak_confs']) / len(stats['subthreshold_peak_confs'])
        print(f"  Avg peak confidence in these   : {avg_peak:.2f}  "
              f"(threshold is 0.45)")
        if avg_peak > 0.35:
            print(f"  → These are close to the threshold. "
                  f"Lowering to 0.35 may capture them.")
        else:
            print(f"  → Peak confidence is low. Lowering threshold may add noise.")
    else:
        print("  None found.")

    # ── Per-camera summary ────────────────────────────────────────────────
    print("""
PER-CAMERA BREAKDOWN
  Count of events per type for each camera.
  'confirmed' = alert sent, 'missed' = nothing detected, 'detected' = below threshold.
""")
    for cam_id in sorted(stats['camera_summary']):
        s = stats['camera_summary'][cam_id]
        print(f"  {cam_id}")
        print(f"    confirmed={s.get('confirmed',0)}  "
              f"missed={s.get('miss',0)}  "
              f"detected-only={s.get('subthreshold',0)}")

    # ── Verdict ───────────────────────────────────────────────────────────
    print("""
VERDICT
  Genuine miss rate is evaluated per camera, not globally.
  A street camera will always have a high miss rate (cars, wind) —
  that does not mean people are being missed. Only cameras that also
  have confirmed detections are flagged if their miss rate is high,
  because those are the ones where YOLO is actively working and still
  dropping events.
""")

    subthreshold_total = uncertain_total + weak_total
    uw_dropout = ud + wd
    dropout_rate = pct(uw_dropout, subthreshold_total)

    print(f"  Dropout rate (global): {dropout_rate:.1f}%  "
          f"(sub-threshold frames that belong to a confirmed event)")
    if dropout_rate > 60:
        print("  [OK] Most sub-threshold frames are harmless dropouts inside confirmed")
        print("       events. The person was caught — just not on every frame.")
        print("       Lowering the alert threshold would add Telegram noise with")
        print("       little benefit.")
    else:
        print("  [NOTE] Many sub-threshold frames are NOT inside confirmed events.")
        print("         Review the detected-only events above.")

    # Per-camera genuine miss rate
    print(f"\n  Genuine miss rate — per camera:")
    print(f"  {'Camera':<30}  {'Miss':>5}  {'Total':>5}  {'Rate':>6}  Verdict")
    print(f"  {'-'*30}  {'-'*5}  {'-'*5}  {'-'*6}  -------")

    any_camera_warned = False
    for cam_id in sorted(stats['camera_summary']):
        s = stats['camera_summary'][cam_id]
        cam_miss      = s.get('miss', 0)
        cam_confirmed = s.get('confirmed', 0)
        cam_total     = s.get('total', 0)
        cam_miss_rate = pct(cam_miss, cam_total)

        # Classify each camera's miss rate with context:
        #
        # skip  — zero confirmed detections: YOLO never fired on a person here,
        #          so a high miss rate just means non-person motion (cars, wind).
        #          No action needed.
        #
        # traffic — many misses AND misses >> confirmed (ratio > 10x): the camera
        #          watches a busy scene. YOLO is still catching people (confirmed > 0)
        #          but most motion is non-person. Raising min_contour_area would
        #          reduce noise; lowering it would make things worse.
        #
        # warn  — miss rate > 10% AND at least 5 absolute misses AND misses are
        #          not dominated by traffic. Something is genuinely slipping through.
        #
        # ok    — everything else.
        if cam_confirmed == 0:
            verdict = "skip — no confirmed detections, likely noise/traffic camera"
        elif cam_miss > 0 and cam_confirmed > 0 and (cam_miss / cam_confirmed) > 10:
            verdict = "traffic — misses >> confirmed, busy scene (raise min_contour_area)"
        elif cam_miss_rate > 10 and cam_miss >= 5:
            verdict = "[WARN] high — review motion gate or confidence threshold"
            any_camera_warned = True
        else:
            verdict = "[OK]"

        print(f"  {cam_id:<30}  {cam_miss:>5}  {cam_total:>5}  "
              f"{cam_miss_rate:>5.1f}%  {verdict}")

    if any_camera_warned:
        print("\n  [WARN] One or more cameras have a genuinely high miss rate")
        print("         (not explained by traffic or noise volume).")
        print("         Suggested steps:")
        print("         1. Review frames in motion_no_detection/ for those cameras.")
        print("         2. Check if the camera angle cuts off parts of the scene.")
        print("         3. Consider adjusting min_contour_area for those cameras only.")
    else:
        print("\n  [OK] No camera has a concerning genuine miss rate.")
        print("       Cameras marked 'traffic' have high motion volume but are")
        print("       still catching people — consider raising their min_contour_area")
        print("       in config.yaml to reduce noise events.")

    if stats['subthreshold_peak_confs']:
        avg_peak = sum(stats['subthreshold_peak_confs']) / len(stats['subthreshold_peak_confs'])
        if avg_peak > 0.35 and ss > 0:
            print(f"\n  [NOTE] {ss} events were detected (peak avg {avg_peak:.2f}) but never")
            print(f"         alerted. Consider testing with alert threshold lowered to 0.35.")

    print("\n" + "=" * W + "\n")


def main():
    args = parse_args()

    snapshots_dir = Path(args.snapshots_dir)
    if not snapshots_dir.exists():
        print(f"ERROR: Directory not found: {snapshots_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\nSnapshots directory : {snapshots_dir}", file=sys.stderr)
    if args.camera:
        print(f"Filtering to camera : {args.camera}", file=sys.stderr)

    records, folder_counts = walk_snapshots(snapshots_dir, args.camera)

    total = len(records)
    print(f"\n  {total} frames loaded in total", file=sys.stderr)

    if total < 10:
        print(f"\nNot enough data for meaningful analysis ({total} frames found, need >= 10).",
              file=sys.stderr)
        sys.exit(0)

    print(f"\nGrouping frames into events (gap threshold: {args.gap}s) ...", file=sys.stderr)
    sequences_by_camera = cluster_sequences(records, args.gap)

    total_seq = sum(len(s) for s in sequences_by_camera.values())
    print(f"  {total_seq} events found across "
          f"{len(sequences_by_camera)} camera(s)", file=sys.stderr)

    print("Classifying events and frames ...", file=sys.stderr)
    enriched = classify_sequences(sequences_by_camera)

    stats = compute_stats(enriched)
    print_report(stats, args.gap, folder_counts)


if __name__ == '__main__':
    main()
