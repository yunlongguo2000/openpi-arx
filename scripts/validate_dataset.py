#!/usr/bin/env python3
"""
ARX LeRobot 数据集验证工具

在训练前检查数据集完整性，参考 debug_data_collection.py 的验证思路。

检查项:
  1. 目录结构 (meta/, data/, videos/)
  2. info.json 格式和维度定义
  3. Parquet 数据完整性 (NaN/Inf, 维度匹配)
  4. 视频文件存在性和可读性
  5. Episode 连续性

用法:
  python scripts/validate_dataset.py --repo-id deepcybo/arx_lift_task_20260312_v03
  python scripts/validate_dataset.py --local-dir /path/to/dataset
"""

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# ARX 数据集期望维度
EXPECTED_STATE_DIM = 59
EXPECTED_ACTION_DIM = 32
EXPECTED_CAMERAS = ["left_wrist_image", "right_wrist_image"]


def ok(msg: str):
    print(f"  {GREEN}PASS{RESET}  {msg}")


def fail(msg: str):
    print(f"  {RED}FAIL{RESET}  {msg}")


def warn(msg: str):
    print(f"  {YELLOW}WARN{RESET}  {msg}")


def find_dataset_dir(repo_id: str | None, local_dir: str | None) -> Path:
    """定位数据集目录（本地路径或 HuggingFace 缓存）。"""
    if local_dir:
        return Path(local_dir)

    # 常见本地路径
    candidates = [
        Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id,
        Path.home() / "data" / "lerobot" / repo_id.split("/")[-1],
        Path("data") / "lerobot" / repo_id.split("/")[-1],
    ]
    for p in candidates:
        if p.exists():
            return p

    print(f"{RED}Cannot find dataset for '{repo_id}'. Use --local-dir to specify path.{RESET}")
    sys.exit(1)


def check_directory_structure(dataset_dir: Path) -> tuple[int, int]:
    """检查 LeRobot v2 目录结构。"""
    print(f"\n{CYAN}[1/5] Directory structure{RESET}")
    errors = 0
    warnings = 0

    # meta/
    meta_dir = dataset_dir / "meta"
    if meta_dir.is_dir():
        ok(f"meta/ directory exists")
        info_path = meta_dir / "info.json"
        if info_path.exists():
            ok("meta/info.json exists")
        else:
            fail("meta/info.json missing")
            errors += 1
    else:
        # LeRobot v1 style: info.json at root
        info_path = dataset_dir / "info.json"
        if info_path.exists():
            warn("Using legacy layout (info.json at root, no meta/ dir)")
            warnings += 1
        else:
            fail("Neither meta/info.json nor info.json found")
            errors += 1

    # data/
    data_dir = dataset_dir / "data"
    if data_dir.is_dir():
        parquets = list(data_dir.rglob("*.parquet"))
        if parquets:
            ok(f"data/ contains {len(parquets)} parquet file(s)")
        else:
            fail("data/ exists but no .parquet files found")
            errors += 1
    else:
        fail("data/ directory missing")
        errors += 1

    # videos/
    videos_dir = dataset_dir / "videos"
    if videos_dir.is_dir():
        mp4s = list(videos_dir.rglob("*.mp4"))
        ok(f"videos/ contains {len(mp4s)} video file(s)")
        if len(mp4s) == 0:
            warn("No .mp4 files in videos/")
            warnings += 1
    else:
        warn("videos/ directory missing (images may be in parquet)")
        warnings += 1

    return errors, warnings


def check_info_json(dataset_dir: Path) -> tuple[int, int, dict | None]:
    """检查 info.json 维度定义。"""
    print(f"\n{CYAN}[2/5] info.json validation{RESET}")
    errors = 0
    warnings = 0

    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        info_path = dataset_dir / "info.json"
    if not info_path.exists():
        fail("info.json not found, skipping")
        return 1, 0, None

    with open(info_path) as f:
        info = json.load(f)

    # FPS
    fps = info.get("fps")
    if fps:
        ok(f"fps = {fps}")
        if fps != 15:
            warn(f"Expected fps=15 for ARX, got {fps}")
            warnings += 1
    else:
        warn("fps not specified in info.json")
        warnings += 1

    # Features / shapes
    features = info.get("features", {})

    # State dimension
    state_feat = features.get("observation.state", {})
    state_shape = state_feat.get("shape")
    if state_shape:
        state_dim = state_shape[-1] if isinstance(state_shape, list) else state_shape
        if state_dim == EXPECTED_STATE_DIM:
            ok(f"observation.state dim = {state_dim}")
        else:
            fail(f"observation.state dim = {state_dim}, expected {EXPECTED_STATE_DIM}")
            errors += 1
    else:
        warn("observation.state shape not found in features")
        warnings += 1

    # Action dimension
    action_feat = features.get("action", {})
    action_shape = action_feat.get("shape")
    if action_shape:
        action_dim = action_shape[-1] if isinstance(action_shape, list) else action_shape
        if action_dim == EXPECTED_ACTION_DIM:
            ok(f"action dim = {action_dim}")
        else:
            fail(f"action dim = {action_dim}, expected {EXPECTED_ACTION_DIM}")
            errors += 1
    else:
        warn("action shape not found in features")
        warnings += 1

    # Camera keys
    for cam in EXPECTED_CAMERAS:
        key = f"observation.images.{cam}"
        if key in features:
            shape = features[key].get("shape", "?")
            ok(f"{key} present (shape={shape})")
        else:
            warn(f"{key} not found in features")
            warnings += 1

    # Episode count
    total_eps = info.get("total_episodes", info.get("num_episodes"))
    total_frames = info.get("total_frames", info.get("num_frames"))
    if total_eps is not None:
        ok(f"Episodes: {total_eps}, Frames: {total_frames}")
    else:
        warn("Episode/frame counts not found")
        warnings += 1

    return errors, warnings, info


def check_parquet_data(dataset_dir: Path) -> tuple[int, int]:
    """检查 parquet 数据质量 (NaN, Inf, 维度)。"""
    print(f"\n{CYAN}[3/5] Parquet data quality{RESET}")

    data_dir = dataset_dir / "data"
    parquets = sorted(data_dir.rglob("*.parquet")) if data_dir.is_dir() else []
    if not parquets:
        warn("No parquet files to check")
        return 0, 1

    try:
        import pyarrow.parquet as pq
        import numpy as np
    except ImportError:
        warn("pyarrow not installed, skipping parquet checks")
        return 0, 1

    errors = 0
    warnings = 0
    total_rows = 0

    for pf in parquets[:5]:  # 最多检查前5个文件
        table = pq.read_table(pf)
        n_rows = len(table)
        total_rows += n_rows

        # 检查 state 列
        if "observation.state" in table.column_names:
            states = table["observation.state"].to_pylist()
            arr = np.array(states, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] != EXPECTED_STATE_DIM:
                fail(f"{pf.name}: state dim={arr.shape[1]}, expected {EXPECTED_STATE_DIM}")
                errors += 1
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            if nan_count > 0:
                fail(f"{pf.name}: {nan_count} NaN values in state")
                errors += 1
            if inf_count > 0:
                fail(f"{pf.name}: {inf_count} Inf values in state")
                errors += 1

        # 检查 action 列
        if "action" in table.column_names:
            actions = table["action"].to_pylist()
            arr = np.array(actions, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] != EXPECTED_ACTION_DIM:
                fail(f"{pf.name}: action dim={arr.shape[1]}, expected {EXPECTED_ACTION_DIM}")
                errors += 1
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            if nan_count > 0:
                fail(f"{pf.name}: {nan_count} NaN values in action")
                errors += 1
            if inf_count > 0:
                fail(f"{pf.name}: {inf_count} Inf values in action")
                errors += 1

    if errors == 0:
        ok(f"Checked {len(parquets[:5])} parquet file(s), {total_rows} rows, no NaN/Inf")
    return errors, warnings


def check_videos(dataset_dir: Path) -> tuple[int, int]:
    """检查视频文件可读性。"""
    print(f"\n{CYAN}[4/5] Video files{RESET}")

    videos_dir = dataset_dir / "videos"
    if not videos_dir.is_dir():
        warn("No videos/ directory")
        return 0, 1

    errors = 0
    warnings = 0

    for cam in EXPECTED_CAMERAS:
        cam_dir = videos_dir / f"observation.images.{cam}"
        if not cam_dir.is_dir():
            warn(f"Missing video directory: observation.images.{cam}")
            warnings += 1
            continue

        mp4s = sorted(cam_dir.rglob("*.mp4"))
        if not mp4s:
            fail(f"No .mp4 files in {cam_dir.name}")
            errors += 1
            continue

        # 尝试读取第一个视频的第一帧验证可读性
        try:
            import cv2
            cap = cv2.VideoCapture(str(mp4s[0]))
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ok(f"{cam}: {len(mp4s)} file(s), first={w}x{h} @ {fps:.1f}fps, {frame_count} frames")
            else:
                fail(f"{cam}: cannot read first frame from {mp4s[0].name}")
                errors += 1
            cap.release()
        except ImportError:
            ok(f"{cam}: {len(mp4s)} file(s) (cv2 not available for frame check)")
        except Exception as e:
            fail(f"{cam}: error reading video: {e}")
            errors += 1

    return errors, warnings


def check_episodes(dataset_dir: Path, info: dict | None) -> tuple[int, int]:
    """检查 episode 连续性。"""
    print(f"\n{CYAN}[5/5] Episode consistency{RESET}")

    if info is None:
        warn("No info.json loaded, skipping episode check")
        return 0, 1

    errors = 0
    warnings = 0

    # 检查 episodes.jsonl (LeRobot v2)
    episodes_path = dataset_dir / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        episodes_path = dataset_dir / "episodes.jsonl"

    if episodes_path.exists():
        lines = episodes_path.read_text().strip().split("\n")
        ep_count = len(lines)
        total_expected = info.get("total_episodes", info.get("num_episodes"))
        if total_expected and ep_count == total_expected:
            ok(f"episodes.jsonl: {ep_count} episodes (matches info.json)")
        elif total_expected:
            warn(f"episodes.jsonl has {ep_count} episodes, info.json says {total_expected}")
            warnings += 1
        else:
            ok(f"episodes.jsonl: {ep_count} episodes")
    else:
        warn("episodes.jsonl not found")
        warnings += 1

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description="ARX LeRobot Dataset Validator")
    parser.add_argument("--repo-id", type=str, help="HuggingFace repo id (e.g. deepcybo/arx_lift_task_v03)")
    parser.add_argument("--local-dir", type=str, help="Local dataset directory path")
    args = parser.parse_args()

    if not args.repo_id and not args.local_dir:
        parser.error("Specify --repo-id or --local-dir")

    dataset_dir = find_dataset_dir(args.repo_id, args.local_dir)
    print(f"\n{CYAN}Validating dataset: {dataset_dir}{RESET}")

    total_errors = 0
    total_warnings = 0

    e, w = check_directory_structure(dataset_dir)
    total_errors += e
    total_warnings += w

    e, w, info = check_info_json(dataset_dir)
    total_errors += e
    total_warnings += w

    e, w = check_parquet_data(dataset_dir)
    total_errors += e
    total_warnings += w

    e, w = check_videos(dataset_dir)
    total_errors += e
    total_warnings += w

    e, w = check_episodes(dataset_dir, info)
    total_errors += e
    total_warnings += w

    # 汇总
    print(f"\n{'=' * 60}")
    if total_errors == 0:
        print(f"{GREEN}PASSED{RESET} - {total_warnings} warning(s)")
        print("Dataset is ready for training.")
    else:
        print(f"{RED}FAILED{RESET} - {total_errors} error(s), {total_warnings} warning(s)")
        print("Fix the errors above before training.")
    print()

    sys.exit(1 if total_errors > 0 else 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
