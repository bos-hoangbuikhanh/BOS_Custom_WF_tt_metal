#!/usr/bin/env python3
"""
Drive multiple test commands from a single Python script, store per-test logs,
and produce summary reports.

Usage:
    python scripts/test_suite_runner.py
"""

from __future__ import annotations

import argparse
import os
import queue
import re
import select
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
import io
import json
from pathlib import Path
from threading import Thread
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    TERMINAL_WIDTH = shutil.get_terminal_size().columns
except Exception:
    TERMINAL_WIDTH = 80


ROOT = Path(__file__).resolve().parent.parent
LOG_ROOT = ROOT / "logs" / "test_runs"
DEFAULT_HANG_TIMEOUT = 120.0  # treat 10s of no output as hang
DEFAULT_MAX_EXECUTION_TIME = 30 * 60  # auto-pass after 30 minutes
FULL_LOG_PATH = ROOT / "logs" / "test_runs" / "full_run.log"

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-9;?]*[ -/]*[@-~]")
PYTEST_STATUS_RE = re.compile(
    r"^(?P<item>.+?)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS|XSKIP)(?:\s+\[[^\]]+\])?$",
)
GTEST_STATUS_RE = re.compile(r"^\[\s*(RUN|OK|FAILED|PASSED|SKIPPED|DISABLED)\s*\].+")
GTEST_RUN_RE = re.compile(r"^\[\s*RUN\s+\]\s+(.+)$")
GTEST_OK_RE = re.compile(r"^\[\s+OK\s+\]\s+(.+?)(?:\s+\(.+\))?$")
GTEST_FAILED_RE = re.compile(r"^\[\s*FAILED\s+\]\s+(.+?)(?:\s+\(.+\))?$")
GTEST_SKIPPED_RE = re.compile(r"^\[\s*SKIPPED\s+\]\s+(.+?)(?:\s+\(.+\))?$")
GTEST_SUMMARY_RE = re.compile(r"^\[\s*PASSED\s+\]\s+(\d+)\s+tests?\.?$")
GTEST_FAILED_SUMMARY_RE = re.compile(r"^\[\s*FAILED\s+\]\s+(\d+)\s+test")
# Pytest summary patterns: "X passed", "X failed", "X skipped", "X error", "X passed in Y.YYs", etc.
# Matches patterns like: "5 passed, 2 failed, 1 skipped in 10.5s" or "10 passed in 5.2s"
PYTEST_SUMMARY_RE = re.compile(
    r"(\d+)\s+(passed|failed|skipped|error|warnings?)(?:\s+in\s+[\d.]+s)?",
    re.IGNORECASE,
)

# Performance metrics patterns
PERF_FPS_RE = re.compile(r"(?:FPS|fps)\s*[=:]?\s*(\d+\.?\d*)", re.IGNORECASE)
PERF_THROUGHPUT_RE = re.compile(r"(?:throughput|t/s/u|tokens?/s|images?/s)\s*[=:]?\s*(\d+\.?\d*)", re.IGNORECASE)
PERF_TTFT_RE = re.compile(r"(?:TTFT|time\s+to\s+first\s+token)\s*[=:]?\s*(\d+\.?\d*)\s*(?:ms|s)?", re.IGNORECASE)
PERF_INFERENCE_TIME_RE = re.compile(
    r"(?:inference\s+time|inference_time|avg\s+inference)\s*[=:]?\s*(\d+\.?\d*)\s*(?:s|ms)?", re.IGNORECASE
)
PERF_COMPILE_TIME_RE = re.compile(
    r"(?:compile\s+time|compile_time|avg\s+compile)\s*[=:]?\s*(\d+\.?\d*)\s*(?:s|ms)?", re.IGNORECASE
)
PERF_DURATION_RE = re.compile(r"(?:duration|elapsed|total\s+time)\s*[=:]?\s*(\d+\.?\d*)\s*(?:s|ms)?", re.IGNORECASE)

YOLO_TRACE_RE = re.compile(r"\[\s*\d+\s*\]\(Trace\)\s*FPS\s*=\s*(\d+\.?\d*)", re.IGNORECASE)
YOLO_TRACE_E2E_RE = re.compile(r"\[\s*\d+\s*\]\(Trace-e2e\)\s*FPS\s*=\s*(\d+\.?\d*)", re.IGNORECASE)
YOLO_FINAL_RE = re.compile(r"Final\s*FPS\s*=\s*(\d+\.?\d*)", re.IGNORECASE)
TOK_PER_USER_RE = re.compile(r"@\s*(\d+\.?\d*)\s*tok/s/user", re.IGNORECASE)
TOK_TOTAL_RE = re.compile(r"\(\s*(\d+\.?\d*)\s*tok/s\s+throughput\s*\)", re.IGNORECASE)
PREFILL_COMPILE_RE = re.compile(r"prefill\s+compile\s+time\s*[=:]?\s*(\d+\.?\d*)\s*(?:s|ms)?", re.IGNORECASE)
DECODE_COMPILE_RE = re.compile(r"decode\s+compile\s+time\s*[=:]?\s*(\d+\.?\d*)\s*(?:s|ms)?", re.IGNORECASE)


@dataclass
class GTestResult:
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    failed_tests: List[str] = field(default_factory=list)
    skipped_tests: List[str] = field(default_factory=list)


@dataclass
class PytestResult:
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    error: int = 0
    known_issue: int = 0
    failed_tests: List[str] = field(default_factory=list)
    known_issue_tests: List[str] = field(default_factory=list)


@dataclass
class PerformanceMetrics:
    fps: Optional[float] = None
    trace_fps: Optional[float] = None
    trace_e2e_fps: Optional[float] = None
    final_fps: Optional[float] = None
    throughput: Optional[float] = None
    throughput_per_user: Optional[float] = None
    ttft_ms: Optional[float] = None
    inference_time_s: Optional[float] = None
    compile_time_s: Optional[float] = None
    prefill_compile_time_s: Optional[float] = None
    decode_compile_time_s: Optional[float] = None
    duration_s: Optional[float] = None
    raw_lines: List[str] = field(default_factory=list)  # Store lines containing metrics

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "fps": self.fps,
            "trace_fps": self.trace_fps,
            "trace_e2e_fps": self.trace_e2e_fps,
            "final_fps": self.final_fps,
            "throughput": self.throughput,
            "throughput_per_user": self.throughput_per_user,
            "ttft_ms": self.ttft_ms,
            "inference_time_s": self.inference_time_s,
            "compile_time_s": self.compile_time_s,
            "prefill_compile_time_s": self.prefill_compile_time_s,
            "decode_compile_time_s": self.decode_compile_time_s,
            "duration_s": self.duration_s,
        }

    def to_summary_string(self, baseline: Optional[PerformanceMetrics] = None, enable_color: bool = False) -> str:
        parts = []

        def _format_with_comparison(
            label: str,
            current: Optional[float],
            baseline_val: Optional[float],
            higher_is_better: bool = True,
            unit: str = "",
        ) -> str:
            if current is None:
                return ""
            text = f"{label}: {current:.2f}{unit}"
            if baseline_val is not None and baseline_val > 0:
                change_pct = ((current - baseline_val) / baseline_val) * 100
                if abs(change_pct) < 0.01:  # Less than 0.01% change, consider as same
                    return text
                if higher_is_better:
                    is_improvement = change_pct > 0
                else:
                    is_improvement = change_pct < 0
                sign = "+" if change_pct > 0 else ""
                color_code = "\033[32m" if is_improvement else "\033[31m"  # Green for improvement, red for regression
                reset_code = "\033[0m" if enable_color else ""
                text += (
                    f" ({sign}{change_pct:.1f}%)"
                    if not enable_color
                    else f" ({color_code}{sign}{change_pct:.1f}%{reset_code})"
                )
            return text

        if self.trace_fps is not None:
            baseline_trace = baseline.trace_fps if baseline else None
            parts.append(_format_with_comparison("Trace FPS", self.trace_fps, baseline_trace, higher_is_better=True))
        if self.trace_e2e_fps is not None:
            baseline_trace_e2e = baseline.trace_e2e_fps if baseline else None
            parts.append(
                _format_with_comparison("Trace-e2e FPS", self.trace_e2e_fps, baseline_trace_e2e, higher_is_better=True)
            )
        if self.final_fps is not None:
            baseline_final = baseline.final_fps if baseline else None
            parts.append(_format_with_comparison("Final FPS", self.final_fps, baseline_final, higher_is_better=True))
        if self.fps is not None and self.trace_fps is None:
            baseline_fps = baseline.fps if baseline else None
            parts.append(_format_with_comparison("FPS", self.fps, baseline_fps, higher_is_better=True))
        if self.throughput is not None:
            baseline_throughput = baseline.throughput if baseline else None
            parts.append(
                _format_with_comparison("Throughput", self.throughput, baseline_throughput, higher_is_better=True)
            )
        if self.throughput_per_user is not None:
            baseline_throughput_user = baseline.throughput_per_user if baseline else None
            parts.append(
                _format_with_comparison(
                    "Throughput/user", self.throughput_per_user, baseline_throughput_user, higher_is_better=True
                )
            )
        if self.ttft_ms is not None:
            baseline_ttft = baseline.ttft_ms if baseline else None
            parts.append(
                _format_with_comparison("TTFT", self.ttft_ms, baseline_ttft, higher_is_better=False, unit="ms")
            )
        if self.inference_time_s is not None:
            baseline_inf = baseline.inference_time_s if baseline else None
            parts.append(
                _format_with_comparison(
                    "Inference", self.inference_time_s, baseline_inf, higher_is_better=False, unit="s"
                )
            )
        if self.prefill_compile_time_s is not None:
            baseline_prefill = baseline.prefill_compile_time_s if baseline else None
            parts.append(
                _format_with_comparison(
                    "Prefill Compile", self.prefill_compile_time_s, baseline_prefill, higher_is_better=False, unit="s"
                )
            )
        if self.decode_compile_time_s is not None:
            baseline_decode = baseline.decode_compile_time_s if baseline else None
            parts.append(
                _format_with_comparison(
                    "Decode Compile", self.decode_compile_time_s, baseline_decode, higher_is_better=False, unit="s"
                )
            )
        if self.compile_time_s is not None:
            baseline_comp = baseline.compile_time_s if baseline else None
            parts.append(
                _format_with_comparison("Compile", self.compile_time_s, baseline_comp, higher_is_better=False, unit="s")
            )
        if self.duration_s is not None:
            baseline_dur = baseline.duration_s if baseline else None
            parts.append(
                _format_with_comparison("Duration", self.duration_s, baseline_dur, higher_is_better=False, unit="s")
            )
        return ", ".join([p for p in parts if p]) if parts else ""


def _get_submodule_git_info(submodule_path: Path) -> Optional[Dict[str, any]]:
    """Get git information for a submodule."""
    if not submodule_path.exists():
        return None

    # Check if it's a git repository (either .git file or .git directory)
    git_dir = submodule_path / ".git"
    if not git_dir.exists():
        # Try checking if it's a git repository using git command
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=submodule_path,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode != 0:
            return None

    submodule_info: Dict[str, any] = {
        "path": str(submodule_path.relative_to(ROOT)),
        "branch": "unknown",
        "commit_hash": "unknown",
        "commit_hash_short": "unknown",
        "commit_date": "unknown",
        "commit_message": "unknown",
        "status": "unknown",
        "status_summary": "",
    }

    try:
        # Get current branch name - try multiple methods to handle detached HEAD
        branch_name = None

        # Method 1: Try git branch --show-current (Git 2.22+)
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=submodule_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            branch_name = result.stdout.strip()

        # Method 2: If that fails, try symbolic-ref (works when on a branch)
        if not branch_name or branch_name == "HEAD":
            result = subprocess.run(
                ["git", "symbolic-ref", "--short", "HEAD"],
                cwd=submodule_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                branch_name = result.stdout.strip()

        # Method 3: If still HEAD (detached), try to find the closest branch
        if not branch_name or branch_name == "HEAD":
            # Try to find which branch contains this commit
            result = subprocess.run(
                ["git", "branch", "-a", "--contains", "HEAD"],
                cwd=submodule_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                branches = result.stdout.strip().split("\n")
                # Prefer local branches, then remote tracking branches
                for branch in branches:
                    branch = branch.strip().lstrip("*").strip()
                    if branch.startswith("remotes/"):
                        # Extract branch name from remote branch (e.g., "remotes/origin/feature/v63" -> "feature/v63")
                        parts = branch.split("/")
                        if len(parts) >= 3:
                            branch_name = "/".join(parts[2:])
                            break
                    elif branch and not branch.startswith("remotes/"):
                        branch_name = branch
                        break

        # Method 4: If still no branch, try to get tag
        if not branch_name or branch_name == "HEAD":
            result = subprocess.run(
                ["git", "describe", "--tags", "--exact-match", "HEAD"],
                cwd=submodule_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                branch_name = f"tag:{result.stdout.strip()}"

        if branch_name and branch_name != "HEAD":
            submodule_info["branch"] = branch_name
        else:
            submodule_info["branch"] = "detached HEAD"

        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=submodule_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            submodule_info["commit_hash"] = result.stdout.strip()
            submodule_info["commit_hash_short"] = submodule_info["commit_hash"][:8]

        # Get commit date
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=submodule_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            submodule_info["commit_date"] = result.stdout.strip()

        # Get commit message
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=submodule_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            submodule_info["commit_message"] = result.stdout.strip()

        # Get git status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=submodule_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            status_lines = result.stdout.strip().split("\n")
            status_lines = [line for line in status_lines if line]
            if status_lines:
                submodule_info["status"] = "dirty"
                modified_count = sum(1 for line in status_lines if line.startswith(" M") or line.startswith("M "))
                added_count = sum(1 for line in status_lines if line.startswith("A ") or line.startswith("??"))
                deleted_count = sum(1 for line in status_lines if line.startswith(" D") or line.startswith("D "))
                parts = []
                if modified_count > 0:
                    parts.append(f"{modified_count} modified")
                if added_count > 0:
                    parts.append(f"{added_count} added")
                if deleted_count > 0:
                    parts.append(f"{deleted_count} deleted")
                submodule_info["status_summary"] = ", ".join(parts) if parts else "changes detected"
            else:
                submodule_info["status"] = "clean"
                submodule_info["status_summary"] = "no uncommitted changes"

        return submodule_info
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return None


def _get_git_info() -> Dict[str, any]:
    """Get git information for logging purposes."""
    git_info: Dict[str, any] = {
        "branch": "unknown",
        "remote_branch": "unknown",
        "commit_hash": "unknown",
        "commit_hash_short": "unknown",
        "commit_date": "unknown",
        "commit_message": "unknown",
        "status": "unknown",
        "status_summary": "",
        "remote_sync": "unknown",
        "remote_url": "unknown",
        "modified_files": [],
        "added_files": [],
        "deleted_files": [],
        "untracked_files": [],
        "staged_files": [],
        "diff_stats": {"insertions": 0, "deletions": 0},
        "recent_commits": [],
        "tags": [],
        "submodules": [],
    }

    try:
        # Get current branch
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        # Get remote tracking branch
        # First try to get upstream tracking branch (@{u})
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        remote_branch = None
        if result.returncode == 0 and result.stdout.strip():
            remote_branch = result.stdout.strip()
        else:
            # If no upstream tracking branch, check if origin/{branch_name} exists locally
            # This handles cases where local branch exists but tracking is not set up
            # Use local refs first (faster, no network request)
            if git_info["branch"] != "unknown":
                # Check if remote branch ref exists locally (from previous fetch)
                result_ref = subprocess.run(
                    ["git", "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{git_info['branch']}"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result_ref.returncode == 0:
                    # Remote branch exists locally, construct the name
                    remote_branch = f"origin/{git_info['branch']}"
                else:
                    # If not found locally, try ls-remote (requires network, but handles first-time case)
                    result_origin = subprocess.run(
                        ["git", "ls-remote", "--heads", "origin", git_info["branch"]],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result_origin.returncode == 0 and result_origin.stdout.strip():
                        # Remote branch exists, construct the name
                        remote_branch = f"origin/{git_info['branch']}"

        if remote_branch:
            # Extract remote name and branch name
            if "/" in remote_branch:
                remote_name, remote_branch_name = remote_branch.split("/", 1)
                git_info["remote_branch"] = f"{remote_name}/{remote_branch_name}"

                # Get remote URL
                result_url = subprocess.run(
                    ["git", "config", f"remote.{remote_name}.url"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result_url.returncode == 0:
                    git_info["remote_url"] = result_url.stdout.strip()

                # Check remote sync status (ahead/behind) only if remote branch exists
                try:
                    result_ahead = subprocess.run(
                        ["git", "rev-list", "--left-right", "--count", f"HEAD...{git_info['remote_branch']}"],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result_ahead.returncode == 0:
                        ahead_behind = result_ahead.stdout.strip().split()
                        if len(ahead_behind) == 2:
                            ahead = int(ahead_behind[0])
                            behind = int(ahead_behind[1])
                            sync_parts = []
                            if ahead > 0:
                                sync_parts.append(f"{ahead} ahead")
                            if behind > 0:
                                sync_parts.append(f"{behind} behind")
                            if sync_parts:
                                git_info["remote_sync"] = ", ".join(sync_parts)
                            else:
                                git_info["remote_sync"] = "synced"
                except (ValueError, IndexError):
                    pass

        # Get commit hash (full)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()
            git_info["commit_hash_short"] = git_info["commit_hash"][:8]

        # Get commit date
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit_date"] = result.stdout.strip()

        # Get commit message (first line)
        result = subprocess.run(
            ["git", "log", "-1", "--format=%s"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit_message"] = result.stdout.strip()

        # Get git status (clean/dirty) with detailed file lists
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            status_lines = result.stdout.strip().split("\n")
            status_lines = [line for line in status_lines if line]
            if status_lines:
                git_info["status"] = "dirty"
                modified_files = []
                added_files = []
                deleted_files = []
                untracked_files = []
                staged_files = []

                for line in status_lines:
                    if len(line) < 3:
                        continue
                    status_code = line[:2]
                    file_path = line[3:].strip()

                    # Staged files
                    if status_code[0] != " " and status_code[0] != "?":
                        staged_files.append(file_path)

                    # Modified files
                    if status_code in [" M", "MM", "MD"]:
                        modified_files.append(file_path)
                    elif status_code in ["M ", "MM", "AM"]:
                        modified_files.append(file_path)

                    # Added files
                    if status_code in ["A ", "AM", "AD"]:
                        added_files.append(file_path)
                    elif status_code == "??":
                        untracked_files.append(file_path)

                    # Deleted files
                    if status_code in [" D", "DD", "MD"]:
                        deleted_files.append(file_path)
                    elif status_code in ["D ", "DD", "AD"]:
                        deleted_files.append(file_path)

                git_info["modified_files"] = modified_files
                git_info["added_files"] = added_files
                git_info["deleted_files"] = deleted_files
                git_info["untracked_files"] = untracked_files
                git_info["staged_files"] = staged_files

                # Summary: count of modified/added/deleted files
                parts = []
                if modified_files:
                    parts.append(f"{len(modified_files)} modified")
                if added_files:
                    parts.append(f"{len(added_files)} added")
                if deleted_files:
                    parts.append(f"{len(deleted_files)} deleted")
                if untracked_files:
                    parts.append(f"{len(untracked_files)} untracked")
                git_info["status_summary"] = ", ".join(parts) if parts else "changes detected"
            else:
                git_info["status"] = "clean"
                git_info["status_summary"] = "no uncommitted changes"

        # Get diff statistics (lines added/removed)
        if git_info["status"] == "dirty":
            try:
                result_diff = subprocess.run(
                    ["git", "diff", "--stat"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result_diff.returncode == 0:
                    # Parse diff stat summary line (last line)
                    diff_lines = result_diff.stdout.strip().split("\n")
                    if diff_lines:
                        # Look for summary line like "X files changed, Y insertions(+), Z deletions(-)"
                        for line in reversed(diff_lines):
                            if "files changed" in line or "insertions" in line or "deletions" in line:
                                # Extract numbers
                                insertions_match = re.search(r"(\d+)\s+insertions?", line)
                                deletions_match = re.search(r"(\d+)\s+deletions?", line)
                                if insertions_match:
                                    git_info["diff_stats"]["insertions"] = int(insertions_match.group(1))
                                if deletions_match:
                                    git_info["diff_stats"]["deletions"] = int(deletions_match.group(1))
                                break
            except (ValueError, subprocess.TimeoutExpired):
                pass

        # Get recent commits (last 5)
        try:
            result_commits = subprocess.run(
                ["git", "log", "-5", "--format=%h|%ci|%s"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result_commits.returncode == 0:
                recent_commits = []
                for line in result_commits.stdout.strip().split("\n"):
                    if "|" in line:
                        parts = line.split("|", 2)
                        if len(parts) == 3:
                            recent_commits.append(
                                {
                                    "hash": parts[0],
                                    "date": parts[1],
                                    "message": parts[2],
                                }
                            )
                git_info["recent_commits"] = recent_commits
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

        # Get tags for current commit
        try:
            result_tags = subprocess.run(
                ["git", "tag", "--points-at", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result_tags.returncode == 0:
                tags = [tag.strip() for tag in result_tags.stdout.strip().split("\n") if tag.strip()]
                git_info["tags"] = tags
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            pass

        # Collect submodule git information
        submodule_paths = [
            ROOT / "tt_metal" / "third_party" / "tt_llk",
            ROOT / "tt_metal" / "third_party" / "umd",
            ROOT / "tt_metal" / "third_party" / "tracy",
            ROOT / "tt_metal" / "third_party" / "bos-metal",
        ]
        for submodule_path in submodule_paths:
            submodule_info = _get_submodule_git_info(submodule_path)
            if submodule_info:
                git_info["submodules"].append(submodule_info)

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        # Git not available or not a git repository
        pass

    return git_info


def _format_detailed_git_info(git_info: Dict[str, any]) -> str:
    """Format detailed git information for git_info.log file."""
    lines = [
        "=" * 80,
        "Detailed Git Information",
        "=" * 80,
        "",
        "Basic Information",
        "-" * 80,
        f"Branch:           {git_info['branch']}",
    ]

    # Add remote branch info if available (may not exist if branch is not pushed)
    if git_info.get("remote_branch", "unknown") != "unknown":
        lines.append(f"Remote Branch:    {git_info['remote_branch']}")
        if git_info.get("remote_sync", "unknown") != "unknown":
            lines.append(f"Remote Sync:      {git_info['remote_sync']}")
        if git_info.get("remote_url", "unknown") != "unknown":
            lines.append(f"Remote URL:       {git_info['remote_url']}")
    else:
        lines.append("Remote Branch:    (not tracked - local only)")

    lines.extend(
        [
            f"Commit Hash:      {git_info['commit_hash']} ({git_info['commit_hash_short']})",
            f"Commit Date:      {git_info['commit_date']}",
            f"Commit Message:   {git_info['commit_message']}",
        ]
    )

    # Tags
    if git_info.get("tags"):
        lines.append(f"Tags:             {', '.join(git_info['tags'])}")

    lines.extend(
        [
            "",
            "Repository Status",
            "-" * 80,
            f"Status:           {git_info['status']}",
        ]
    )
    if git_info["status_summary"]:
        lines.append(f"Summary:          {git_info['status_summary']}")

    # Detailed file lists
    if git_info["status"] == "dirty":
        lines.append("")
        lines.append("Changed Files")
        lines.append("-" * 80)

        if git_info.get("staged_files"):
            lines.append(f"\nStaged Files ({len(git_info['staged_files'])}):")
            for f in git_info["staged_files"]:
                lines.append(f"  + {f}")

        if git_info.get("modified_files"):
            lines.append(f"\nModified Files ({len(git_info['modified_files'])}):")
            for f in git_info["modified_files"]:
                lines.append(f"  M {f}")

        if git_info.get("added_files"):
            lines.append(f"\nAdded Files ({len(git_info['added_files'])}):")
            for f in git_info["added_files"]:
                lines.append(f"  A {f}")

        if git_info.get("deleted_files"):
            lines.append(f"\nDeleted Files ({len(git_info['deleted_files'])}):")
            for f in git_info["deleted_files"]:
                lines.append(f"  D {f}")

        if git_info.get("untracked_files"):
            lines.append(f"\nUntracked Files ({len(git_info['untracked_files'])}):")
            for f in git_info["untracked_files"]:
                lines.append(f"  ? {f}")

        # Diff statistics
        if (
            git_info.get("diff_stats", {}).get("insertions", 0) > 0
            or git_info.get("diff_stats", {}).get("deletions", 0) > 0
        ):
            lines.append("")
            lines.append("Diff Statistics")
            lines.append("-" * 80)
            stats = git_info["diff_stats"]
            lines.append(f"Insertions:       {stats.get('insertions', 0)} lines")
            lines.append(f"Deletions:        {stats.get('deletions', 0)} lines")
            net_change = stats.get("insertions", 0) - stats.get("deletions", 0)
            if net_change != 0:
                sign = "+" if net_change > 0 else ""
                lines.append(f"Net Change:       {sign}{net_change} lines")

    # Recent commits
    if git_info.get("recent_commits"):
        lines.append("")
        lines.append("Recent Commits")
        lines.append("-" * 80)
        for commit in git_info["recent_commits"]:
            lines.append(f"  {commit['hash']}  {commit['date']}  {commit['message']}")

    # Submodules
    if git_info.get("submodules"):
        lines.append("")
        lines.append("Submodules")
        lines.append("-" * 80)
        for submodule in git_info["submodules"]:
            lines.append(f"\nSubmodule: {submodule['path']}")
            lines.append(f"  Branch:           {submodule['branch']}")
            lines.append(f"  Commit Hash:      {submodule['commit_hash']} ({submodule['commit_hash_short']})")
            lines.append(f"  Commit Date:      {submodule['commit_date']}")
            lines.append(f"  Commit Message:   {submodule['commit_message']}")
            lines.append(f"  Repository Status: {submodule['status']}")
            if submodule.get("status_summary"):
                lines.append(f"  Status Summary:   {submodule['status_summary']}")

    lines.extend(
        [
            "",
            "=" * 80,
            "",
        ]
    )
    return "\n".join(lines)


def _format_git_header(git_info: Dict[str, any]) -> str:
    """Format git information as a header for log files."""
    lines = [
        "=" * 80,
        "Git Information",
        "=" * 80,
        f"Branch:           {git_info['branch']}",
    ]

    # Add remote branch info if available (may not exist if branch is not pushed)
    if git_info.get("remote_branch", "unknown") != "unknown":
        lines.append(f"Remote Branch:    {git_info['remote_branch']}")
        if git_info.get("remote_sync", "unknown") != "unknown":
            lines.append(f"Remote Sync:      {git_info['remote_sync']}")
        if git_info.get("remote_url", "unknown") != "unknown":
            lines.append(f"Remote URL:       {git_info['remote_url']}")
    else:
        lines.append("Remote Branch:    (not tracked - local only)")

    lines.extend(
        [
            f"Commit Hash:      {git_info['commit_hash']} ({git_info['commit_hash_short']})",
            f"Commit Date:      {git_info['commit_date']}",
            f"Commit Message:   {git_info['commit_message']}",
            f"Repository Status: {git_info['status']}",
        ]
    )
    if git_info["status_summary"]:
        lines.append(f"Status Details:   {git_info['status_summary']}")

    # Add submodule summary
    if git_info.get("submodules"):
        submodule_summary = []
        for submodule in git_info["submodules"]:
            submodule_name = submodule["path"].split("/")[-1]
            branch_name = submodule["branch"]
            status_indicator = "✓" if submodule["status"] == "clean" else "✗"
            status_text = f"{submodule['status']}" + (
                f" ({submodule['status_summary']})" if submodule.get("status_summary") else ""
            )
            submodule_summary.append(
                f"{submodule_name}: {submodule['commit_hash_short']} | branch: {branch_name} | status: {status_text} {status_indicator}"
            )
        if submodule_summary:
            lines.append("Submodules:")
            for summary in submodule_summary:
                lines.append(f"  {summary}")

    lines.extend(
        [
            "=" * 80,
            "",
        ]
    )
    return "\n".join(lines)


def _load_baseline_performance_metrics(baseline_path: Optional[Path] = None) -> Dict[str, PerformanceMetrics]:
    """Load baseline performance metrics from specified file or default location."""
    if baseline_path is None:
        return {}

    if not baseline_path.exists():
        return {}

    try:
        with baseline_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        baseline_dict: Dict[str, PerformanceMetrics] = {}
        for item in data:
            test_name = item.get("test", "")
            metrics_dict = item.get("metrics", {})
            baseline_dict[test_name] = PerformanceMetrics(
                fps=metrics_dict.get("fps"),
                trace_fps=metrics_dict.get("trace_fps"),
                trace_e2e_fps=metrics_dict.get("trace_e2e_fps"),
                final_fps=metrics_dict.get("final_fps"),
                throughput=metrics_dict.get("throughput"),
                throughput_per_user=metrics_dict.get("throughput_per_user"),
                ttft_ms=metrics_dict.get("ttft_ms"),
                inference_time_s=metrics_dict.get("inference_time_s"),
                compile_time_s=metrics_dict.get("compile_time_s"),
                prefill_compile_time_s=metrics_dict.get("prefill_compile_time_s"),
                decode_compile_time_s=metrics_dict.get("decode_compile_time_s"),
                duration_s=metrics_dict.get("duration_s"),
            )
        return baseline_dict
    except Exception as e:
        print(f"Warning: Failed to load baseline performance metrics from {baseline_path}: {e}")
        return {}


def _parse_performance_metrics(output: str, test_name: Optional[str] = None) -> PerformanceMetrics:
    """Parse performance metrics from test output."""
    metrics = PerformanceMetrics()
    lines = output.split("\n")

    fps_values: List[float] = []
    trace_fps_values: List[float] = []
    trace_e2e_values: List[float] = []
    final_fps_values: List[float] = []

    for line in lines:
        clean = _strip_ansi(line).strip()
        if not clean:
            continue

        # YOLO-style metrics (Trace / Trace-e2e / Final)
        yolo_match = YOLO_TRACE_RE.search(clean)
        if yolo_match:
            try:
                value = float(yolo_match.group(1))
                trace_fps_values.append(value)
                metrics.raw_lines.append(clean)
            except ValueError:
                pass
            continue

        yolo_trace_e2e_match = YOLO_TRACE_E2E_RE.search(clean)
        if yolo_trace_e2e_match:
            try:
                value = float(yolo_trace_e2e_match.group(1))
                trace_e2e_values.append(value)
                metrics.raw_lines.append(clean)
            except ValueError:
                pass
            continue

        yolo_final_match = YOLO_FINAL_RE.search(clean)
        if yolo_final_match:
            try:
                value = float(yolo_final_match.group(1))
                final_fps_values.append(value)
                metrics.raw_lines.append(clean)
            except ValueError:
                pass
            continue

        handled_throughput = False
        if "tok/s" in clean.lower():
            per_user_match = TOK_PER_USER_RE.search(clean)
            total_match = TOK_TOTAL_RE.search(clean)
            if per_user_match or total_match:
                try:
                    if per_user_match:
                        metrics.throughput_per_user = float(per_user_match.group(1))
                    if total_match:
                        metrics.throughput = float(total_match.group(1))
                    metrics.raw_lines.append(clean)
                    handled_throughput = True
                except ValueError:
                    pass

        # FPS
        match = PERF_FPS_RE.search(clean)
        if match:
            try:
                fps_val = float(match.group(1))
                fps_values.append(fps_val)
                metrics.raw_lines.append(clean)
            except ValueError:
                pass

        # Throughput
        if not handled_throughput:
            match = PERF_THROUGHPUT_RE.search(clean)
            if match:
                try:
                    throughput_val = float(match.group(1))
                    if metrics.throughput is None or throughput_val > metrics.throughput:
                        metrics.throughput = throughput_val
                    metrics.raw_lines.append(clean)
                except ValueError:
                    pass

        # TTFT
        match = PERF_TTFT_RE.search(clean)
        if match:
            try:
                ttft_val = float(match.group(1))
                # Convert to ms if needed (assume ms if no unit specified)
                if "s" in clean.lower() and "ms" not in clean.lower():
                    ttft_val *= 1000
                metrics.ttft_ms = ttft_val if metrics.ttft_ms is None else min(metrics.ttft_ms, ttft_val)
                metrics.raw_lines.append(clean)
            except ValueError:
                pass

        # Inference time
        match = PERF_INFERENCE_TIME_RE.search(clean)
        if match:
            try:
                inf_time = float(match.group(1))
                # Convert to seconds if needed
                if "ms" in clean.lower():
                    inf_time /= 1000
                metrics.inference_time_s = (
                    inf_time if metrics.inference_time_s is None else min(metrics.inference_time_s, inf_time)
                )
                metrics.raw_lines.append(clean)
            except ValueError:
                pass

        # Compile time
        match = PREFILL_COMPILE_RE.search(clean)
        if match:
            try:
                comp_time = float(match.group(1))
                if "ms" in clean.lower():
                    comp_time /= 1000
                metrics.prefill_compile_time_s = comp_time
                metrics.raw_lines.append(clean)
            except ValueError:
                pass

        match = DECODE_COMPILE_RE.search(clean)
        if match:
            try:
                comp_time = float(match.group(1))
                if "ms" in clean.lower():
                    comp_time /= 1000
                metrics.decode_compile_time_s = comp_time
                metrics.raw_lines.append(clean)
            except ValueError:
                pass

        match = PERF_COMPILE_TIME_RE.search(clean)
        if match:
            try:
                comp_time = float(match.group(1))
                # Convert to seconds if needed
                if "ms" in clean.lower():
                    comp_time /= 1000
                metrics.compile_time_s = (
                    comp_time if metrics.compile_time_s is None else min(metrics.compile_time_s, comp_time)
                )
                metrics.raw_lines.append(clean)
            except ValueError:
                pass

        # Duration
        match = PERF_DURATION_RE.search(clean)
        if match:
            try:
                duration_val = float(match.group(1))
                # Convert to seconds if needed
                if "ms" in clean.lower():
                    duration_val /= 1000
                metrics.duration_s = (
                    duration_val if metrics.duration_s is None else min(metrics.duration_s, duration_val)
                )
                metrics.raw_lines.append(clean)
            except ValueError:
                pass

    if trace_fps_values:
        metrics.trace_fps = sum(trace_fps_values) / len(trace_fps_values)
        metrics.fps = metrics.trace_fps
    if trace_e2e_values:
        metrics.trace_e2e_fps = sum(trace_e2e_values) / len(trace_e2e_values)
    if final_fps_values:
        metrics.final_fps = sum(final_fps_values) / len(final_fps_values)
    if fps_values and metrics.fps is None:
        metrics.fps = sum(fps_values) / len(fps_values)
    if metrics.prefill_compile_time_s is not None or metrics.decode_compile_time_s is not None:
        total_compile = 0.0
        if metrics.prefill_compile_time_s is not None:
            total_compile += metrics.prefill_compile_time_s
        if metrics.decode_compile_time_s is not None:
            total_compile += metrics.decode_compile_time_s
        if total_compile > 0:
            metrics.compile_time_s = total_compile

    return metrics


def _parse_gtest_output(output: str) -> Optional[GTestResult]:
    lines = output.split("\n")
    result = GTestResult()
    has_summary = False

    for line in lines:
        clean = _strip_ansi(line).strip()
        if not clean:
            continue

        # Summary lines - these come at the end and give us totals
        match = GTEST_SUMMARY_RE.match(clean)
        if match:
            result.passed = int(match.group(1))
            has_summary = True
            continue

        match = GTEST_FAILED_SUMMARY_RE.match(clean)
        if match:
            result.failed = int(match.group(1))
            has_summary = True
            continue

        # Individual test results - collect test names
        match = GTEST_OK_RE.match(clean)
        if match:
            # Just track names, don't count here if we have summary
            continue

        match = GTEST_FAILED_RE.match(clean)
        if match:
            test_name = match.group(1).strip()
            if test_name not in result.failed_tests:
                result.failed_tests.append(test_name)
            continue

        match = GTEST_SKIPPED_RE.match(clean)
        if match:
            test_name = match.group(1).strip()
            # Filter out invalid entries like "15 tests, listed below:"
            if "tests" in test_name.lower() and "listed" in test_name.lower():
                continue
            if test_name not in result.skipped_tests:
                result.skipped_tests.append(test_name)
            result.skipped += 1
            continue

    # If no summary found, count individual tests
    if not has_summary:
        for line in lines:
            clean = _strip_ansi(line).strip()
            if GTEST_OK_RE.match(clean):
                result.passed += 1
        result.failed = len(result.failed_tests)

    result.total = result.passed + result.failed + result.skipped
    return result if result.total > 0 else None


def _collect_pytest_tests(test_command: str, env: Dict[str, str] = None) -> Optional[PytestResult]:
    """
    Collect pytest tests without running them using --collect-only.
    Returns statistics about total tests and expected skips.
    """
    import subprocess

    # Extract pytest command and arguments
    # Handle cases like "pytest tests/..." or "source env.sh && pytest tests/..."
    if "pytest" not in test_command:
        return None

    # Extract just the pytest part
    parts = test_command.split()
    pytest_idx = next((i for i, p in enumerate(parts) if "pytest" in p), None)
    if pytest_idx is None:
        return None

    # Build collect-only command
    pytest_cmd = parts[pytest_idx:]
    # Remove any existing --collect-only if present
    pytest_cmd = [p for p in pytest_cmd if p != "--collect-only"]
    pytest_cmd.insert(1, "--collect-only")  # Insert after "pytest"

    # Reconstruct full command with collect-only
    if pytest_idx > 0:
        # There's a prefix command (e.g., "source env.sh &&")
        prefix = " ".join(parts[:pytest_idx])
        full_cmd = f"{prefix} {' '.join(pytest_cmd)}"
    else:
        full_cmd = " ".join(pytest_cmd)

    command = f"cd {ROOT} && {full_cmd}"

    env_dict = os.environ.copy()
    if env:
        env_dict.update(env)

    try:
        proc = subprocess.run(
            ["bash", "-lc", command],
            cwd=str(ROOT),
            env=env_dict,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,  # Collection should be fast
        )

        output = proc.stdout

        # Parse the collection output
        # Format: "<Module tests/path/to/test.py>"
        #         "  <Function test_name[param1-param2]>"
        #         "  <Function test_name[param3-param4]>"
        # Or: "  <Function test_name[param1-param2]> SKIPPED"

        result = PytestResult()
        lines = output.split("\n")

        for line in lines:
            clean = _strip_ansi(line).strip()
            if not clean:
                continue

            # Count total tests (Function items)
            if "<Function" in clean or "<TestCaseFunction" in clean:
                result.total += 1

                # Check if it's skipped
                if "SKIPPED" in clean.upper():
                    result.skipped += 1
                    # Check if it's a known issue
                    if "KNOWNISSUE" in clean.upper():
                        result.known_issue += 1
                        # Try to extract test name
                        test_match = re.search(r"([^\s]+::[^\s\[\]]+(?:\[[^\]]+\])?)", clean)
                        if test_match:
                            test_name = test_match.group(1)
                            if test_name not in result.known_issue_tests:
                                result.known_issue_tests.append(test_name)

        # Adjust skipped count (known_issue is part of skipped)
        result.skipped = max(0, result.skipped - result.known_issue)

        return result if result.total > 0 else None

    except (subprocess.TimeoutExpired, Exception) as e:
        # Collection failed, return None (don't fail the test)
        return None


def _parse_pytest_output(output: str) -> Optional[PytestResult]:
    """Parse pytest output to extract test statistics."""
    lines = output.split("\n")
    result = PytestResult()

    # Track test names we've seen to avoid double counting
    seen_tests = set()

    # Look for summary line (usually at the end)
    # Format: "X passed, Y failed, Z skipped in W.WWs" or "X passed in Y.YYs"
    # Note: We parse summary line first, but will adjust skipped count after parsing individual tests
    # to exclude KNOWNISSUE tests from skipped count
    summary_found = False
    summary_skipped_count = 0
    for line in reversed(lines):  # Start from the end
        clean = _strip_ansi(line).strip()
        if not clean:
            continue

        # Try to match pytest summary pattern
        matches = PYTEST_SUMMARY_RE.findall(clean)
        if matches:
            summary_found = True
            for count_str, status in matches:
                try:
                    count = int(count_str)
                    status_lower = status.lower()
                    if "pass" in status_lower:
                        result.passed = count
                    elif "fail" in status_lower:
                        result.failed = count
                    elif "skip" in status_lower:
                        # Store skipped count from summary, will adjust after parsing individual tests
                        summary_skipped_count = count
                    elif "error" in status_lower:
                        result.error = count
                except ValueError:
                    continue
            break

    # Always parse individual test results to get failed test names
    # This is important even if summary was found, because we need the actual test IDs
    # Parse individual test results to get test names and status
    # This is done regardless of whether summary was found, to capture failed test IDs
    for i, line in enumerate(lines):
        clean = _strip_ansi(line).strip()
        if not clean:
            continue

        # Try to match pytest status pattern (test_name STATUS format)
        match = PYTEST_STATUS_RE.match(clean)
        if match:
            status = match.group(2).upper()
            test_name = match.group(1).strip()
            if test_name not in seen_tests:
                seen_tests.add(test_name)
                if status == "PASSED":
                    result.passed += 1
                elif status == "FAILED":
                    result.failed += 1
                    if test_name not in result.failed_tests:
                        result.failed_tests.append(test_name)
                elif status == "SKIPPED":
                    # IMPORTANT: Check if this test was already marked as PASSED
                    # If so, don't count it as KNOWNISSUE or SKIPPED
                    is_passed = False
                    if test_name:
                        # Check if we've seen this test as PASSED in nearby lines
                        for j in range(max(0, i - 30), min(i + 5, len(lines))):
                            check_line = _strip_ansi(lines[j]).strip()
                            if "PASSED" in check_line and test_name in check_line:
                                is_passed = True
                                break

                    if not is_passed:
                        # Check if this skip is due to KNOWNISSUE
                        # Look back a few lines to find KNOWNISSUE message
                        is_known_issue = False
                        for j in range(max(0, i - 10), i + 1):
                            check_line = _strip_ansi(lines[j]).strip()
                            if "KNOWNISSUE" in check_line.upper():
                                is_known_issue = True
                                break
                        if is_known_issue:
                            result.known_issue += 1
                            if test_name not in result.known_issue_tests:
                                result.known_issue_tests.append(test_name)
                            # Don't count as skipped for known issues
                        else:
                            result.skipped += 1
                elif status == "ERROR":
                    result.error += 1
                    if test_name not in result.failed_tests:
                        result.failed_tests.append(test_name)
        else:
            # Check for pytest test name pattern (tests/path/to/test.py::test_name[...])
            # and look for PASSED/FAILED in the same line or next line
            if "::" in clean and ("test_" in clean or "/test_" in clean or "tests/" in clean):
                # This looks like a test name line
                # Match full test ID including parameters: tests/path/to/test.py::test_name[param1-param2]
                # Handle multi-line test names by looking ahead for continuation
                test_name_parts = [clean]
                # Look ahead for continuation lines (test name might be split across lines)
                j = i + 1
                bracket_count = clean.count("[") - clean.count("]")  # Track bracket balance
                while j < len(lines) and j < i + 15:  # Check up to 15 lines ahead for long test names
                    next_clean = _strip_ansi(lines[j]).strip()
                    if not next_clean:
                        j += 1
                        continue
                    # If next line looks like continuation (has brackets or parameters but no status)
                    if not re.match(r"^(PASSED|FAILED|SKIPPED|ERROR)", next_clean):
                        # Check if it's part of the test name
                        # Continue if: has brackets/parameters, or is clearly part of test name
                        if (
                            "[" in next_clean or "]" in next_clean or "-" in next_clean or "=" in next_clean
                        ) and "::" not in next_clean:
                            test_name_parts.append(next_clean)
                            bracket_count += next_clean.count("[") - next_clean.count("]")
                            # Stop if brackets are balanced
                            if bracket_count == 0:
                                break
                            j += 1
                        elif "::" in next_clean and ("tests/" in next_clean or "test_" in next_clean):
                            # Another test name starts, stop here
                            break
                        else:
                            # Might be continuation, check if brackets are still open
                            if bracket_count > 0:
                                test_name_parts.append(next_clean)
                                j += 1
                            else:
                                break
                    else:
                        # Status line found, stop
                        break

                # Combine test name parts
                combined_test_name = " ".join(test_name_parts)
                # Match full test ID including parameters: tests/path/to/test.py::test_name[param1-param2]
                # Handle nested brackets and long parameter lists by matching balanced brackets
                # Pattern: tests/path/to/test.py::test_name[...]
                # First extract the base path and test name
                base_match = re.search(r"(tests/[^\s]+::[^\s\[\]]+)", combined_test_name)
                if base_match:
                    base_test_id = base_match.group(1)
                    # Now find the opening bracket after the test name
                    remaining = combined_test_name[len(base_test_id) :]
                    if remaining.strip().startswith("["):
                        # Find matching closing bracket (handle nested brackets)
                        bracket_count = 0
                        end_pos = -1
                        for idx, char in enumerate(remaining):
                            if char == "[":
                                bracket_count += 1
                            elif char == "]":
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_pos = idx + 1
                                    break
                        if end_pos > 0:
                            # Complete test ID with parameters
                            test_name = base_test_id + remaining[:end_pos]
                        else:
                            # No closing bracket found, use base
                            test_name = base_test_id
                    else:
                        # No brackets, use base
                        test_name = base_test_id
                    test_name_match = True
                else:
                    test_name_match = None
                if test_name_match:
                    # Check current line and next few lines for status
                    # Look further ahead since test name might span multiple lines
                    status_found = False
                    status = None
                    # Check from current line to several lines ahead (accounting for multi-line test names)
                    check_start = i
                    check_end = min(i + 10, len(lines))  # Check up to 10 lines ahead
                    for k in range(check_start, check_end):
                        check_line = _strip_ansi(lines[k]).strip()
                        if "PASSED" in check_line:
                            status = "PASSED"
                            status_found = True
                            break
                        elif "FAILED" in check_line:
                            status = "FAILED"
                            status_found = True
                            break
                        elif "SKIPPED" in check_line:
                            # Check if this skip is due to KNOWNISSUE
                            # Look back a few lines to find KNOWNISSUE message
                            is_known_issue = False
                            for m in range(max(0, k - 10), k + 1):
                                check_prev_line = _strip_ansi(lines[m]).strip()
                                if "KNOWNISSUE" in check_prev_line.upper():
                                    is_known_issue = True
                                    status = "KNOWNISSUE"
                                    break
                            if not is_known_issue:
                                status = "SKIPPED"
                            status_found = True
                            break
                        elif "ERROR" in check_line:
                            status = "ERROR"
                            status_found = True
                            break

                    if status_found:
                        if test_name not in seen_tests:
                            seen_tests.add(test_name)
                            if status == "PASSED":
                                result.passed += 1
                            elif status == "FAILED":
                                result.failed += 1
                                if test_name not in result.failed_tests:
                                    result.failed_tests.append(test_name)
                            elif status == "SKIPPED":
                                # Check if this test was already marked as PASSED
                                is_passed = False
                                if test_name:
                                    for j in range(max(0, k - 30), min(k + 5, len(lines))):
                                        check_line = _strip_ansi(lines[j]).strip()
                                        if "PASSED" in check_line and test_name in check_line:
                                            is_passed = True
                                            break
                                if not is_passed:
                                    result.skipped += 1
                            elif status == "KNOWNISSUE":
                                # Check if this test was already marked as PASSED
                                is_passed = False
                                if test_name:
                                    for j in range(max(0, k - 30), min(k + 5, len(lines))):
                                        check_line = _strip_ansi(lines[j]).strip()
                                        if "PASSED" in check_line and test_name in check_line:
                                            is_passed = True
                                            break
                                if not is_passed:
                                    result.known_issue += 1
                                    if test_name not in result.known_issue_tests:
                                        result.known_issue_tests.append(test_name)
                            elif status == "ERROR":
                                result.error += 1
                                if test_name not in result.failed_tests:
                                    result.failed_tests.append(test_name)
            else:
                # Check for standalone status lines like "PASSED2025-11-11..." or "PASSED"
                # These appear after test name lines
                # Try to extract test name from previous lines to avoid duplicate counting
                # Look back further to find the actual test name with full parameters
                test_name_from_prev = None
                most_recent_test_name = None
                most_recent_test_passed = False

                for j in range(max(0, i - 30), i):
                    prev_line = _strip_ansi(lines[j]).strip()

                    # Track test names as we go
                    if "::" in prev_line and ("test_" in prev_line or "/test_" in prev_line or "tests/" in prev_line):
                        # Match full test ID including parameters: tests/path/to/test.py::test_name[param1-param2]
                        # Try to match complete test name with all parameters
                        test_name_match = re.search(r"(tests/[^\s]+::[^\s\[\]]+(?:\[[^\]]+\])?)", prev_line)
                        if test_name_match:
                            most_recent_test_name = test_name_match.group(1)
                            # Check if this line also shows PASSED
                            if "PASSED" in prev_line:
                                most_recent_test_passed = True
                            else:
                                most_recent_test_passed = False

                    # If we see a standalone PASSED line with the most recent test name, mark it as passed
                    elif "PASSED" in prev_line and most_recent_test_name:
                        if most_recent_test_name in prev_line:
                            most_recent_test_passed = True

                # Only use test name if it hasn't passed
                if most_recent_test_name and not most_recent_test_passed:
                    test_name_from_prev = most_recent_test_name

                # Only count if we haven't seen this test before
                should_count = test_name_from_prev is None or test_name_from_prev not in seen_tests

                if re.match(r"^PASSED", clean):
                    # Standalone PASSED line (from previous test)
                    if should_count:
                        seen_tests.add(test_name_from_prev) if test_name_from_prev else None
                        result.passed += 1
                elif re.match(r"^FAILED", clean):
                    if should_count:
                        seen_tests.add(test_name_from_prev) if test_name_from_prev else None
                        result.failed += 1
                        if test_name_from_prev and test_name_from_prev not in result.failed_tests:
                            result.failed_tests.append(test_name_from_prev)
                elif re.match(r"^SKIPPED", clean) or re.search(r"\bSKIPPED\b", clean):
                    # Check if this skip is due to KNOWNISSUE
                    is_known_issue = "KNOWNISSUE" in clean.upper()
                    if not is_known_issue:
                        for j in range(max(0, i - 10), i + 1):
                            check_line = _strip_ansi(lines[j]).strip()
                            if "KNOWNISSUE" in check_line.upper():
                                is_known_issue = True
                                break

                    # Try to extract test name from current line
                    # Format 1: "SKIPPED [1] tests/path/to/test.py::test_name[...]: KNOWNISSUE: ..."
                    # Format 2: "SKIPPED [1] tests/path/to/test.py:130: KNOWNISSUE: ..."
                    # Format 3: "SKIPPED (KNOWNISSUE: ...)" - need to find test name from previous lines
                    test_name_from_line = None

                    # First try to find test name with :: (full test ID)
                    if "::" in clean:
                        test_name_match = re.search(r"(tests/[^\s]+::[^\s\[\]]+(?:\[[^\]]+\])?)", clean)
                        if test_name_match:
                            test_name_from_line = test_name_match.group(1)

                    # If not found and this is "SKIPPED (KNOWNISSUE: ...)" format, look at immediate previous line
                    if not test_name_from_line and re.match(r"^SKIPPED\s*\(", clean):
                        # Look at the immediate previous line for test name
                        if i > 0:
                            prev_line = _strip_ansi(lines[i - 1]).strip()
                            if "::" in prev_line and ("test_" in prev_line or "tests/" in prev_line):
                                test_name_match = re.search(r"(tests/[^\s]+::[^\s\[\]]+(?:\[[^\]]+\])?)", prev_line)
                                if test_name_match:
                                    test_name_from_line = test_name_match.group(1)

                    # If not found, try to extract from "SKIPPED [1] tests/path/to/test.py:130:" format
                    # and look for the actual test name in previous lines
                    if not test_name_from_line and re.match(r"^SKIPPED\s+\[", clean):
                        # Extract file path from "SKIPPED [1] tests/path/to/test.py:130:"
                        file_match = re.search(r"(tests/[^\s]+\.py)", clean)
                        if file_match:
                            file_path = file_match.group(1)
                            # Look back further to find the actual test name with parameters
                            # But make sure we don't pick up a PASSED test name
                            for j in range(max(0, i - 20), i):
                                prev_line = _strip_ansi(lines[j]).strip()
                                # Skip if this line shows PASSED for the test
                                if "PASSED" in prev_line and file_path in prev_line:
                                    continue
                                if file_path in prev_line and "::" in prev_line:
                                    # Found a line with the file path and ::, extract full test name
                                    test_name_match = re.search(r"(tests/[^\s]+::[^\s\[\]]+(?:\[[^\]]+\])?)", prev_line)
                                    if test_name_match:
                                        test_name_from_line = test_name_match.group(1)
                                        break

                    # Use test name from line if available, otherwise use from previous lines
                    final_test_name = test_name_from_line or test_name_from_prev

                    # IMPORTANT: Check if this test was already marked as PASSED
                    # If so, don't count it as KNOWNISSUE or SKIPPED
                    is_passed = False
                    if final_test_name:
                        # Check if we've seen this test as PASSED in nearby lines
                        for j in range(max(0, i - 30), min(i + 5, len(lines))):
                            check_line = _strip_ansi(lines[j]).strip()
                            if "PASSED" in check_line and final_test_name in check_line:
                                is_passed = True
                                break

                    should_count_final = final_test_name is None or final_test_name not in seen_tests

                    if should_count_final and not is_passed:
                        if final_test_name:
                            seen_tests.add(final_test_name)
                        if is_known_issue:
                            result.known_issue += 1
                            if final_test_name and final_test_name not in result.known_issue_tests:
                                result.known_issue_tests.append(final_test_name)
                        else:
                            result.skipped += 1
                elif re.match(r"^ERROR", clean) or re.search(r"\bERROR\b", clean):
                    if should_count:
                        seen_tests.add(test_name_from_prev) if test_name_from_prev else None
                        result.error += 1
                        if test_name_from_prev and test_name_from_prev not in result.failed_tests:
                            result.failed_tests.append(test_name_from_prev)
                # Also check for status in the middle of line (only if not already matched above)
                elif re.search(r"\bPASSED\b", clean) and not re.match(r"^PASSED", clean):
                    if should_count:
                        seen_tests.add(test_name_from_prev) if test_name_from_prev else None
                        result.passed += 1
                elif re.search(r"\bFAILED\b", clean) and not re.match(r"^FAILED", clean):
                    if should_count:
                        seen_tests.add(test_name_from_prev) if test_name_from_prev else None
                        result.failed += 1
                        if test_name_from_prev and test_name_from_prev not in result.failed_tests:
                            result.failed_tests.append(test_name_from_prev)

    # Use known_issue_tests list length as the source of truth for known_issue count
    # Remove duplicates from known_issue_tests first
    unique_known_issue_tests = []
    seen_known_issue = set()
    for test_name in result.known_issue_tests:
        test_name_clean = " ".join(test_name.split())
        if test_name_clean not in seen_known_issue:
            unique_known_issue_tests.append(test_name_clean)
            seen_known_issue.add(test_name_clean)
    result.known_issue_tests = unique_known_issue_tests
    result.known_issue = len(unique_known_issue_tests)

    # Adjust skipped count: subtract known_issue from skipped
    # This is because KNOWNISSUE tests are reported as skipped by pytest,
    # but we want to track them separately
    # If we parsed individual tests, use that count (which already excludes KNOWNISSUE)
    # Otherwise, if we only have summary line, subtract known_issue from summary skipped count
    if result.skipped == 0 and summary_skipped_count > 0:
        # We didn't parse individual skipped tests, so use summary count
        # But subtract known_issue if we found any
        result.skipped = max(0, summary_skipped_count - result.known_issue)
    elif result.known_issue > 0 and result.skipped > 0:
        # We parsed individual tests, but summary line might have included KNOWNISSUE
        # Individual parsing should have already excluded KNOWNISSUE, but double-check
        # If skipped count seems too high compared to what we parsed, adjust it
        pass  # Individual parsing already handled it correctly

    result.total = result.passed + result.failed + result.skipped + result.error + result.known_issue

    # Return result even if total is 0, as long as we found something
    # This helps with manual_pass/interrupted cases where summary might be missing
    if result.total > 0:
        return result

    # If no summary found and no individual tests counted, return None
    # But if we're in a manual_pass/interrupted scenario, we might want to return empty result
    return None


def _format_pytest_stats(pytest_result: PytestResult) -> str:
    """Format pytest statistics as a compact string."""
    parts = []
    if pytest_result.total > 0:
        parts.append(f"total: {pytest_result.total}")
    if pytest_result.passed > 0:
        parts.append(f"passed: {pytest_result.passed}")
    if pytest_result.failed > 0:
        parts.append(f"failed: {pytest_result.failed}")
    if pytest_result.error > 0:
        parts.append(f"error: {pytest_result.error}")
    if pytest_result.skipped > 0:
        parts.append(f"skipped: {pytest_result.skipped}")
    if pytest_result.known_issue > 0:
        parts.append(f"known_issue: {pytest_result.known_issue}")
    return f"pytest({', '.join(parts)})" if parts else ""


def _format_gtest_stats(gtest_result: GTestResult) -> str:
    """Format gtest statistics as a compact string."""
    parts = []
    # For interrupted tests, show total as "?" but still show failed if any
    if gtest_result.total == -1:
        parts.append("total: ?")
        if gtest_result.passed > 0:
            parts.append(f"passed: {gtest_result.passed}")
        if gtest_result.failed > 0:
            parts.append(f"failed: {gtest_result.failed}")
        if gtest_result.skipped > 0:
            parts.append(f"skipped: {gtest_result.skipped}")
    else:
        if gtest_result.total > 0:
            parts.append(f"total: {gtest_result.total}")
        if gtest_result.passed > 0:
            parts.append(f"passed: {gtest_result.passed}")
        if gtest_result.failed > 0:
            parts.append(f"failed: {gtest_result.failed}")
        if gtest_result.skipped > 0:
            parts.append(f"skipped: {gtest_result.skipped}")
    return f"gtest({', '.join(parts)})" if parts else ""


def _format_pytest_summary(test: TestCase, pytest_result: PytestResult) -> str:
    """Format pytest summary with failed test re-run commands."""
    # First, filter and deduplicate known_issue_tests to get accurate count
    # This ensures known_issue count matches the actual list length
    unique_known_issue_tests_for_count = []
    if pytest_result.known_issue_tests:
        # Remove duplicates and incomplete test names (those that are prefixes of others)
        seen_full_names = set()

        # First, collect complete test names (with full parameters)
        complete_names = []
        incomplete_names = []

        for test_name in pytest_result.known_issue_tests:
            test_name_clean = " ".join(test_name.split())
            # Check if this is a complete test name (has closing bracket or no brackets)
            if "[" in test_name_clean:
                # Check if it has a closing bracket
                if "]" in test_name_clean:
                    # Count brackets to ensure it's balanced
                    open_count = test_name_clean.count("[")
                    close_count = test_name_clean.count("]")
                    if open_count == close_count:
                        complete_names.append(test_name_clean)
                    else:
                        incomplete_names.append(test_name_clean)
                else:
                    incomplete_names.append(test_name_clean)
            else:
                complete_names.append(test_name_clean)

        # Use complete names first, then check if incomplete names are actually different
        for test_name in complete_names:
            if test_name not in seen_full_names:
                unique_known_issue_tests_for_count.append(test_name)
                seen_full_names.add(test_name)

        # For incomplete names, check if they're not already covered by complete names
        for test_name in incomplete_names:
            # Check if this incomplete name is a prefix of any complete name
            is_covered = False
            for complete_name in complete_names:
                if complete_name.startswith(test_name):
                    is_covered = True
                    break
            if not is_covered and test_name not in seen_full_names:
                unique_known_issue_tests_for_count.append(test_name)
                seen_full_names.add(test_name)

        # Update known_issue count to match the actual unique list length
        pytest_result.known_issue = len(unique_known_issue_tests_for_count)

    lines = [
        f"Pytest Summary for: {test.name}",
        f"Command: {test.command}",
        "",
        "=" * 80,
        "Test Statistics:",
        f"  Total:  {pytest_result.total}",
        f"  PASSED: {pytest_result.passed}",
        f"  FAILED: {pytest_result.failed}",
    ]
    if pytest_result.error > 0:
        lines.append(f"  ERROR:  {pytest_result.error}")
    if pytest_result.skipped > 0:
        lines.append(f"  SKIPPED: {pytest_result.skipped}")
    if pytest_result.known_issue > 0:
        lines.append(f"  KNOWNISSUE: {pytest_result.known_issue}")
    lines.extend(
        [
            "=" * 80,
            "",
        ]
    )

    # Show failed/error tests if any
    has_failures = pytest_result.failed_tests or pytest_result.failed > 0 or pytest_result.error > 0
    if has_failures:
        lines.append("Failed tests:")
        # Show individual failed tests if we have them
        if pytest_result.failed_tests:
            for test_name in pytest_result.failed_tests:
                # Clean up test name (remove newlines, extra spaces)
                test_name_clean = " ".join(test_name.split())
                # Create pytest filter command
                # For very long test names (especially with long parameter lists), use -k filter instead
                # Extract just the test function name for -k filter
                if "::" in test_name_clean:
                    # Extract test function name (without parameters)
                    test_func_match = re.search(r"::([^\[]+)", test_name_clean)
                    if test_func_match:
                        test_func_name = test_func_match.group(1)
                        # Use -k filter for readability and to avoid command length issues
                        # This is more practical for very long parameter lists
                        base_cmd = test.command
                        if "pytest" in base_cmd:
                            base_cmd_clean = re.sub(r"\s+--lf\b|\s+--last-failed\b", "", base_cmd)
                            # Extract just the file path part
                            file_match = re.search(r"(tests/[^\s]+)", test_name_clean)
                            if file_match:
                                file_path = file_match.group(1)
                                filter_cmd = f"{base_cmd_clean} -k '{test_func_name}'"
                            else:
                                filter_cmd = f"{base_cmd_clean} -k '{test_func_name}'"
                        else:
                            filter_cmd = f"pytest -k '{test_func_name}'"
                    else:
                        # Fallback: use full test name with quotes
                        filter_cmd = f'pytest "{test_name_clean}"'
                else:
                    filter_cmd = f"pytest -k '{test_name_clean}'"
                lines.append(f"  - {test_name_clean}")
                lines.append(f"    Re-run: {filter_cmd}")
                # Also show note about --lf option for convenience
                if len(test_name_clean) > 200:  # Very long test name
                    base_cmd = test.command
                    if "pytest" in base_cmd:
                        base_cmd_clean = re.sub(r"\s+--lf\b|\s+--last-failed\b", "", base_cmd)
                        lines.append(f"    (or use: {base_cmd_clean} --lf to run all failed tests)")
        # If we have failures/errors but no specific test names, use --lf
        elif pytest_result.failed > 0 or pytest_result.error > 0:
            lines.append(f"  - {pytest_result.failed + pytest_result.error} test(s) failed/errored")
            # Use --lf option to rerun failed tests
            base_cmd = test.command
            if "pytest" in base_cmd:
                base_cmd_clean = re.sub(r"\s+--lf\b|\s+--last-failed\b", "", base_cmd)
                filter_cmd = f"{base_cmd_clean} --lf"
            else:
                filter_cmd = f"pytest --lf"
            lines.append(f"    Re-run: {filter_cmd}")
        lines.append("")

        # Combined filter for all failed tests
        # Use --lf (last-failed) option for pytest - simplest way to rerun only failed tests
        # Extract base pytest command from test.command
        base_cmd = test.command
        if "pytest" in base_cmd:
            # Remove any existing --lf or --last-failed
            base_cmd_clean = re.sub(r"\s+--lf\b|\s+--last-failed\b", "", base_cmd)
            combined_cmd = f"{base_cmd_clean} --lf"
        else:
            # Fallback: use direct paths or -k filter
            if pytest_result.failed_tests and all("::" in t for t in pytest_result.failed_tests):
                combined_cmd = f"pytest {' '.join(pytest_result.failed_tests)}"
            elif pytest_result.failed_tests:
                combined_filter = " or ".join([t if "::" not in t else f"'{t}'" for t in pytest_result.failed_tests])
                combined_cmd = f"pytest -k '{combined_filter}'"
            else:
                combined_cmd = f"pytest --lf"
        lines.append(f"Re-run all failed tests:")
        lines.append(f"  {combined_cmd}")
        lines.append("")

    # Show known issue tests if any
    # Filter out duplicate/incomplete test names
    if pytest_result.known_issue_tests:
        # Remove duplicates and incomplete test names (those that are prefixes of others)
        unique_known_issue_tests = []
        seen_full_names = set()

        # First, collect complete test names (with full parameters)
        complete_names = []
        incomplete_names = []

        for test_name in pytest_result.known_issue_tests:
            test_name_clean = " ".join(test_name.split())
            # Check if this is a complete test name (has closing bracket or no brackets)
            if "[" in test_name_clean:
                # Check if it has a closing bracket
                if "]" in test_name_clean:
                    # Count brackets to ensure it's balanced
                    open_count = test_name_clean.count("[")
                    close_count = test_name_clean.count("]")
                    if open_count == close_count:
                        complete_names.append(test_name_clean)
                    else:
                        incomplete_names.append(test_name_clean)
                else:
                    incomplete_names.append(test_name_clean)
            else:
                complete_names.append(test_name_clean)

        # Use complete names first, then check if incomplete names are actually different
        for test_name in complete_names:
            if test_name not in seen_full_names:
                unique_known_issue_tests.append(test_name)
                seen_full_names.add(test_name)

        # For incomplete names, check if they're not already covered by complete names
        for test_name in incomplete_names:
            # Check if this incomplete name is a prefix of any complete name
            is_covered = False
            for complete_name in complete_names:
                if complete_name.startswith(test_name):
                    is_covered = True
                    break
            if not is_covered and test_name not in seen_full_names:
                unique_known_issue_tests.append(test_name)
                seen_full_names.add(test_name)

        # Update known_issue count to match the actual unique list length
        # This ensures the count matches what's actually displayed
        pytest_result.known_issue = len(unique_known_issue_tests)

        if unique_known_issue_tests:
            lines.append("Known Issue tests:")
            for test_name in unique_known_issue_tests:
                lines.append(f"  - {test_name} (KNOWNISSUE - skipped)")
            lines.append("")

    return "\n".join(lines) + "\n"


def _format_unittest_summary(
    test: TestCase, gtest_result: Optional[GTestResult] = None, pytest_result: Optional[PytestResult] = None
) -> str:
    """Format unified unittest summary (gtest or pytest) with failed test re-run commands."""
    if gtest_result:
        return _format_gtest_summary(test, gtest_result)
    elif pytest_result:
        return _format_pytest_summary(test, pytest_result)
    return ""


def _format_gtest_summary(test: TestCase, gtest_result: GTestResult) -> str:
    lines = [
        f"GTest Summary for: {test.name}",
        f"Command: {test.command}",
        "",
        "=" * 80,
        "Test Statistics:",
    ]
    # For interrupted tests, show total as "?" but still show failed if any
    if gtest_result.total == -1:
        lines.append("  Total:  ? (interrupted)")
        lines.append(f"  PASSED: {gtest_result.passed}")
        if gtest_result.failed > 0:
            lines.append(f"  FAILED: {gtest_result.failed}")
        lines.append(f"  SKIPPED: {gtest_result.skipped}")
    else:
        lines.append(f"  Total:  {gtest_result.total}")
        lines.append(f"  PASSED: {gtest_result.passed}")
        lines.append(f"  FAILED: {gtest_result.failed}")
        lines.append(f"  SKIPPED: {gtest_result.skipped}")
    lines.extend(
        [
            "=" * 80,
            "",
        ]
    )

    # Show failed tests if any (including interrupted tests)
    if gtest_result.failed_tests:
        lines.append("Failed tests:")
        for test_name in gtest_result.failed_tests:
            base_cmd = test.command.split()[0] if test.command.split() else test.command
            filter_cmd = f"{base_cmd} --gtest_filter={test_name}"
            lines.append(f"  - {test_name}")
            lines.append(f"    Re-run: {filter_cmd}")
        lines.append("")

        # Combined filter for all failed tests
        combined_filter = ":".join(gtest_result.failed_tests)
        base_cmd = test.command.split()[0] if test.command.split() else test.command
        combined_cmd = f"{base_cmd} --gtest_filter={combined_filter}"
        lines.append(f"Re-run all failed tests:")
        lines.append(f"  {combined_cmd}")
        lines.append("")

    if gtest_result.skipped_tests:
        lines.append("Skipped tests:")
        for test_name in gtest_result.skipped_tests:
            # Filter out invalid entries like "15 tests, listed below:"
            if "tests" in test_name.lower() and "listed" in test_name.lower():
                continue
            lines.append(f"  - {test_name}")
        lines.append("")

    # Ensure summary ends with a newline
    return "\n".join(lines) + "\n"


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def _extract_progress_line(line: str) -> Optional[str]:
    clean = _strip_ansi(line).replace("\r", "").strip()
    if not clean:
        return None
    # pytest style (module::test)
    if "::" in clean:
        lower = clean.lower()
        if lower.startswith("collected ") or lower.startswith("= "):
            return None
        if PYTEST_STATUS_RE.match(clean):
            return clean
        return clean
    # gtest style ([ RUN ] ...)
    if GTEST_STATUS_RE.match(clean):
        return clean
    return None


FAIL_PATTERNS = [
    r"\bFAIL\b",
    r"\bFAILED\b",
    r"Traceback \(most recent call last\)",
    r"\bERROR\b",
]


@dataclass
class TestCase:
    name: str
    command: str
    env: Dict[str, str] = field(default_factory=dict)
    fail_patterns: Sequence[str] = field(default_factory=list)
    success_patterns: Sequence[str] = field(default_factory=list)
    hang_timeout: float = DEFAULT_HANG_TIMEOUT
    max_execution_time: Optional[float] = DEFAULT_MAX_EXECUTION_TIME


pytest_success = [r"\d+\s+passed"]

TEST_CASES: Dict[str, TestCase] = {
    "yolov8s-trace-320": TestCase(
        name="yolov8s-trace-320",
        command=("source env_set.sh && " "python ./models/bos_model/yolov8s/run_yolov8s.py --trace -i 320 -n 22"),
        fail_patterns=[r"\bFAIL\b", r"\bERROR\b"],
    ),
    "yolov8s-trace-256": TestCase(
        name="yolov8s-trace-256",
        command=("source env_set.sh && " "python ./models/bos_model/yolov8s/run_yolov8s.py --trace -i 256 -n 22"),
        fail_patterns=[r"\bFAIL\b", r"\bERROR\b"],
    ),
    "qwen25-vl-3b-batch2-trace": TestCase(
        name="qwen25-vl-3b-batch2-trace",
        command=(
            "source env_set.sh && "
            "HF_MODEL=Qwen/Qwen2.5-VL-3B-Instruct "
            "pytest models/bos_model/qwen25_vl/demo/vision_demo.py -k 'batch2-trace'"
        ),
        success_patterns=pytest_success,
    ),
    "qwen25-vl-7b-batch2-trace": TestCase(
        name="qwen25-vl-7b-batch2-trace",
        command=(
            "source env_set.sh && "
            "HF_MODEL=Qwen/Qwen2.5-VL-7B-Instruct "
            "pytest models/bos_model/qwen25_vl/demo/vision_demo.py -k 'batch2-trace'"
        ),
        success_patterns=pytest_success,
    ),
    "llama32-1b-performance-batch-one": TestCase(
        name="llama32-1b-performance-batch-one",
        command=(
            "source env_set.sh && "
            "HF_MODEL=meta-llama/Llama-3.2-1B-Instruct "
            "pytest models/bos_model/llama32/demo/demo.py -k 'performance and batch-one'"
        ),
        success_patterns=pytest_success,
    ),
    "llama32-3b-performance-batch-one": TestCase(
        name="llama32-3b-performance-batch-one",
        command=(
            "source env_set.sh && "
            "HF_MODEL=meta-llama/Llama-3.2-3B-Instruct "
            "pytest models/bos_model/llama32/demo/demo.py -k 'performance and batch-one'"
        ),
        success_patterns=pytest_success,
    ),
    "ssr-demo-performance-1": TestCase(
        name="ssr-demo-performance-1",
        command="source env_set.sh ssr && bash $WORKING_DIR/scripts/run_ssr_demo.sh performance",
    ),
    "unit_tests_device": TestCase(
        name="unit_tests_device",
        command="./build/test/tt_metal/unit_tests_device",
        env={"TT_METAL_SLOW_DISPATCH_MODE": "1"},
    ),
    "unit_tests_api": TestCase(
        name="unit_tests_api",
        command="./build/test/tt_metal/unit_tests_api",
        env={"TT_METAL_SLOW_DISPATCH_MODE": "1"},
    ),
    "unit_tests_llk": TestCase(
        name="unit_tests_llk",
        command="./build/test/tt_metal/unit_tests_llk",
        env={"TT_METAL_SLOW_DISPATCH_MODE": "1"},
    ),
    "unit_tests_debug_tools": TestCase(
        name="unit_tests_debug_tools",
        command="./build/test/tt_metal/unit_tests_debug_tools",
        env={"TT_METAL_SLOW_DISPATCH_MODE": "1"},
    ),
    "run bos tests": TestCase(
        name="run bos tests",
        command="pytest tests/ttnn/bos_tests/",
        success_patterns=pytest_success,
        hang_timeout=300.0,
    ),
    "run bos tests unit operations": TestCase(
        name="run bos tests unit operations",
        command="source env_set.sh ssr && pytest tests/ttnn/unit_tests/operations/bos/",
        success_patterns=pytest_success,
    ),
    "run bos_yolo": TestCase(
        name="run bos_yolo",
        command="pytest models/bos_model/yolov8s/run_yolov8s.py",
        success_patterns=pytest_success,
    ),
    "run setitem": TestCase(
        name="run setitem",
        command="pytest tests/ttnn/unit_tests/operations/bos/test_bos_setitem_device_op_rm.py",
        success_patterns=pytest_success,
    ),
    "run deformable attention": TestCase(
        name="run deformable attention",
        command="source env_set.sh ssr && pytest tests/ttnn/unit_tests/operations/bos/test_bos_deformable_attention.py",
        success_patterns=pytest_success,
    ),
    "run resnet50": TestCase(
        name="run resnet50",
        command=(
            "pytest --disable-warnings "
            "models/bos_model/resnet50/tests/test_ttnn_functional_resnet50.py::test_resnet_50 -k batch_4"
        ),
        success_patterns=pytest_success,
    ),
    "run pdl": TestCase(
        name="run pdl",
        command=("pytest models/bos_model/panoptic_deeplab/tests/pcc/test_tt_model.py::test_model_panoptic_deeplab"),
        success_patterns=pytest_success,
    ),
    "run pdl pipeline e2e": TestCase(
        name="run pdl pipeline e2e",
        command=("pytest models/bos_model/panoptic_deeplab/tests/test_pipeline_e2e.py"),
        success_patterns=pytest_success,
    ),
    "run oft": TestCase(
        name="run oft",
        command="pytest models/bos_model/fastoft/demo/demo.py",
        success_patterns=pytest_success,
        env={"CHECKPOINTS_PATH": "./models/bos_model/fastoft/resources/checkpoint-best-no-dist_01.pth.gz"},
        hang_timeout=60.0,
    ),
    "run oft full performance test": TestCase(
        name="run oft full performance test",
        command="pytest models/bos_model/fastoft/tests/test_device_perf_oft.py::test_device_perf_oft",
        success_patterns=pytest_success,
        env={"CHECKPOINTS_PATH": "./models/bos_model/fastoft/resources/checkpoint-best-no-dist_01.pth.gz"},
        hang_timeout=60.0,
    ),
    "run oft trace_2cq": TestCase(
        name="run oft trace_2cq",
        command="pytest models/bos_model/fastoft/tests/test_perf_e2e_oft.py::test_perf_oft -k oft_trace_2cq",
        success_patterns=pytest_success,
        env={"CHECKPOINTS_PATH": "./models/bos_model/fastoft/resources/checkpoint-best-no-dist_01.pth.gz"},
        hang_timeout=60.0,
    ),
    "run vit": TestCase(
        name="run vit",
        command="pytest models/bos_model/vit/tests/test_vit_layerwise_pcc.py",
        success_patterns=pytest_success,
    ),
}

METAL_UNIT_TEST_KEYS = [
    "unit_tests_device",
    "unit_tests_api",
    "unit_tests_llk",
    "unit_tests_debug_tools",
]

MODEL_OP_UNIT_TEST_KEYS = [
    "run bos tests",
    "run bos tests unit operations",
]

MODEL_TEST_KEYS = [
    "yolov8s-trace-320",
    "yolov8s-trace-256",
    "qwen25-vl-3b-batch2-trace",
    "qwen25-vl-7b-batch2-trace",
    "llama32-1b-performance-batch-one",
    "llama32-3b-performance-batch-one",
    "ssr-demo-performance-1",
    "run bos_yolo",
    "run resnet50",
    "run pdl",
    "run pdl pipeline e2e",
    "run oft",
    "run oft full performance test",
    "run oft trace_2cq",
    "run vit",
]

MENU_STRUCTURE = [
    (
        "===== METAL UNIT TEST ======",
        [
            ("device", [METAL_UNIT_TEST_KEYS[0]]),
            ("api", [METAL_UNIT_TEST_KEYS[1]]),
            ("llk", [METAL_UNIT_TEST_KEYS[2]]),
            ("debug tools", [METAL_UNIT_TEST_KEYS[3]]),
            ("ALL (metal unit)", METAL_UNIT_TEST_KEYS),
        ],
    ),
    (
        "===== MODEL OP UNIT TEST ======",
        [
            ("bos tests (ttnn/bos_tests)", [MODEL_OP_UNIT_TEST_KEYS[0]]),
            ("bos tests (ttnn/unit_tests/operations/bos)", [MODEL_OP_UNIT_TEST_KEYS[1]]),
            ("ALL (model op unit)", MODEL_OP_UNIT_TEST_KEYS),
        ],
    ),
    (
        "===== MODEL ======",
        [
            ("yolov8s trace 320", ["yolov8s-trace-320"]),
            ("yolov8s trace 256", ["yolov8s-trace-256"]),
            ("qwen2.5 VL 3B batch2 trace", ["qwen25-vl-3b-batch2-trace"]),
            ("qwen2.5 VL 7B batch2 trace", ["qwen25-vl-7b-batch2-trace"]),
            ("llama3.2 1B performance batch-one", ["llama32-1b-performance-batch-one"]),
            ("llama3.2 3B performance batch-one", ["llama32-3b-performance-batch-one"]),
            ("ssr demo performance", ["ssr-demo-performance-1"]),
            ("bos yolo pytest", ["run bos_yolo"]),
            ("bos resnet50 pytest", ["run resnet50"]),
            ("panoptic deeplab (pdl)", ["run pdl"]),
            ("panoptic deeplab pipeline e2e", ["run pdl pipeline e2e"]),
            ("oft demo", ["run oft"]),
            ("oft full performance test", ["run oft full performance test"]),
            ("oft trace_2cq", ["run oft trace_2cq"]),
            ("bos vit pytest", ["run vit"]),
            ("ALL (model)", MODEL_TEST_KEYS),
        ],
    ),
    (
        "===== ALL ======",
        [
            ("ALL TESTS", METAL_UNIT_TEST_KEYS + MODEL_OP_UNIT_TEST_KEYS + MODEL_TEST_KEYS),
        ],
    ),
]


def _print_two_column_menu(items: List[Tuple[int, str]], width: int = None) -> None:
    """Print menu items in two columns."""
    if width is None:
        try:
            width = shutil.get_terminal_size().columns
        except Exception:
            width = TERMINAL_WIDTH

    if not items:
        return

    # Calculate column widths
    max_num_width = max(len(str(num)) for num, _ in items)
    max_label_width = max(len(label) for _, label in items)

    # Column width: number + ". " + label + padding
    col_width = max_num_width + 2 + max_label_width + 4
    cols_per_row = max(2, (width - 4) // col_width)  # Leave some margin

    # Print in rows
    for i in range(0, len(items), cols_per_row):
        row_items = items[i : i + cols_per_row]
        row_parts = []
        for num, label in row_items:
            part = f" {num}. {label}".ljust(col_width)
            row_parts.append(part)
        print("".join(row_parts))


def _load_last_unittest_info() -> Optional[Dict[str, any]]:
    """Load last unittest execution info including failed tests (gtest or pytest).
    Finds the most recent unittest execution that has failed tests."""
    unittest_summary_path = ROOT / "unittest_summary.log"
    if not unittest_summary_path.exists():
        # Fallback to old gtest_summary.log for backward compatibility
        unittest_summary_path = ROOT / "gtest_summary.log"
        if not unittest_summary_path.exists():
            return None

    try:
        content = unittest_summary_path.read_text(encoding="utf-8")
        # Parse all unittest summaries
        # Split by separator line (80 equals signs)
        separator = "=" * 80
        # Split summaries by looking for "GTest Summary for:" or "Pytest Summary for:" patterns
        # Each summary starts with one of these patterns
        summaries = []
        current_summary = []
        for line in content.split("\n"):
            if line.startswith("GTest Summary for:") or line.startswith("Pytest Summary for:"):
                # Start a new summary
                if current_summary:
                    summaries.append("\n".join(current_summary))
                current_summary = [line]
            elif current_summary:
                current_summary.append(line)
        # Add the last summary
        if current_summary:
            summaries.append("\n".join(current_summary))

        if not summaries:
            return None

        # Search from the most recent to oldest for a summary with failed tests
        for summary in reversed(summaries):
            summary = summary.strip()
            if not summary:
                continue

            # Extract test name and failed tests
            lines = summary.split("\n")
            test_name = None
            command = None
            test_type = None  # "gtest" or "pytest"
            failed_tests: List[str] = []
            in_failed_section = False

            for line in lines:
                line = line.strip()
                if line.startswith("GTest Summary for:") or line.startswith("Pytest Summary for:"):
                    test_name = line.replace("GTest Summary for:", "").replace("Pytest Summary for:", "").strip()
                    test_type = "gtest" if "GTest" in line else "pytest"
                elif line.startswith("Command:"):
                    command = line.replace("Command:", "").strip()
                elif line == "Failed tests:":
                    in_failed_section = True
                elif in_failed_section:
                    if line.startswith("- "):
                        # Extract test name from "  - TestSuite.TestName" or "  - tests/path/to/test.py::test_name"
                        test_line = line.replace("- ", "").strip()
                        # Skip lines that contain "Re-run:" or are empty
                        if test_line and "Re-run:" not in test_line and not test_line.startswith("("):
                            failed_tests.append(test_line)
                    elif line.startswith("Re-run all failed tests:") or line.startswith("Re-run:"):
                        # Continue reading failed tests, don't stop here
                        pass
                    elif line.startswith("Skipped tests:") or (not line and failed_tests):
                        # Stop if we hit skipped tests section or empty line after failed tests
                        in_failed_section = False
                    elif not line:
                        # Empty line might be separator, but continue if we're still in failed section
                        pass

            # Return the first summary with failed tests (most recent)
            if test_name and failed_tests:
                return {
                    "test_name": test_name,
                    "command": command,
                    "test_type": test_type,
                    "failed_tests": failed_tests,
                }
    except Exception:
        pass

    return None


def _load_recent_failures() -> List[str]:
    recent_path = ROOT / "failures.json"
    if not recent_path.exists():
        return []
    try:
        data = json.loads(recent_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    names: List[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                if item in TEST_CASES:
                    names.append(item)
            elif isinstance(item, dict):
                name = item.get("test")
                if isinstance(name, str) and name in TEST_CASES:
                    names.append(name)
    # deduplicate while preserving order
    seen: set = set()
    deduped: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            deduped.append(name)
    return deduped


def select_tests_from_menu(
    args: argparse.Namespace,
    last_run_info: Optional[Dict[str, str]] = None,
    predefined_selection: Optional[str] = None,
) -> List[TestCase]:
    if predefined_selection is not None:
        recent_failures = _load_recent_failures()
        entries_cli: List[Tuple[int, List[str]]] = []
        # METAL UNIT TEST: 1-9 (9가 all)
        metal_unit_entries = MENU_STRUCTURE[0][1]
        for idx_unit, (_, keys) in enumerate(metal_unit_entries[:-1], start=1):
            entries_cli.append((idx_unit, keys))
        metal_unit_all_label, metal_unit_all_keys = metal_unit_entries[-1]
        entries_cli.append((9, metal_unit_all_keys))
        # MODEL OP UNIT TEST: 10-39 (39가 all, 고정)
        model_op_unit_entries = MENU_STRUCTURE[1][1]
        model_op_unit_start = 10
        for idx, (_, keys) in enumerate(model_op_unit_entries):
            if idx == len(model_op_unit_entries) - 1:
                # 마지막 항목(ALL)은 39로 고정
                entries_cli.append((39, keys))
            else:
                number = model_op_unit_start + idx
                entries_cli.append((number, keys))
        # MODEL: 40-89 (89가 all, 고정)
        model_entries = MENU_STRUCTURE[2][1]
        model_start = 40
        for idx, (_, keys) in enumerate(model_entries):
            if idx == len(model_entries) - 1:
                # 마지막 항목(ALL)은 89로 고정
                entries_cli.append((89, keys))
            else:
                number = model_start + idx
                entries_cli.append((number, keys))

        # RECENT FAILURES: 91~99
        recent_start = 91
        recent_end = 99
        recent_idx = 0
        last_unittest_info = _load_last_unittest_info()
        if last_unittest_info and last_unittest_info["failed_tests"]:
            # Find matching test case key
            # Handle test names like "unit_tests_device::TestSuite.TestName"
            test_name_from_summary = last_unittest_info["test_name"]
            if "::" in test_name_from_summary:
                # Extract base test name (before ::)
                test_name_from_summary = test_name_from_summary.split("::")[0]

            test_key = None
            for key in TEST_CASES.keys():
                if (
                    key == test_name_from_summary
                    or _slugify(key) == _slugify(test_name_from_summary)
                    or key == last_unittest_info["test_name"]
                    or _slugify(key) == _slugify(last_unittest_info["test_name"])
                ):
                    test_key = key
                    break

            if test_key:
                # 91: Retry first failed unittest
                if recent_idx < (recent_end - recent_start + 1):
                    entries_cli.append((recent_start + recent_idx, [f"__unittest_first_failed__{test_key}"]))
                    recent_idx += 1
                # 92: Retry all failed unittest
                if recent_idx < (recent_end - recent_start + 1):
                    entries_cli.append((recent_start + recent_idx, [f"__unittest_failed__{test_key}"]))
                    recent_idx += 1
        # 93: Retry all failed scenarios
        if recent_failures and recent_idx < (recent_end - recent_start + 1):
            entries_cli.append((recent_start + recent_idx, list(recent_failures)))
            recent_idx += 1
        # Individual recent failures (94~)
        if recent_failures:
            for name in recent_failures:
                if recent_idx >= (recent_end - recent_start + 1):
                    break
                entries_cli.append((recent_start + recent_idx, [name]))
                recent_idx += 1
        entries_cli.append((999, METAL_UNIT_TEST_KEYS + MODEL_OP_UNIT_TEST_KEYS + MODEL_TEST_KEYS))

        selected_keys, error = _resolve_selection_expression(predefined_selection, entries_cli)
        if error:
            print(f"Invalid selection '{predefined_selection}': {error}")
            return []

        # Process selected keys and handle gtest failed tests
        tests: List[TestCase] = []
        last_unittest_info = _load_last_unittest_info()
        for key in selected_keys:
            if key.startswith("__unittest_first_failed__"):
                # Handle retry first failed unittest (first failed test only)
                if last_unittest_info and last_unittest_info["failed_tests"]:
                    test_name = key.replace("__unittest_first_failed__", "")
                    base_test = TEST_CASES.get(test_name)
                    if base_test and last_unittest_info["failed_tests"]:
                        # Only the first failed test
                        first_failed = last_unittest_info["failed_tests"][0]
                        test_type = last_unittest_info.get("test_type", "gtest")
                        if test_type == "pytest":
                            # For pytest, if test_name contains ::, use it directly; otherwise use -k filter
                            if "::" in first_failed:
                                filter_cmd = f"pytest {first_failed}"
                            else:
                                filter_cmd = f"pytest -k '{first_failed}'"
                        else:
                            # For gtest, use --gtest_filter
                            base_cmd = base_test.command.split()[0] if base_test.command.split() else base_test.command
                            filter_cmd = f"{base_cmd} --gtest_filter={first_failed}"
                        test_case = TestCase(
                            name=f"{test_name}::{first_failed}",
                            command=filter_cmd,
                            env=base_test.env,
                            fail_patterns=base_test.fail_patterns,
                            success_patterns=base_test.success_patterns,
                            hang_timeout=base_test.hang_timeout,
                            max_execution_time=base_test.max_execution_time,
                        )
                        tests.append(test_case)
            elif key.startswith("__unittest_failed__"):
                # Handle retry all failed unittest
                if last_unittest_info and last_unittest_info["failed_tests"]:
                    test_name = key.replace("__unittest_failed__", "")
                    base_test = TEST_CASES.get(test_name)
                    if base_test:
                        test_type = last_unittest_info.get("test_type", "gtest")
                        if test_type == "pytest":
                            # For pytest, use --lf (last-failed) option to run only failed tests
                            # This is simpler and more reliable than manually filtering
                            base_cmd = base_test.command
                            # Add --lf if not already present
                            if "--lf" not in base_cmd and "--last-failed" not in base_cmd:
                                filter_cmd = f"{base_cmd} --lf"
                            else:
                                filter_cmd = base_cmd
                            test_case = TestCase(
                                name=f"{test_name} (failed only)",
                                command=filter_cmd,
                                env=base_test.env,
                                fail_patterns=base_test.fail_patterns,
                                success_patterns=base_test.success_patterns,
                                hang_timeout=base_test.hang_timeout,
                                max_execution_time=base_test.max_execution_time,
                            )
                            tests.append(test_case)
                        else:
                            # For gtest, create test cases for each failed test
                            for failed_test in last_unittest_info["failed_tests"]:
                                base_cmd = (
                                    base_test.command.split()[0] if base_test.command.split() else base_test.command
                                )
                                filter_cmd = f"{base_cmd} --gtest_filter={failed_test}"
                                test_case = TestCase(
                                    name=f"{test_name}::{failed_test}",
                                    command=filter_cmd,
                                    env=base_test.env,
                                    fail_patterns=base_test.fail_patterns,
                                    success_patterns=base_test.success_patterns,
                                    hang_timeout=base_test.hang_timeout,
                                    max_execution_time=base_test.max_execution_time,
                                )
                                tests.append(test_case)
            elif key in TEST_CASES:
                tests.append(TEST_CASES[key])
        return tests

    if not sys.stdin.isatty():
        return [TEST_CASES[key] for key in TEST_CASES.keys()]

    while True:
        entries: List[Tuple[int, List[str]]] = []
        recent_failures = _load_recent_failures()
        print(
            "\nSelect the test numbers to run. "
            "Separate multiple numbers with commas to execute them sequentially. "
            "To repeat a set, prefix the selection with a multiplier such as '3x'."
        )
        print(
            "Examples:\n"
            "    '1,3,10,40'  -> device → llk → model op unit → model\n"
            "    '40x2'        -> run the 'model' item twice\n"
            "    '2x1,4'       -> api once, debug tools once\n"
            "    '1-12'        -> run options 1 through 12\n"
            "    '1-5,!3'      -> run 1-5 except item 3\n"
        )

        print("\n===== METAL UNIT TEST ======")
        metal_unit_entries = MENU_STRUCTURE[0][1]
        metal_unit_menu_items: List[Tuple[int, str]] = []
        for idx_unit, (label, keys) in enumerate(metal_unit_entries[:-1], start=1):
            metal_unit_menu_items.append((idx_unit, label))
            entries.append((idx_unit, keys))
        metal_unit_all_label, metal_unit_all_keys = metal_unit_entries[-1]
        metal_unit_menu_items.append((9, metal_unit_all_label))
        entries.append((9, metal_unit_all_keys))
        _print_two_column_menu(metal_unit_menu_items)

        print("\n===== MODEL OP UNIT TEST ======")
        model_op_unit_entries = MENU_STRUCTURE[1][1]
        model_op_unit_menu_items: List[Tuple[int, str]] = []
        # MODEL OP UNIT TEST: 10~39 (39가 all, 고정)
        model_op_unit_start = 10
        for idx, (label, keys) in enumerate(model_op_unit_entries):
            if idx == len(model_op_unit_entries) - 1:
                # 마지막 항목(ALL)은 39로 고정
                number = 39
            else:
                number = model_op_unit_start + idx
            model_op_unit_menu_items.append((number, label))
            entries.append((number, keys))
        _print_two_column_menu(model_op_unit_menu_items)

        print("\n===== MODEL ======")
        model_entries = MENU_STRUCTURE[2][1]
        model_menu_items: List[Tuple[int, str]] = []
        # MODEL: 40~89 (89가 all, 고정)
        model_start = 40
        for idx, (label, keys) in enumerate(model_entries):
            if idx == len(model_entries) - 1:
                # 마지막 항목(ALL)은 89로 고정
                number = 89
            else:
                number = model_start + idx
            model_menu_items.append((number, label))
            entries.append((number, keys))
        _print_two_column_menu(model_menu_items)

        # RECENT FAILURES: 91~99 (max 9 items)
        recent_start = 91
        recent_end = 99
        recent_menu_items: List[Tuple[int, str]] = []
        recent_idx = 0

        # Add unittest retry options if available (gtest or pytest)
        last_unittest_info = _load_last_unittest_info()
        if last_unittest_info and last_unittest_info["failed_tests"]:
            # Find matching test case key
            # Handle test names like "unit_tests_device::TestSuite.TestName"
            test_name_from_summary = last_unittest_info["test_name"]
            if "::" in test_name_from_summary:
                # Extract base test name (before ::)
                test_name_from_summary = test_name_from_summary.split("::")[0]

            test_key = None
            for key in TEST_CASES.keys():
                if (
                    key == test_name_from_summary
                    or _slugify(key) == _slugify(test_name_from_summary)
                    or key == last_unittest_info["test_name"]
                    or _slugify(key) == _slugify(last_unittest_info["test_name"])
                ):
                    test_key = key
                    break

            if test_key:
                # 91: Retry first failed unittest
                if recent_idx < (recent_end - recent_start + 1):
                    number = recent_start + recent_idx
                    recent_menu_items.append((number, "Retry first failed unittest"))
                    entries.append((number, [f"__unittest_first_failed__{test_key}"]))
                    recent_idx += 1

                # 92: Retry all failed unittest
                if recent_idx < (recent_end - recent_start + 1):
                    number = recent_start + recent_idx
                    recent_menu_items.append((number, "Retry all failed unittest"))
                    entries.append((number, [f"__unittest_failed__{test_key}"]))
                    recent_idx += 1

                # 93: Run full unittest test scenario (original test)
                if recent_idx < (recent_end - recent_start + 1):
                    number = recent_start + recent_idx
                    test_type = last_unittest_info.get("test_type", "gtest")
                    test_type_label = "gtest" if test_type == "gtest" else "pytest"
                    recent_menu_items.append((number, f"Run full {test_type_label} test scenario"))
                    entries.append((number, [test_key]))
                    recent_idx += 1

        # Retry all failed test scenarios (moved to after unittest options)
        if recent_failures and recent_idx < (recent_end - recent_start + 1):
            number = recent_start + recent_idx
            recent_menu_items.append((number, "Retry all failed scenarios"))
            entries.append((number, list(recent_failures)))
            recent_idx += 1

        # Add individual recent failures (94~)
        if recent_failures:
            for name in recent_failures:
                if recent_idx >= (recent_end - recent_start + 1):
                    break  # Max 9 items
                number = recent_start + recent_idx
                label = f"{name} (last run)"
                recent_menu_items.append((number, label))
                entries.append((number, [name]))
                recent_idx += 1

        if recent_menu_items:
            print("\n===== RECENT FAILURES ======")
            _print_two_column_menu(recent_menu_items)

        if last_run_info:
            print("\n===== LAST RUN ======")
            # Display LAST RUN info - each line on separate line
            print(f" log dir: {last_run_info['run_dir']}")
            print(f" summary: {last_run_info['summary_path']}")
            if "unittest_summary_path" in last_run_info:
                print(f" unittest: {last_run_info['unittest_summary_path']}")
            print(f" tests: {last_run_info['total']} (failures {last_run_info['failures']})")

        print("\n 999. ALL TESTS")
        entries.append((999, METAL_UNIT_TEST_KEYS + MODEL_OP_UNIT_TEST_KEYS + MODEL_TEST_KEYS))

        print("\n 0. Exit\n")

        try:
            choice = input("Select: ").strip()
        except EOFError:
            return []
        except KeyboardInterrupt:
            print("\nSelection cancelled by user.")
            return []

        if not choice:
            print("No selection entered. Please try again.\n")
            continue

        lowered = choice.lower()
        if lowered in {"0", "exit", "quit", "q"}:
            return []

        selected_keys, error = _resolve_selection_expression(choice, entries)
        if error:
            print(f"{error}\n")
            continue

        # Process selected keys and handle gtest failed tests
        tests: List[TestCase] = []
        last_unittest_info = _load_last_unittest_info()
        for key in selected_keys:
            if key.startswith("__unittest_first_failed__"):
                # Handle retry first failed unittest (first failed test only)
                if last_unittest_info and last_unittest_info["failed_tests"]:
                    test_name = key.replace("__unittest_first_failed__", "")
                    base_test = TEST_CASES.get(test_name)
                    if base_test and last_unittest_info["failed_tests"]:
                        # Only the first failed test
                        first_failed = last_unittest_info["failed_tests"][0]
                        test_type = last_unittest_info.get("test_type", "gtest")
                        if test_type == "pytest":
                            # For pytest, if test_name contains ::, use it directly; otherwise use -k filter
                            if "::" in first_failed:
                                filter_cmd = f"pytest {first_failed}"
                            else:
                                filter_cmd = f"pytest -k '{first_failed}'"
                        else:
                            # For gtest, use --gtest_filter
                            base_cmd = base_test.command.split()[0] if base_test.command.split() else base_test.command
                            filter_cmd = f"{base_cmd} --gtest_filter={first_failed}"
                        test_case = TestCase(
                            name=f"{test_name}::{first_failed}",
                            command=filter_cmd,
                            env=base_test.env,
                            fail_patterns=base_test.fail_patterns,
                            success_patterns=base_test.success_patterns,
                            hang_timeout=base_test.hang_timeout,
                            max_execution_time=base_test.max_execution_time,
                        )
                        tests.append(test_case)
            elif key.startswith("__unittest_failed__"):
                # Handle retry all failed unittest
                if last_unittest_info and last_unittest_info["failed_tests"]:
                    test_name = key.replace("__unittest_failed__", "")
                    base_test = TEST_CASES.get(test_name)
                    if base_test:
                        test_type = last_unittest_info.get("test_type", "gtest")
                        if test_type == "pytest":
                            # For pytest, use --lf (last-failed) option to run only failed tests
                            # This is simpler and more reliable than manually filtering
                            base_cmd = base_test.command
                            # Add --lf if not already present
                            if "--lf" not in base_cmd and "--last-failed" not in base_cmd:
                                filter_cmd = f"{base_cmd} --lf"
                            else:
                                filter_cmd = base_cmd
                            test_case = TestCase(
                                name=f"{test_name} (failed only)",
                                command=filter_cmd,
                                env=base_test.env,
                                fail_patterns=base_test.fail_patterns,
                                success_patterns=base_test.success_patterns,
                                hang_timeout=base_test.hang_timeout,
                                max_execution_time=base_test.max_execution_time,
                            )
                            tests.append(test_case)
                        else:
                            # For gtest, create test cases for each failed test
                            for failed_test in last_unittest_info["failed_tests"]:
                                base_cmd = (
                                    base_test.command.split()[0] if base_test.command.split() else base_test.command
                                )
                                filter_cmd = f"{base_cmd} --gtest_filter={failed_test}"
                                test_case = TestCase(
                                    name=f"{test_name}::{failed_test}",
                                    command=filter_cmd,
                                    env=base_test.env,
                                    fail_patterns=base_test.fail_patterns,
                                    success_patterns=base_test.success_patterns,
                                    hang_timeout=base_test.hang_timeout,
                                    max_execution_time=base_test.max_execution_time,
                                )
                                tests.append(test_case)
            elif key in TEST_CASES:
                tests.append(TEST_CASES[key])
        return tests


def _resolve_selection_expression(
    expr: str,
    entries: Sequence[Tuple[int, List[str]]],
) -> Tuple[Optional[List[str]], Optional[str]]:
    expr = expr.strip()
    if not expr:
        return None, "No selection provided."

    multiplier = 1
    match = re.match(r"^\s*(\d+)\s*x\s*(.+)$", expr, re.IGNORECASE)
    if match:
        multiplier = max(1, int(match.group(1)))
        body = match.group(2)
    else:
        body = expr

    tokens = [token.strip() for token in re.split(r"[,\s]+", body) if token.strip()]
    if not tokens:
        return None, "No selection provided."

    numbers_available = {num for num, _ in entries}
    include_numbers: List[int] = []
    exclude_numbers: List[int] = []

    for token in tokens:
        if token.lower() == "all":
            include_numbers.extend(sorted(numbers_available))
            continue

        exclude = False
        if token.startswith("!"):
            exclude = True
            token = token[1:].strip()
            if not token:
                return None, "'!' must be followed by a numeric value or range."

        range_match = re.match(r"^(\d+)-(\d+)$", token)
        resolved: List[int]
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = 1 if end >= start else -1
            resolved = list(range(start, end + step, step))
        elif token.isdigit():
            resolved = [int(token)]
        else:
            return None, "Only numeric values, ranges (e.g. '1-5'), '!N' exclusions, or 'all' are allowed."

        for number in resolved:
            if number not in numbers_available:
                return None, f"Choice out of range: {number}"
            if exclude:
                exclude_numbers.append(number)
            else:
                include_numbers.append(number)

    if not include_numbers:
        return None, "No tests were selected."

    number_to_keys = {num: keys for num, keys in entries}
    selected_keys: List[str] = []
    for num in include_numbers:
        selected_keys.extend(number_to_keys.get(num, []))

    for num in exclude_numbers:
        for key in number_to_keys.get(num, []):
            if key in selected_keys:
                selected_keys.remove(key)

    if not selected_keys:
        return None, "All selected tests were excluded."

    if multiplier > 1:
        selected_keys = selected_keys * multiplier

    return selected_keys, None


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "test"


def _ensure_log_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LOG_ROOT / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_file(path: Path, content: str, git_info: Optional[Dict[str, str]] = None) -> None:
    """Write content to file, optionally prepending git information header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if git_info:
        git_header = _format_git_header(git_info)
        content = git_header + content
    path.write_text(content, encoding="utf-8")


class InputMonitor:
    def __init__(self) -> None:
        self._enabled = sys.stdin.isatty()
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._stop = False
        self._thread: Optional[Thread] = None
        self.manual_pass = False
        self.manual_quit = False
        if self._enabled:
            self._thread = Thread(target=self._worker, daemon=True)
            self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _worker(self) -> None:
        while not self._stop:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            except Exception:
                break
            if not ready:
                continue
            try:
                line = sys.stdin.readline()
            except Exception:
                break
            if line == "":
                self._enabled = False
                break
            if self._stop:
                break
            normalized = line.strip().lower()
            if normalized in {"p", "ㅔ"}:
                self.manual_pass = True
            elif normalized in {"q", "ㅂ"}:
                self.manual_quit = True
            else:
                self._queue.put(line)

    def poll(self) -> Optional[str]:
        if not self._enabled:
            return None
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stop = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.1)


def _colorize_status(status: str, enable: bool) -> str:
    if not enable:
        return status
    mapping = {
        "PASS": "\033[32mPASS\033[0m",
        "PASS_MANUAL": "\033[32mPASS_MANUAL\033[0m",
        "SUCCESS": "\033[32mSUCCESS\033[0m",
        "FAIL": "\033[31mFAIL\033[0m",
        "HANG": "\033[33mHANG\033[0m",
        "INTERRUPTED": "\033[35mINTERRUPTED\033[0m",
        "TIMEOUT": "\033[33mTIMEOUT\033[0m",
    }
    upper = status.upper()
    return mapping.get(upper, status)


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format_row(columns: Sequence[str], sep: str = "|") -> str:
        cells = [f" {col.ljust(widths[idx])} " for idx, col in enumerate(columns)]
        return sep + sep.join(cells) + sep

    def _format_border(char: str = "-") -> str:
        cells = [char * (width + 2) for width in widths]
        return "+" + "+".join(cells) + "+"

    lines = [
        _format_border("-"),
        _format_row(headers),
        _format_border("="),
    ]
    for row in rows:
        lines.append(_format_row(row))
    lines.append(_format_border("-"))
    return "\n".join(lines)


def _detect_failures(output: str, extra_patterns: Iterable[str]) -> Optional[str]:
    patterns = list(FAIL_PATTERNS) + list(extra_patterns)
    for pattern in patterns:
        if re.search(pattern, output):
            return f"Matched failure pattern: {pattern}"
    return None


def _detect_success(output: str, patterns: Iterable[str]) -> Optional[str]:
    for pattern in patterns:
        if re.search(pattern, output):
            return pattern
    return None


def _terminate_process(proc: subprocess.Popen, timeout: float = 2.0) -> None:
    try:
        if proc.poll() is not None:
            return
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_test(
    test: TestCase,
    run_dir: Path,
    progress_callback: Optional[Callable[[float], None]] = None,
    stream_stdout: bool = False,
    allow_user_skip: bool = False,
    full_log_file: Optional[io.TextIOBase] = None,
    input_monitor: Optional[InputMonitor] = None,
) -> Dict[str, str]:
    log_name = f"{_slugify(test.name)}.log"
    log_path = run_dir / log_name
    rel_log_path = str(log_path.relative_to(ROOT))

    command = f"cd {ROOT} && {test.command}"

    env = os.environ.copy()
    env.update(test.env)

    start_time = time.monotonic()
    output_chunks: List[str] = []
    status = "PASS"
    detail = ""
    hang_detected = False
    interrupted = False
    user_skipped = False
    auto_pass = False
    manual_pass = False
    max_exec_time = test.max_execution_time if test.max_execution_time and test.max_execution_time > 0 else None
    last_output_time = start_time
    is_pytest_flag = "pytest" in test.command

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect pytest tests before running (if pytest command)
    # Note: Collection shows function count, but actual execution includes parametrize combinations
    # So collection total may be lower than actual execution total
    collection_stats = None
    if is_pytest_flag:
        collection_stats = _collect_pytest_tests(test.command, env)
        if collection_stats:
            stats_parts = []
            if collection_stats.total > 0:
                stats_parts.append(f"functions: {collection_stats.total}")
            if collection_stats.skipped > 0:
                stats_parts.append(f"skipped: {collection_stats.skipped}")
            if collection_stats.known_issue > 0:
                stats_parts.append(f"known_issue: {collection_stats.known_issue}")
            if stats_parts:
                print(
                    f"    [Collection] pytest({', '.join(stats_parts)}) - Note: actual test count may be higher due to parametrize"
                )

    try:
        proc = subprocess.Popen(
            ["bash", "-lc", command],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        if proc.stdin:
            proc.stdin.close()
    except OSError as exc:
        duration = time.monotonic() - start_time
        duration_str = f"{duration:.1f}"
        error_msg = f"Failed to start process: {exc}"
        _write_file(log_path, error_msg)
        return {
            "Test": test.name,
            "Status": "FAIL",
            "Duration (s)": duration_str,
            "Log": str(log_path.relative_to(ROOT)),
            "Detail": error_msg,
            "Command": test.command,
        }

    assert proc.stdout is not None  # for type checkers

    # Get git info once per test run (cache it)
    if not hasattr(run_test, "_git_info_cache"):
        run_test._git_info_cache = _get_git_info()
    git_info = run_test._git_info_cache

    with log_path.open("w", encoding="utf-8") as log_file:
        # Write git information header
        git_header = _format_git_header(git_info)
        log_file.write(git_header)
        log_file.flush()
        while True:
            wait_time = 0.2

            read_fds: List[io.TextIOBase] = []
            stdout_handle = proc.stdout
            if stdout_handle is not None and not stdout_handle.closed:
                read_fds.append(stdout_handle)
            skip_enabled = bool(allow_user_skip and input_monitor and input_monitor.enabled)

            try:
                ready, _, _ = select.select(read_fds, [], [], wait_time)
            except KeyboardInterrupt:
                interrupted = True
                if proc.poll() is None:
                    _terminate_process(proc)
                break

            if stdout_handle in ready:
                line = stdout_handle.readline()
                if line:
                    stripped_line = line.rstrip("\n")
                    progress_line = _extract_progress_line(stripped_line)
                    if stream_stdout:
                        print(line, end="")
                    log_file.write(line)
                    log_file.flush()
                    if full_log_file is not None:
                        full_log_file.write(line)
                        full_log_file.flush()
                    output_chunks.append(line)
                    if progress_callback is not None:
                        progress_callback(
                            time.monotonic() - start_time,
                            latest_line=progress_line,
                        )
                    last_output_time = time.monotonic()
                    continue
                else:
                    if proc.poll() is not None:
                        break

            now = time.monotonic()
            if progress_callback is not None:
                progress_callback(now - start_time, latest_line=None)

            if skip_enabled and input_monitor:
                # Drain any stray characters
                while input_monitor.poll() is not None:
                    pass
                if input_monitor.manual_pass:
                    input_monitor.manual_pass = False
                    manual_pass = True
                    if proc.poll() is None:
                        _terminate_process(proc)
                    break
                if input_monitor.manual_quit:
                    input_monitor.manual_quit = False
                    interrupted = True
                    if proc.poll() is None:
                        _terminate_process(proc)
                    break

            if max_exec_time and now - start_time >= max_exec_time:
                auto_pass = True
                if proc.poll() is None:
                    _terminate_process(proc)
                break

            if test.hang_timeout and now - last_output_time >= test.hang_timeout:
                hang_detected = True
                if proc.poll() is None:
                    _terminate_process(proc)
                break

            if proc.poll() is not None and not read_fds:
                break

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _terminate_process(proc)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    duration = time.monotonic() - start_time
    duration_str = f"{duration:.1f}"

    output = "".join(output_chunks)

    if progress_callback is not None:
        progress_callback(time.monotonic() - start_time, latest_line=None)

    # Parse performance metrics for model tests (exclude MODEL OP UNIT TEST)
    perf_metrics: Optional[PerformanceMetrics] = None
    is_model_test = any(
        keyword in test.command.lower()
        for keyword in ["pytest", "yolov8s", "qwen", "llama", "ssr", "demo", "performance"]
    )
    # Exclude MODEL OP UNIT TEST from performance metrics collection
    is_model_op_unit_test = test.name in MODEL_OP_UNIT_TEST_KEYS
    if is_model_test and not is_model_op_unit_test and not hang_detected:
        perf_metrics = _parse_performance_metrics(output, test.name)

    # Parse gtest output if this is a gtest command
    # Parse even if interrupted to show partial results
    gtest_result: Optional[GTestResult] = None
    is_gtest = "./build/test" in test.command or "unit_tests" in test.command
    if is_gtest and not hang_detected:
        gtest_result = _parse_gtest_output(output)
        if interrupted and gtest_result:
            gtest_result.total = -1  # Mark as incomplete

    # Parse pytest output if this is a pytest command
    # Parse even if manual_pass or interrupted to show partial results
    pytest_result: Optional[PytestResult] = None
    is_pytest = "pytest" in test.command
    if is_pytest and not hang_detected:
        # First try parsing from output
        pytest_result = _parse_pytest_output(output)
        # For manual_pass or interrupted, always try reading from log file
        # as the log file has more complete information
        if manual_pass or interrupted:
            try:
                if log_path.exists():
                    log_output = log_path.read_text(encoding="utf-8")
                    log_result = _parse_pytest_output(log_output)
                    # Use log file result if it has more data or if output parsing failed
                    if log_result and (pytest_result is None or log_result.total > pytest_result.total):
                        pytest_result = log_result
            except Exception:
                pass
        # If still None, try reading from log file
        elif pytest_result is None:
            try:
                if log_path.exists():
                    log_output = log_path.read_text(encoding="utf-8")
                    pytest_result = _parse_pytest_output(log_output)
            except Exception:
                pass

    if interrupted:
        status = "INTERRUPTED"
        detail = "Interrupted by user (Ctrl+C)"
    elif auto_pass:
        status = "PASS"
        detail = "Auto-pass: exceeded max execution time"
    elif manual_pass:
        status = "PASS_MANUAL"
        detail = "Manually marked as pass"
    elif user_skipped:
        status = "PASS"
        detail = "Manually marked as pass"
    elif hang_detected:
        status = "HANG"
        detail = (
            f"No output for {test.hang_timeout:.0f}s; marked as hang"
            if test.hang_timeout >= 1.0
            else "Terminated due to stalled output"
        )
    else:
        return_code = proc.returncode if proc.returncode is not None else 0
        if return_code != 0:
            status = "FAIL"
            detail = f"Process exited with code {return_code}"
        else:
            # For gtest, check if there are failures
            if gtest_result and gtest_result.failed > 0:
                status = "FAIL"
                detail = f"GTest: {gtest_result.failed} test(s) failed"
            # For pytest, check if there are failures
            elif pytest_result and (pytest_result.failed > 0 or pytest_result.error > 0):
                status = "FAIL"
                failed_count = pytest_result.failed + pytest_result.error
                detail = f"Pytest: {failed_count} test(s) failed"
            else:
                fail_hit = _detect_failures(output, test.fail_patterns)
                if fail_hit:
                    status = "FAIL"
                    detail = fail_hit
                elif test.success_patterns:
                    success_hit = _detect_success(output, test.success_patterns)
                    if success_hit is None:
                        status = "FAIL"
                        detail = "Required success pattern not found"

    rel_log = log_path.relative_to(ROOT)
    result = {
        "Test": test.name,
        "Status": status,
        "Duration (s)": duration_str,
        "Log": str(rel_log),
        "Detail": detail,
        "Command": test.command,
    }
    if gtest_result:
        result["gtest_result"] = gtest_result
    if pytest_result:
        result["pytest_result"] = pytest_result
    elif is_pytest and (manual_pass or interrupted) and not hang_detected:
        # For pytest tests that were manually passed or interrupted,
        # try one more time to parse from log file if available
        try:
            if log_path.exists():
                log_output = log_path.read_text(encoding="utf-8")
                pytest_result = _parse_pytest_output(log_output)
                if pytest_result:
                    result["pytest_result"] = pytest_result
        except Exception:
            pass
    if perf_metrics:
        result["perf_metrics"] = perf_metrics
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TT test suite runner")
    parser.add_argument("selections", nargs="*", help="predefined menu selections to run")
    parser.add_argument("--no-tail-pane", action="store_true", help="disable automatic tmux tail pane")
    parser.add_argument("--no-external-tail", action="store_true", help="disable spawning external terminal tail")
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline performance_metrics.json file for performance comparison",
    )
    return parser.parse_args()


def _setup_tmux_tail(full_log_path: Path, args: argparse.Namespace) -> Optional[str]:
    if args.no_tail_pane:
        return None
    if shutil.which("tmux") is None:
        print("tmux is not installed. Install it to enable automatic full-log tailing (e.g. 'sudo apt install tmux').")
        return None
    if not os.environ.get("TMUX"):
        return None
    tail_cmd = f"tail -F {full_log_path}"
    result = subprocess.run(
        ["tmux", "split-window", "-v", "-P", "-F", "#{pane_id}", tail_cmd],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return None
    pane_id = result.stdout.strip()
    subprocess.run(["tmux", "select-pane", "-U"], check=False)
    return pane_id or None


def _teardown_tmux_tail(pane_id: Optional[str]) -> None:
    if not pane_id:
        return
    if shutil.which("tmux") is None:
        return
    subprocess.run(["tmux", "kill-pane", "-t", pane_id], check=False)


def _spawn_external_tail(full_log_path: Path, args: argparse.Namespace) -> Optional[subprocess.Popen]:
    if args.no_external_tail:
        return None
    if os.environ.get("TMUX"):
        return None
    terminals = [
        ["gnome-terminal", "--", "tail", "-F", str(full_log_path)],
        ["konsole", "-e", "tail", "-F", str(full_log_path)],
        ["alacritty", "-e", "tail", "-F", str(full_log_path)],
        ["xterm", "-e", f"tail -F {full_log_path}"],
    ]
    for cmd in terminals:
        if shutil.which(cmd[0]) is None:
            continue
        try:
            proc = subprocess.Popen(cmd)
            return proc
        except Exception:
            continue
    print(
        "Could not spawn an external terminal for full-run logs. "
        "Please open another terminal and run 'tail -f logs/test_runs/full_run.log' manually."
    )
    return None


def _teardown_external_tail(proc: Optional[subprocess.Popen]) -> None:
    if not proc:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_test_suite(
    tests: List[TestCase],
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, str]]:
    run_dir = _ensure_log_dir()
    results: List[Dict[str, str]] = []

    # Load baseline performance metrics for comparison if specified
    baseline_perf_metrics: Dict[str, PerformanceMetrics] = {}
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.is_absolute():
            baseline_path = ROOT / baseline_path
        baseline_perf_metrics = _load_baseline_performance_metrics(baseline_path)
        if baseline_perf_metrics:
            print(f"Loaded baseline performance metrics from {baseline_path} ({len(baseline_perf_metrics)} tests)")
        else:
            print(f"Warning: No baseline performance metrics found in {baseline_path}")

    total = len(tests)
    print(f"Test log directory: {run_dir}")

    # Get git information once per test suite run
    git_info = _get_git_info()

    if FULL_LOG_PATH.exists():
        FULL_LOG_PATH.unlink()
    full_log_file = FULL_LOG_PATH.open("w", encoding="utf-8")
    # Write git information header
    git_header = _format_git_header(git_info)
    full_log_file.write(git_header)
    full_log_file.write(f"=== Test session started at {datetime.now().isoformat()} ===\n")
    full_log_file.flush()

    tail_pane_id = _setup_tmux_tail(FULL_LOG_PATH, args)

    is_tty = sys.stdout.isatty()

    input_monitor = InputMonitor()
    exit_code = 0
    try:
        print(
            "Tip: open another terminal or tmux pane and run "
            "'tail -f logs/test_runs/full_run.log' to watch the combined logs."
        )
        for idx, test in enumerate(tests, start=1):
            if "pytest" in test.command:
                progress_label = "pytest"
            elif "./build/test" in test.command or "unit_tests" in test.command:
                progress_label = "gtest"
            else:
                progress_label = "progress"
            progress_status_holder = {"line": ""}
            status_line_holder = {"last_len": 0}
            last_elapsed = {"value": -1.0}
            fail_lines: List[Tuple[float, str]] = []  # List of (elapsed, fail_line) tuples

            def _write_status_line(text: str, is_fail: bool = False) -> None:
                padding = max(0, status_line_holder["last_len"] - len(text))
                status_line_holder["last_len"] = len(text)
                if is_fail:
                    # Red color for FAIL lines
                    colored_text = f"\033[31m{text}\033[0m"  # Red
                    sys.stdout.write("\r" + colored_text + (" " * padding) + "\033[K\n")
                else:
                    sys.stdout.write("\r" + text + (" " * padding) + "\033[K")
                sys.stdout.flush()

            def on_progress(
                elapsed: float,
                *,
                _last=last_elapsed,
                latest_line: Optional[str] = None,
            ) -> None:
                if elapsed < 0:
                    return

                force_update = False
                if latest_line:
                    # Check if this line indicates a FAIL
                    if latest_line and "FAILED" in latest_line.upper():
                        # Print FAIL line in red and move to next line
                        fail_line = f"    elapsed: {elapsed:.1f}s"
                        display = latest_line[:70]
                        label = progress_label if progress_label != "progress" else "status"
                        fail_line += f"    {label}: {display}"
                        fail_lines.append((elapsed, fail_line))
                        _write_status_line(fail_line, is_fail=True)
                        # Clear status line holder length for new line
                        status_line_holder["last_len"] = 0
                        force_update = True
                    if latest_line != progress_status_holder["line"]:
                        progress_status_holder["line"] = latest_line
                        force_update = True
                if not force_update and _last["value"] >= 0 and elapsed - _last["value"] < 1.0:
                    return
                _last["value"] = elapsed
                line = f"    elapsed: {elapsed:.1f}s"
                if progress_status_holder["line"]:
                    display = progress_status_holder["line"][:70]
                    label = progress_label if progress_label != "progress" else "status"
                    line += f"    {label}: {display}"
                _write_status_line(line)

            callback = on_progress

            log_name = f"{_slugify(test.name)}.log"
            rel_log_path = str((run_dir / log_name).relative_to(ROOT))

            print(f"\n[{idx}/{total}] Starting test: {test.name}")
            print(f"    command: {test.command}")
            log_lines = [f"    log: {rel_log_path}"]
            is_unittest = ("./build/test" in test.command or "unit_tests" in test.command) or "pytest" in test.command
            if is_unittest:
                unittest_log_path = str((run_dir / "unittest_summary.log").relative_to(ROOT))
                log_lines.append(f"    unittest_log: {unittest_log_path}")
            print("\n".join(log_lines))
            if is_tty:
                print("    (Type 'p' + Enter to mark as pass, 'q' + Enter to skip remaining tests)")

            callback(0.0)
            test_start_time = time.monotonic()

            try:
                result = run_test(
                    test,
                    run_dir,
                    progress_callback=callback,
                    stream_stdout=False,
                    allow_user_skip=is_tty,
                    full_log_file=full_log_file,
                    input_monitor=input_monitor,
                )
            except KeyboardInterrupt:
                # Handle Ctrl+C during test execution
                # Mark current test as interrupted
                duration = time.monotonic() - test_start_time
                result = {
                    "Test": test.name,
                    "Status": "INTERRUPTED",
                    "Duration (s)": f"{duration:.1f}",
                    "Log": str((run_dir / f"{_slugify(test.name)}.log").relative_to(ROOT)),
                    "Detail": "Interrupted by user (Ctrl+C)",
                    "Command": test.command,
                    "Test Statistics": "-",
                }
                # Try to parse partial unittest output if available
                is_unittest = (
                    "./build/test" in test.command or "unit_tests" in test.command
                ) or "pytest" in test.command
                if is_unittest:
                    log_path = run_dir / f"{_slugify(test.name)}.log"
                    if log_path.exists():
                        try:
                            partial_output = log_path.read_text(encoding="utf-8")
                            if "./build/test" in test.command or "unit_tests" in test.command:
                                gtest_result = _parse_gtest_output(partial_output)
                                if gtest_result:
                                    gtest_result.total = -1  # Mark as incomplete
                                    result["gtest_result"] = gtest_result
                            elif "pytest" in test.command:
                                pytest_result = _parse_pytest_output(partial_output)
                                if pytest_result:
                                    result["pytest_result"] = pytest_result
                        except Exception:
                            pass

                # Add test statistics for interrupted tests if available
                if "gtest_result" in result:
                    gtest_result: GTestResult = result["gtest_result"]
                    stats_parts = []
                    if gtest_result.total == -1:
                        stats_parts.append("T:?")
                    else:
                        stats_parts.append(f"T:{gtest_result.total}")
                    if gtest_result.passed > 0:
                        stats_parts.append(f"P:{gtest_result.passed}")
                    if gtest_result.failed > 0:
                        stats_parts.append(f"F:{gtest_result.failed}")
                    if gtest_result.skipped > 0:
                        stats_parts.append(f"S:{gtest_result.skipped}")
                    result["Test Statistics"] = " ".join(stats_parts) if stats_parts else "-"
                elif "pytest_result" in result:
                    pytest_result: PytestResult = result["pytest_result"]
                    stats_parts = []
                    if pytest_result.total > 0:
                        stats_parts.append(f"T:{pytest_result.total}")
                    if pytest_result.passed > 0:
                        stats_parts.append(f"P:{pytest_result.passed}")
                    if pytest_result.failed > 0:
                        stats_parts.append(f"F:{pytest_result.failed}")
                    if pytest_result.error > 0:
                        stats_parts.append(f"E:{pytest_result.error}")
                    if pytest_result.skipped > 0:
                        stats_parts.append(f"S:{pytest_result.skipped}")
                    if pytest_result.known_issue > 0:
                        stats_parts.append(f"KI:{pytest_result.known_issue}")
                    result["Test Statistics"] = " ".join(stats_parts) if stats_parts else "-"

            # Add test statistics if available (gtest or pytest)
            if "gtest_result" in result:
                gtest_result: GTestResult = result["gtest_result"]
                stats_parts = []
                if gtest_result.total == -1:
                    stats_parts.append("T:?")
                else:
                    stats_parts.append(f"T:{gtest_result.total}")
                if gtest_result.passed > 0:
                    stats_parts.append(f"P:{gtest_result.passed}")
                if gtest_result.failed > 0:
                    stats_parts.append(f"F:{gtest_result.failed}")
                if gtest_result.skipped > 0:
                    stats_parts.append(f"S:{gtest_result.skipped}")
                result["Test Statistics"] = " ".join(stats_parts) if stats_parts else "-"
            elif "pytest_result" in result:
                pytest_result: PytestResult = result["pytest_result"]
                stats_parts = []
                if pytest_result.total > 0:
                    stats_parts.append(f"T:{pytest_result.total}")
                if pytest_result.passed > 0:
                    stats_parts.append(f"P:{pytest_result.passed}")
                if pytest_result.failed > 0:
                    stats_parts.append(f"F:{pytest_result.failed}")
                if pytest_result.error > 0:
                    stats_parts.append(f"E:{pytest_result.error}")
                if pytest_result.skipped > 0:
                    stats_parts.append(f"S:{pytest_result.skipped}")
                if pytest_result.known_issue > 0:
                    stats_parts.append(f"KI:{pytest_result.known_issue}")
                result["Test Statistics"] = " ".join(stats_parts) if stats_parts else "-"
            else:
                result["Test Statistics"] = "-"

            # Add performance comparison to Detail if available
            if "perf_metrics" in result:
                perf_metrics: PerformanceMetrics = result["perf_metrics"]
                baseline = baseline_perf_metrics.get(result["Test"])
                perf_summary = perf_metrics.to_summary_string(baseline=baseline, enable_color=is_tty)
                if perf_summary:
                    detail = result.get("Detail", "")
                    if detail:
                        result["Detail"] = f"{detail} | {perf_summary}"
                    else:
                        result["Detail"] = perf_summary
            # Print final status (fail_lines are already printed in red)
            if progress_status_holder["line"]:
                label = progress_label if progress_label != "progress" else "status"
                line = f"    elapsed: {result['Duration (s)']}s    {label}: (done)"
                _write_status_line(line)
            else:
                line = f"    elapsed: {result['Duration (s)']}s (done)"
                _write_status_line(line)
            sys.stdout.write("\n")
            sys.stdout.flush()
            status_line_holder["last_len"] = 0

            results.append(result)
            status_for_log = "SUCCESS" if result["Status"] in {"PASS", "PASS_MANUAL"} else result["Status"]
            full_log_file.write(f"[{idx}/{total}] END {test.name} -> {status_for_log} ({result['Duration (s)']}s)\n")
            if result["Detail"]:
                full_log_file.write(f"detail: {result['Detail']}\n")
            full_log_file.flush()

            # Update unittest_summary.log after each unittest test completes (gtest or pytest)
            if "gtest_result" in result or "pytest_result" in result:
                test_case = next((t for t in tests if t.name == result["Test"]), None)
                if test_case:
                    unittest_summary = _format_unittest_summary(
                        test_case,
                        gtest_result=result.get("gtest_result"),
                        pytest_result=result.get("pytest_result"),
                    )
                    if unittest_summary:
                        unittest_summary_path = run_dir / "unittest_summary.log"
                        # Read existing content if file exists
                        existing_summaries: List[str] = []
                        if unittest_summary_path.exists():
                            existing_content = unittest_summary_path.read_text(encoding="utf-8")
                            # Split by separator and filter out empty parts
                            parts = existing_content.split("=" * 80)
                            existing_summaries = [p.strip() for p in parts if p.strip()]
                        # Add new summary
                        existing_summaries.append(unittest_summary.strip())
                        # Write updated content with clear separators between summaries
                        separator = "\n" + "=" * 80 + "\n"
                        unittest_summary_content = separator.join(existing_summaries) + "\n"
                        _write_file(unittest_summary_path, unittest_summary_content, git_info=git_info)
                        _write_file(ROOT / "unittest_summary.log", unittest_summary_content, git_info=git_info)

            display_status = status_for_log
            colored_status = _colorize_status(display_status, is_tty)
            result_line = f"    result: {colored_status} (duration {result['Duration (s)']}s)"
            if "gtest_result" in result:
                gtest_stats = _format_gtest_stats(result["gtest_result"])
                if gtest_stats:
                    result_line += f" {gtest_stats}"
            if "pytest_result" in result:
                pytest_stats = _format_pytest_stats(result["pytest_result"])
                if pytest_stats:
                    result_line += f" {pytest_stats}"
            print(result_line)
            if result["Detail"]:
                print(f"    detail: {result['Detail']}")
            if result["Status"] in {"INTERRUPTED"}:
                print("Skipping remaining tests because execution was interrupted.")
                full_log_file.write("Execution interrupted by user. Remaining tests skipped.\n")
                full_log_file.flush()
                break

        if len(results) < len(tests):
            for remaining_test in tests[len(results) :]:
                results.append(
                    {
                        "Test": remaining_test.name,
                        "Status": "NOT_RUN",
                        "Duration (s)": "0.0",
                        "Log": "-",
                        "Detail": "Not executed",
                        "Command": remaining_test.command,
                        "Test Statistics": "-",
                    }
                )

        headers = ["Test", "Status", "Duration (s)", "Log", "Test Statistics", "Detail"]
        rows = [[result[h] for h in headers] for result in results]
        summary = _format_table(headers, rows)

        summary_path = run_dir / "summary.log"
        _write_file(summary_path, summary, git_info=git_info)
        _write_file(ROOT / "summary.log", summary, git_info=git_info)

        # Write detailed git information to git_info.log
        git_info_log_content = _format_detailed_git_info(git_info)
        git_info_log_path = run_dir / "git_info.log"
        _write_file(git_info_log_path, git_info_log_content, git_info=None)  # Don't add header twice
        _write_file(ROOT / "git_info.log", git_info_log_content, git_info=None)

        # Save performance metrics to separate JSON file
        perf_metrics_data: Dict[str, any] = {
            "git_info": git_info,
            "test_run_timestamp": datetime.now().isoformat(),
            "metrics": [],
        }
        for result in results:
            if "perf_metrics" in result:
                perf_metrics: PerformanceMetrics = result["perf_metrics"]
                perf_metrics_data["metrics"].append(
                    {
                        "test": result["Test"],
                        "status": result["Status"],
                        "metrics": perf_metrics.to_dict(),
                        "raw_lines": perf_metrics.raw_lines,
                    }
                )
        if perf_metrics_data:
            perf_metrics_path = run_dir / "performance_metrics.json"
            perf_metrics_json = json.dumps(perf_metrics_data, indent=2, ensure_ascii=False)
            _write_file(perf_metrics_path, perf_metrics_json)
            _write_file(ROOT / "performance_metrics.json", perf_metrics_json)

        # Generate unittest summary files (gtest or pytest)
        unittest_summaries: List[str] = []
        for result in results:
            if "gtest_result" in result or "pytest_result" in result:
                test_case = next((t for t in tests if t.name == result["Test"]), None)
                if test_case:
                    unittest_summary = _format_unittest_summary(
                        test_case,
                        gtest_result=result.get("gtest_result"),
                        pytest_result=result.get("pytest_result"),
                    )
                    if unittest_summary:
                        unittest_summaries.append(unittest_summary)

        if unittest_summaries:
            unittest_summary_path = run_dir / "unittest_summary.log"
            # Use same separator format as in the per-test update
            separator = "\n" + "=" * 80 + "\n"
            unittest_summary_content = separator.join([s.strip() for s in unittest_summaries]) + "\n"
            _write_file(unittest_summary_path, unittest_summary_content, git_info=git_info)
            _write_file(ROOT / "unittest_summary.log", unittest_summary_content, git_info=git_info)

        failures = [r for r in results if r["Status"] not in {"PASS", "PASS_MANUAL", "SUCCESS", "NOT_RUN"}]
        fail_summary_lines: List[str] = []
        if failures:
            fail_headers = ["Test", "Status", "Duration (s)", "Log", "Test Statistics", "Detail"]
            fail_rows = [[result[h] for h in fail_headers] for result in failures]
            fail_summary = _format_table(fail_headers, fail_rows)
            fail_summary_lines.append(fail_summary)
        fail_path = run_dir / "failures.log"
        fail_json_path = run_dir / "failures.json"
        root_fail_json_path = ROOT / "failures.json"
        if fail_summary_lines:
            _write_file(fail_path, "\n\n".join(fail_summary_lines), git_info=git_info)
            _write_file(ROOT / "failures.log", "\n\n".join(fail_summary_lines), git_info=git_info)
            fail_json_payload = [
                {
                    "test": result["Test"],
                    "status": result["Status"],
                    "duration_s": result["Duration (s)"],
                    "log": result["Log"],
                    "detail": result["Detail"],
                }
                for result in failures
            ]
            fail_json_path.write_text(json.dumps(fail_json_payload, indent=2), encoding="utf-8")
            root_fail_json_path.write_text(json.dumps(fail_json_payload, indent=2), encoding="utf-8")
        else:
            for path in [fail_path, ROOT / "failures.log", fail_json_path, root_fail_json_path]:
                if path.exists():
                    path.unlink()

        print("\nFinal results:")
        status_counts = {"PASS": 0, "PASS_MANUAL": 0, "FAIL": 0, "HANG": 0, "INTERRUPTED": 0, "NOT_RUN": 0}
        for result in results:
            status_counts[result["Status"]] = status_counts.get(result["Status"], 0) + 1
        for status in ["PASS", "PASS_MANUAL", "FAIL", "HANG", "INTERRUPTED", "NOT_RUN"]:
            count = status_counts.get(status, 0)
            if count:
                label = _colorize_status(status, is_tty)
                print(f"  {label}: {count}")
        for idx, result in enumerate(results, start=1):
            display_status = "SUCCESS" if result["Status"] in {"PASS", "PASS_MANUAL"} else result["Status"]
            colored_status = _colorize_status(display_status, is_tty)
            result_line = f"  {idx}. {result['Test']} -> {colored_status} (duration {result['Duration (s)']}s)"
            if "gtest_result" in result:
                gtest_stats = _format_gtest_stats(result["gtest_result"])
                if gtest_stats:
                    result_line += f" {gtest_stats}"
            if "pytest_result" in result:
                pytest_stats = _format_pytest_stats(result["pytest_result"])
                if pytest_stats:
                    result_line += f" {pytest_stats}"
            print(result_line)
            # Print performance metrics with comparison if available
            if "perf_metrics" in result:
                perf_metrics: PerformanceMetrics = result["perf_metrics"]
                baseline = baseline_perf_metrics.get(result["Test"])
                perf_summary = perf_metrics.to_summary_string(baseline=baseline, enable_color=is_tty)
                if perf_summary:
                    print(f"      Performance: {perf_summary}")
        print(f"\nSummary table saved to: {summary_path}")
        print(f"Latest summary file: {ROOT / 'summary.log'}")
        if unittest_summaries:
            print(f"Unittest summary saved to: {run_dir / 'unittest_summary.log'}")
            print(f"Latest unittest summary file: {ROOT / 'unittest_summary.log'}")
        perf_metrics_path = run_dir / "performance_metrics.json"
        if perf_metrics_path.exists():
            print(f"Performance metrics saved to: {perf_metrics_path}")
            print(f"Latest performance metrics file: {ROOT / 'performance_metrics.json'}")
        if failures:
            print(f"Failures/Hangs summary: {fail_path}")
            print(f"Latest failure summary file: {ROOT / 'failures.log'}")

        full_log_file.write("\n=== Test session finished ===\n")
        exit_code = 0 if results and all(r["Status"] in {"PASS", "PASS_MANUAL", "SUCCESS"} for r in results) else 1
        run_info = {
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
            "total": str(len(results)),
            "failures": str(len(failures)),
        }
        if unittest_summaries:
            run_info["unittest_summary_path"] = str(run_dir / "unittest_summary.log")
    finally:
        input_monitor.close()
        _teardown_tmux_tail(tail_pane_id)
        full_log_file.close()

    return exit_code, run_info


def main(args: argparse.Namespace) -> int:
    interactive = sys.stdin.isatty() and not args.selections
    last_run_info: Optional[Dict[str, str]] = None
    last_exit_code = 0

    selection_plan: List[str] = [sel for sel in args.selections if sel.strip()]

    if not interactive and not selection_plan:
        tests = [TEST_CASES[key] for key in TEST_CASES.keys()]
        external_tail_proc = _spawn_external_tail(FULL_LOG_PATH, args)
        try:
            last_exit_code, last_run_info = run_test_suite(tests, args)
        finally:
            _teardown_external_tail(external_tail_proc)
        return last_exit_code

    external_tail_proc = _spawn_external_tail(FULL_LOG_PATH, args)
    try:
        while True:
            if selection_plan:
                selection_expr = selection_plan.pop(0)
                tests = select_tests_from_menu(args, last_run_info, predefined_selection=selection_expr)
            else:
                tests = select_tests_from_menu(args, last_run_info)
            if not tests:
                if selection_plan:
                    continue
                return last_exit_code
            last_exit_code, last_run_info = run_test_suite(tests, args)
            if not selection_plan and not interactive:
                return last_exit_code
    finally:
        _teardown_external_tail(external_tail_proc)


if __name__ == "__main__":
    cli_args = _parse_args()
    raise SystemExit(main(cli_args))
