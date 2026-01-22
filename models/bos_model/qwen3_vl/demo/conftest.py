# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.bos_model.qwen25_vl.tt.model_config import parse_optimizations


def pytest_addoption(parser):
    parser.addoption(
        "--optimizations",
        action="store",
        default=None,
        type=parse_optimizations,
        help="Precision and fidelity configuration diffs over default (i.e., accuracy)",
    )
    parser.addoption(
        "--res",
        action="store",
        default=None,
        help="Set input resolution (single int, e.g. --res 128 for [128, 128])",
    )
