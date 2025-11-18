#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Utility helpers replicated from Upkie for the standalone Zepto package."""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger("zepto")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def clamp_and_warn(value: float, lower: float, upper: float, label: str) -> float:
    """Clamp `value` to [`lower`, `upper`] and warn if saturation occurs."""

    clamped = max(lower, min(upper, value))
    if clamped != value:
        logger.warning(
            "%s saturated to %.3f (bounds %.3f, %.3f)", label, clamped, lower, upper
        )
    return clamped


def low_pass_filter(
    prev_output: float, cutoff_period: float, new_input: float, dt: float
) -> float:
    """Simple exponential smoothing filter."""

    if cutoff_period <= 0.0:
        return new_input
    alpha = dt / (cutoff_period + dt)
    return prev_output + alpha * (new_input - prev_output)


def saturate(value: float, bounds: Tuple[float, float]) -> float:
    """Clamp without logging; helper for controller code."""

    lower, upper = bounds
    return max(lower, min(upper, value))

