#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone MuJoCo test that drives the Upkie MPC balancer."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from mujoco_balancer import MujocoWheelBalancer

ASSETS_DIR = Path(__file__).parent / "assets"
MODEL_PATH = r"/home/albin/upkie/zepto/assets/Bipedal.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Duration to run the simulation (seconds)",
    )
    parser.add_argument(
        "--target-velocity",
        type=float,
        default=0.0,
        help="Sagittal ground velocity target in m/s",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without launching the MuJoCo viewer",
    )
    return parser.parse_args()


def run_simulation(args: argparse.Namespace) -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    balancer = MujocoWheelBalancer(
        model, wheel_radius=0.09, target_ground_velocity=args.target_velocity
    )

    def step():
        ctrl = balancer.compute_ctrl(model, data, dt=model.opt.timestep)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

    if args.headless:
        steps = int(args.duration / model.opt.timestep)
        for _ in range(steps):
            step()
    else:
        try:
            import mujoco.viewer as mujoco_viewer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mujoco.viewer is unavailable; rerun with --headless"
            ) from exc

        with mujoco_viewer.launch_passive(model, data) as viewer:
            deadline = time.time() + args.duration
            while viewer.is_running() and time.time() < deadline:
                step()
                viewer.sync()


if __name__ == "__main__":
    run_simulation(parse_args())

