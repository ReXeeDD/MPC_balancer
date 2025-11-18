#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo interface layer around the Upkie MPC balancer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import mujoco
import numpy as np

from controllers.mpc_balancer import MPCBalancer


def quat_to_pitch(quat: np.ndarray) -> float:
    """Return the pitch (rotation around body Y) from a quaternion."""
    w, x, y, z = quat
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    return math.asin(sinp)


@dataclass
class MujocoObservation:
    """Subset of the spine observation required by MPCBalancer."""
    base_pitch: float
    base_pitch_rate: float
    ground_position: float
    ground_velocity: float
    has_contact: bool

    def to_spine_dict(self) -> Dict[str, Dict[str, float]]:
        """Convert to the dictionary format expected by MPCBalancer."""
        return {
            "floor_contact": {"contact": self.has_contact},
            "base_orientation": {
                "pitch": self.base_pitch,
                "angular_velocity": np.array([0.0, self.base_pitch_rate, 0.0]),
            },
            "wheel_odometry": {
                "position": self.ground_position,
                "velocity": self.ground_velocity,
            },
        }


class MujocoWheelBalancer:
    """Use the Upkie MPC balancer in a MuJoCo simulation loop."""

    def __init__(
        self,
        model: mujoco.MjModel,
        wheel_radius: float = 0.09,
        target_ground_velocity: float = 0.0,
        target_turn_velocity: float = 0.0,
    ):
        self.controller = MPCBalancer(leg_length=0.40)

        self.left_wheel_act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel"
        )
        self.right_wheel_act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel"
        )
        
        self.left_wheel_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_wheel_link"
        )
        self.right_wheel_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "right_wheel_link"
        )

        # --- NEW: Get Joint IDs for Odometry ---
        self.left_wheel_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "left_wheel"
        )
        self.right_wheel_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "right_wheel"
        )
        self.left_wheel_qpos_adr = model.jnt_qposadr[self.left_wheel_joint]
        self.right_wheel_qpos_adr = model.jnt_qposadr[self.right_wheel_joint]
        self.left_wheel_qvel_adr = model.jnt_dofadr[self.left_wheel_joint]
        self.right_wheel_qvel_adr = model.jnt_dofadr[self.right_wheel_joint]
        # ---------------------------------------

        self.wheel_radius = wheel_radius
        self.target_ground_velocity = target_ground_velocity
        self.target_turn_velocity = target_turn_velocity

    def _observe(self, model: mujoco.MjModel, data: mujoco.MjData) -> MujocoObservation:
        quat = data.qpos[3:7]
        pitch = quat_to_pitch(quat)
        base_pitch_rate = data.qvel[4]  # rotational velocity around Y

        # --- FIX: USE WHEEL ODOMETRY INSTEAD OF GLOBAL X ---
        # Read the angular position (rad) and velocity (rad/s) of the wheels directly
        left_theta = data.qpos[self.left_wheel_qpos_adr]
        right_theta = data.qpos[self.right_wheel_qpos_adr]
        left_omega = data.qvel[self.left_wheel_qvel_adr]
        right_omega = data.qvel[self.right_wheel_qvel_adr]

        # Calculate linear distance and speed (average of both wheels)
        # Note: We average them to filter out the turning component
        ground_position = ((left_theta + right_theta) / 2.0) * self.wheel_radius
        ground_velocity = ((left_omega + right_omega) / 2.0) * self.wheel_radius
        # ---------------------------------------------------

        wheel_clearance = 0.01  # meters
        left_height = data.xpos[self.left_wheel_body][2]
        right_height = data.xpos[self.right_wheel_body][2]
        left_contact = left_height <= self.wheel_radius + wheel_clearance
        right_contact = right_height <= self.wheel_radius + wheel_clearance
        has_contact = left_contact or right_contact

        return MujocoObservation(
            base_pitch=pitch,
            base_pitch_rate=base_pitch_rate,
            ground_position=ground_position,
            ground_velocity=ground_velocity,
            has_contact=has_contact,
        )

    def compute_ctrl(
        self, model: mujoco.MjModel, data: mujoco.MjData, dt: float
    ) -> np.ndarray:
        """Return actuator commands for the current step."""

        observation = self._observe(model, data)
        spine_dict = observation.to_spine_dict()
        commanded_ground_velocity = self.controller.compute_ground_velocity(
            target_ground_velocity=self.target_ground_velocity,
            observation=spine_dict,
            dt=dt,
        )

        # Base velocity for balancing (Forward/Backward)
        linear_wheel_vel = commanded_ground_velocity / self.wheel_radius
        
        # Differential velocity for turning
        turn_offset = self.target_turn_velocity 

        ctrl = np.zeros(model.nu)
        ctrl[self.left_wheel_act_id] = linear_wheel_vel - turn_offset
        ctrl[self.right_wheel_act_id] = linear_wheel_vel + turn_offset
        return ctrl