#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone wheel-balancing MPC controller."""

from __future__ import annotations

import numpy as np
from qpmpc import MPCQP, MPCProblem, Plan
from qpmpc.systems import WheeledInvertedPendulum
from qpsolvers import solve_problem

from utils import clamp_and_warn, logger, low_pass_filter

from .proxqp_workspace import ProxQPWorkspace


def get_target_states(
    pendulum: WheeledInvertedPendulum,
    state: np.ndarray,
    target_ground_velocity: float,
):
    """Reference state trajectory over the prediction horizon."""

    nx = pendulum.STATE_DIM
    T = pendulum.sampling_period
    target_states = np.zeros((pendulum.nb_timesteps + 1) * nx)
    for k in range(pendulum.nb_timesteps + 1):
        target_states[k * nx] = state[0] + (k * T) * target_ground_velocity
        target_states[k * nx + 2] = target_ground_velocity
    return target_states


class MPCBalancer:
    """Model-predictive sagittal balancer."""

    def __init__(
        self,
        fall_pitch: float = 1.0,
        leg_length: float = 0.58,
        max_ground_accel: float = 10.0,
        max_ground_velocity: float = 3.0,
        nb_timesteps: int = 50,
        sampling_period: float = 0.02,
        stage_input_cost_weight: float = 1e-2,
        stage_state_cost_weight: float = 1e-3,
        terminal_cost_weight: float = 1.0,
        warm_start: bool = True,
    ):
        pendulum = WheeledInvertedPendulum(
            length=leg_length,
            max_ground_accel=max_ground_accel,
            nb_timesteps=nb_timesteps,
            sampling_period=sampling_period,
        )
        mpc_problem = pendulum.build_mpc_problem(
            terminal_cost_weight=terminal_cost_weight,
            stage_state_cost_weight=stage_state_cost_weight,
            stage_input_cost_weight=stage_input_cost_weight,
        )
        mpc_problem.initial_state = np.zeros(4)
        mpc_qp = MPCQP(mpc_problem)
        workspace = ProxQPWorkspace(mpc_qp)
        self.commanded_velocity = 0.0
        self.fall_pitch = fall_pitch
        self.fallen = False
        self.max_ground_velocity = max_ground_velocity
        self.mpc_problem = mpc_problem
        self.mpc_qp = mpc_qp
        self.pendulum = pendulum
        self.warm_start = warm_start
        self.workspace = workspace

    def compute_ground_velocity(
        self,
        target_ground_velocity: float,
        observation: dict,
        dt: float,
    ) -> float:
        """Compute a new ground velocity command."""

        floor_contact = observation["floor_contact"]["contact"]
        base_orientation = observation["base_orientation"]
        base_pitch = base_orientation["pitch"]
        base_angular_velocity = base_orientation["angular_velocity"][1]
        ground_position = observation["wheel_odometry"]["position"]
        ground_velocity = observation["wheel_odometry"]["velocity"]

        fallen = abs(base_pitch) > self.fall_pitch
        if fallen and not self.fallen:
            logger.warning("Base angle %.3f rad denotes a fall", base_pitch)
        self.fallen = fallen

        cur_state = np.array(
            [
                ground_position,
                base_pitch,
                ground_velocity,
                base_angular_velocity,
            ]
        )

        nx = self.pendulum.STATE_DIM
        target_states = get_target_states(
            self.pendulum, cur_state, target_ground_velocity
        )
        self.mpc_problem.update_initial_state(cur_state)
        self.mpc_problem.update_goal_state(target_states[-nx:])
        self.mpc_problem.update_target_states(target_states[:-nx])

        self.mpc_qp.update_cost_vector(self.mpc_problem)
        if self.warm_start:
            qpsol = self.workspace.solve(self.mpc_qp)
        else:
            qpsol = solve_problem(self.mpc_qp.problem, solver="proxqp")
        if not qpsol.found:
            logger.warning("No solution found to the MPC problem")
        plan = Plan(self.mpc_problem, qpsol)

        if fallen or not floor_contact:
            self.commanded_velocity = low_pass_filter(
                prev_output=self.commanded_velocity,
                cutoff_period=0.1,
                new_input=0.0,
                dt=dt,
            )
        elif plan.is_empty:
            logger.error("Solver found no solution to the MPC problem")
            logger.info("Re-sending previous ground velocity")
        else:
            self.pendulum.state = cur_state
            commanded_accel = plan.first_input[0]
            self.commanded_velocity = clamp_and_warn(
                self.commanded_velocity + commanded_accel * dt / 2.0,
                lower=-self.max_ground_velocity,
                upper=+self.max_ground_velocity,
                label="MPCBalancer.commanded_velocity",
            )
        return self.commanded_velocity

