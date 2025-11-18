#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0

"""Local copy of the ProxQP workspace helper used by Upkie."""

from __future__ import annotations

import qpsolvers
from proxsuite import proxqp
from qpmpc import MPCQP


class ProxQPWorkspace:
    """Minimal wrapper that keeps a warm-started ProxQP solver instance."""

    def __init__(
        self,
        mpc_qp: MPCQP,
        update_preconditioner: bool = True,
        verbose: bool = False,
    ):
        n_eq = 0
        n_in = mpc_qp.h.size // 2
        n = mpc_qp.P.shape[1]
        solver = proxqp.dense.QP(
            n,
            n_eq,
            n_in,
            dense_backend=proxqp.dense.DenseBackend.PrimalDualLDLT,
        )
        solver.settings.eps_abs = 1e-3
        solver.settings.eps_rel = 0.0
        solver.settings.verbose = verbose
        solver.settings.compute_timings = False
        solver.settings.primal_infeasibility_solving = True
        solver.init(
            H=mpc_qp.P,
            g=mpc_qp.q,
            C=mpc_qp.G[::2, :],
            l=-mpc_qp.h[1::2],
            u=mpc_qp.h[::2],
        )
        solver.solve()
        self.update_preconditioner = update_preconditioner
        self.solver = solver

    def solve(self, mpc_qp: MPCQP) -> qpsolvers.Solution:
        """Solve the updated MPC QP with the warm-started solver."""

        self.solver.update(
            g=mpc_qp.q,
            update_preconditioner=self.update_preconditioner,
        )
        self.solver.solve()
        result = self.solver.results
        qpsol = qpsolvers.Solution(mpc_qp.problem)
        qpsol.found = result.info.status == proxqp.QPSolverOutput.PROXQP_SOLVED
        qpsol.x = result.x
        return qpsol

