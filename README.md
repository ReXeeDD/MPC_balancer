# Zepto Standalone Balancer

This directory mirrors the MPC-based wheel balancing logic from Upkie but removes
all dependencies on the rest of the codebase. It contains:

- `controllers/mpc_balancer`: copy of the MPC controller plus the ProxQP
  workspace wrapper.
- `assets/Bipedal.xml`: MuJoCo model built from the CAD meshes.
- `mujoco_balancer.py`: bridge that converts MuJoCo state into the observation
  dictionary expected by the balancer.
- `testing.py`: simple executable that loads the XML in MuJoCo and commands the
  MPC balancer (headless or with viewer).

## Requirements

Install the runtime dependencies into any Python environment:

```bash
pip install mujoco numpy proxsuite qpmpc qpsolvers
```

## Running the test

From the repo root (or any directory with `PYTHONPATH=/path/to/upkie`), run:

```bash
python3 -m zepto.testing  --target-velocity 0.0 --target-turn -0.5
```
You can implemnt keyboard or controller control if you want better controling 

I will provide hardware implemenation after i complete my project

Also big thanks to upkie team for providing the template for mpc balancer i ported it to mujoco for my RL training but you can simple chnage it to another simlator with minor tweeks but i hope this can be helpfull to anyone in need 



