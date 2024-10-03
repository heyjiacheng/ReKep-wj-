import mujoco
import numpy as np

# Load the model from an XML file (e.g., 'humanoid.xml')
model = mujoco.load_model_from_path('humanoid.xml')
sim = mujoco.MjSim(model)

# Run the simulation for a few steps
for i in range(1000):
    sim.step()
    print(f"Step {i}:", sim.data.qpos)
