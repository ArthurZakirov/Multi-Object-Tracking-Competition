from src.tracker.visualization import get_visualization_functions
import matplotlib.pyplot as plt

# ENTER PATH TO SEQUENCE
sequence_dir = "results\\tracker\\11-04-2022_00-28\\MOT16-02-mini"

# ACCESS PLOT FUNCTION VIA KEY ["detections", "lost_tracks"]
vis_functions = get_visualization_functions(sequence_dir)
fig = vis_functions["detections"](frame_id=6, type="FP")

plt.show()
