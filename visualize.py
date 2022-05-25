from src.tracker.visualization import get_visualization_functions
import matplotlib.pyplot as plt

# ENTER PATH TO SEQUENCE
sequence_dir = "results\\tracker\\25-04-2022_17-18\\MOT16-02-mini"

# ACCESS PLOT FUNCTION VIA KEY ["detections", "lost_tracks"]
vis_functions = get_visualization_functions(sequence_dir)
fig = vis_functions["lost_tracks"](lost_idx=0, show_boxes=["track", "fut"])

plt.show()
