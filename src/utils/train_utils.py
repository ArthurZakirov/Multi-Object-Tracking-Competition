import os
from matplotlib.figure import Figure


def log_to_tensorboard(metric_dict, step, summary_writer, mode="train"):
    if metric_dict:
        for metric_name, metric_value in metric_dict.items():
            if isinstance(metric_value, Figure):
                summary_writer.add_figure(
                    tag=os.path.join(mode, metric_name),
                    figure=metric_value,
                    global_step=step,
                )

            else:
                summary_writer.add_scalar(
                    tag=os.path.join(mode, metric_name),
                    scalar_value=metric_value,
                    global_step=step,
                )
