import os


def log_to_tensorboard(metric_dict, step, summary_writer, mode="train"):
    for metric_name, metric_value in metric_dict.items():
        summary_writer.add_scalar(
            tag=os.path.join(mode, metric_name),
            scalar_value=metric_value,
            global_step=step,
        )
