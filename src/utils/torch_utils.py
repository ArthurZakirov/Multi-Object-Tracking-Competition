import torch
import numpy as np


def run_model_on_list(
    model, input_list, device="cpu", train=False, concat=True
):
    outputs = []
    model.to(device)
    if train:
        model.train()
        for inp in input_list:
            inp = inp.to(device)
            out = model(inp).cpu().clone()
            outputs.append(outputs)
    else:
        model.eval()
        with torch.no_grad():
            for inp in input_list:
                inp = inp.to(device)
                out = model(inp).cpu().clone()
                outputs.append(outputs)
    if concat:
        outputs = torch.cat(outputs, dim=0)
    return outputs


def set_dict_to_torch_device(numpy_dict, device):
    tensor_dict = {}
    for key, array in numpy_dict.items():
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
        else:
            tensor = array
        tensor_dict[key] = tensor.to(device)
    return tensor_dict


def set_list_to_torch_device(numpy_list, device):
    tensor_list = {}
    for array in numpy_list:
        if isinstance(array, np.ndarray):
            tensor = torch.from_numpy(array)
        else:
            tensor = array
        tensor_list.append(tensor.to(device))
    return tensor_list


def dict2keys_and_items(d, out_fmt):
    keys = list(d.keys())
    items = [item for (_, item) in d.items()]

    if out_fmt == "lists":
        return keys, items

    if out_fmt == "numpy":
        return np.array(keys), np.stack(items, axis=0)

    if out_fmt == "torch":
        return (
            torch.from_numpy(np.array(keys)),
            torch.from_numpy(np.stack(items, axis=0)),
        )
