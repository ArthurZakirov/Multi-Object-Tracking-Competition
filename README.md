# tracker_challenge

Note:<br>
- you can put the skript that you want to run inside a package and run them with ```python -m src.gnn.train_gnn```
- you can put the skript that you want to run in the parent folder and run them with ```python train_gnn.py```
- you can **not** put the skript that you want to run inside the package and run them with ```python -m src/gnn/train_gnn.py```<br>


| help | command |
|------|---------|
| evaluate tracker | python evaluate_tracker.py --tracker_config_path "config/tracker/tracker.json" --split reid --save_evaluation |
| train gnn | python train_gnn.py --assign_model_config_path ""config/assign_model.json --eval_tracker_config_path "config/tracker/gnn_eval_tracker.json" --train_split train_wo_val2 --eval_split val2 |

