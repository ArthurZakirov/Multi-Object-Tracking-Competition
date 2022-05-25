# **DOLOTraPed**: **D**on't **O**nly **L**ook **O**nce when **Tra**cking **Ped**estrians
This repository contains the Multi-Object-Tracking Model of *Arthur Zakirov* (Technical University of Munich) for the *MOT16* Challenge.

<img src="test.gif" width="500"/>

## **The Method explained**
*DOLOTraPed* uses [MaskRCNN](https://arxiv.org/abs/1703.06870) as an object detector. The data association is based on [DeepSORT](https://arxiv.org/abs/1703.07402). The major contribution of this work are these 4 modules,
that can be added to any state of the art Multi-Object-Tracker to potentially improve the performance.<br>
<img src="TrackerOverview.png" width="500"/>


#### **opNMS**: **o**cclusion **p**reserving **N**on **M**aximum **S**upression<br>
One major dilemma in object detection is tradeoff between recall and precision in non maximium supression. 
This work contributes a novel non maxmimum supression algorithm, which increases recall without sacrificing precision.
This is achieved taking instance segmentation masks of object proposals into account. 

#### **BOXCORM**: **b**ox **c**orrection **m**odule
*opNMS* and *daTRAFIM* significantly increase the amount of recognized pedestrians without major precision drawbacks. However one problem still remains: It is not enough to recognize occluded pedestrians, their location must be predicted accurately.
This is a huge challenge for most object detectors, since they focus on the visible part of the pedestrian when predicting the box. 
*BOXCORM* adjusts the regressed box by the detector, using the knowledge of neighbour pedestrian positions in the scene.

#### **daTRAFIM**: **d**istractor **a**ware **t**rack **f**iltering **m**odule<br> 
Maximizing the recall of a Multi-Object-Tracker always comes at a risk of reducing the precision. 
State of the art MOT algorithms mainly rely on the detector score, to determine wether to consider a detection.
*daTRAFIM* is an alternative disrimination, that uses domain knowledge to test low score detections for their potential to be a pedestrian. It also checks already tracked object hypotheses for certain criteria and conditionally drops them, to prevent false positives.
<img src="MatchingSystempng.png" width="500"/>

#### **attReID**: **att**entive **Re**-**ID**entification
State of the art Multi-Object-Trackers use appearance based features for matching between frames in the data association step.
Conventionally ReID-features are computed from an image crop inside the regressed box by the detector. This leads to major issues, when a single box contains multiple pedestrians. This work contributes an attentive ReID module, which uses the mask of an instance segmentation model, to remove pixels from the image crop, that don't belong to the detected object.


## **Performance**
I evaluate *DOLOTraPed* on the *MOT16* dataset. Keep in mind that these results are NOT comparable to the official [Leaderboard](https://paperswithcode.com/sota/multi-object-tracking-on-mot16). This is because I have no access to the official evaluation function of the Challenge, which does not penalize the detection of static pedestrians, pedestrians inside buildings, pedestrians on vehicles. In my evaluation these detections are penalized. So the competition metrics of *DOLOTraPed* are expected to be higher.

|               |   idf1 |   idp |   idr |   recall |   precision |   mota |   motp |
|:--------------|-------:|------:|------:|---------:|------------:|-------:|-------:|
| MOT16-02-mini |   0.57 |  0.63 |  0.52 |     0.53 |        0.64 |   0.22 |   0.26 |
| OVERALL       |   0.57 |  0.63 |  0.52 |     0.53 |        0.64 |   0.22 |   0.26 |


## **Training**
In order to prove generalizability I use a *COCO* pretrained MaskRCNN from [here](https://pytorch.org/vision/main/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html). The other components are trained using the commands below.

| component | training command  |
|-----------|-------------------|
| Detector  | ```python train_detector.py --obj_detect_path "config/obj_detect/maskrcnn.json" --data_root_dir "data/MOT16" --train_split "train_wo_val2" --eval_split "val2" --vis_threshold 0.25 --batch_size 2 --num_epochs 20 --learning_rate 0.005``` |
| ReID      | ```python train_reid.py --reid_model_config_path "config/reid_model/reid_on_seg.json" --data_root_dir "data/market" --num_epochs 20 --batch_size_train 32 --learning_rate 0.0003``` |
| Assign Model | ```python train_gnn.py --assign_model_config_path "config/assign_model/gnn.json" --data_root_dir "data/MOT16" --eval_split "val2" --train_split "train_wo_val2" --precomputed_data_root_dir "data/precomputed_detection/default"```|


## **Evaluation**
| component | training command  |
|-----------|-------------------|
| Detector | ```python evaluate_detector.py --obj_detect_path "config/obj_detect/coco_maskrcnn_recall.json" --data_root_dir "data/MOT16" --split "mini"``` |
| ReID | ```python evaluate_reid.py --reid_model_dir "models/reid_model/default_reid" --original_data_root_dir "data/MOT16" --precomputed_data_root_dir "data/precomputed_detection/coco_maskrcnn_recall --split "mini"``` |
| Tracker | ```python evaluate_tracker.py --original_data_root_dir "data/MOT16" --use_precomputed --precomputed_data_root_dir "data/precomputed_detection/coco_maskrcnn_recall" --split "mini" --tracker_config_path "config/tracker/tracker.json"```|
