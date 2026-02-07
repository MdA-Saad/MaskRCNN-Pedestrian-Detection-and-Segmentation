# MaskRCNN-Pedestrian-Detection-and-Segmentation
Pedestrian detection and segmentation using Mask R-CNN on the Penn-Fudan dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MdA-Saad/MaskRCNN-Pedestrian-Detection-and-Segmentation/blob/main/pedestriansDetection.ipynb)
[PennFudanPed](https://www.cis.upenn.edu/~jshi/ped_html/) dataset for pedestrian detection and instance segmentation.

## Highlights
- **71.3% mAP@[0.5:0.95]** for bounding box detection  
- **69.0% mAP@[0.5:0.95]** for instance segmentation  
- Full COCO-compatible evaluation using `pycocotools`  
- Custom dataset class with mask-to-bbox conversion and size metadata  
- Visualization of predictions with `torchvision.utils.draw_bounding_boxes` and `draw_segmentation_masks`

## Tech Stack
PyTorch • TorchVision • COCO API • NumPy • Matplotlib
