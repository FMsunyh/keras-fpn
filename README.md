# keras_frcnn
## Requirements
Basically, this code supports python3.5+, the following package should installed:
- python 3.5
- tensorflow 1.10
- keras 2.2.4
- cv2
- numpy (pip install numpy --user)

That's all, help you enjoy!

## Train RPN
```bash
cd tools
python train_rpn.py pascal /home/syh/train_data/VOCdevkit/VOC2007
```

## Train Faster RCNN
config the args_setting file in experiments/cfgs.
the context may look like as following:
```bash
TRAIN:
  gpu: 0
  weight_path: /home/syh/keras_frcnn/snapshots/voc/voc_04.h5
  epochs: 50
  tag: voc
  pascal_path: ['/home/syh/train_data/VOCdevkit/VOC2007']
  classes_path: /home/syh/keras_frcnn/voc_classes.txt
```

and then
```bash
cd tools
python train_rcnn.py
```

# evaluate
save path: .../experiments/eval_output

# Implemente
