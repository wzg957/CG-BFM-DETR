# CG-BFM-DETR: Context-Guided Bidirectional Feature Modulation for UAV Object Detection

* This repository contains the official PyTorch implementation for our proposed **CG-BFM-DETR**. 
* Our work focuses on tiny and small object detection in UAV scenarios, and the code is built upon the excellent [DQ-DETR](https://github.com/hoiliu-0801/DQ-DETR) and [DINO](https://github.com/IDEA-Research/DINO) repositories.

> **Note:** This repository currently provides the core implementation of the CG-BFM-DETR architecture used in our experiments. The code is provided for research reference to demonstrate the feasibility of our proposed modules. The full training pipeline and comprehensive data processing scripts will be updated gradually.

## 🚀 News
* The core source code for CG-BFM-DETR has been released!

## 📦 Installation
The environment configuration is highly similar to the official DINO-DETR and DQ-DETR.

```bash
conda create -n cgbfm python=3.9 -y
conda activate cgbfm
pip install -r requirements.txt
# Compiling CUDA operators
bash install.sh
```

## 📊 Dataset Preparation
We evaluate our model on the public **VisDrone2019** dataset. 

* **Step 1:** Download the dataset from the [official VisDrone repository](https://github.com/VisDrone/VisDrone-Dataset).
* **Step 2:** Organize the downloaded files in the following way:

```text
├─ datasets
│   └─ visdrone
│       ├─ annotations
│       ├─ train
│       ├─ val
│       └─ test
├─ CG-BFM-DETR
```

## 🧩 Context Templates Preparation (Important!)
The CG-BFM module relies on offline context templates (spatial and relation priors) for bidirectional feature modulation. Before training on any dataset, you **must** extract the context templates first by running:

```bash
python mine_context_templates.py
```
*(Note: Please ensure the annotation and image paths in the script point to your corresponding dataset directories).*

## 🏋️ Pretrained Weights
The Pretrained Weights and best model weights evaluated on the VisDrone dataset are hosted externally:

* [Baidu Netdisk Link] (Password: xxxx) *(Link to be updated after blind review)*

## 💻 Training
Our training strategy adopts a meticulous **two-stage progressive fine-tuning** approach to ensure the stability of the bidirectional feature modulation.
https://drive.google.com/drive/folders/1oqlVAMqEKrxeCYp-p3-gn3cmXPuyyKFQ?usp=sharing.

**Stage 1: Initial Feature Alignment**
```bash
CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=2 --master_port=29501 main_aitod.py \
  -c config/DQ_5scale.py \
  --output_dir /path/to/output_stage1 \
  --coco_path /path/to/your/dataset \
  --pretrain_model_path /path/to/pretrain_model.pth \
  --finetune_ignore class_embed label_enc \
  --options lr=1e-4 lr_backbone=1e-5 num_classes=10 dn_labelbook_size=10 print_freq=2000 ccm_coeff=1.0 clip_max_norm=0.1 find_unused_parameters=True epochs=12
```

**Stage 2: Progressive Micro-Finetuning**
```bash
CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=2 --master_port=29501 main_aitod.py \
  -c config/DQ_5scale.py \
  --output_dir /path/to/output_stage2 \
  --coco_path /path/to/your/dataset \
  --resume /path/to/output_stage1/checkpoint.pth \
  --options lr=1e-4 lr_backbone=1e-5 num_classes=10 dn_labelbook_size=10 ccm_coeff=0.01 clip_max_norm=0.1 find_unused_parameters=True epochs=24 multi_step_lr=True lr_drop_list=[13,23]
```

## 📈 Evaluation
To evaluate the model performance on the test set, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1 CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=2 --master_port=29522 main_aitod.py \
  -c config/DQ_5scale.py \
  --output_dir /path/to/eval_output \
  --coco_path /path/to/your/dataset \
  --resume /path/to/your/checkpoint_best.pth \
  --eval \
  --options num_classes=10 dn_labelbook_size=10 ccm_coeff=0.01 use_cgfe=False find_unused_parameters=True
```

## 🙏 Acknowledgements
This code is heavily inspired by and built upon [DQ-DETR](https://github.com/hoiliu-0801/DQ-DETR). We sincerely thank the authors for their excellent open-source contributions to the tiny object detection community.
