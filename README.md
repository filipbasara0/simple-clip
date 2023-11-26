# simple-clip
Simple implementation of [CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pretraining) in PyTorch.

<img width="1099" alt="3d5d1009-6e3d-4570-8fd9-ee8f588003e7" src="https://github.com/filipbasara0/simple-clip/assets/29043871/27e708ac-0ced-4382-bcc4-e0db5fc2d115">

# CLIP
[CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pretraining) by OpenAI is a model that unifies text and image understanding through a contrastive learning approach. It employs two neural networks, one for image processing and another for text processing, which are jointly trained on a large dataset of images and their corresponding textual descriptions. This training enables the model to understand and link visual content with natural language. CLIP's distinctive feature is its zero-shot learning capability, allowing it to generalize across various visual tasks without task-specific training, solely based on textual prompts. This makes it highly adaptable for diverse applications in AI, from image classification to complex visual reasoning tasks.

# Results

All experiments used ResNet50 and Distill BERT as respectively image and text encoders. Models were first trained on smaller datasets, such as COCO to validate the approach. Later on, they were trained on combined COCO and sbucaptions data and a yfcc7m subset.

Models were evaluating in zero-shot fashion, where text queries were constructed as "a photo of {label_name}". For ImageNet, we used the 50k validation dataset.

ImageNet results surpassed the [zero-shot scaling trend](https://github.com/mlfoundations/open_clip/blob/main/docs/LOW_ACC.md), by a few points, signalling a potential for smaller but more diverse and information dense datasets. This is in line with https://arxiv.org/abs/2205.01397, where authors determined that the main contributing factor in model quality and robustness for the CLIP objective are more diverse training distribution. In other words, data quality and diversity >> data quantity.

| Training Datasets           | Text Encoder            | Image Encoder | Feature dim | Training steps     | Eval dataset | Top1 % |
|-----------------------------|-------------------------|---------------|-------------|--------------------|--------------|--------|
| yfcc7m + coco + sbucaptions | distilbert-base-uncased | ResNet-50     | 768         | 57,800             | STL-10       | 93.75  |
| yfcc7m + coco + sbucaptions | distilbert-base-uncased | ResNet-50     | 768         | 57,800             | ImageNet     | 34.61  |


# Usage

### Instalation
To setup the code, clone the repository, optionally create a venv and install requirements:

1. `git clone git@github.com:filipbasara0/simple-clip.git`
2. create virtual environment: `virtualenv -p python3.10 env`
3. activate virtual environment: `source env/bin/activate`
4. install requirements: `pip install -r requirements.txt`

Code currently supports ResNet18, ResNet50 and an experimental version of the EfficientNet model as image encoders. Resnet50 was used in all experiments as the image encoder.
Distill BERT (`distilbert-base-uncased`) was used as the text encoder in all experiments.

Supported datasets are textcap, coco, sbucaptions and yfcc7m.

### Examples
`yfcc7m` CLIP was trained with this command (around 7M samples):

`python run_training.py --dataset_name yfcc7m --fp16_precision --batch_size 256  --log_every_n_steps 50 --image_size 224 --learning_rate 1e-4 --imagenet_eval`

Combined `coco + textcaptions + sbucaptions` CLIP was trained using (around 1M samples):

`python run_training.py --dataset_name combined --fp16_precision --batch_size 256  --log_every_n_steps 50 --image_size 224 --learning_rate 1e-4 --imagenet_eval`


### Detailed options
Once the code is setup, run the following command with optinos listed below:
`python run_training.py [args...]⬇️`

```
options:
  -h, --help            show this help message and exit
  --dataset_path DATASET_PATH
                        Path where datasets will be saved
  --dataset_name {textcap,coco,sbucaptions,combined,yfcc7m}
                        Dataset name
  --image_encoder_name {resnet18,resnet50,efficientnet}
                        image model architecture: resnet18, resnet50 or efficientnet (default: resnet50)
  --text_encoder_name {distilbert-base-uncased}
                        text model architecture: distilbert-base-uncased (default: distilbert-base-uncased)
  -save_model_dir SAVE_MODEL_DIR
                        Path where models
  --num_epochs NUM_EPOCHS
                        Number of epochs for training
  --image_size IMAGE_SIZE
                        Image size
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
  --fp16_precision      Whether to use 16-bit precision for GPU training
  --imagenet_eval       Whether to evaluate on imagenet validation dataset. Required huggingface imagenet-1k dataset.
  --imagenet_eval_steps IMAGENET_EVAL_STEPS
                        Evaluate on imagenet every N steps
  --log_every_n_steps LOG_EVERY_N_STEPS
                        Log every n steps
  --ckpt_path CKPT_PATH
                        Specify path to relic_model.pth to resume training
```

# Citation
```
@misc{radford2021learning,
      title={Learning Transferable Visual Models From Natural Language Supervision}, 
      author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
      year={2021},
      eprint={2103.00020},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
