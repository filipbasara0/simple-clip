import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import random
import numpy as np
from tqdm.auto import tqdm
from torchinfo import summary
import webdataset as wds

from simple_clip import CLIP, contrastive_loss, siglip_loss

from simple_clip.utils import accuracy, get_dataset, get_image_encoder, get_text_encoder
from simple_clip.custom_datasets.clip_datasets import get_image_tranforms
from simple_clip.imagenet_eval import ImageNetValidation

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def train_clip(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_encoder = get_image_encoder(args.image_encoder_name)
    text_encoder = get_text_encoder(args.text_encoder_name)
    model = CLIP(image_encoder, text_encoder)

    img = torch.rand(1, 3, args.image_size, args.image_size)
    txt = torch.randint(high=20000, size=(1, 100))
    att_mask = torch.randint(high=1, size=(1, 100))
    summary(model, input_data=[img, txt, att_mask])

    if args.ckpt_path:
        model_state = torch.load(args.ckpt_path)
        model.load_state_dict(model_state)
    model = model.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    transforms_inference = get_image_tranforms(
        (args.image_size, args.image_size))
    ds = get_dataset(args.dataset_name,
                     args.dataset_path,
                     transforms=transforms_inference)
    if args.dataset_name == "yfcc7m":
        train_loader = wds.WebLoader(ds, num_workers=2, batch_size=args.batch_size)
        # hardcoded for num images = 7329280 :/
        steps_per_epcoch = 7329280 // args.batch_size
    else:
        train_loader = DataLoader(ds,
                              batch_size=args.batch_size,
                              num_workers=12,
                              drop_last=True,
                              shuffle=True)
        steps_per_epcoch = len(train_loader)

    tmax = args.num_epochs * steps_per_epcoch + steps_per_epcoch // 4
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(tmax), eta_min=1e-8)

    # Always use coco val dataset for standardization purposes
    ds_val = get_dataset("coco",
                         args.dataset_path,
                         transforms=transforms_inference,
                         split="validation",
                         shuffle_captions=False)
    val_loader = DataLoader(ds_val,
                            batch_size=min(args.batch_size, 256),
                            num_workers=4)
    
    if args.imagenet_eval:
        imgnet_val = ImageNetValidation(transforms_inference)

    scaler = GradScaler(enabled=args.fp16_precision)

    global_step = 0
    total_loss = 0.0
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", total=steps_per_epcoch)
        for step, (batch) in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'image']}

            with autocast(enabled=args.fp16_precision):
                logits = model(**batch)
                if args.use_siglip:
                    loss = siglip_loss(logits)
                else:
                    loss = contrastive_loss(logits)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            epoch_loss += loss.item()
            avg_loss = total_loss / (global_step + 1)
            ep_loss = epoch_loss / (step + 1)

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_description(
                f"Epoch {epoch+1}/{args.num_epochs} | "
                f"Step {global_step+1} | "
                f"Epoch Loss: {ep_loss:.4f} |"
                f"Total Loss: {avg_loss:.4f} |"
                f"Lr: {current_lr:.8f}")

            global_step += 1
            if global_step % args.log_every_n_steps == 0:
                batch_size = logits.shape[0]
                labels = torch.arange(batch_size).to(logits.device)
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                print('acc/top1 images', top1[0].item())
                print('acc/top5 images', top5[0].item())
                top1, top5 = accuracy(logits.t(), labels, topk=(1, 5))
                print('acc/top1 texts', top1[0].item())
                print('acc/top5 texts', top5[0].item())

                torch.save(model.state_dict(),
                           f"{args.save_model_dir}/clip_model.pth")

            if global_step % (args.log_every_n_steps * 10) == 0:
                validate(model, val_loader, device)
            
            if args.imagenet_eval and global_step % args.imagenet_eval_steps == 0:
                imgnet_val.evaluate(model)

            lr_scheduler.step()


def validate(model, val_loader, device):
    top1_acc_images = []
    top5_acc_images = []
    top1_acc_texts = []
    top5_acc_texts = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(**batch)

            labels = torch.arange(logits.size(0)).to(device)
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            top1_acc_images.append(top1[0].item())
            top5_acc_images.append(top5[0].item())
            top1, top5 = accuracy(logits.t(), labels, topk=(1, 5))
            top1_acc_texts.append(top1[0].item())
            top5_acc_texts.append(top5[0].item())
    print("#" * 100)
    print('eval acc/top1 images', np.mean(top1_acc_images))
    print('eval acc/top5 images', np.mean(top5_acc_images))
    print('eval acc/top1 texts', np.mean(top1_acc_texts))
    print('eval acc/top5 texts', np.mean(top5_acc_texts))
