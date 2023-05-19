# %%
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transforms import RandomAdjustBrightness, RandomAdjustContrast, RandomVerticalFlip, RandomHorizontalFlip, \
    RandomAdjustColor
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset, get_coco, get_coco_kp
import transforms as T
import utils


def get_dataset(name, image_set, transform, data_path):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2),
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes


def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
          'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
          'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# dataset, num_classes = get_dataset("coco", "train", get_transform(),"data/coco")
# dataset_test, num_classes = get_dataset("coco", "val", get_transform(), "data/coco")
dataset_test, num_classes = get_dataset("coco", "val", get_transform(), r"E:\BaiduNetdiskDownload\coco2017")
# train_sampler = torch.utils.data.RandomSampler(dataset)
test_sampler = torch.utils.data.SequentialSampler(dataset_test)

# %%
# data_loader = torch.utils.data.DataLoader(
#         dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
#         collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1,
    sampler=test_sampler,
    collate_fn=utils.collate_fn)

# %%
# from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.to(device)
model.train()
1


# %%
def ifgsm_attack(image, target, epsilon, model, alpha, num_iter, momentum=0.3):
    model.train()

    if isinstance(target, list):
        target = target[0]
    rf_hori = RandomHorizontalFlip(0.5)
    rf_vert = RandomVerticalFlip(0.5)
    rf_hori_ = RandomHorizontalFlip(1)
    rf_vert_ = RandomVerticalFlip(1)
    rf_contrast = RandomAdjustContrast(0.4, 0.5)
    rf_brightness = RandomAdjustBrightness(0.4, 0.5)
    rf_color = RandomAdjustColor(0.3, 0.3)
    image.requires_grad = True
    last_grad = None
    for i in range(num_iter):
        image_hori, _, hori_trans = rf_hori(image, target)
        image_hori = image_hori.detach()
        image_hori.requires_grad = True
        image_vert, _, vert_trans = rf_vert(image, target)
        image_vert = image_vert.detach()
        image_vert.requires_grad = True
        image_contrast, _, contrast_trans = rf_contrast(image, target)
        image_contrast = image_contrast.detach()
        image_contrast.requires_grad = True
        image_brightness, _, brightness_trans = rf_brightness(image, target)
        image_brightness = image_brightness.detach()
        image_brightness.requires_grad = True
        image_color, _, color_trans = rf_color(image, target)
        image_color = image_color.detach()
        image_color.requires_grad = True

        loss_dict = model(image, [target])
        loss_hori_dict = model(image_hori, [target]) if hori_trans else {"0": 0}
        loss_vert_dict = model(image_vert, [target]) if vert_trans else {"0": 0}
        loss_contrast_dict = model(image_contrast, [target]) if contrast_trans else {"0": 0}
        loss_brightness_dict = model(image_brightness, [target]) if brightness_trans else {"0": 0}
        loss_color_dict = model(image_color, [target]) if color_trans else {"0": 0}
        loss = sum(loss for loss in loss_dict.values())
        loss_hori = sum(loss for loss in loss_hori_dict.values())
        loss_vert = sum(loss for loss in loss_vert_dict.values())
        loss_contrast = sum(loss for loss in loss_contrast_dict.values())
        loss_brightness = sum(loss for loss in loss_brightness_dict.values())
        loss_color = sum(loss for loss in loss_color_dict.values())
        loss = loss + loss_hori + loss_vert + loss_contrast + loss_brightness + loss_color
        loss = loss / len([l for l in [loss, loss_hori, loss_vert, loss_contrast, loss_brightness, loss_color] if l])
        loss.backward()
        image_grad = image.grad.data
        image_hori_grad = image_hori.grad.data if hori_trans else None
        image_vert_grad = image_vert.grad.data if vert_trans else None
        image_contrast_grad = image_contrast.grad.data if contrast_trans else None
        image_brightness_grad = image_brightness.grad.data if brightness_trans else None
        image_color_grad = image_color.grad.data if color_trans else None
        if hori_trans:
            image_hori_grad, _, hori_trans = rf_hori_(image_hori_grad, target)
        if vert_trans:
            image_vert_grad, _, vert_trans = rf_vert_(image_vert_grad, target)
        grad_list = [image_grad]
        if hori_trans:
            grad_list.append(image_hori_grad)
        if vert_trans:
            grad_list.append(image_vert_grad)
        if contrast_trans:
            grad_list.append(image_contrast_grad)
        if brightness_trans:
            grad_list.append(image_brightness_grad)
        if color_trans:
            grad_list.append(image_color_grad)
        image_grad = torch.cat(grad_list, dim=0).mean(dim=0)
        if last_grad is not None:
            image_grad = image_grad + momentum * last_grad
            last_grad = image_grad
        else:
            last_grad = image_grad
        perturbed_image = image + alpha * image_grad.sign()
        eta = torch.clamp(perturbed_image - image, min=-epsilon, max=epsilon)
        image = torch.clamp(image + eta, min=0, max=1).detach_()
        image.requires_grad = True
    model.eval()
    return image


# %%
def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


# %%
#from tqdm.notebook import tqdm
from tqdm import tqdm


def test_2000_samples(model, data_loader_test, device, num_iter=1, epsilon=0.1, alpha=0.1, momentum=0.3):
    model.eval()
    cpu_device = torch.device("cpu")
    iou_types = _get_iou_types(model)
    coco = get_coco_api_from_dataset(data_loader_test.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    for idx, (images, targets) in tqdm(enumerate(data_loader_test), total=10):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        images = ifgsm_attack(images[0].unsqueeze(0), targets, epsilon, model, alpha, num_iter, momentum=momentum)
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images, targets)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)
        if idx == 2000:
            break
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


# %%
test_2000_samples(model, data_loader_test, device, num_iter=5, epsilon=0.001, alpha=0.001)

# %%
from matplotlib import patches

cpu_device = torch.device("cpu")
for idx, (images, targets) in enumerate(data_loader_test):
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    images_adv = ifgsm_attack(images[0].unsqueeze(0), targets, 0.001, model, 0.001, 5, True, True, True, True)
    images_adv = list(image.to(device) for image in images_adv)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    outputs = model(images, targets)
    outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
    outputs_adv = model(images_adv, targets)
    outputs_adv = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs_adv]
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(images[0].cpu().detach().permute(1, 2, 0))
    for item in outputs:
        boxes = item['boxes']
        labels = item['labels']
        scores = item['scores']
        keep_thresh = 0.2
        keep = scores > keep_thresh
        boxes = boxes[keep].cpu().detach().numpy()
        labels = labels[keep].cpu().detach().numpy()
        for box, label in zip(boxes, labels):
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
            # plt.text(box[0],box[1],labels[label-1],fontsize=12,color='r')
            # axes[0].text(box[0],box[1],labels[label-1],fontsize=12,color='r')
            axes[0].add_patch(rect)
    axes[1].imshow(images_adv[0].cpu().detach().permute(1, 2, 0))
    for item in outputs_adv:
        boxes = item['boxes']
        labels = item['labels']
        scores = item['scores']
        keep_thresh = 0.2
        keep = scores > keep_thresh
        boxes = boxes[keep].cpu().detach().numpy()
        labels = labels[keep].cpu().detach().numpy()
        for box, label in zip(boxes, labels):
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
            # plt.text(box[0],box[1],labels[label-1],fontsize=12,color='r')
            # axes[1].text(box[0],box[1],labels[label-1],fontsize=12,color='r')
            axes[1].add_patch(rect)
    plt.show()
    #plt.savefig
    break

# %%
