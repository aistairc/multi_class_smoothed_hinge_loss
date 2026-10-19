import os
import argparse
import time
import csv
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import WarmUpLR

from segmentation_datasets import build_dataset, IGNORE_INDEX


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _raw_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def configure_transfer_mode(model, transfer_mode):
    """Select which part of the segmentation model is trainable.

    head:
        Freeze the ImageNet-pretrained backbone and train only the newly
        initialized segmentation classifier and auxiliary classifier.
    full:
        Fine-tune both backbone and segmentation head.
    """
    transfer_mode = transfer_mode.lower()
    if transfer_mode not in {'head', 'full'}:
        raise ValueError('transfer_mode must be head or full')

    for parameter in model.parameters():
        parameter.requires_grad = transfer_mode == 'full'

    if transfer_mode == 'head':
        if not hasattr(model, 'backbone'):
            raise ValueError('The segmentation model has no backbone attribute')
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

        for module_name in ('classifier', 'aux_classifier'):
            module = getattr(model, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        'transfer mode: {} | trainable parameters: {:,} / {:,} ({:.2f}%)'.format(
            transfer_mode,
            trainable,
            total,
            100.0 * trainable / max(total, 1),
        )
    )


def keep_frozen_backbone_in_eval(model, transfer_mode):
    """Prevent frozen BatchNorm running statistics from changing.

    Calling model.train() recursively switches the backbone to training mode.
    For a true fixed-feature experiment, put the frozen backbone back in eval
    mode after every model.train() call.
    """
    if transfer_mode.lower() == 'head':
        _raw_model(model).backbone.eval()


def trainable_parameters(model):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise RuntimeError('No trainable parameters were selected')
    return params


def get_num_classes(dataset):
    dataset_to_classes = {
        'voc': 21,
        'voc2012': 21,
        'pascal_voc': 21,
        'ade': 150,
        'ade20k': 150,
        'pannuke': 6,
        'pannuke_hf': 6,
        'pannuke_npy': 6,
    }
    if dataset not in dataset_to_classes:
        raise ValueError('Unsupported segmentation dataset: {}'.format(dataset))
    return dataset_to_classes[dataset]


def _canonical_net_name(net_name):
    """Keep the old ResNet command names as DeepLabV3 aliases."""
    aliases = {
        'resnet50': 'deeplabv3_resnet50',
        'resnet101': 'deeplabv3_resnet101',
        'deeplab50': 'deeplabv3_resnet50',
        'deeplab101': 'deeplabv3_resnet101',
        'fcn50': 'fcn_resnet50',
        'fcn101': 'fcn_resnet101',
        'lraspp': 'lraspp_mobilenet_v3_large',
    }
    return aliases.get(net_name, net_name)


def build_segmentation_network(net_name, num_classes, pretrained):
    """
    Build a torchvision semantic-segmentation network.

    pretrained='pytorch' loads an ImageNet-pretrained backbone only.  A custom
    classification checkpoint is loaded into the segmentation backbone later.
    The segmentation classifier/decoder is always initialized for num_classes.
    """
    net_name = _canonical_net_name(net_name)
    use_pytorch_backbone = str(pretrained).lower() == 'pytorch'

    if net_name == 'deeplabv3_resnet50':
        backbone_weights = (
            torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            if use_pytorch_backbone else None
        )
        net = torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            aux_loss=True,
        )
    elif net_name == 'deeplabv3_resnet101':
        backbone_weights = (
            torchvision.models.ResNet101_Weights.IMAGENET1K_V2
            if use_pytorch_backbone else None
        )
        net = torchvision.models.segmentation.deeplabv3_resnet101(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            aux_loss=True,
        )
    elif net_name == 'fcn_resnet50':
        backbone_weights = (
            torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            if use_pytorch_backbone else None
        )
        net = torchvision.models.segmentation.fcn_resnet50(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            aux_loss=True,
        )
    elif net_name == 'fcn_resnet101':
        backbone_weights = (
            torchvision.models.ResNet101_Weights.IMAGENET1K_V2
            if use_pytorch_backbone else None
        )
        net = torchvision.models.segmentation.fcn_resnet101(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
            aux_loss=True,
        )
    elif net_name == 'lraspp_mobilenet_v3_large':
        backbone_weights = (
            torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            if use_pytorch_backbone else None
        )
        net = torchvision.models.segmentation.lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=backbone_weights,
            num_classes=num_classes,
        )
    else:
        raise ValueError(
            'Unsupported segmentation network: {}\n'
            'Supported: resnet50, resnet101, deeplabv3_resnet50, '
            'deeplabv3_resnet101, fcn_resnet50, fcn_resnet101, '
            'lraspp_mobilenet_v3_large'.format(net_name)
        )

    return net


def _extract_state_dict(checkpoint):
    """Accept a raw state_dict or common checkpoint dictionary layouts."""
    if not isinstance(checkpoint, dict):
        raise TypeError('Checkpoint must be a state_dict or a dictionary')

    for key in ('state_dict', 'model_state_dict', 'model', 'net'):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            checkpoint = checkpoint[key]
            break

    cleaned = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith('module.'):
            key = key[len('module.'):]
        cleaned[key] = value
    return cleaned


def _resolve_pretrained_path(pretrained):
    path = Path(pretrained).expanduser()
    candidates = [path, Path('./pretrained') / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        'Custom pretrained checkpoint was not found. Checked: {}'.format(
            ', '.join(str(x) for x in candidates)
        )
    )


def load_classification_pretrained_to_backbone(net, pretrained):
    """
    Load a classification checkpoint into only the segmentation backbone.

    ResNet classification checkpoints generally contain keys such as conv1,
    bn1, layer1, ..., layer4, fc.  The matching convolutional keys are copied
    into net.backbone; classifier/head keys and shape mismatches are ignored.
    """
    pretrained_lower = str(pretrained).lower()
    if pretrained_lower in {'pytorch', 'none', '', 'random'}:
        return

    checkpoint_path = _resolve_pretrained_path(pretrained)
    print('loading classification pretrained backbone from {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    source_state = _extract_state_dict(checkpoint)
    target_state = net.backbone.state_dict()

    loaded_state = {}
    skipped_shape = []

    for source_key, value in source_state.items():
        candidates = [source_key]

        # Also support checkpoints whose keys already contain a backbone/body prefix.
        for prefix in ('backbone.', 'backbone.body.', 'body.', 'model.', 'encoder.'):
            if source_key.startswith(prefix):
                candidates.append(source_key[len(prefix):])

        matched = False
        for candidate_key in candidates:
            if candidate_key not in target_state:
                continue
            if target_state[candidate_key].shape != value.shape:
                skipped_shape.append(
                    (candidate_key, tuple(value.shape), tuple(target_state[candidate_key].shape))
                )
                continue
            loaded_state[candidate_key] = value
            matched = True
            break

        if matched:
            continue

    if len(loaded_state) == 0:
        raise RuntimeError(
            'No checkpoint tensor matched the segmentation backbone. '
            'Check that the classification architecture matches the segmentation backbone.'
        )

    incompatible = net.backbone.load_state_dict(loaded_state, strict=False)
    print(
        'backbone load - matched: {}, missing: {}, unexpected: {}, shape-skipped: {}'.format(
            len(loaded_state),
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
            len(skipped_shape),
        )
    )


class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=255):
        super(MultiClassFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            ignore_index=self.ignore_index,
            reduction='none',
        )
        valid = targets != self.ignore_index
        if not valid.any():
            return inputs.sum() * 0.0

        ce_loss = ce_loss[valid]
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


def _resize_output(outputs, labels):
    if outputs.shape[-2:] != labels.shape[-2:]:
        outputs = F.interpolate(
            outputs,
            size=labels.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
    return outputs


def _main_output(outputs):
    if isinstance(outputs, dict):
        return outputs['out']
    return outputs


def _single_segmentation_loss(outputs, labels):
    """Apply the target-task segmentation loss.

    For the intended ImageNet-backbone transfer comparison, use CE for every
    run.  The older focal/MCSH/hinge paths remain available only for controlled
    ablations and should not be mixed into the primary comparison.
    """
    outputs = _resize_output(outputs, labels)

    if args.loss_name.lower() in {'ce', 'focal'}:
        return loss_function(outputs, labels)

    # Existing classification MCSH/CS/WW implementations expect [N,C] and [N].
    num_classes = outputs.shape[1]
    outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, num_classes)
    labels_flat = labels.reshape(-1)
    valid = labels_flat != IGNORE_INDEX

    if not valid.any():
        return outputs.sum() * 0.0

    return loss_function(outputs_flat[valid], labels_flat[valid])


def calculate_loss(outputs, labels):
    if isinstance(outputs, dict):
        loss = _single_segmentation_loss(outputs['out'], labels)
        if 'aux' in outputs and outputs['aux'] is not None:
            loss = loss + args.aux_weight * _single_segmentation_loss(
                outputs['aux'], labels
            )
        return loss
    return _single_segmentation_loss(outputs, labels)


def _update_confusion_matrix(confusion_matrix, predictions, labels, num_classes):
    valid = labels != IGNORE_INDEX
    labels_valid = labels[valid]
    predictions_valid = predictions[valid]

    if labels_valid.numel() == 0:
        return 0

    encoded = labels_valid * num_classes + predictions_valid
    bincount = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    confusion_matrix += bincount.cpu()
    return int(labels_valid.numel())


def _metrics_from_confusion(confusion_matrix):
    confusion = confusion_matrix.double()
    diagonal = torch.diag(confusion)
    ground_truth = confusion.sum(dim=1)
    predicted = confusion.sum(dim=0)
    union = ground_truth + predicted - diagonal

    iou = torch.full_like(diagonal, float('nan'))
    valid_iou = union > 0
    iou[valid_iou] = diagonal[valid_iou] / union[valid_iou]

    class_accuracy = torch.full_like(diagonal, float('nan'))
    valid_class = ground_truth > 0
    class_accuracy[valid_class] = diagonal[valid_class] / ground_truth[valid_class]

    total = confusion.sum()
    pixel_accuracy = (diagonal.sum() / total).item() if total > 0 else 0.0
    mean_accuracy = class_accuracy[valid_class].mean().item() if valid_class.any() else 0.0
    miou = iou[valid_iou].mean().item() if valid_iou.any() else 0.0

    no_background = valid_iou.clone()
    if len(no_background) > 0:
        no_background[0] = False
    miou_no_background = iou[no_background].mean().item() if no_background.any() else 0.0

    return {
        'pixel_accuracy': pixel_accuracy,
        'mean_accuracy': mean_accuracy,
        'miou': miou,
        'miou_no_background': miou_no_background,
        'class_iou': iou,
    }


def train(epoch):

    start = time.time()
    net.train()
    keep_frozen_backbone_in_eval(net, args.transfer_mode)
    running_loss = 0.0

    for batch_index, batch in enumerate(trainval_dataloader):
        images = batch['image']
        labels = batch['target']

        if args.gpu:
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)

        if args.optimizer == 'sam':
            optimizer.zero_grad()
            outputs = net(images)
            loss = calculate_loss(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            second_outputs = net(images)
            second_loss = calculate_loss(second_outputs, labels)
            second_loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            outputs = net(images)
            loss = calculate_loss(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        n_iter = (epoch - 1) * len(trainval_dataloader) + batch_index + 1

        # update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        if epoch <= args.warm:
            warmup_scheduler.step()

    if not args.no_histogram:
        for name, param in net.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram('{}/{}'.format(layer, attr), param, epoch)

    finish = time.time()
    print(
        'epoch {} training time consumed: {:.2f}s LR: {:0.6f} Loss: {:.4f}'.format(
            epoch,
            finish - start,
            optimizer.param_groups[0]['lr'],
            running_loss / max(len(trainval_dataloader), 1),
        )
    )


@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0
    loss_batches = 0
    confusion_matrix = torch.zeros(
        (num_classes, num_classes), dtype=torch.int64
    )

    true_confidence = 0.0
    true_count = 0
    false_confidence = 0.0
    false_count = 0

    for batch in test_dataloader:
        images = batch['image']
        labels = batch['target']

        if args.gpu:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        output_dict = net(images)
        loss = calculate_loss(output_dict, labels)
        outputs = _resize_output(_main_output(output_dict), labels)

        test_loss += loss.item()
        loss_batches += 1

        predictions = outputs.argmax(dim=1)
        _update_confusion_matrix(
            confusion_matrix,
            predictions,
            labels,
            num_classes,
        )

        valid = labels != IGNORE_INDEX
        correct = predictions.eq(labels) & valid
        incorrect = predictions.ne(labels) & valid
        confidence = torch.softmax(outputs, dim=1).amax(dim=1)

        if correct.any():
            true_confidence += confidence[correct].sum().item()
            true_count += int(correct.sum().item())
        if incorrect.any():
            false_confidence += confidence[incorrect].sum().item()
            false_count += int(incorrect.sum().item())

    metrics = _metrics_from_confusion(confusion_matrix)
    average_loss = test_loss / max(loss_batches, 1)
    true_confidence_mean = true_confidence / (true_count + 1e-7)
    false_confidence_mean = false_confidence / (false_count + 1e-7)

    finish = time.time()
    print('Evaluating Network.....')
    print(
        'Validation set: Epoch: {}, Average loss: {:.4f}, '
        'mIoU: {:.4f}, mIoU(no-bg): {:.4f}, Pixel Acc: {:.4f}, '
        'Mean Acc: {:.4f}, Time consumed: {:.2f}s'.format(
            epoch,
            average_loss,
            metrics['miou'],
            metrics['miou_no_background'],
            metrics['pixel_accuracy'],
            metrics['mean_accuracy'],
            finish - start,
        )
    )
    print()

    if tb:
        writer.add_scalar('Validation/Average loss', average_loss, epoch)
        writer.add_scalar('Validation/mIoU', metrics['miou'], epoch)
        writer.add_scalar(
            'Validation/mIoU_no_background',
            metrics['miou_no_background'],
            epoch,
        )
        writer.add_scalar(
            'Validation/Pixel_accuracy', metrics['pixel_accuracy'], epoch
        )
        writer.add_scalar(
            'Validation/Mean_class_accuracy', metrics['mean_accuracy'], epoch
        )
        writer.add_scalar(
            'Validation/True_confidence', true_confidence_mean, epoch
        )
        writer.add_scalar(
            'Validation/False_confidence', false_confidence_mean, epoch
        )

    return (
        metrics['miou'],
        metrics['miou_no_background'],
        metrics['pixel_accuracy'],
        true_confidence_mean,
        false_confidence_mean,
    )


def _pannuke_npy_fold_source(base_source, fold):
    """Resolve fold1/fold2/fold3 against common original PanNuke folder names."""
    base = Path(base_source).expanduser()
    fold_number = int(fold[-1])
    candidates = [
        base / fold,
        base / ('Fold ' + str(fold_number)),
        base / ('Fold' + str(fold_number)),
        base / ('fold_' + str(fold_number)),
        base / str(fold_number),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(base)


def build_segmentation_datasets(args):
    dataset_name = args.dataset.lower()
    root = Path(args.data_root).expanduser()

    if dataset_name in {'voc', 'voc2012', 'pascal_voc'}:
        train_dataset = build_dataset(
            name='voc',
            root=root,
            split=args.voc_train_split,
            train=True,
            crop_size=args.imgsize,
            download=args.download,
        )
        validation_dataset = build_dataset(
            name='voc',
            root=root,
            split='val',
            train=False,
            crop_size=args.val_imgsize,
            download=False,
        )

    elif dataset_name in {'ade', 'ade20k'}:
        train_dataset = build_dataset(
            name='ade20k',
            root=root,
            split='training',
            train=True,
            crop_size=args.imgsize,
        )
        validation_dataset = build_dataset(
            name='ade20k',
            root=root,
            split='validation',
            train=False,
            crop_size=args.val_imgsize,
        )

    elif dataset_name in {'pannuke', 'pannuke_hf', 'pannuke_npy'}:
        folds = ['fold1', 'fold2', 'fold3']
        validation_fold = folds[args.foldnum % len(folds)]
        training_folds = [fold for fold in folds if fold != validation_fold]

        pannuke_source = args.pannuke_source
        if args.pannuke_backend == 'hf':
            if not pannuke_source:
                offline_source = root / 'PanNuke_hf'
                pannuke_source = str(offline_source) if offline_source.exists() else None
        elif not pannuke_source:
            raise ValueError(
                '-pannuke_backend npy requires -pannuke_source pointing to the '
                'original PanNuke fold root'
            )

        train_parts = []
        for fold in training_folds:
            fold_source = pannuke_source
            if args.pannuke_backend == 'npy':
                fold_source = _pannuke_npy_fold_source(pannuke_source, fold)
            train_parts.append(
                build_dataset(
                    name=dataset_name,
                    root=root / 'hf_cache',
                    split=fold,
                    train=True,
                    crop_size=args.imgsize,
                    pannuke_backend=args.pannuke_backend,
                    pannuke_source=fold_source,
                )
            )
        train_dataset = ConcatDataset(train_parts)

        validation_source = pannuke_source
        if args.pannuke_backend == 'npy':
            validation_source = _pannuke_npy_fold_source(
                pannuke_source, validation_fold
            )
        validation_dataset = build_dataset(
            name=dataset_name,
            root=root / 'hf_cache',
            split=validation_fold,
            train=False,
            crop_size=args.val_imgsize,
            pannuke_backend=args.pannuke_backend,
            pannuke_source=validation_source,
        )
        print(
            'PanNuke fold split - train: {}, validation: {}'.format(
                training_folds, validation_fold
            )
        )
    else:
        raise ValueError('Unsupported dataset: {}'.format(args.dataset))

    return train_dataset, validation_dataset


def _pretrained_tag(pretrained):
    value = str(pretrained)
    if value.lower() in {'pytorch', 'none', 'random', ''}:
        return value.lower() or 'none'
    return Path(value).stem


def _load_resume_weights(model, resume_path):
    checkpoint_path = Path(resume_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError('resume checkpoint not found: {}'.format(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)

    start_epoch = 0
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        start_epoch = int(checkpoint['epoch'])
    else:
        # Old-style filename: network-123-regular.pth
        for token in checkpoint_path.stem.split('-'):
            if token.isdigit():
                start_epoch = int(token)
    print('loaded resume weights: {} (epoch {})'.format(checkpoint_path, start_epoch))
    return start_epoch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='segmentation net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=4, help='training batch size')
    parser.add_argument('-val_b', type=int, default=1, help='validation batch size')
    parser.add_argument('-num_classes', type=int, default=0, help='0: infer from dataset')
    parser.add_argument('-warm', type=int, default=5, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-resume_path', type=str, default='', help='segmentation checkpoint path')
    parser.add_argument('-imgsize', type=int, default=512, help='training crop size')
    parser.add_argument('-val_imgsize', type=int, default=512, help='validation resize size')
    parser.add_argument('-dataset', type=str, required=True, help='voc / ade20k / pannuke')
    parser.add_argument('-data_root', type=str, default='./data', help='dataset root')
    parser.add_argument('-workers', type=int, default=16, help='dataloader workers')
    parser.add_argument('-download', action='store_true', default=False, help='VOC download')
    parser.add_argument('-voc_train_split', type=str, default='trainaug', choices=['train', 'trainaug'])
    parser.add_argument('-pannuke_backend', type=str, default='hf', choices=['hf', 'npy'])
    parser.add_argument('-pannuke_source', type=str, default='', help='PanNuke save_to_disk or NPY root')
    parser.add_argument('-weight_decay', default=2e-5, type=float)
    parser.add_argument('-label_smooth', default=0.0, type=float)
    parser.add_argument('-optimizer', default='adam', type=str, choices=['adam', 'sam'])
    parser.add_argument('-rho', type=float, default=0.05, help='rho for sam')
    parser.add_argument('-mcsh_margin', default=0.0, type=float)
    parser.add_argument('-loss_name', default='CE', type=str, metavar='loss name')
    parser.add_argument('-pretrained', type=str, default='pytorch', help='ImageNet checkpoint: pytorch / none / checkpoint')
    parser.add_argument('-transfer_mode', type=str, default='head', choices=['head', 'full'], help='head: freeze backbone; full: fine-tune all')
    parser.add_argument('-seed', type=int, default=1, help='random seed; keep equal across pretrained checkpoints')
    parser.add_argument('-foldnum', type=int, default=0, help='PanNuke validation fold index: 0/1/2')
    parser.add_argument('-epochs', type=int, default=200)
    parser.add_argument('-milestones', type=str, default='100,150,180')
    parser.add_argument('-aux_weight', type=float, default=0.4)
    parser.add_argument('-save_best_after', type=int, default=0)
    parser.add_argument('-best_metric', type=str, default='miou', choices=['miou', 'miou_nobg'])
    parser.add_argument('-no_histogram', action='store_true', default=False)

    args = parser.parse_args()

    set_random_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    if args.gpu and not use_cuda:
        raise RuntimeError('GPU was requested but CUDA is not available')

    num_classes = args.num_classes if args.num_classes > 0 else get_num_classes(args.dataset.lower())
    net_name = _canonical_net_name(args.net)

    net = build_segmentation_network(
        net_name=net_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
    )
    load_classification_pretrained_to_backbone(net, args.pretrained)
    configure_transfer_mode(net, args.transfer_mode)

    if args.gpu:
        net = net.cuda()

    print('loading network -done')
    print('network: {}, classes: {}'.format(net_name, num_classes))
    print('ImageNet backbone checkpoint: {}'.format(args.pretrained))
    print('target segmentation loss: {}'.format(args.loss_name))
    print('Training with: ' + str(torch.cuda.device_count()) + ' gpus')
    if args.gpu and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    EPOCH = args.epochs
    MILESTONES = [int(x) for x in args.milestones.split(',') if x.strip()]

    trainval_dataset, test_dataset = build_segmentation_datasets(args)
    print(
        '==> Segmentation dataset {} / train (size: {})'.format(
            args.dataset, len(trainval_dataset)
        )
    )
    print(
        '==> Segmentation dataset {} / validation (size: {})'.format(
            args.dataset, len(test_dataset)
        )
    )

    trainval_dataloader = DataLoader(
        trainval_dataset,
        batch_size=args.b,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.gpu,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.val_b,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.gpu,
        drop_last=False,
    )

    if args.loss_name.lower() != 'ce':
        print(
            'WARNING: primary transfer comparison should keep -loss_name CE '
            'for every ImageNet checkpoint. Current target loss: {}'.format(args.loss_name)
        )

    if args.loss_name.upper() == 'MCSH':
        from custom_loss import MultiClassMarginHingeHuber
        loss_function = MultiClassMarginHingeHuber(
            margin=args.mcsh_margin
        )
    elif args.loss_name.lower() == 'focal':
        loss_function = MultiClassFocalLoss(ignore_index=IGNORE_INDEX)
    elif args.loss_name.lower() == 'cs':
        from custom_loss import CS
        loss_function = CS()
    elif args.loss_name.lower() == 'ww':
        from custom_loss import WW
        loss_function = WW()
    else:
        loss_function = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX,
            label_smoothing=args.label_smooth,
        )

    if args.gpu:
        loss_function = loss_function.cuda()

    if args.optimizer == 'sam':
        from optim_sam import SAM
        optimizer = SAM(
            trainable_parameters(net),
            torch.optim.SGD,
            rho=args.rho,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.Adam(
            trainable_parameters(net),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    train_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=MILESTONES, gamma=0.2
    )
    iter_per_epoch = len(trainval_dataloader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    pretrained_tag = _pretrained_tag(args.pretrained)
    path_name = (
        pretrained_tag + '_' + args.loss_name + str(args.mcsh_margin)
        + '_data' + args.dataset
        + '_fold' + str(args.foldnum)
        + '_' + args.optimizer
        + '_ls' + str(args.label_smooth)
        + '_' + net_name
        + '_' + args.transfer_mode
        + '_seed' + str(args.seed)
    )

    checkpoint_directory = os.path.join(
        settings.CHECKPOINT_PATH,
        args.dataset,
        path_name,
        settings.TIME_NOW,
    )

    csv_path = './Segmentation_Result_' + path_name + '.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                [
                    'name', 'description', 'pretrained', 'transfer_mode', 'seed',
                    'best_mIoU', 'last_mIoU',
                    'best_mIoU_no_background', 'last_mIoU_no_background',
                    'best_pixel_accuracy', 'last_pixel_accuracy',
                ]
            )

    if not os.path.exists(settings.LOG_DIR):
        os.makedirs(settings.LOG_DIR, exist_ok=True)

    writer = SummaryWriter(
        log_dir=os.path.join(
            settings.LOG_DIR,
            args.net,
            path_name,
            settings.TIME_NOW,
        )
    )

    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    checkpoint_path = os.path.join(
        checkpoint_directory, '{net}-{epoch}-{type}.pth'
    )

    best_miou = 0.0
    best_miou_no_background = 0.0
    best_pixel_accuracy = 0.0
    best_save_score = -1.0
    resume_epoch = 0

    raw_net = _raw_model(net)
    if args.resume:
        if not args.resume_path:
            raise ValueError('-resume requires -resume_path')
        resume_epoch = _load_resume_weights(raw_net, args.resume_path)

    last_miou = 0.0
    last_miou_no_background = 0.0
    last_pixel_accuracy = 0.0

    for epoch in range(1, EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume and epoch <= resume_epoch:
            continue

        train(epoch)
        (
            miou,
            miou_no_background,
            pixel_accuracy,
            true_confidence,
            false_confidence,
        ) = eval_training(epoch)

        last_miou = miou
        last_miou_no_background = miou_no_background
        last_pixel_accuracy = pixel_accuracy

        best_miou = max(best_miou, miou)
        best_miou_no_background = max(
            best_miou_no_background, miou_no_background
        )
        best_pixel_accuracy = max(best_pixel_accuracy, pixel_accuracy)

        score = miou if args.best_metric == 'miou' else miou_no_background

        if epoch > args.save_best_after and best_save_score < score:
            weights_path = checkpoint_path.format(
                net=args.net, epoch=epoch, type='best'
            )
            print('saving weights file to {}'.format(weights_path))
            torch.save(raw_net.state_dict(), weights_path)
            best_save_score = score
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(
                net=args.net, epoch=epoch, type='regular'
            )
            print('saving weights file to {}'.format(weights_path))
            torch.save(raw_net.state_dict(), weights_path)

    writer.close()

    with open(csv_path, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
                args.net,
                path_name,
                args.pretrained,
                args.transfer_mode,
                args.seed,
                '{:.7f}'.format(best_miou),
                '{:.7f}'.format(last_miou),
                '{:.7f}'.format(best_miou_no_background),
                '{:.7f}'.format(last_miou_no_background),
                '{:.7f}'.format(best_pixel_accuracy),
                '{:.7f}'.format(last_pixel_accuracy),
            ]
        )
