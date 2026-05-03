import argparse
import json
import logging
import os
import os.path
import time

import torch
import yaml

from util.seeding import set_seed


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('LeakyReLU') != -1:
        m.inplace = True


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def build_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-08,
                                 weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer: {name}")


def build_lr_lambda(cfg):
    """Returns a function epoch -> lr_multiplier matching cfg's schedule."""
    gamma = cfg["lr_decay_gamma"]
    if cfg["lr_decay"] == "step":
        step = cfg["lr_decay_step_size"]
        if step <= 0:
            raise ValueError(
                f"lr_decay=step requires lr_decay_step_size > 0, got {step}"
            )
        return lambda e: gamma ** (e // step)
    if cfg["lr_decay"] == "multistep":
        milestones = sorted(cfg["lr_decay_milestones"])
        def f(e):
            n = sum(1 for m in milestones if e >= m)
            return gamma ** n
        return f
    raise ValueError(f"unsupported lr_decay: {cfg['lr_decay']}")


def main():
    parser = argparse.ArgumentParser(description="LGANet training")
    parser.add_argument("--config", default="configs/paper.yaml",
                        help="Path to YAML config file (default: configs/paper.yaml)")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for logs and checkpoints (default: logs/<timestamp>)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Deterministic seeding — must happen before any randomness
    set_seed(cfg["seed"])

    # Resolve output directory
    log_path = 'weight_' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if args.output_dir is not None:
        logs_dir = args.output_dir
    else:
        logs_dir = os.path.join('logs', log_path)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Logger setup
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file = os.path.join(logs_dir, 'log_embedding.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log the resolved config
    logger.info("Resolved config: %s", json.dumps(cfg, indent=2))

    # Defer heavy imports until after argparse (so --help works without CUDA ext)
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from model.sem.GraphAttention import graphAttention_seg_repro as Model
    from util.data_util import collate_fn
    from util.sealingNails_npz import SealingNailDatasetNPZ

    # Unpack config values
    root = cfg["data_root"]
    num_class = cfg["num_classes"]
    num_points = cfg["num_points"]
    end_epoch = cfg["epoch"]
    learning_rate = cfg["learning_rate"]
    weight_decay = cfg["weight_decay"]
    batch_size = cfg["batch_size"]

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = cfg.get("lr_decay_step_size", 20)

    # Change device selection to prefer CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_label = {'Background1': 0, 'Burst': 1, 'Pit': 2, 'Stain': 3, 'Warpage': 4, 'Background2': 5, 'Burst2': 6, 'Pinhole': 7}
    feat_dim = 6

    """Load Dataset"""
    TRAIN_SET = SealingNailDatasetNPZ(root=root, npoints=num_points, split='train', use_cache=True)
    # Reduce num_workers for better stability
    num_workers = min(4, os.cpu_count() or 1)
    trainDataLoader = DataLoader(TRAIN_SET, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_fn, pin_memory=True)
    TEST_SET = SealingNailDatasetNPZ(root=root, npoints=num_points, split='test', use_cache=True)
    testDataLoader = DataLoader(TEST_SET, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True, collate_fn=collate_fn, pin_memory=True)

    """Calculate Weight"""
    weight = [float('{:.4f}'.format(i)) for i in TRAIN_SET.weight.tolist()]
    l_weight = [float('{:.4f}'.format(i)) for i in TRAIN_SET.l_weight.tolist()]
    logger.info('The weight of class: %s' % weight)
    logger.info('The weight of loss function: %s' % l_weight)
    weight = torch.tensor(TRAIN_SET.l_weight, dtype=torch.float).to(device)

    # Model Parameter
    classifier = Model(c=feat_dim, k=num_class)
    # Move model to device before applying any operations
    classifier = classifier.to(device)
    classifier.apply(inplace_relu)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    # loss_fn = FocalLoss(weight=weight, device=device)

    # No resume from checkpoint — start fresh
    start_epoch = 0

    # Optimizer
    optimizer = build_optimizer(cfg["optimizer"], classifier.parameters(), learning_rate, weight_decay)

    global_epoch = 0
    best_mean_class_IoU = 0
    for epoch in range(start_epoch, end_epoch):
        logger.info('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, end_epoch))
        """Performance parameter"""
        train_intersection = torch.zeros(num_class, device=device)
        train_union = torch.zeros(num_class, device=device)
        train_targets = torch.zeros(num_class, device=device)
        train_loss = 0
        eval_intersection = torch.zeros(num_class, device=device)
        eval_union = torch.zeros(num_class, device=device)
        eval_targets = torch.zeros(num_class, device=device)
        eval_loss = 0

        '''Adjust learning rate and BN momentum'''
        lr_mult = build_lr_lambda(cfg)(epoch)
        lr = max(learning_rate * lr_mult, 1e-5)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        logger.info('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (coords, feats, targets, offset) in tqdm(enumerate(trainDataLoader), desc=f'Epoch: {epoch + 1}/{end_epoch}', total=len(trainDataLoader), smoothing=0.9):
            # Remove these lines as they're redundant and cause the error
            # coords = coords.cuda()
            # feats = feats.cuda()
            # labels = labels.cuda()  # This line caused the error

            # Correct data movement to device
            coords, feats, targets, offset = coords.float().to(device), feats.float().to(device), targets.long().to(device), offset.to(device)

            """Training"""
            optimizer.zero_grad()
            seg_pred = classifier([coords, feats, offset])
            seg_pred = seg_pred.contiguous().view(-1, num_class)
            targets = targets.view(-1, 1)[:, 0]
            preds = seg_pred.data.max(1)[1]
            loss = loss_fn(seg_pred, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            """Training data statistics"""
            for label in class_label.values():
                train_intersection[label] += torch.logical_and(preds == label, targets == label).sum().data.cpu().item()
                train_union[label] += torch.logical_or(preds == label, targets == label).sum().data.cpu().item()
                train_targets[label] += torch.sum(targets == label).data.cpu().item()

        train_mAcc = (train_intersection / train_targets).mean().item()
        train_OA = train_intersection.sum() / train_targets.sum()
        train_IoU = train_intersection / train_union
        train_mIoU = train_IoU.mean().item()

        """print"""
        logger.info('Training loss is: %s %.5f' % (' ' * 12, (train_loss / len(trainDataLoader))))
        logger.info('Training overall accuracy: %s %.5f' % (' ' * 3, train_OA))
        logger.info('Training mean accuracy is: %s %.5f' % (' ' * 3, train_mAcc))
        for cat in class_label.keys():
            logger.info('Training mIoU of %s %f' % (cat + ' ' * (16 - len(cat)), train_IoU[class_label[cat]]))
        logger.info('Training mean class mIoU %s %f' % (' ' * 8, train_mIoU))

        if 1:
            with torch.no_grad():
                classifier = classifier.eval()
                """Validation"""
                for i, (coords, feats, targets, offset) in tqdm(enumerate(testDataLoader), desc='Validation', total=len(testDataLoader), smoothing=0.9):
                    coords, feats, targets, offset = coords.float().to(device), feats.float().to(device), targets.long().to(device), offset.to(device)
                    seg_pred = classifier([coords, feats, offset])
                    seg_pred = seg_pred.contiguous().view(-1, num_class)
                    targets = targets.view(-1, 1)[:, 0]
                    preds = seg_pred.data.max(1)[1]
                    loss = loss_fn(seg_pred, targets)
                    eval_loss += loss.item()
                    """Validation data statistics"""
                    for label in class_label.values():
                        eval_intersection[label] += torch.logical_and(preds == label, targets == label).sum().data.cpu().item()
                        eval_union[label] += torch.logical_or(preds == label, targets == label).sum().data.cpu().item()
                        eval_targets[label] += torch.sum(targets == label).data.cpu().item()

            eval_mAcc = (eval_intersection / eval_targets).mean().item()
            eval_OA = eval_intersection.sum() / eval_targets.sum()
            eval_IoU = eval_intersection / eval_union
            eval_mIoU = eval_IoU.mean().item()

            """print"""
            logger.info('Validation loss is: %s %.5f' % (' ' * 10, (eval_loss / len(testDataLoader))))
            logger.info('Validation overall accuracy: %s %.5f' % (' ', eval_OA))
            logger.info('Validation mean accuracy is: %s %.5f' % (' ', eval_mAcc))
            for cat in class_label.keys():
                logger.info('Validation mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), eval_IoU[class_label[cat]]))
            logger.info('Validation mean class mIoU %s %f' % (' ' * 6, eval_mIoU))

            logger.info('Save last model...')
            save_path = logs_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            logger.info('Saving in %s' % save_path)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            last_model = os.path.join(save_path, 'last_model.pth')
            torch.save(state, last_model)
            logger.info('Completed.')
            if eval_mIoU >= best_mean_class_IoU:
                best_mean_class_IoU = eval_mIoU
                logger.info('Save best model...')
                save_path = logs_dir
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                logger.info('Saving in %s' % save_path)
                state = {
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                best_model = os.path.join(save_path, 'best_model.pth')
                torch.save(state, best_model)
                logger.info('Completed.')

        global_epoch += 1


if __name__ == "__main__":
    main()
