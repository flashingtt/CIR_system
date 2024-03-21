import random
from pathlib import Path
from typing import Union, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import math

from MPAC.dataset.data_utils import CIRRDataset, FashionIQDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def extract_index_features(args, dataset: Union[CIRRDataset, FashionIQDataset], model) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR database in 'classic' mode
    :param clip_model: CLIP model
    :return: a tensor of features and a list of images
    """
    feature_dim = model.clip.visual.output_dim
    classic_val_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                    pin_memory=True, collate_fn=collate_fn)

    tar_indexfeats = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []

    if isinstance(dataset, CIRRDataset):
        print(f"extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    for names, images in tqdm(classic_val_loader):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            if not args.asynchronous:
                if args.final:
                    if args.txt2img:
                        tar_prompts, tar_tokenized_prompts, tar_shared_ctx, tar_deep_compound_prompts_text, \
                            tar_deep_compound_prompts_vision = model.prompt_learner()
                    else:
                        tar_prompts, tar_tokenized_prompts, tar_shared_ctx, tar_deep_compound_prompts_vision, \
                            tar_deep_compound_prompts_text = model.prompt_learner()
                else:
                    if args.txt2img:
                        tar_prompts, tar_tokenized_prompts, tar_shared_ctx, tar_deep_compound_prompts_text, \
                            tar_deep_compound_prompts_vision = model.prompt_learner()
                    else:
                        tar_prompts, tar_tokenized_prompts, tar_shared_ctx, tar_deep_compound_prompts_vision, \
                            tar_deep_compound_prompts_text = model.prompt_learner()
            if args.asynchronous:
                # tar_batchfeats = model.clip_image_encoder(images.type(model.dtype), tar_shared_ctx, tar_deep_compound_prompts_vision)
                tar_batchfeats = model.clip_image_encoder(images.type(model.dtype))
            else:
                tar_batchfeats = model.image_encoder(images.type(model.dtype), tar_shared_ctx,
                                                     tar_deep_compound_prompts_vision)
            tar_indexfeats = torch.vstack((tar_indexfeats, tar_batchfeats))
            index_names.extend(names)
    return tar_indexfeats, index_names


def element_wise_sum(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    """
    Normalized element-wise sum of image features and text features
    :param image_features: non-normalized image features
    :param text_features: non-normalized text features
    :return: normalized element-wise sum of image and text features
    """
    return F.normalize(image_features + text_features, dim=-1)


def element_wise_times(image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
    return F.normalize(image_features * text_features, dim=-1)


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path):
    """
    Save the weights of the model during training
    :param name: name of the file
    :param cur_epoch: current epoch
    :param model_to_save: pytorch model to be saved
    :param training_path: path associated with the training run
    """
    models_path = training_path / "models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
    }, str(models_path / f'{name}.pt'))


class ShowBest(object):
    def __init__(self):
        super(ShowBest, self).__init__()
        self.epoch = -1

    def __call__(self, epoch, best_avg_recall):
        if epoch > self.epoch:
            print(f"\n-----previous best: {best_avg_recall}-----")
            self.epoch = epoch


class LR_Scheduler(object):
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        super(LR_Scheduler, self).__init__()
        self.mode = mode
        print(f"Using {self.mode} LR Scheduler!")
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


class ShowBestCIRR(object):
    def __init__(self):
        super(ShowBestCIRR, self).__init__()
        self.epoch = -1

    def __call__(self, epoch, best_avg_recall, best_harmonic, best_geometric, best_arithmetic):
        if epoch > self.epoch:
            print(f"\n-----best_avg_recall: {best_avg_recall}\
                  \tbest_harmonic: {best_harmonic}\
                  \tbest_geometric: {best_geometric}\
                  \tbestarithmetic: {best_arithmetic}-----")
            self.epoch = epoch