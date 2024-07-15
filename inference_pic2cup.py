import os
import pdb
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image

from Pic2CuP.model.clip import _transform, load
from Pic2CuP.model.model import convert_weights, CLIP, IM2TEXT, IM2MultiTEXT, PromptLearner
from Pic2CuP.src.data import CustomFolder, ImageList, CsvCOCO, FashionIQ, CIRR
from Pic2CuP.src.data import targetpad_transform
from Pic2CuP.src.params import parse_args, get_project_root, get_dataset_root
from Pic2CuP.src.utils import is_master, TargetPad

from tqdm import tqdm
from Pic2CuP.model.clip import tokenize
import numpy as np
import pickle


def load_p2cmodel(args, model_path):
    model, _, preprocess_val = load(args.model, jit=False)

    img2text = IM2TEXT(embed_dim=model.embed_dim,
                       middle_dim=args.middle_dim,
                       output_dim=model.token_embedding.weight.shape[1],
                       n_layer=args.n_layer)
    prompt_learner = PromptLearner(args, model)
    model.cuda(args.gpu)
    img2text.cuda(args.gpu)
    prompt_learner.cuda(args.gpu)

    ckpt = torch.load(model_path, map_location=f'cuda:{args.gpu}')
    sd = ckpt['state_dict']
    sd_img2text = ckpt['state_dict_img2text']
    sd_prompt_learner = ckpt['state_dict_prompt_learner']

    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    if next(iter(sd_img2text.items()))[0].startswith('module'):
        sd_img2text = {k[len('module.'):]: v for k, v in sd_img2text.items()}
    if next(iter(sd_prompt_learner.items()))[0].startswith('module'):
        sd_prompt_learner = {k[len('module.'):]: v for k, v in sd_prompt_learner.items()}
    model.load_state_dict(sd)
    img2text.load_state_dict(sd_img2text)
    prompt_learner.load_state_dict(sd_prompt_learner)

    return model, img2text, prompt_learner, preprocess_val


def fashioniq_retrieval(model, img2text, args, ref_imgName, ref_image, mod_text, dress_type,
                        target_loader, prompt_learner):
    model.eval()
    img2text.eval()
    prompt_learner.eval()

    all_image_features = []
    all_target_paths = []
    all_composed_features = []

    prompt = " ".join(["X"] * args.n_ctx)
    img_token = " ".join(["Y"] * args.n_img)
    text = prompt + " " + img_token
    if len(mod_text) == 2:
        cap1, cap2 = mod_text[0], mod_text[1]
        text_with_blank = '{} , {} and {}'.format(text, cap2, cap1)
    elif len(mod_text) == 1:
        text_with_blank = f'{text}, {mod_text[0]}'
    token_texts = tokenize(text_with_blank)[0]

    if not os.path.exists(f'cache_data/fiq_p2c_{dress_type}_index_features.pkl'):
        with torch.no_grad():
            for batch in tqdm(target_loader):
                target_images, target_paths = batch
                if args.gpu is not None:
                    target_images = target_images.cuda(args.gpu, non_blocking=True)
                image_features = model.encode_image(target_images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_image_features.append(image_features)
                for path in target_paths:
                    all_target_paths.append(path)
            database_imgFeats = torch.cat(all_image_features)
        with open(f'cache_data/fiq_p2c_{dress_type}_index_features.pkl', 'wb') as f:
            pickle.dump(database_imgFeats, f)
        with open(f'cache_data/fiq_p2c_{dress_type}_index_names.pkl', 'wb') as g:
            pickle.dump(all_target_paths, g)
    else:
        with open(f'cache_data/fiq_p2c_{dress_type}_index_features.pkl', 'rb') as f:
            database_imgFeats = pickle.load(f)
        with open(f'cache_data/fiq_p2c_{dress_type}_index_names.pkl', 'rb') as g:
            all_target_paths = pickle.load(g)

    with torch.no_grad():
        ref_image = ref_image.cuda(args.gpu, non_blocking=True)
        token_texts = token_texts.cuda(args.gpu, non_blocking=True)
        ref_image = ref_image.unsqueeze(0)
        ref_imgFeats = model.encode_image(ref_image)

        query_image_tokens = img2text(ref_imgFeats)

        text_embedding = prompt_learner(query_image_tokens)
        composed_feature = model.encode_text_img_retrieval_learnable(args.n_ctx, args.n_img, text_embedding,
                                                                     token_texts)

        composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
        all_composed_features.append(composed_feature)
        query_features = torch.cat(all_composed_features)

        res = get_fiq_results(args, ref_imgName, database_imgFeats, query_features, all_target_paths)

    return res


def get_fiq_results(args, ref_names, image_features, ref_features, target_names):
    metrics = {}
    distances = 1 - ref_features @ image_features.T  # the distance is more closer to 1, the better (2038, 6346)
    sorted_indices = torch.argsort(distances, dim=-1).cpu()  # from small to large for each line
    sorted_index_names = np.array(target_names)[sorted_indices]

    tmp = sorted_index_names[0][:6]
    res = []
    for t in tmp:
        r = os.path.splitext(os.path.basename(t))[0]
        res.append(r)

    return res


def cirr_retrieval(model, img2text, args, ref_imgName, ref_image, mod_text, target_loader, prompt_learner):
    model.eval()
    img2text.eval()
    prompt_learner.eval()

    all_image_features = []
    all_target_paths = []
    all_composed_features = []

    with torch.no_grad():
        if not os.path.exists(f'cache_data/cirr_p2c_index_features.pkl'):
            for batch in tqdm(target_loader):
                target_images, target_paths = batch
                if args.gpu is not None:
                    target_images = target_images.cuda(args.gpu, non_blocking=True)
                image_features = model.encode_image(target_images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_image_features.append(image_features)
                for path in target_paths:
                    all_target_paths.append(path)
            with open(f'cache_data/cirr_p2c_index_features.pkl', 'wb') as f:
                pickle.dump(all_image_features, f)
            with open(f'cache_data/cirr_p2c_index_names.pkl', 'wb') as g:
                pickle.dump(all_target_paths, g)
        else:
            with open(f'cache_data/cirr_p2c_index_features.pkl', 'rb') as f:
                all_image_features = pickle.load(f)
            with open(f'cache_data/cirr_p2c_index_names.pkl', 'rb') as g:
                all_target_paths = pickle.load(g)

        ref_image = ref_image.cuda(args.gpu, non_blocking=True)
        prompt = " ".join(["X"] * args.n_ctx)
        img_token = " ".join(["Y"] * args.n_img)
        text = prompt + " " + img_token
        # print(prompt)
        text_with_blank = '{}, {}'.format(text, mod_text)
        ref_text_tokens = tokenize(text_with_blank)[0]
        ref_text_tokens = ref_text_tokens.cuda(args.gpu, non_blocking=True)
        ref_image = ref_image.unsqueeze(0)
        query_image_features = model.encode_image(ref_image)
        query_image_tokens = img2text(query_image_features)
        text_embedding = prompt_learner(query_image_tokens)
        composed_feature = model.encode_text_img_retrieval_learnable(args.n_ctx, args.n_img,
                                                                     text_embedding, ref_text_tokens)
        composed_feature = composed_feature / composed_feature.norm(dim=-1, keepdim=True)
        all_composed_features.append(composed_feature)

        query_features = torch.cat(all_composed_features)
        database_features = torch.cat(all_image_features)

    res = get_cirr_result(args, database_features, query_features, ref_imgName, all_target_paths)
    return res

def get_cirr_result(args, database_features, query_features, ref_imgName, index_names):
    distances = 1 - query_features @ database_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    reference_mask = torch.tensor(sorted_index_names != np.repeat(np.array(ref_imgName),
                                                                  len(index_names)).reshape(distances.shape[0], -1))

    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    res = sorted_index_names[0][:6]

    return res



def inference_pic2cup(dataset, model_path, mode, ref_imgPath, mod_text, ref_imgPath_list):
    args = parse_args()
    args.retrieval_mode = mode

    model, img2text, prompt_learner, preprocess_val = load_p2cmodel(args, model_path)

    preprocess_val = targetpad_transform(args.target_ratio, model.visual.input_resolution)

    ref_image = preprocess_val(Image.open(ref_imgPath))
    ref_imgName = os.path.splitext(os.path.basename(ref_imgPath))[0]

    if dataset.lower() == 'fashioniq':
        dress_type = os.path.dirname(ref_imgPath).split("/")[-1]
        target_dataset = FashionIQ(args, cloth=dress_type,
                                   transforms=preprocess_val,
                                   root=get_dataset_root(),
                                   mode='imgs')
        target_dataloader = DataLoader(target_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.workers,
                                       pin_memory=True,
                                       drop_last=False)

        res = fashioniq_retrieval(model, img2text, args, ref_imgName, ref_image, mod_text, dress_type,
                                  target_dataloader, prompt_learner)
    if dataset.lower() == 'cirr':
        target_dataset = CIRR(args, transforms=preprocess_val,
                              root=get_dataset_root(),
                              mode='imgs')
        target_dataloader = DataLoader(target_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.workers,
                                       pin_memory=True,
                                       drop_last=False)
        res = cirr_retrieval(model, img2text, args, ref_imgName, ref_image, mod_text,
                             target_dataloader, prompt_learner)

    return res


def main():
    dataset = 'cirr'
    model_path = 'models/pic2cup.pt'
    ref_imgPath = 'reference_images/CIRR/dev-1-0-img1.png'
    mod_text = ['is a smaller one']
    mode = 't2i'
    ref_imgPath_list = []
    res = inference_pic2cup(dataset, model_path, mode, ref_imgPath, mod_text, ref_imgPath_list)
    print(res)


if __name__ == '__main__':
    main()
