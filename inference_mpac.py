from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List

from MPAC import clip
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from MPAC.dataset.data_utils import FashionIQDataset, targetpad_transform, CIRRDataset
from MPAC.utils.utils import extract_index_features, device
from MPAC.modeling import train_models_map
import os
import pickle


def compute_fiq_val_metrics(args, model, tar_indexfeats: torch.tensor, index_names: List[str],
                            ref_imgName, ref_image, mod_text):

    # Generate predictions
    predicted_features = generate_fiq_val_predictions(args, model, index_names, tar_indexfeats,
                                                      ref_imgName, ref_image, mod_text)

    # Normalize the index features
    tar_indexfeats = F.normalize(tar_indexfeats, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ tar_indexfeats.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
    sorted_index_names != np.repeat(np.array(ref_imgName), len(index_names)).reshape(distances.shape[0], -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)

    result = sorted_index_names[0][:6]
    return result


def generate_fiq_val_predictions(args, model, index_names: List[str], tar_indexfeats: torch.tensor,
                                 ref_imgName, ref_image, mod_text):

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, tar_indexfeats))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, model.clip.visual.output_dim)).to(device, non_blocking=True)
    all_reference_names = []

    # Concatenate the captions in a deterministic way
    if len(mod_text) == 2:
        flattened_captions: list = np.array(mod_text).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
    else:
        input_captions = [mod_text]
    text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True)
    reference_image = ref_image.to(device)
    reference_image = reference_image.unsqueeze(0)
    # Compute the predicted features
    with torch.no_grad():
        fixed_imgfeats = model.fixed_image_encoder(reference_image)
        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
            ref_deep_compound_prompts_text = model.prompt_learner(text_inputs, fixed_imgfeats)

        text_feats = model.text_encoder(ref_prompts, ref_tokenized_prompts, ref_deep_compound_prompts_text)
        ref_imgfeats = itemgetter(*[ref_imgName])(name_to_feat).unsqueeze(0)
        batch_predfeats = model.combiner(ref_imgfeats, text_feats)

    all_reference_names.extend(ref_imgName)
    predicted_features = torch.vstack((predicted_features, F.normalize(batch_predfeats, dim=-1)))

    return predicted_features


def fashioniq_val_retrieval(args, dress_type: str, model, preprocess: callable, ref_imgName, ref_image, mod_text):

    # Define the validation datasets and extract the index features
    if not os.path.exists(f'cache_data/fiq_mpac_{dress_type}_index_features.pkl'):
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
        index_features, index_names = extract_index_features(args, classic_val_dataset, model)

        with open(f'cache_data/fiq_mpac_{dress_type}_index_features.pkl', 'wb') as f:
            pickle.dump(index_features, f)
        with open(f'cache_data/fiq_mpac_{dress_type}_index_names.pkl', 'wb') as g:
            pickle.dump(index_names, g)
    else:
        with open(f'cache_data/fiq_mpac_{dress_type}_index_features.pkl', 'rb') as f:
            index_features = pickle.load(f)
        with open(f'cache_data/fiq_mpac_{dress_type}_index_names.pkl', 'rb') as g:
            index_names = pickle.load(g)

    return compute_fiq_val_metrics(args, model, index_features, index_names, ref_imgName, ref_image, mod_text)


def compute_cirr_val_metrics(args, model, index_features: torch.tensor, index_names: List[str],
                             ref_imgName, ref_image, mod_text):
    # Generate predictions
    predicted_features, group_members = \
        generate_cirr_val_predictions(args, model, index_names, index_features, ref_imgName, ref_image, mod_text)

    print("Compute CIRR validation metrics")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    result = sorted_index_names[0][:6]

    return result

def generate_cirr_val_predictions(args, model, index_names: List[str], index_features: torch.tensor,
                                  ref_imgName, ref_image, mod_text):
    print("Compute CIRR validation predictions")

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, model.clip.visual.output_dim)).to(device, non_blocking=True)
    group_members = []

    text_inputs = clip.tokenize(mod_text).to(device, non_blocking=True)
    batch_reference_image = ref_image.to(device)
    batch_reference_image = batch_reference_image.unsqueeze(0)
    # Compute the predicted features
    with torch.no_grad():
        batch_fixed_features = model.fixed_image_encoder(batch_reference_image)

        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
            ref_deep_compound_prompts_text = model.prompt_learner(text_inputs, batch_fixed_features)

        text_feats = model.text_encoder(ref_prompts, ref_tokenized_prompts, ref_deep_compound_prompts_text)
        ref_imgfeats = itemgetter(*[ref_imgName])(name_to_feat).unsqueeze(0)
        batch_predicted_features = model.combiner(ref_imgfeats, text_feats)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))

    return predicted_features, group_members


def cirr_val_retrieval(args, model, preprocess: callable, ref_imgName, ref_image, mod_text):

    if not os.path.exists('cache_data/cirr_mpac_index_features.pkl'):
        classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
        index_features, index_names = extract_index_features(args, classic_val_dataset, model)

        with open('cache_data/cirr_mpac_index_features.pkl', 'wb') as f:
            pickle.dump(index_features, f)
        with open('cache_data/cirr_mpac_index_names.pkl', 'wb') as g:
            pickle.dump(index_names, g)
    else:
        with open('cache_data/cirr_mpac_index_features.pkl', 'rb') as f:
            index_features = pickle.load(f)
        with open('cache_data/cirr_mpac_index_names.pkl', 'rb') as g:
            index_names = pickle.load(g)

    return compute_cirr_val_metrics(args, model, index_features, index_names, ref_imgName, ref_image, mod_text)


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--database", type=str, default="fashionIQ", help="should be either 'CIRR' or 'fashionIQ'"
    )
    parser.add_argument(
        "--projection-dim", default=4096, type=int, help='Combiner projection dim'
    )
    parser.add_argument(
        "--hidden-dim", default=8192, type=int, help="Combiner hidden dim"
    )
    parser.add_argument(
        "--clip-model-name", default="ViT-B/16", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'"
    )
    parser.add_argument(
        "--clip-model-path", type=Path, help="Path to the fine-tuned CLIP model"
    )
    parser.add_argument(
        "--target-ratio", default=1.25, type=float, help="TargetPad target ratio"
    )
    parser.add_argument(
        "--transform", default="targetpad", type=str,
        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] "
    )
    parser.add_argument(
        "--sum-combiner", default=False, action='store_true'
    )
    parser.add_argument(
        "--maple-n-ctx", type=int, default=3, help=""
    )
    parser.add_argument(
        "--maple-ctx-init", type=str, default="a photo of", help=""
    )
    parser.add_argument(
        "--maple-prompt-depth", type=int, default=9, help=""
    )
    parser.add_argument(
        "--input-size", type=int, default=224, help=""
    )
    parser.add_argument(
        "--final", default=True
    )
    parser.add_argument(
        "--model-s1-path", type=str
    )
    parser.add_argument(
        "--model-s2-path", type=str
    )
    parser.add_argument(
        "--network", type=str, default='clip4cir_maple_final_s2'
    )
    parser.add_argument(
        "--asynchronous", default=True
    )
    parser.add_argument(
        "--combiner", type=str, default='combiner_v5'
    )
    parser.add_argument(
        "--aligner", default=False, action='store_true'
    )
    parser.add_argument(
        "--batch-size", default=32, type=int
    )
    parser.add_argument(
        "--num-workers", default=4, type=int
    )
    parser.add_argument(
        "--fixed-image-encoder", default=True
    )
    parser.add_argument(
        "--txt2img", default=False, action='store_true'
    )
    parser.add_argument(
        "--embed-size", default=512, type=int
    )
    parser.add_argument(
        "--mu", default=0.1, type=float
    )
    parser.add_argument(
        "--router", default=False, action='store_true'
    )
    parser.add_argument(
        "--optimizer", default='combiner', type=str
    )
    parser.add_argument(
        "--bsc-loss", default=False, action='store_true'
    )
    parser.add_argument(
        "--cross-attn-layer", default=4, type=int
    )
    parser.add_argument(
        "--cross-attn-head", default=2, type=int
    )
    parser.add_argument(
        "--img2txt-model-path", default='', type=str
    )
    parser.add_argument(
        "--val-bs", default=1, type=int,
    )
    parser.add_argument(
        "--visual-best", default=False, action="store_true"
    )

    args = parser.parse_args()

    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": args.maple_n_ctx}
    args.design_details = design_details
    return args


def inference_mpac(dataset, model_path, ref_imgPath, mod_text):
    args = get_args()

    if args.network in train_models_map.keys():
        model = train_models_map[args.network](args)
    else:
        raise NotImplementedError(f"Unkonw model types: {args.network}")

    input_dim = model.clip.visual.input_resolution

    print("Trying to load the model")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['state_dict_mpac'])

    print("Trying to load combiner")
    model.combiner.load_state_dict(ckpt['state_dict_combiner'])

    model = model.to(device)
    model.eval().float()

    preprocess = targetpad_transform(args.target_ratio, input_dim)

    ref_image = preprocess(Image.open(ref_imgPath))
    ref_imgName = os.path.splitext(os.path.basename(ref_imgPath))[0]

    if dataset.lower() == 'cirr':
        res = cirr_val_retrieval(args, model, preprocess, ref_imgName, ref_image, mod_text)

    elif dataset.lower() == 'fashioniq':
        dress_type = os.path.dirname(ref_imgPath).split("/")[-1]
        print(dress_type)
        res = fashioniq_val_retrieval(args, dress_type, model, preprocess, ref_imgName, ref_image, mod_text)

    return res

def main():
    dataset = 'fashioniq'
    model_path = 'models/fiq_mpac.pt'
    ref_imgPath = 'database/fashionIQ/images/B000A38K4A.png'
    mod_text = ['is a smaller size', 'is lighter']
    res = inference_mpac(dataset, model_path, ref_imgPath, mod_text)


if __name__ == '__main__':
    main()
