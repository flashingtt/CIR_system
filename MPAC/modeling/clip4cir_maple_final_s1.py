from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
import math

from MPAC import clip as clip
from MPAC.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from MPAC.utils.utils import device, _get_clones


class CLIP4cirMaPLeFinalS1(nn.Module):
    def __init__(self, args):
        super(CLIP4cirMaPLeFinalS1, self).__init__()
        self.args = args

        # load CLIP model
        if args.fixed_image_encoder:
            if args.asynchronous:
                self.clip, self.clip_image_encoder, self.fixed_image_encoder, self.clip_preprocess = \
                    clip.load(args, args.clip_model_name, args.design_details, device=device, jit=False)
            else:
                self.clip, self.fixed_image_encoder, self.clip_preprocess = clip.load(args, args.clip_model_name,
                                                                                      args.design_details,
                                                                                      device=device, jit=False)
        else:
            if args.asynchronous:
                self.clip, self.fixed_image_encoder, self.clip_preprocess = \
                    clip.load(args, args.clip_model_name, args.design_details, device=device, jit=False)
            else:
                self.clip, self.clip_preprocess = clip.load(args, args.clip_model_name, args.design_details,
                                                            device=device, jit=False)

        # load multi-modal prompt learner
        self.prompt_learner = MultiModalPromptLearner(args, self.clip)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip.visual
        self.text_encoder = TextEncoder(self.clip)
        self.combiner = Combiner()
        self.crossentropy_criterion = nn.CrossEntropyLoss()

        if args.aligner:
            self.aligner = Alignment(args, self.image_encoder.output_dim, args.cross_attn_layer, args.cross_attn_head)
        # self.logit_scale = self.clip.logit_scale
        self.dtype = self.clip.dtype

    def forward(self, reference_image, text_inputs, target_images):
        if self.args.fixed_image_encoder:
            with torch.no_grad():
                fixed_imgfeats = self.fixed_image_encoder(reference_image)

            # get
            if self.args.txt2img:
                ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_text, \
                    ref_deep_compound_prompts_vision = self.prompt_learner(text_inputs, fixed_imgfeats)
            else:
                ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
                    ref_deep_compound_prompts_text = self.prompt_learner(text_inputs, fixed_imgfeats)
        else:
            if self.args.txt2img:
                ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_text, \
                    ref_deep_compound_prompts_vision = self.prompt_learner(text_inputs)
            else:
                ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
                    ref_deep_compound_prompts_text = self.prompt_learner(text_inputs)

        if not self.args.asynchronous:
            if self.args.txt2img:
                tar_prompts, tar_tokenized_prompts, tar_shared_ctx, tar_deep_compound_prompts_text, \
                    tar_deep_compound_prompts_vision = self.prompt_learner()
            else:
                tar_prompts, tar_tokenized_prompts, tar_shared_ctx, tar_deep_compound_prompts_vision, \
                    tar_deep_compound_prompts_text = self.prompt_learner()

        # text_features
        text_feats = \
            self.text_encoder(ref_prompts, ref_tokenized_prompts, ref_deep_compound_prompts_text)

        # reference features
        ref_imgfeats = \
            self.image_encoder(reference_image.type(self.dtype), ref_shared_ctx, ref_deep_compound_prompts_vision)

        # target features
        if self.args.asynchronous:
            # tar_imgfeats = \
            #     self.clip_image_encoder(target_images.type(self.dtype), tar_shared_ctx, tar_deep_compound_prompts_vision)
            tar_imgfeats = self.clip_image_encoder(target_images.type(self.dtype))
        else:
            tar_imgfeats = \
                self.image_encoder(target_images.type(self.dtype), tar_shared_ctx, tar_deep_compound_prompts_vision)

        # calculate composed features and logits
        composed_feats = self.combiner(ref_imgfeats, text_feats)
        logits = 100 * composed_feats @ F.normalize(tar_imgfeats, dim=-1).T

        # calculate alignment logits
        if self.args.aligner:
            align_logits = self.aligner(ref_imgfeats, text_feats, tar_imgfeats)
            logits = logits + align_logits

        # return loss
        ground_truth = torch.arange(reference_image.shape[0], dtype=torch.long, device=device)
        loss = self.crossentropy_criterion(logits, ground_truth)
        return loss


class MultiModalPromptLearner(nn.Module):
    def __init__(self, args, clip_model):
        super().__init__()
        self.args = args
        n_cls = 1
        n_ctx = args.maple_n_ctx
        ctx_init = args.maple_ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = args.input_size

        self.clip_model = clip_model

        # Default is 1, which is compound shallow prompting
        assert args.maple_prompt_depth >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = args.maple_prompt_depth  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(ctx_dim, 768)
        self.ctx = nn.Parameter(ctx_vectors)  # change prompt to learnable

        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        if args.txt2img:
            self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                           for _ in range(self.compound_prompts_depth - 1)])
            for single_para in self.compound_prompts_text:
                nn.init.normal_(single_para, std=0.02)
            # Also make corresponding projection layers, for each prompt
            txt2img = nn.Linear(ctx_dim, 768)
            self.compound_prompt_projections = _get_clones(txt2img, self.compound_prompts_depth - 1)  # 8
        else:
            self.compound_prompts_visual = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                                             for _ in range(self.compound_prompts_depth - 1)])

            for single_para in self.compound_prompts_visual:
                nn.init.normal_(single_para, std=0.02)
            # Also make corresponding projection layers, for each prompt
            img2txt = nn.Linear(768, ctx_dim)
            self.i2t_compound_prompt_projections = _get_clones(img2txt, self.compound_prompts_depth - 1)  # 8

        prompts = [prompt_prefix + 'resemblance']  # "a photo of" + "sth" + "." ["a photo of sth."]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  # [1, 77, 512]

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        # transmit global image information to text as a token, which can be applied knowledge distiilation if I have time
        if args.fixed_image_encoder:
            self.img2token = Img2Token()

    def construct_prompts(self, ctx, prefix, suffix, image_token_embedding, text_embedding, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        if text_embedding is not None:
            bsz = text_embedding.shape[0]
            if image_token_embedding is not None:
                image_token_embedding = image_token_embedding.reshape(bsz, 1, -1)

                prompts = torch.cat(
                    [
                        prefix.expand(bsz, -1, -1),  # (dim0, 1, dim)
                        ctx.expand(bsz, -1, -1),  # (dim0, n_ctx, dim)
                        image_token_embedding,
                        text_embedding[:, 1: 76 - ctx.shape[1], :]
                        # 后面接上text_embedding内容, embedding of "a photo of similar one has longer sleeves ..."
                    ],
                    dim=1
                )
            else:
                prompts = torch.cat(
                    [
                        prefix.expand(bsz, -1, -1),  # (dim0, 1, dim)
                        ctx.expand(bsz, -1, -1),  # (dim0, n_ctx, dim)
                        suffix[:, 1, :],  # (dim0, *, dim)
                        text_embedding[:, 1: 76 - ctx.shape[1], :]  # 后面接上text_embedding内容

                    ],
                    dim=1,
                )
        else:
            prompts = torch.cat(
                [
                    prefix,  # (dim0, 1, dim)
                    ctx,  # (dim0, n_ctx, dim)
                    suffix  # (dim0, *, dim)
                ],
                dim=1,
            )
        # return embedding
        return prompts

    def construct_token(self, init_token, text_inputs):
        bsz = text_inputs.shape[0]
        tokenized_prompts = torch.cat(
            [
                init_token.expand(bsz, -1)[:, :5],
                text_inputs[:, 1: 73]
            ],
            dim=1
        )
        return tokenized_prompts

    def forward(self, text_inputs=None, fixed_imgfeats=None):
        ctx = self.ctx  # embedding tensor of "a photo of similar one"

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix  # SOS
        suffix = self.token_suffix  # CLS, EOS

        if text_inputs is not None:
            text_embedding = self.clip_model.token_embedding(text_inputs).type(self.clip_model.dtype)
        else:
            text_embedding = None

        if fixed_imgfeats is not None:
            image_token_embedding = self.img2token(fixed_imgfeats)
        else:
            image_token_embedding = None

        prompts = self.construct_prompts(ctx, prefix, suffix, image_token_embedding, text_embedding)
        # pdb.set_trace()

        # Before returning, need to transform
        # prompts to 768 for the visual side
        # TODO: 也许根据captions来映射到图像prompt？有待讨论
        if self.args.txt2img:
            visual_deep_prompts = []
            for index, layer in enumerate(self.compound_prompt_projections):
                visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        else:
            text_deep_prompts = []
            for index, layer in enumerate(self.i2t_compound_prompt_projections):
                text_deep_prompts.append(layer(self.compound_prompts_visual[index]))

        if text_inputs is not None:
            tokenized_prompts = self.construct_token(self.tokenized_prompts, text_inputs)
        else:
            tokenized_prompts = self.tokenized_prompts
        # pdb.set_trace()
        shared_ctx = prompts[:, 1: 1 + self.n_ctx, :]  # [B, n_ctx, 512]
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        if self.args.txt2img:
            return prompts, tokenized_prompts, self.proj(shared_ctx), self.compound_prompts_text, visual_deep_prompts
        else:
            return prompts, tokenized_prompts, self.proj(
                shared_ctx), self.compound_prompts_visual, text_deep_prompts  # pass here original, as for visual 768 is required


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # TODO:
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class Combiner(nn.Module):
    def __init__(self):
        super(Combiner, self).__init__()

    def forward(self, ref_imgfeats, text_feats):
        composed_feats = ref_imgfeats + text_feats
        # composed_feats = text_feats
        return F.normalize(composed_feats, dim=-1)


class Alignment(nn.Module):
    def __init__(self, args, clip_feature_dim, cross_attn_layer, cross_attn_head):
        super(Alignment, self).__init__()
        self.args = args
        self.corss_attn = _get_clones(CrossAttention(clip_feature_dim, cross_attn_layer, cross_attn_head), 2)
        if args.bsc_loss:
            self.bsc_loss = BSCLoss()

    def forward(self, image_feats, text_feats, target_feats):
        relative_feats = self.text_align(image_feats, target_feats)
        text_align_logits = 100 * relative_feats @ F.normalize(text_feats, dim=-1).T
        if self.args.bsc_loss:
            bsc_loss = self.bsc_loss(relative_feats, F.normalize(text_feats, dim=-1))
            text_align_logits = text_align_logits + bsc_loss
        return text_align_logits

    def text_align(self, image_feats, target_feats):
        image_feats = image_feats.unsqueeze(1)
        target_feats = target_feats.unsqueeze(1)
        cross_feats_0 = self.corss_attn[0](image_feats, target_feats)
        cross_feats_1 = self.corss_attn[1](target_feats, image_feats)
        relative_features = (cross_feats_0 + cross_feats_1) / 2
        relative_features = relative_features.squeeze(1)
        return F.normalize(relative_features)


class CrossAttention(nn.Module):
    def __init__(self, clip_feature_dim, n_layers, n_heads, attn_mask=None):
        super(CrossAttention, self).__init__()
        self.n_layers = n_layers
        self.resblocks = _get_clones(ResidualCrossAttentionBlock(clip_feature_dim, n_heads, attn_mask), n_layers)

    def forward(self, x, y):
        for i in range(self.n_layers):
            x = self.resblocks[i](x, y)
        return x


class ResidualCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super(ResidualCrossAttentionBlock, self).__init__()

        self.attn = CrossAttentionLayer(d_model, n_head)
        self.ln_x1 = nn.LayerNorm(d_model)
        self.ln_y1 = nn.LayerNorm(d_model)
        self.mlp_ratio = 4
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, int(d_model * self.mlp_ratio))),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(int(d_model * self.mlp_ratio), d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, y):
        return self.attn(x, y)

    def forward(self, x, y):
        x = x + self.attention(self.ln_x1(x), self.ln_y1(y))
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head):
        super(CrossAttentionLayer, self).__init__()
        self.h = n_head
        self.d_model = d_model
        self.d_k = d_model // n_head
        self.projections = _get_clones(nn.Linear(d_model, d_model), 3)

    def forward(self, x, y):
        nbatches = x.size(0)
        query, key, value = [l(v).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, v in zip(self.projections, (y, x, x))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        p_attn = F.softmax(scores, dim=-1)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class BSCLoss(nn.Module):
    def __init__(self):
        super(BSCLoss, self).__init__()

    def forward(self, composed_feats, target_feats):
        query_matrix = composed_feats @ composed_feats.T
        target_maxtrix = target_feats @ target_feats.T
        bsc_loss = nn.MSELoss()(query_matrix, target_maxtrix)
        return bsc_loss


class Img2Token(nn.Module):
    def __init__(self, embed_dim=512, middle_dim=512, output_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        self.fc_out = nn.Linear(middle_dim, output_dim)
        layers = []
        dim = embed_dim
        for _ in range(n_layer):
            block = []
            block.append(nn.Linear(dim, middle_dim))
            block.append(nn.Dropout(dropout))
            block.append(nn.ReLU())
            dim = middle_dim
            layers.append(nn.Sequential(*block))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

