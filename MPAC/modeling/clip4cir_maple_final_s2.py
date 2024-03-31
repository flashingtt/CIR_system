import pdb
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from MPAC import clip as clip
from MPAC.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
from MPAC.utils.utils import device, _get_clones
import math


class CLIP4cirMaPLeFinalS2(nn.Module):
    def __init__(self, args):
        super(CLIP4cirMaPLeFinalS2, self).__init__()
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
                self.clip, self.clip_image_encoder, self.clip_preprocess = \
                    clip.load(args, args.clip_model_name, args.design_details, device=device, jit=False)
            else:
                self.clip, self.clip_preprocess = clip.load(args, args.clip_model_name, args.design_details,
                                                            device=device, jit=False)

        # load multi-modal prompt learner
        self.prompt_learner = MultiModalPromptLearner(args, self.clip)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip.visual
        self.text_encoder = TextEncoder(self.clip)
        if args.combiner == 'combiner':
            self.combiner = Combiner(self.image_encoder.output_dim, args.projection_dim, args.hidden_dim)
        elif args.combiner == 'fine_grained_weighted_sum_combiner':
            self.combiner = FineGrainedWeightedSum(args, self.image_encoder.output_dim, args.projection_dim,
                                                   args.hidden_dim)
        elif args.combiner == 'combiner_v3':
            self.combiner = CombinerV3(args, self.image_encoder.output_dim, args.projection_dim, args.hidden_dim)
        elif args.combiner == 'combiner_v4':
            self.combiner = CombinerV4(args, self.image_encoder.output_dim, args.projection_dim, args.hidden_dim)
        elif args.combiner == 'combiner_v5':
            self.combiner = CombinerV5(args, self.image_encoder.output_dim, args.projection_dim, args.hidden_dim)
        elif args.combiner == 'combiner_qm_qc':
            self.combiner = CombinerQmQc(args, self.image_encoder.output_dim, args.projection_dim, args.hidden_dim)
        elif args.combiner == 'combiner_wio_qc':
            self.combiner = CombinerwioQc(args, self.image_encoder.output_dim, args.projection_dim, args.hidden_dim)
        else:
            raise NotImplementedError(f"unknow combiner {args.combiner}")
        if args.aligner:
            self.aligner = Alignment(args, self.image_encoder.output_dim, args.cross_attn_layer, args.cross_attn_head)
        self.dtype = self.clip.dtype

    def forward(self, reference_image, text_inputs, target_images):
        with torch.no_grad():
            textinput_list = torch.split(text_inputs, self.args.clip_bs)
            refimg_list = torch.split(reference_image, self.args.clip_bs)
            tarimg_list = torch.split(target_images, self.args.clip_bs)

            feat_dim = self.image_encoder.output_dim
            text_feats = torch.empty((0, feat_dim)).to(device, non_blocking=True)
            ref_imgfeats = torch.empty((0, feat_dim)).to(device, non_blocking=True)
            tar_imgfeats = torch.empty((0, feat_dim)).to(device, non_blocking=True)

            for i in range(len(textinput_list)):
                if self.args.fixed_image_encoder:
                    fixed_imgfeat = self.fixed_image_encoder(refimg_list[i])

                    # get prompts
                    if self.args.txt2img:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_text, \
                            ref_deep_compound_prompts_vision = self.prompt_learner(textinput_list[i], fixed_imgfeat)
                    else:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
                            ref_deep_compound_prompts_text = self.prompt_learner(textinput_list[i], fixed_imgfeat)
                else:
                    if self.args.txt2img:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_text, \
                            ref_deep_compound_prompts_vision = self.prompt_learner(textinput_list[i])
                    else:
                        ref_prompts, ref_tokenized_prompts, ref_shared_ctx, ref_deep_compound_prompts_vision, \
                            ref_deep_compound_prompts_text = self.prompt_learner(textinput_list[i])

                if not self.args.asynchronous:
                    if self.args.txt2img:
                        tar_prompts, tar_tokenized_prompts, tar_shared_ctx, tar_deep_compound_prompts_text, \
                            tar_deep_compound_prompts_vision = self.prompt_learner()
                    else:
                        tar_prompts, tar_tokenized_prompts, tar_shared_ctx, tar_deep_compound_prompts_vision, \
                            tar_deep_compound_prompts_text = self.prompt_learner()

                # text features
                text_feat = \
                    self.text_encoder(ref_prompts, ref_tokenized_prompts, ref_deep_compound_prompts_text)
                text_feats = torch.vstack((text_feats, text_feat))

                # reference image features
                ref_imgfeat = \
                    self.image_encoder(refimg_list[i].type(self.dtype), \
                                       ref_shared_ctx, ref_deep_compound_prompts_vision)
                ref_imgfeats = torch.vstack((ref_imgfeats, ref_imgfeat))

                # target image features
                if self.args.asynchronous:
                    # tar_imgfeat = \
                    #     self.clip_image_encoder(tarimg_list[i].type(self.dtype), \
                    #                             tar_shared_ctx, tar_deep_compound_prompts_vision)
                    tar_imgfeat = self.clip_image_encoder(tarimg_list[i])
                else:
                    tar_imgfeat = \
                        self.image_encoder(tarimg_list[i].type(self.dtype), \
                                           tar_shared_ctx, tar_deep_compound_prompts_vision)
                tar_imgfeats = torch.vstack((tar_imgfeats, tar_imgfeat))

        with torch.cuda.amp.autocast():
            composed_feats = self.combiner(ref_imgfeats, text_feats)

        logits = 100 * composed_feats @ F.normalize(tar_imgfeats, dim=-1).T
        if self.args.aligner:
            align_logits = self.aligner(ref_imgfeats, text_feats, tar_imgfeats)
            logits = logits + 0.1 * align_logits
        return logits


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
        # self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
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
            single_layer = nn.Linear(ctx_dim, 768)
            self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)
        else:
            self.compound_prompts_visual = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                                             for _ in range(self.compound_prompts_depth - 1)])

            for single_para in self.compound_prompts_visual:
                nn.init.normal_(single_para, std=0.02)
            # Also make corresponding projection layers, for each prompt
            img2txt = nn.Linear(768, ctx_dim)
            self.i2t_compound_prompt_projections = _get_clones(img2txt, self.compound_prompts_depth - 1)  # 8
        # TODO:
        # classnames = ['dress', 'toptee', 'shirt']
        prompts = [prompt_prefix + 'resemblance']

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
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
                        text_embedding[:, 1: 76 - ctx.shape[1], :]  # 后面接上text_embedding内容
                        # suffix,  # (dim0, *, dim)
                    ],
                    dim=1,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix.expand(bsz, -1, -1),  # (dim0, 1, dim)
                        ctx.expand(bsz, -1, -1),  # (dim0, n_ctx, dim)
                        # suffix[:, 1, :],  # (dim0, *, dim)
                        text_embedding[:, 1: 77 - ctx.shape[1], :]  # 后面接上text_embedding内容

                    ],
                    dim=1,
                )
        else:
            prompts = torch.cat(
                [
                    prefix,  # (dim0, 1, dim)
                    ctx,  # (dim0, n_ctx, dim)
                    suffix,  # (dim0, *, dim)
                ],
                dim=1,
            )

        return prompts

    def construct_token(self, init_token, text_inputs):
        bsz = text_inputs.shape[0]
        tokenized_prompts = torch.cat([
            init_token.expand(bsz, -1)[:, :5],
            text_inputs[:, 1: 73]
        ], dim=1)
        return tokenized_prompts

    def forward(self, text_inputs=None, fixed_imgfeats=None):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if text_inputs is not None:
            text_embedding = self.clip_model.token_embedding(text_inputs).type(self.clip_model.dtype)
        else:
            text_embedding = None

        if fixed_imgfeats is not None:
            image_token_embedding = self.img2token(fixed_imgfeats)
        else:
            image_token_embedding = None

        prompts = self.construct_prompts(ctx, prefix, suffix, image_token_embedding, text_embedding)

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

        shared_ctx = prompts[:, 1: 1 + self.n_ctx, :]

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
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class Combiner(nn.Module):
    def __init__(self, clip_feature_dim, projection_dim, hidden_dim):
        super(Combiner, self).__init__()
        self.text_projection_layer = nn.Linear(clip_feature_dim, projection_dim)
        self.image_projection_layer = nn.Linear(clip_feature_dim, projection_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.combiner_layer = nn.Linear(projection_dim * 2, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, clip_feature_dim)

        self.dropout3 = nn.Dropout(0.5)
        self.dynamic_scalar = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, 1), nn.Sigmoid())

        self.logit_scale = 100

    def forward(self, ref_imgfeats, text_feats):
        text_projfeats = self.dropout1(F.relu(self.text_projection_layer(text_feats)))
        img_projfeats = self.dropout2(F.relu(self.image_projection_layer(ref_imgfeats)))

        raw_combfeats = torch.cat((text_projfeats, img_projfeats), dim=-1)
        combined_feats = self.dropout3(F.relu(self.combiner_layer(raw_combfeats)))
        dynamic_scalar = self.dynamic_scalar(raw_combfeats)
        output = self.output_layer(combined_feats) + \
                 dynamic_scalar * text_feats + (1 - dynamic_scalar) * ref_imgfeats
        return F.normalize(output, dim=-1)


class FineGrainedWeightedSum(Combiner):
    def __init__(self, args, clip_feature_dim, projection_dim, hidden_dim):
        super(FineGrainedWeightedSum, self).__init__(clip_feature_dim, projection_dim, hidden_dim)
        self.dynamic_vector = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, clip_feature_dim), nn.Sigmoid())
        self.vector_norm = nn.LayerNorm(clip_feature_dim)
        self.mu = args.mu

    def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        dynamic_vector = self.dynamic_vector(raw_combined_features)
        dynamic_vector = F.normalize(dynamic_vector, dim=-1)
        dynamic_vector = 2 * (dynamic_vector - 0.5)  # when activate function == sigmoid()
        dynamic_vector = self.vector_norm(dynamic_vector)

        output = self.output_layer(combined_features) + \
                 (dynamic_scalar + self.mu * dynamic_vector) * text_features + \
                 (1 - (dynamic_scalar + self.mu * dynamic_vector)) * image_features
        return F.normalize(output, dim=-1)


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


class CombinerV3(Combiner):
    '''
    self-attention(i, t) + (a + 0.1w)t + (1 - a - 0.1w)i
    '''

    def __init__(self, args, clip_feature_dim, projection_dim, hidden_dim):
        super(CombinerV3, self).__init__(clip_feature_dim, projection_dim, hidden_dim)
        self.args = args
        self.dynamic_vector = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, clip_feature_dim), nn.Sigmoid())
        self.vector_norm = nn.LayerNorm(clip_feature_dim)
        self.self_attn = SelfAttentionCell(args)
        if args.router:
            self.q_weight_layer = Router(2, projection_dim, hidden_dim)
        self.mu = args.mu

    def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))

        if self.args.router:
            q_weights = self.q_weight_layer(raw_combined_features)  # [B, 2]

        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        dynamic_vector = self.dynamic_vector(raw_combined_features)
        dynamic_vector = F.normalize(dynamic_vector, dim=-1)
        dynamic_vector = 2 * (dynamic_vector - 0.5)  # when activate function == sigmoid()
        dynamic_vector = self.vector_norm(dynamic_vector)

        cat_feats = self.output_layer(combined_features)
        self_attn_feats = self.self_attn(cat_feats.unsqueeze(1)).squeeze(1)
        if self.args.router:
            output = q_weights[:, 0].unsqueeze(-1) * self_attn_feats + \
                     q_weights[:, 1].unsqueeze(-1) * ((dynamic_scalar + self.mu * dynamic_vector) * text_features + \
                                                      (1 - (
                                                                  dynamic_scalar + self.mu * dynamic_vector)) * image_features)
        else:
            output = self_attn_feats + (dynamic_scalar + self.mu * dynamic_vector) * text_features + \
                     (1 - (dynamic_scalar + self.mu * dynamic_vector)) * image_features
        return F.normalize(output, dim=-1)


class SelfAttentionCell(nn.Module):
    def __init__(self, args):
        super(SelfAttentionCell, self).__init__()
        self.h = 8
        self.drop = 0.0
        self.mlp_ratio = 4
        mlp_hidden_dim = int(args.embed_size * self.mlp_ratio)
        self.att_layer = AttentionLayer(args.embed_size, self.h, drop=self.drop)
        self.feed_forward_layer = FeedForward(args.embed_size, mlp_hidden_dim, drop=self.drop)
        self.dropout = nn.Dropout(self.drop)
        self.norm1 = nn.LayerNorm(args.embed_size)
        self.norm2 = nn.LayerNorm(args.embed_size)

    def forward(self, local_emb):
        mask = None
        self_att_emb = self.dropout(self.att_layer(self.norm1(local_emb), mask=mask))
        out = self_att_emb + self.dropout(self.feed_forward_layer(self.norm2(self_att_emb)))
        return out


class AttentionLayer(nn.Module):
    def __init__(self, embed_size, h, is_share=False, drop=0.0):
        super(AttentionLayer, self).__init__()
        self.is_share = is_share
        self.h = h
        self.embed_size = embed_size
        self.d_k = embed_size // h
        self.drop_p = drop
        if is_share:
            self.linear = nn.Linear(embed_size, embed_size)
            self.linears = [self.linear, self.linear, self.linear]
        else:
            self.linears = _get_clones(nn.Linear(embed_size, embed_size), 3)
        if self.drop_p > 0:
            self.dropout = nn.Dropout(drop)

    def forward(self, inp, mask=None):
        nbatches = inp.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (inp, inp, inp))]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.drop_p > 0:
            p_attn = self.dropout(p_attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden, drop=0.0):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden)
        self.fc2 = nn.Linear(hidden, embed_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class CombinerV4(Combiner):
    def __init__(self, args, clip_feature_dim, projection_dim, hidden_dim):
        super().__init__(clip_feature_dim, projection_dim, hidden_dim)
        self.dynamic_vector = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, clip_feature_dim), nn.Sigmoid())
        self.vector_norm = nn.LayerNorm(clip_feature_dim)
        self.self_attn = SelfAttentionCell(args)
        self.alpha = nn.Sequential(nn.Linear(clip_feature_dim, clip_feature_dim), )
        self.beta = nn.Sequential(nn.Linear(clip_feature_dim, clip_feature_dim), )
        # self.modified_ln = nn.LayerNorm(clip_feature_dim)
        self.mu = args.mu

    def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))

        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        dynamic_vector = self.dynamic_vector(raw_combined_features)
        dynamic_vector = F.normalize(dynamic_vector, dim=-1)
        dynamic_vector = 2 * (dynamic_vector - 0.5)  # when activate function == sigmoid()
        dynamic_vector = self.vector_norm(dynamic_vector)

        cat_feats = self.output_layer(combined_features)
        alpha = self.alpha(cat_feats)
        beta = self.beta(cat_feats)
        # mod_imgfeats = self.modified_ln(alpha * image_features + beta)
        mod_imgfeats = alpha * image_features + beta
        self_attn_feats = self.self_attn(cat_feats.unsqueeze(1)).squeeze(1)
        output = mod_imgfeats + self_attn_feats + (dynamic_scalar + self.mu * dynamic_vector) * text_features + \
                 (1 - (dynamic_scalar + self.mu * dynamic_vector)) * image_features
        return F.normalize(output, dim=-1)


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


class CombinerQmQc(Combiner):
    def __init__(self, args, clip_feature_dim, projection_dim, hidden_dim):
        super().__init__(clip_feature_dim, projection_dim, hidden_dim)
        self.args = args
        self.self_attn = SelfAttentionCell(args)
        self.alpha = nn.Sequential(nn.Linear(clip_feature_dim, clip_feature_dim), )
        self.beta = nn.Sequential(nn.Linear(clip_feature_dim, clip_feature_dim), )
        # self.modified_ln = nn.LayerNorm(clip_feature_dim)
        if args.router:
            self.q_weight_layer = Router(3, projection_dim, hidden_dim)
        # self.mu = args.mu

    def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))
        if self.args.router:
            q_weights = self.q_weight_layer(raw_combined_features)  # [B, 3]

        cat_feats = self.output_layer(combined_features)
        alpha = self.alpha(cat_feats)
        beta = self.beta(cat_feats)
        # mod_imgfeats = self.modified_ln(alpha * image_features + beta)
        mod_imgfeats = alpha * image_features + beta
        self_attn_feats = self.self_attn(cat_feats.unsqueeze(1)).squeeze(1)
        if self.args.router:
            output = q_weights[:, 0].unsqueeze(-1) * mod_imgfeats + \
                     q_weights[:, 1].unsqueeze(-1) * self_attn_feats + \
                     q_weights[:, 2].unsqueeze(-1) * (image_features + text_features)
        else:
            output = mod_imgfeats + self_attn_feats + image_features + text_features
        return F.normalize(output, dim=-1)


class CombinerwioQc(Combiner):
    def __init__(self, args, clip_feature_dim, projection_dim, hidden_dim):
        super().__init__(clip_feature_dim, projection_dim, hidden_dim)
        self.args = args
        self.dynamic_vector = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, clip_feature_dim), nn.Sigmoid())
        self.vector_norm = nn.LayerNorm(clip_feature_dim)
        self.alpha = nn.Sequential(nn.Linear(clip_feature_dim, clip_feature_dim), )
        self.beta = nn.Sequential(nn.Linear(clip_feature_dim, clip_feature_dim), )
        # self.modified_ln = nn.LayerNorm(clip_feature_dim)
        if args.router:
            self.q_weight_layer = Router(2, projection_dim, hidden_dim)
        self.mu = args.mu

    def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))

        if self.args.router:
            q_weights = self.q_weight_layer(raw_combined_features)  # [B, 2]

        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        dynamic_vector = self.dynamic_vector(raw_combined_features)
        dynamic_vector = F.normalize(dynamic_vector, dim=-1)
        dynamic_vector = 2 * (dynamic_vector - 0.5)  # when activate function == sigmoid()
        dynamic_vector = self.vector_norm(dynamic_vector)

        cat_feats = self.output_layer(combined_features)
        alpha = self.alpha(cat_feats)
        beta = self.beta(cat_feats)
        # mod_imgfeats = self.modified_ln(alpha * image_features + beta)
        mod_imgfeats = alpha * image_features + beta
        # self_attn_feats = self.self_attn(cat_feats.unsqueeze(1)).squeeze(1)
        if self.args.router:
            output = q_weights[:, 0].unsqueeze(-1) * mod_imgfeats + \
                     q_weights[:, 1].unsqueeze(-1) * ((dynamic_scalar + self.mu * dynamic_vector) * text_features + \
                                                      (1 - (
                                                                  dynamic_scalar + self.mu * dynamic_vector)) * image_features)
        else:
            output = mod_imgfeats + (dynamic_scalar + self.mu * dynamic_vector) * text_features + \
                     (1 - (dynamic_scalar + self.mu * dynamic_vector)) * image_features
        return F.normalize(output, dim=-1)


class CombinerV5(Combiner):
    def __init__(self, args, clip_feature_dim, projection_dim, hidden_dim):
        super().__init__(clip_feature_dim, projection_dim, hidden_dim)
        self.dynamic_vector = nn.Sequential(nn.Linear(projection_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
                                            nn.Linear(hidden_dim, clip_feature_dim), nn.Sigmoid())
        self.vector_norm = nn.LayerNorm(clip_feature_dim)

        self.self_attn = SelfAttentionCell(args)
        self.alpha = nn.Sequential(nn.Linear(clip_feature_dim, clip_feature_dim), )
        self.beta = nn.Sequential(nn.Linear(clip_feature_dim, clip_feature_dim), )
        # self.modified_ln = nn.LayerNorm(clip_feature_dim)
        self.q_weight_layer = Router(3, projection_dim, hidden_dim)
        self.mu = args.mu

    def forward(self, image_features: torch.tensor, text_features: torch.tensor) -> torch.tensor:
        text_projected_features = self.dropout1(F.relu(self.text_projection_layer(text_features)))
        image_projected_features = self.dropout2(F.relu(self.image_projection_layer(image_features)))

        raw_combined_features = torch.cat((text_projected_features, image_projected_features), -1)
        combined_features = self.dropout3(F.relu(self.combiner_layer(raw_combined_features)))

        q_weights = self.q_weight_layer(raw_combined_features)  # [B, 3]

        dynamic_scalar = self.dynamic_scalar(raw_combined_features)
        dynamic_vector = self.dynamic_vector(raw_combined_features)
        dynamic_vector = F.normalize(dynamic_vector, dim=-1)
        dynamic_vector = 2 * (dynamic_vector - 0.5)  # when activate function == sigmoid()
        dynamic_vector = self.vector_norm(dynamic_vector)

        cat_feats = self.output_layer(combined_features)
        # pdb.set_trace()
        alpha = self.alpha(cat_feats)
        beta = self.beta(cat_feats)
        # mod_imgfeats = self.modified_ln(alpha * image_features + beta)
        mod_imgfeats = alpha * image_features + beta
        self_attn_feats = self.self_attn(cat_feats.unsqueeze(1)).squeeze(1)
        # pdb.set_trace()
        output = q_weights[:, 0].unsqueeze(-1) * mod_imgfeats + \
                 q_weights[:, 1].unsqueeze(-1) * self_attn_feats + \
                 q_weights[:, 2].unsqueeze(-1) * ((dynamic_scalar + self.mu * dynamic_vector) * text_features + \
                                                  (1 - (dynamic_scalar + self.mu * dynamic_vector)) * image_features)
        return F.normalize(output, dim=-1)


class Router(nn.Module):
    def __init__(self, num_out_path, embed_size, hid):
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.mlp = nn.Sequential(nn.Linear(embed_size * 2, hid, bias=False),
                                 nn.LayerNorm(normalized_shape=hid),
                                 nn.ReLU(True),
                                 nn.Linear(hid, num_out_path, bias=False))

    def forward(self, x):  # [B, clip_feature_dim]
        # x = x.mean(-2)#b,k,d
        # pdb.set_trace()
        x = self.mlp(x)  # [B, 3]
        soft_g = torch.sigmoid(x)
        return soft_g