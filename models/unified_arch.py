import torch
from abc import ABC, abstractmethod
from torch import nn
from ipdb import set_trace

from models.multimodal_encoder import (
    VisualEncoder,
    AudioEncoder,
    VLProjector,
    ALProjector,
    SegModule,
    MaskEncoder
)

class UnifiedMetaModel:

    def __init__(self, config):
        super(UnifiedMetaModel, self).__init__(config)
        self.config = config


    def init_multimodal_modules(
        self,
        d_model = 3584,
        # visual
        vit_ckpt_path = '/l/users/panwen.hu/jinxing.zhou/mllm/Crab/pretrained_weights/clip-vit-large-patch14',
        select_layer_list = [-11,-1],
        select_feature = 'patch',
        image_size = 224,
        patch_size = 14,
        visual_query_token_nums = 32,
        # audio
        BEATs_ckpt_path = '/l/users/panwen.hu/jinxing.zhou/mllm/Crab/pretrained_weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
        audio_query_token_nums = 32,
        # seg
        image_scale_nums = 2,
        token_nums_per_scale = 3,
        avs_query_num = 300,
        num_classes = 1,
        query_generator_num_layers = 2,
        prompt_embed_dim = 256,
        mask_decoder_transformer_depth = 2,
        low_res_mask_size = 112,
        dice_loss_weight = 0.5,
        bce_loss_weight = 2.0,
        vit_image_embedding_dim = 1024,
        visual_branch = False,
        audio_branch = False,
        segment_branch = False,
        # vqgan
        use_vqgan = False,
    ):
        
        if visual_branch:
            image_token_nums = (image_size//patch_size) * (image_size//patch_size)
            self.visual_encoder = VisualEncoder(model_name_or_path=vit_ckpt_path,select_layer_list=select_layer_list,
                                                select_feature=select_feature)
            self.vl_projector = VLProjector(hidden_size=1024, d_model=d_model, depth=2, image_token_nums=image_token_nums,
                                            num_query_token=visual_query_token_nums, num_hidden_layers=2,)
            print('init visual_encoder, vl_projector finished...')

        if audio_branch:
            self.audio_encoder =  AudioEncoder(ckpt_path=BEATs_ckpt_path)
            self.al_projector = ALProjector(hidden_size=768, d_model=d_model, depth=2, num_query_token=audio_query_token_nums,
                                            num_hidden_layers=2)
            print('init audio_encoder, al_projector finished...')


        if segment_branch:
            self.low_res_mask_size = low_res_mask_size
            self.seg_module = SegModule(
                d_model=d_model,
                prompt_embed_dim=prompt_embed_dim,
                image_scale_nums=image_scale_nums,
                token_nums_per_scale=token_nums_per_scale,
                mask_decoder_transformer_depth=mask_decoder_transformer_depth,
                vit_image_embedding_dim=vit_image_embedding_dim,
                avs_query_num=avs_query_num,
                num_classes=num_classes,
                query_generator_num_layers=query_generator_num_layers,
                image_size=image_size,
                patch_size=patch_size,
                image_embedding_size=(image_size // patch_size),
                dice_loss_weight=dice_loss_weight,
                bce_loss_weight=bce_loss_weight,
            )
            print('init seg_module finished...')

        if use_vqgan:
            self.mask_encoder = MaskEncoder(token_shift=32000+20)


    def encode_video(self,visual,batch_question=None):
        vit_feature_list = self.visual_encoder(visual)  # [(b,t*n,d),(b,t*n,d),...]
        qformer_feature_list = []
        for vit_feature in vit_feature_list:
            qformer_feature = self.vl_projector(vit_feature,batch_question) # b,t*256 -> b,t*32
            qformer_feature_list.append(qformer_feature)
        return vit_feature_list,qformer_feature_list


    def encode_audio(self,audio,batch_qustion=None):
        audio_feature = self.audio_encoder(audio)
        audio_feature = self.al_projector(audio_feature,batch_qustion)
        return audio_feature


    def encode_mask(self,mask):
        return self.mask_encoder(mask)


    def postprocess_seg(
        self,
        pred_embeddings, # bs,nums,dim
        multi_scale_image_feature_list,
        gt_mask = None,
        batch_task_names = [],
    ):
        low_res_mask_size = self.low_res_mask_size
        return self.seg_module(
            pred_embeddings = pred_embeddings,
            multi_scale_image_feature_list = multi_scale_image_feature_list,
            low_res_mask_size = low_res_mask_size,
            gt_mask = gt_mask,
            batch_task_names = batch_task_names,
        )


class UnifiedMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self) -> UnifiedMetaModel:
        pass

    def encode_audio(self,audio,batch_qustion=None, batch_first=True):
        if not batch_first:
            audio = audio.unsqueeze(0)
        audio_feature = self.get_model().encode_audio(audio,batch_qustion=batch_qustion)
        if not batch_first:
            audio_feature = audio_feature.squeeze(0)
        return audio_feature


    def encode_video(self,video,batch_question=None,batch_first=True):
        if not batch_first:
            video = video.unsqueeze(0)
        vit_feature_list, qformer_feature_list = self.get_model().encode_video(video,batch_question=batch_question)
        if not batch_first:
            vit_feature_list = [item.squeeze(0) for item in vit_feature_list]
            qformer_feature_list = [item.squeeze(0) for item in qformer_feature_list]
        return vit_feature_list,qformer_feature_list


    def encode_mask(self,mask,batch_first=False):
        if not batch_first:
            mask = mask.unsqueeze(0)
        indices = self.get_model().encode_mask(mask) # b,n
        if not batch_first:
            indices = indices.squeeze(0)
        return indices
    

    def encode_ids(self,ids):
        return self.get_model().embed_tokens(ids)
    
    
    def prepare_multimodal_inputs(
        self,
        batch_input_ids,
        batch_labels,
        batch_X_modals,
        batch_audio_question = None,
        batch_visual_question = None,
        batch_task_names = None,
        return_multi_scale_features = False,
        return_gt_mask = False,
    ):
        device = self.device
        bs = len(batch_input_ids)

        gt_mask = []
        scale = 2
        multi_scale_image_features = [[] for _ in range(scale)]
        if return_gt_mask:
            for i, X_modals in enumerate(batch_X_modals):
                is_avs_task = batch_task_names[i] in ['ms3','s4','avss','ref-avs']
                if not is_avs_task:
                    continue
                for key, X_modal in X_modals.items():
                    if key == '<mask>':
                        gt_mask.append(X_modal) # [(1,224,224), ...]
                    elif key == '<image>':
                        vit_feature_list, qformer_feature_list = self.encode_video(batch_X_modals[i][key],batch_question=None,batch_first=False)
                        if return_multi_scale_features and is_avs_task:
                            for _scale in range(scale):
                                multi_scale_image_features[_scale].append(vit_feature_list[_scale])
            if len(gt_mask) > 0:
                gt_mask = torch.stack(gt_mask,dim=0) # b,1,224,224

        max_length = 0
        new_batch_inputs_embeds = []
        new_batch_attention_mask = []
        new_batch_labels = []
        batch_mask_token_indices = []
        keys = self.KEYS
        for i in range(bs):
            input_ids = batch_input_ids[i]
            labels = batch_labels[i]
            
            task_name = batch_task_names[i]
            
            X_token_indices = torch.where(torch.any(torch.stack([input_ids == self.SPECIAL_TOKEN_2_IDS[key] for key in keys]), dim=0))[0]
            X_token_indices = X_token_indices.tolist()            

            inputs_embeds_seg=[]
            labels_seg=[]
            pre_indice=0
            for idx,indice in enumerate(X_token_indices):
                inputs_embeds_seg.append(self.encode_ids(input_ids[pre_indice:indice]))
                labels_seg.append(labels[pre_indice:indice])
                
                special_token = self.IDS_2_SPECIAL_TOKEN[input_ids[indice].item()]
                if special_token == '<audio>':
                    feature = self.encode_audio(batch_X_modals[i][special_token],batch_qustion=None,batch_first=False)
                    inputs_embeds_seg.append(feature)
                    labels_seg.append(torch.full((feature.shape[0],),-100,dtype=torch.long,device=device))
                elif special_token == '<video>':
                    vit_feature_list, qformer_feature_list = self.encode_video(batch_X_modals[i][special_token],batch_question=None,batch_first=False)
                    feature = qformer_feature_list[-1] # last layer qformer feature
                    inputs_embeds_seg.append(feature)  
                    labels_seg.append(torch.full((feature.shape[0],),-100,dtype=torch.long,device=device))
                elif special_token == '<image>':
                    vit_feature_list, qformer_feature_list = self.encode_video(batch_X_modals[i][special_token],batch_question=None,batch_first=False)
                    feature = qformer_feature_list[-1] # last layer qformer feature
                    inputs_embeds_seg.append(feature)
                    labels_seg.append(torch.full((feature.shape[0],),-100,dtype=torch.long,device=device))
                elif special_token == '<mask>': 
                    indices = self.encode_mask(batch_X_modals[i][special_token],batch_first=False)
                    feature = self.encode_ids(indices)
                    inputs_embeds_seg.append(feature)
                    labels_seg.append(indices)
                
                pre_indice = indice + 1

            # add last tokens
            inputs_embeds_seg.append(self.encode_ids(input_ids[pre_indice:]))
            labels_seg.append(labels[pre_indice:])
            # concat segs
            inputs_embeds_seg = torch.cat(inputs_embeds_seg,dim=0)
            attention_mask_seg = torch.ones(inputs_embeds_seg.shape[0],dtype=torch.int32,device=device)
            labels_seg = torch.cat(labels_seg,dim=0)
            
            new_batch_inputs_embeds.append(inputs_embeds_seg)
            new_batch_attention_mask.append(attention_mask_seg)
            new_batch_labels.append(labels_seg)

            max_length = max(max_length,inputs_embeds_seg.shape[0])


        if return_multi_scale_features:
            multi_scale_image_features = [torch.stack(item,dim=0) for item in multi_scale_image_features if len(item) > 0]

        ### left padding
        padding_inputs_embeds = []
        padding_attention_mask = []
        padding_labels = []
        for i in range(bs):
            embeds = new_batch_inputs_embeds[i]
            mask = new_batch_attention_mask[i]
            labels = new_batch_labels[i]
            
            L,d = embeds.shape
            pad_embeds = self.encode_ids(torch.full((max_length-L,),self.get_model().pad_token_id,dtype=torch.long,device=device))
            padding_inputs_embeds.append(torch.cat([pad_embeds,embeds],dim=0))
            padding_attention_mask.append(torch.cat([torch.zeros((max_length-L),dtype=torch.int32,device=device),mask],dim=0)) 
            padding_labels.append(torch.cat([torch.full((max_length-L,),-100,dtype=torch.long,device=device),labels],dim=0))
            
        padding_inputs_embeds = torch.stack(padding_inputs_embeds,dim=0)
        padding_attention_mask = torch.stack(padding_attention_mask,dim=0)
        padding_labels = torch.stack(padding_labels,dim=0)

        position_ids = torch.cumsum(padding_attention_mask,dim=-1) - 1
        position_ids[position_ids==-1] = 0

        dict_data = {
            'input_ids':None,
            'inputs_embeds':padding_inputs_embeds,
            'attention_mask':padding_attention_mask,
            'labels':padding_labels,
            'position_ids':position_ids,
        }
        if return_multi_scale_features:
            dict_data['multi_scale_image_features'] = multi_scale_image_features
        if return_gt_mask:
            dict_data['gt_mask'] = gt_mask
        
        return dict_data
    

    def initialize_MM_tokenizer(self, tokenizer, mask_token_nums = 6, output_embeddings_require_grad = False, use_vqgan = False):
        vocab_nums=len(tokenizer)
        added_tokens = []
        image_tokens = ['<image>','<image_start>','<image_end>']
        added_tokens += image_tokens
        video_tokens = ['<video>','<video_start>','<video_end>']
        added_tokens += video_tokens
        audio_tokens = ['<audio>','<audio_start>','<audio_end>']
        added_tokens += audio_tokens
        cot_tokens = [
            "<think>", "</think>",
            "<answer>", "</answer>",
            "<f_object>", "</f_object>",
            "<s_object>", "</s_object>",
            "<frame>", "</frame>",
            "<box>", "</box>"
        ]
        added_tokens += cot_tokens
        tokenizer.add_tokens(added_tokens,special_tokens=True)


        self.KEYS = ['<image>','<video>','<audio>']
        
        self.SPECIAL_TOKEN_2_IDS={
            token : i + vocab_nums for i,token in enumerate(added_tokens)
        }
        self.IDS_2_SPECIAL_TOKEN={
            i + vocab_nums:token for i,token in enumerate(added_tokens)
        }
        '''
            {'<image>': 151646, '<image_start>': 151647, '<image_end>': 151648, '<video>': 151649, '<video_start>': 151650, 
            '<video_end>': 151651, '<audio>': 151652, '<audio_start>': 151653, '<audio_end>': 151654, '<mask_start>': 151655, 
            '<mask_end>': 151656, '<mask_0>': 151657, '<mask_1>': 151658, '<mask_2>': 151659, '<mask_3>': 151660, 
            '<mask_4>': 151661, '<mask_5>': 151662}
        '''
        self.resize_token_embeddings(len(tokenizer))


    @property
    def device(self):
        return list(self.parameters())[0].device


