import json
import torch
from torch import nn
from typing import Optional,List,Tuple
from transformers import AutoModelForCausalLM,LlamaConfig,AutoConfig


from transformers import LlamaForCausalLM,LlamaModel
from models.unified_arch import UnifiedMetaModel,UnifiedMetaForCausalLM
from ipdb import set_trace

class UnifiedConfig(LlamaConfig):
    model_type = "unified_llm"


class UnifiedModel(UnifiedMetaModel,LlamaModel):
    config_class = UnifiedConfig

    def __init__(self, config: LlamaConfig):
        super(UnifiedModel, self).__init__(config)
        self.config = config

def is_avs_task(task_name):
    return task_name in ['ms3','s4','avss','ref-avs']


class UnifiedForCausalLM(LlamaForCausalLM,UnifiedMetaForCausalLM):
    config_class = UnifiedConfig

    def __init__(self, config: LlamaConfig, **kwargs):
        super().__init__(config)
        self.config=config
        self.model = UnifiedModel(config,**kwargs)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

        self.is_avs_task = False


    def get_model(self) -> UnifiedModel:
        return self.model


    def forward(
        self,
        batch_input_ids = None,
        batch_labels = None,
        batch_X_modals = None,
        batch_task_names = None,
        
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        
        self.is_avs_task = False
        
        if input_ids is not None and input_ids.shape[1]==1:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            input_ids = None

        elif inputs_embeds is None and batch_input_ids is not None:
            inputs = self.prepare_multimodal_inputs(
                batch_input_ids=batch_input_ids,
                batch_labels=batch_labels,
                batch_X_modals=batch_X_modals,
                batch_task_names=batch_task_names,
                return_multi_scale_features=False, 
                return_gt_mask=False,
            )
            input_ids = inputs['input_ids']
            inputs_embeds = inputs['inputs_embeds']
            attention_mask = inputs['attention_mask']
            labels = inputs['labels']
            position_ids = inputs['position_ids']

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            
        )
        return output


    def forward_avs(
        self,
        batch_input_ids = None,
        batch_labels = None,
        batch_X_modals = None,
        batch_task_names = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):

        if input_ids is not None and input_ids.shape[1]==1:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            input_ids = None

        elif inputs_embeds is None and batch_input_ids is not None:
            inputs = self.prepare_multimodal_inputs(
                batch_input_ids=batch_input_ids,
                batch_labels=batch_labels,
                batch_X_modals=batch_X_modals,
                batch_task_names=batch_task_names,
                return_multi_scale_features=True, 
                return_gt_mask=True,
            )
            input_ids = inputs['input_ids']
            inputs_embeds = inputs['inputs_embeds']
            attention_mask = inputs['attention_mask']
            labels = inputs['labels']
            position_ids = inputs['position_ids']

            mask_token_mask = inputs.get('mask_token_mask',None)
            multi_scale_image_features = inputs.get('multi_scale_image_features',None)
            
            gt_mask = inputs.get('gt_mask',None)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        
        output_hidden_states = output.hidden_states
        avs_hidden_states = output_hidden_states[-1] 
        avs_nums, seq_len, dim = avs_hidden_states.shape
        pred_embeddings = avs_hidden_states[mask_token_mask]
        pred_embeddings = pred_embeddings.reshape(avs_nums,-1,dim)
        seg_output = self.model.postprocess_seg(
            pred_embeddings = pred_embeddings, 
            multi_scale_image_feature_list = multi_scale_image_features, 
            gt_mask = gt_mask, 
            batch_task_names = batch_task_names,
        )
        mask_loss = seg_output['mask_loss']
        output.loss = mask_loss
        return output
        

    @torch.no_grad()
    def generate(
        self,
        batch_input_ids,
        batch_labels,
        batch_X_modals,
        batch_task_names,
        **kwargs
    ):
        inputs = self.prepare_multimodal_inputs(
            batch_input_ids = batch_input_ids,
            batch_labels = batch_labels,
            batch_X_modals = batch_X_modals,
            return_multi_scale_features=False,
            return_gt_mask=False,
            batch_task_names=batch_task_names
        )
        inputs_embeds = inputs['inputs_embeds']
        return super().generate(
            inputs_embeds = inputs_embeds,
            output_hidden_states=False,
            return_dict_in_generate=False,
            **kwargs
        )


    @torch.no_grad()
    def generate_avs(
        self,
        batch_input_ids,
        batch_labels,
        batch_X_modals,
        batch_task_names,
        **kwargs
    ):
        
        
        self.is_avs_task = True
        
        inputs = self.prepare_multimodal_inputs(
            batch_input_ids = batch_input_ids,
            batch_labels = batch_labels,
            batch_X_modals = batch_X_modals,
            return_multi_scale_features=True,
            return_gt_mask=True,
            batch_task_names=batch_task_names
        )
        input_ids = inputs['input_ids']
        inputs_embeds = inputs['inputs_embeds']
        attention_mask = inputs['attention_mask']
        labels = inputs['labels']
        position_ids = inputs['position_ids']
        mask_token_mask = inputs.get('mask_token_mask',None)
        multi_scale_image_features = inputs.get('multi_scale_image_features',None)
        gt_mask = inputs.get('gt_mask',None)

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states
        avs_hidden_states = output_hidden_states[-1] 
        avs_nums, seq_len, dim = avs_hidden_states.shape
        pred_embeddings = avs_hidden_states[mask_token_mask]
        pred_embeddings = pred_embeddings.reshape(avs_nums,-1,dim)
        seg_output = self.model.postprocess_seg(
            pred_embeddings = pred_embeddings, 
            multi_scale_image_feature_list = multi_scale_image_features, 
            gt_mask = None, 
            batch_task_names = batch_task_names,
        )
        
        return seg_output


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

    
    @property
    def device(self):
        return list(self.parameters())[0].device
    

AutoConfig.register("unified_llm", UnifiedConfig)
AutoModelForCausalLM.register(UnifiedConfig, UnifiedForCausalLM)
