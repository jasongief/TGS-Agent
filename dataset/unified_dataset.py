import json
import ast
import os
from os.path import join,exists
import numpy as np
import pandas as pd
import cv2,csv
from typing import Sequence,Dict
from dataclasses import dataclass
import librosa
from PIL import Image
import torch
import random
import transformers
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from decord import VideoReader
from transformers import CLIPImageProcessor

from dataset.audio_processor import preprocess


class UnifiedDataset(Dataset):
    def __init__(
        self,
        mode='train', # train,val,test
        video_processor: CLIPImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        image_size = 224,
        video_frame_nums = 10,
        # avqa
        avqa_task=False,
        # ave task
        ave_task = False,
        # avvp task
        avvp_task = False,
        # avs task
        avss_task = False,
        ms3_task=False,
        s4_task=False,
        ref_avs_task = False,
        multi_frames = False,
        image_scale_nums = 2,
        token_nums_per_scale = 3,
        # audio referred image grounding task
        arig_task = False,
        # av caption task
        avcap_task = False,
    ) -> None:
        super().__init__()

        self.mode=mode
        self.video_processor = video_processor
        self.multi_frames = multi_frames
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums

        # if avss_task or ms3_task or s4_task or ref_avs_task:
        # token_nums = image_scale_nums * token_nums_per_scale
        # mask_token = [f'<mask_{i}>' for i in range(token_nums)]
        # self.mask_token = ''.join(mask_token)
        # print('mask token: ',self.mask_token)

        self.samples = []
        self.tot = 0

        if ref_avs_task:
            self.add_ref_avs_samples()
        
        print(f'tot training sample nums: {self.tot}')


    def add_ref_avs_samples(self):
        data_root = './REFAVS'
        meta_csv_root = "./R2AVSBench"
        cot_anno_root = "./R2AVSBench/RefThinker_instruction_tuning_set.json"
        with open(cot_anno_root, 'r') as f:
            cot_anno_lines = json.load(f)

        tot = 0
        with open(join(meta_csv_root,'RefAVSBench_metadata.csv'),'r') as f:

            rows = csv.reader(f)
            for row in rows:
                vid, uid, split, fid, exp = row
                if split != 'train':
                    continue
                vid = uid.rsplit('_', 2)[0]  # TODO: use encoded id.
                # obj = uid.rsplit('_',2)[1]
                if uid.startswith('null_'): # 
                    audio_path = join(data_root,'media_cross',vid,'audio.wav')
                else:
                    audio_path = join(data_root,'media',vid,'audio.wav')
                image_path_list = [join(data_root,'media',vid,'frames',str(i)+'.jpg') for i in range(10)]
                mask_path_list = [join(data_root,'gt_mask',vid,'fid_'+str(fid),'0000'+str(i)+'.png') for i in range(10)]
                instruction = (
                    "This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\n"
                    f"Given the referential expression: '{exp.lower()}', analyze the video and audio, and then generate a reasoning chain (<think>) and final answer (<answer>).\n"
                    "Your output must follow this format:\n"
                    "<think>\nYour reasoning here\n</think>\n"
                    "<answer>\n<f_object> ... </f_object>\n<s_object> ... </s_object>\nThe corresponding bbox is: <frame> <box> ... </box> </frame>\n</answer>"
                    # "<answer>\n<f_object> ... </f_object>\n<s_object> ... </s_object> </answer>" # may change to this if do not want the caption about bbox, the prediction of bbox is also not used
                )
                output = cot_anno_lines[uid]['output']

                self.samples.append(
                    {
                        'instruction': instruction,
                        'output': output,
                        'image_path':None,
                        'mask_path':mask_path_list,
                        'audio_path':audio_path,
                        'image_path_list':image_path_list,
                        'vid':vid,
                        'uid':uid,
                        'fid':fid,
                        'task_name':'ref-avs',
                    }
                )
                tot += 1
        
        self.tot += tot
        print(f'ref-avs sample nums: {tot}')


    def read_label(self,label_path):
        with open(label_path,'r') as f:
            label = f.read()
        return label


    def __len__(self):
        return len(self.samples)


    def __getitem__(self,idx):
        sample = self.samples[idx]
        task_name = sample['task_name']
        instruction = sample['instruction']
        output = sample.get('output')

        data = {
            'instruction': "<s>" +instruction,
            'output':output + "</s>",
            'task_name':task_name,
        }
        
        if task_name == 'ref-avs':
            ## video
            image_path_list = sample['image_path_list']
            video = []
            for path in image_path_list:
                image = Image.open(path).convert('RGB')
                image = image.resize((224,224))
                image = self.video_processor.preprocess([image],return_tensors='pt')
                image = image['pixel_values']  
                video.append(image)
            video = torch.cat(video,dim=0) # t,c,h,w
            data['video'] = video

            ## audio
            audio_path = sample['audio_path']
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 10
            nums_per_second = int(length / tot)
            indices = [i for i in range(tot)]
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if len(audio_seg) < 1 * nums_per_second:
                    sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['audio'] = audio_feature

            ## image
            # image_path = sample['image_path']
            # image = Image.open(image_path).convert('RGB')
            # image = image.resize((224,224))
            # image = self.video_processor.preprocess([image],return_tensors='pt')
            # image = image['pixel_values']  # t,c,h,w
            # data['image'] = image
            
            ## mask
            # mask_path = sample['mask_path']
            # mask = cv2.imread(mask_path)
            # gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            # gt_mask = gray_mask > 0
            # gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
            # gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
            # data['mask'] = gt_mask

        return data


class UnifiedTestDataset(Dataset):
    def __init__(
        self,
        mode='test', # train,val,test
        video_processor: CLIPImageProcessor = None,
        tokenizer: PreTrainedTokenizer = None,
        image_size = 224,
        video_frame_nums = 10,
        # avqa
        avqa_task=False,
        # ave task
        ave_task = False,
        # avvp task
        avvp_task = False,
        # avs task
        avss_task = False,
        ms3_task=False,
        s4_task=False,
        multi_frames = False,
        image_scale_nums = 2,
        token_nums_per_scale = 3,
        ref_avs_task = False,
        test_name = 'test_s',  # for ref-avs: test_s, test_u, test_n
        # audio referred image grounding task
        arig_task = False,
        # avcap task
        avcap_task = False,
    ) -> None:
        super().__init__()

        self.mode=mode
        self.video_processor = video_processor
        self.multi_frames = multi_frames
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.video_frame_nums = video_frame_nums
        self.test_name = test_name


        self.samples = []
        self.tot = 0

        if ref_avs_task:
            self.add_ref_avs_samples()
        print(f'tot test sample nums: {self.tot}')

    def add_ref_avs_samples(self):
        data_root = './REFAVS'
        meta_csv_root = "./R2AVSBench"

        tot = 0
        with open(join(meta_csv_root,'RefAVSBench_metadata.csv'),'r') as f: 
        # with open(join(data_root,'R2AVSBench_metadata.csv'),'r') as f: # may change to R^2-AVSBench meta csv
        # with open(join(data_root,'RefAVSBenchRef_R2AVSBenchVideo.csv'),'r') as f: #RefAVSBench reference but using R^2-AVSBench videos
            rows = csv.reader(f)
            for row in rows:
                vid, uid, split, fid, exp = row
                if split != self.test_name:  # test_s,test_u,test_n
                    continue
                vid = uid.rsplit('_', 2)[0]  
                obj = uid.rsplit('_',2)[1]
                if uid.startswith('null_'): # 
                    audio_path = join(data_root,'media_cross',vid,'audio.wav')
                else:
                    audio_path = join(data_root,'media',vid,'audio.wav')
                image_path_list = [join(data_root,'media',vid,'frames',str(i)+'.jpg') for i in range(10)]
                mask_path_list = [join(data_root,'gt_mask',vid,'fid_'+str(fid),'0000'+str(i)+'.png') for i in range(10)]
                instruction = (
                    'This is a video:\n<video_start><video><video_end>\nThis is an audio:\n<audio_start><audio><audio_end>\n'
                    f"Given the referential expression {exp.lower()}, analyze the video and audio, and then generate a reasoning chain (<think>) and final answer (<answer>).\n"
                    "Your output must follow this format:\n"
                    "<think>\nYour reasoning here\n</think>\n"
                    "<answer>\n<f_object> ... </f_object>\n<s_object> ... </s_object>\nThe corresponding bbox is: <frame> <box> ... </box> </frame>\n</answer>"
                )
                output = f'<answer>\n<f_object> ... </f_object>\n<s_object> {obj} </s_object>\nThe corresponding bbox is: <frame> <box> ... </box> </frame>\n</answer>'
                self.samples.append(
                    {
                        'instruction': instruction,
                        'output': output,
                        'image_path':None,
                        'mask_path':None,
                        'audio_path':audio_path,
                        'image_path_list':image_path_list,
                        'mask_path_list':mask_path_list,
                        'uid':uid,
                        'ref':exp,
                        'fid':fid,
                        'task_name':'ref-avs',
                    }
            )
            tot += 1
        self.tot += tot
        print(f'ref-avs {self.test_name} sample nums: {tot}')



    def __len__(self):
        return len(self.samples)


    def read_label(self,label_path):
        if not os.path.exists(label_path):
            return 'no label.'
        with open(label_path,'r') as f:
            label = f.read()
        return label


    def __getitem__(self,idx):
        sample = self.samples[idx]
        uid = sample['uid']
        ref = sample['ref']
        task_name = sample['task_name']
        instruction = sample['instruction']
        output = sample.get('output')
        # if output is None:
        #     label_path = sample['label_path']
        #     output = self.read_label(label_path)


        # if self.tokenizer is not None and hasattr(self.tokenizer,'apply_chat_template'):
        #     messages = [
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": instruction},
        #     ]
        #     instruction = self.tokenizer.apply_chat_template(conversation=messages,add_generation_prompt=True,tokenize=False)
        #     output = output + '</s>'
        
        data = {
            'uid':uid,
            'ref':ref,
            'instruction': "<s>" +instruction,
            'output':output + "</s>",
            'task_name':task_name,
        }
    
        if task_name == 'ref-avs':
            ## video
            image_path_list = sample['image_path_list']
            video = []
            for path in image_path_list:
                image = Image.open(path).convert('RGB')
                image = image.resize((224,224))
                image = self.video_processor.preprocess([image],return_tensors='pt')
                image = image['pixel_values']  
                video.append(image)
            video = torch.cat(video,dim=0) # t,c,h,w
            data['video'] = video

            ## audio
            audio_path = sample['audio_path']
            audio_feature = []
            audio, sr = librosa.load(audio_path,sr=16000,mono=True)
            length = len(audio)
            tot = 10
            nums_per_second = int(length / tot)
            indices = [i for i in range(tot)]
            for indice in indices:
                start_time = max(0, indice)
                end_time = min(tot, indice + 1)
                audio_seg = audio[int(start_time * nums_per_second) : int(nums_per_second * end_time)]
                if len(audio_seg) < 1 * nums_per_second:
                    sil = np.zeros(1 * nums_per_second - len(audio_seg), dtype=float)
                    audio_seg = np.concatenate((audio_seg, sil),axis=0)
                audio_seg = torch.from_numpy(audio_seg).unsqueeze(0)
                fbank = preprocess(audio_seg)
                fbank = fbank.squeeze(0).to(torch.float32) # L,128   1s -> 98 tokens
                audio_feature.append(fbank)
            audio_feature = torch.stack(audio_feature,dim=0) # t,L,128
            data['audio'] = audio_feature
            data['audio_path'] = audio_path

            ## image
            # image_path = sample['image_path']
            # image = Image.open(image_path).convert('RGB')
            # image = image.resize((224,224))
            # image = self.video_processor.preprocess([image],return_tensors='pt')
            # image = image['pixel_values']  # t,c,h,w
            # data['image'] = image
            # data['image_path'] = image_path
            
            ## mask
            # mask_path = sample['mask_path']
            # mask = cv2.imread(mask_path)
            # gray_mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            # gt_mask = gray_mask > 0
            # gt_mask = cv2.resize(gt_mask.astype(np.float32),(224,224),interpolation=cv2.INTER_NEAREST)
            # gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(torch.float32) # (1,224,224)
            # data['mask'] = gt_mask
            # data['mask_path'] = mask_path


        return data



@dataclass
class DataCollatorForUnifiedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_task_names = []

        for instance in instances:
            instruction=instance['instruction']
            output=instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            input_ids = instruction_ids + output_ids
            # print('instruction_ids', instruction_ids)
            # print('output_ids', output_ids)
            label = [-100] * len(instruction_ids) + output_ids
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image
                
            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio
            
            mask = instance.get('mask',None)
            if mask is not None:
                X_modals['<mask>'] = mask
            
            batch_X_modals.append(X_modals)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_task_names':batch_task_names
        }


@dataclass
class DataCollatorForUnifiedTestDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]):
        
        tokenizer=self.tokenizer
        batch_input_ids=[]
        batch_label=[]
        batch_X_modals=[]
        batch_metadata=[] # 比 train多了这个
        batch_task_names = []

        for instance in instances:
            instruction = instance['instruction']
            output = instance['output']
            task_name = instance['task_name']
            batch_task_names.append(task_name)

            uid = instance.get('uid',None)
            ref = instance.get('ref',None)

            metadata = {
                'uid':uid,
                'ref':ref,
                'instruction': instruction,
                'output': output,
            }
            
            if task_name == 'avqa':
                question_type = instance.get('question_type',None)
                vid = instance.get('vid',None)
                qid = instance.get('qid',None)
                metadata.update(
                    {
                        'question_type':question_type,
                        'vid':vid,
                        'qid':qid
                    }
                )
            
            instruction_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(instruction))
            output_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(output))
            

            input_ids = instruction_ids
            label = [-100] * len(instruction_ids)
            batch_input_ids.append(torch.tensor(input_ids,dtype=torch.long))
            batch_label.append(torch.tensor(label,dtype=torch.long))
            X_modals = {}
            image = instance.get('image',None)
            if image is not None:
                X_modals['<image>'] = image
                metadata['image_path'] = instance.get('image_path','')
                
            video = instance.get('video',None)
            if video is not None:
                X_modals['<video>'] = video
                metadata['video_path'] = instance.get('video_path','')

            audio = instance.get('audio',None)
            if audio is not None:
                X_modals['<audio>'] = audio
                metadata['audio_path'] = instance.get('audio_path','')
            
            mask = instance.get('mask',None)
            if mask is not None:
                X_modals['<mask>'] = mask
                metadata['mask_path'] = instance.get('mask_path','')
            
            batch_X_modals.append(X_modals)
            batch_metadata.append(metadata)

        
        return {
            'batch_input_ids':batch_input_ids,
            'batch_labels':batch_label,
            'batch_X_modals':batch_X_modals,
            'batch_metadata':batch_metadata,
            'batch_task_names':batch_task_names,
        }


def get_dataset_collator(
    data_args,tokenizer: transformers.PreTrainedTokenizer,
    image_processor=None,mode='train',
    image_scale_nums = 2, token_nums_per_scale = 3, test_name = 'test_s',
):
    if mode == 'train':
        dataset = UnifiedDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            avqa_task=data_args.avqa_task,
            ave_task=data_args.ave_task,
            avvp_task = data_args.avvp_task,
            arig_task = data_args.arig_task, 
            avss_task=data_args.avss_task,
            ms3_task=data_args.ms3_task,
            s4_task=data_args.s4_task,
            ref_avs_task=data_args.ref_avs_task,
            avcap_task=data_args.avcap_task,
            multi_frames=data_args.multi_frames,
            image_scale_nums=image_scale_nums,
            token_nums_per_scale=token_nums_per_scale
        )
        data_collator = DataCollatorForUnifiedDataset(tokenizer=tokenizer)
    
    elif mode == 'test':
        dataset = UnifiedTestDataset(
            video_processor=image_processor,
            tokenizer=tokenizer,
            avqa_task=data_args.avqa_task,
            ave_task=data_args.ave_task,
            avvp_task = data_args.avvp_task,
            arig_task = data_args.arig_task, 
            avcap_task = data_args.avcap_task,
            avss_task=data_args.avss_task,
            ms3_task=data_args.ms3_task,
            s4_task=data_args.s4_task,
            ref_avs_task=data_args.ref_avs_task,
            test_name=test_name,
            multi_frames=data_args.multi_frames,
            image_scale_nums=image_scale_nums,
            token_nums_per_scale=token_nums_per_scale
        )
        data_collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)
    
    return dataset,data_collator


