#coding=utf8
import os
from PIL import Image
import time
import argparse

from diffusers import FluxFillPipeline, FluxPriorReduxPipeline, FluxTransformer2DModel, BitsAndBytesConfig
from transformers import AutoModelForImageSegmentation
import torch 

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.aigc_run_parsing import Parsing
from tools.utils import resize_only, background_remove, concat_2_img, get_mask_location

class TryOnFluxPipe:
    def __init__(self, flux_fill_model_path, flux_redux_model_path, \
                 background_remove_model_path, openpose_model_path, humanparsing_model_path, \
                 gpu_id = 0, remove_background = True, dtype = "bfloat16"):
        
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        
        self.pipe = FluxFillPipeline.from_pretrained(flux_fill_model_path,torch_dtype=torch_dtype)
        self.pipe.enable_model_cpu_offload()
        print('load flux fill model ok...')
        self.pipe_prior_redux = FluxPriorReduxPipeline.from_pretrained(flux_redux_model_path, torch_dtype=torch_dtype)
        self.pipe_prior_redux.enable_model_cpu_offload()
        print('load flux redux model ok...')
        if openpose_model_path:
            self.openpose_model = OpenPose(gpu_id, openpose_model_path)
            print('load openpose model ok...')
        if humanparsing_model_path:
            self.parsing_model = Parsing(gpu_id, humanparsing_model_path)
            print('load humanparsing model ok...')
        if remove_background and background_remove_model_path:
            self.background_remove_model = AutoModelForImageSegmentation.from_pretrained(background_remove_model_path, trust_remote_code=True)
            torch.set_float32_matmul_precision(['high', 'highest'][0])
            self.background_remove_model.to('cuda')
            self.background_remove_model.eval()
            print('load background removal model ok...')
            
        self.category_dict_utils = ['upper_body', 'lower_body', 'whole_body', 'short_dresses', 'long_dresses', 'short_skirt', 'long_skirt', 'decoration']

        print('load model ok')
            
    def run(self, model_img, cloth_img, model_mask_img = None, category = 0, n_samples=1, n_steps=20, seed = -1, remove_background = True, max_size =1120):

        if remove_background and model_mask_img is None:
            cloth_img, cloth_mask = background_remove(cloth_img, self.background_remove_model)

        if model_mask_img is None:
            keypoints = self.openpose_model(model_img.resize((384, 512)))
            model_parse, _ = self.parsing_model(model_img.resize((384, 512)))
            model_mask,mask_gray = get_mask_location(self.category_dict_utils[category], model_parse, keypoints)
            model_mask = model_mask.resize(model_img.size, Image.NEAREST)
        else:
            model_mask = model_mask_img.resize(model_img.size, Image.NEAREST)

        concImage, concMask, oriSize = concat_2_img(model_img, cloth_img, model_mask)
        #### scale image 
        concImage, resize_scaled = resize_only(concImage, (max_size, max_size))
        concMask, resize_scaled = resize_only(concMask, (max_size, max_size))
        oriSize = [int(x * resize_scaled) for x in oriSize]
        #### flux inference
        t1 = time.time()

        pipe_prior_output = self.pipe_prior_redux(cloth_img)
        w,h = concImage.size
        images = self.pipe(
            image=concImage,
            mask_image=concMask,
            height=h,
            width=w,
            guidance_scale=50,
            num_images_per_prompt = n_samples, 
            num_inference_steps=n_steps,
            max_sequence_length=512,
            **pipe_prior_output,
            generator=torch.Generator("cpu").manual_seed(seed),
            )
        t2 = time.time()

        print("Inference Time:", t2-t1)

        images = [image.crop((0, 0, oriSize[0], oriSize[1])) for image in images.images]
        for idx, image in enumerate(images):
            image.save(str(category) + '_' + str(idx) + ".jpg")
        return images
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='try on based on flux model')
    parser.add_argument('--flux_fill_model_path', type=str, default="./models/models--black-forest-labs--FLUX.1-Fill-dev/snapshots/FLUX.1-Fill-dev", required=True)
    parser.add_argument('--flux_redux_model_path', type=str, default="./models/models--black-forest-labs--FLUX.1-Fill-dev/snapshots/FLUX.1-Redux-dev", required=True)
    parser.add_argument('--background_remove_model_path', type=str, default=None, required=False)
    parser.add_argument('--openpose_model_path', type=str, default=None, required=False)
    parser.add_argument('--humanparsing_model_path', type=str, default=None, required=False)
    parser.add_argument('--remove_background', type=bool, default=True, required=False)
    parser.add_argument('--model_path', type=str, default="./images/models/liuyifei.jpeg", required=False)
    parser.add_argument('--model_mask_path', type=str, default=None, required=False)
    parser.add_argument('--cloth_path', type=str, default="./images/cloth/itachi.png", required=False)
    parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
    parser.add_argument('--max_size', type=int, default=1120, required=False)
    parser.add_argument('--category', '-c', type=int, default=1, required=False)
    parser.add_argument('--dtype', type=str, default="bfloat16", required=False)
    parser.add_argument('--step', type=int, default=10, required=False)
    parser.add_argument('--sample', type=int, default=1, required=False)
    parser.add_argument('--seed', type=int, default=1234, required=False)
    args = parser.parse_args()

    flux_fill_model_path = args.flux_fill_model_path
    flux_redux_model_path = args.flux_redux_model_path
    background_remove_model_path = args.background_remove_model_path
    openpose_model_path = args.openpose_model_path
    humanparsing_model_path = args.humanparsing_model_path

    remove_background = args.remove_background
    cloth_path = args.cloth_path
    model_path = args.model_path
    model_mask_path = args.model_mask_path
    max_size = args.max_size
    
    category = args.category
    gpu_id = args.gpu_id
    dtype = args.dtype
    n_steps = args.step
    n_samples = args.sample
    seed = args.seed

    # load models
    try_on_pipeline = TryOnFluxPipe(flux_fill_model_path, flux_redux_model_path, background_remove_model_path, openpose_model_path, humanparsing_model_path,gpu_id = gpu_id, remove_background = remove_background, dtype = dtype)

    model_img = Image.open(model_path).convert('RGB')
    model_mask_img = None
    if model_mask_path:
        model_mask_img = Image.open(model_mask_path)
    cloth_img = Image.open(cloth_path).convert('RGB')

    kwargs = {
    'model_img': model_img,
    'model_mask_img': model_mask_img,
    'cloth_img': cloth_img,
    'category': category,
    'n_samples': n_samples,
    'n_steps': n_steps,
    'seed': seed,
    'remove_background': remove_background,
    'max_size': max_size,
    }
    # inference
    try_on_pipeline.run(**kwargs)
