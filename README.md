# try-on-flux
The simplest tryon pipeline!!!
* The pipeline is based on flux.1-fill-dev and flux.1-redux-dev. No additional training process required.
* To run this pipeline, you'd better prepare a graphics card with more than 32g of video memory, such as v100-32g, a100-40g, a100-80g.

![demo](assets/demo.png)&nbsp;

## Installation
1. Clone the repository

```sh
git clone xxx
```

2. Create a conda environment and install the required packages

```sh
conda create -n tryonflux python==3.10
conda activate tryonflux
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
pip install -r requirements.txt

```

## Download
1. Download the flux fill model from [FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
2. Download the flux redux model from [FLUX.1-Redux-dev](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev)
* If you want to quick start, then you don't need to download the following 3 models
3. Download the Background Removal model from [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
4. Download the openpose model from [openpose](https://huggingface.co/levihsu/OOTDiffusion/tree/main/checkpoints/openpose)
5. Download the humanparsing model from [humanparsing](https://huggingface.co/levihsu/OOTDiffusion/tree/main/checkpoints/humanparsing)

The checkpoint structure should be like:
```
|-- models
    |-- FLUX.1-Fill-dev
        |-- model_index.json
        |-- scheduler
        |-- text_encoder
        |-- ...
        |-- vae
    |-- FLUX.1-Redux-dev
        |-- feature_extractor
        |-- image_embedder
        |-- ...
        |-- model_index.json
    |-- briaai--RMBG-2.0
        |-- BiRefNet_config.py
        |-- ...
        |-- pytorch_model.bin
        |-- README.md
    |-- humanparsing
        |-- exp-schp-201908261155-lip.pth
        |-- exp-schp-201908301523-atr.pth
    |-- openpose
        |-- ckpts
            |-- body_pose_model.pth
      
```

## Start

### Quick Start
If you want to quick start, please run the following code. We have prepared all materials.
```sh
python tryon_flux_pipe.py \
    --flux_fill_model_path "./models/FLUX.1-Fill-dev" \
    --flux_redux_model_path "./models/FLUX.1-Redux-dev" \
    --model_path "./images/models/liuyifei.jpeg" \
    --model_mask_path "images/models/liuyifei_c2_mask.png" \
    --cloth_path "./images/cloth/itachi.png" \
    --category 2 \
    --step 10 
```
You can also run the demo using the script which contains more demos:
```sh
sh run_quick.sh
```
### Start
If you want to change the models, you can run the following code and change the model image(model_path) or cloth image(cloth_path), or both.
```sh
python tryon_flux_pipe.py \
    --flux_fill_model_path "./models/FLUX.1-Fill-dev" \
    --flux_redux_model_path "./models/FLUX.1-Redux-dev" \
    --background_remove_model_path "./models/briaai--RMBG-2.0" \
    --openpose_model_path "./models/openpose/ckpts" \
    --humanparsing_model_path "./models/humanparsing" \
    --model_path "./images/models/liuyifei.jpeg" \
    --cloth_path "./images/cloth/itachi.png" \
    --category 2 \
    --step 10 
```
You can also run the demo using the script  which contains more demos:
```sh
sh run.sh
```
## License

This repository uses [FLUX](https://github.com/black-forest-labs/flux) as the base model. Users must comply with FLUX's license when using this code. Please refer to [FLUX's License](https://github.com/black-forest-labs/flux/tree/main/model_licenses) for more details.
