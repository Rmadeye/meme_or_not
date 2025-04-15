![Python Version](https://img.shields.io/badge/Python-3.10-blue)![PyTorch Version](https://img.shields.io/badge/PyTorch-2.5.1-red)![CUDA Version](https://img.shields.io/badge/CUDA-12.4-green)
# mem_or_not

`mem_or_not` is a project designed to help users determine whether a given image is a mem or not.

# Usage

1. Create virtual environment either using `venv` or `conda`. For example:
`conda create -n "mem_or_not" python=3.10`

2. Activate your environment:
`conda activate mem_or_not`

3. Install all requirements:
`pip install .`

4. Prepare your images.
`python -m scripts.prepare_images --image_dir <your_dir> --output_dir <dir_of_your_choice>`

5. Download my ResNet50-based model and save it to models directory: https://drive.google.com/file/d/1UZxUk2Lb3UG-2Ca7Pn6yjPjmyliT_68j/view?usp=drive_link

Prediction whether it is meme or not is done using `predict.py` script.

`python predict.py --image_dir <dir_of_your_choice> --model_path models/meme_classifier.ckpt`

At this point, information about each image - whether it is a meme - along with the confidence, is shown in your terminal.
