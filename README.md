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
`python -m scripts.prepare_images --raw_image_dir <your_dir> --output_dir <dir_of_your_choice>
5. Download my ResNet50-based model and save it to models directory: https://drive.google.com/file/d/1UZxUk2Lb3UG-2Ca7Pn6yjPjmyliT_68j/view?usp=drive_link

Prediction whether it is meme or not is done using `predict.py` script.

`python predict.py --image_dir <dir_of_your_choice> --model_path models/meme_classifier.ckpt`

At this point an  information about each image - whether it is meme and how probable is that - is shown in your terminal.
