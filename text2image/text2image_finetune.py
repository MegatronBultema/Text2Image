"""
Project Objectives
1. Convert a model into the right format
2. Finetune a model that looks good to your eyes
3. Auto prompt engineer to get good results

This script (text2image_finetune.py) contains functions for
1. Convert a model into the right format
2. Finetune a model that looks good to your eyes

The script text2image_gpt-assist.py contains functions for
3. Auto prompt engineer to get good results

"""

import subprocess
import os
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from huggingface_hub import notebook_login
import pandas as pd
import argparse

"""
1. Convert a model into the right format
"""
def convert_safetensor_to_diffusion(local_path_chkpt: str, local_save_folder: str, script_path: str = '/Users/megan.bultema/Documents/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py'):
    """
    Converts a Safetensors checkpoint to a Diffusion checkpoint using a provided script.

    Parameters:
    local_path_chkpt (str): The local path to the Safetensors checkpoint file.
    local_save_folder (str): The local folder path to save the new Diffusion checkpoint file.
    script_path (str): The full path to the script to be used for the conversion. Defaults to the provided script path.

    Returns:
    None

    Raises:
    FileNotFoundError: If the local_path_chkpt file is not found.

    """

    # Get the current working directory
    current_directory = os.getcwd()

    # Construct the full path to the Safetensors checkpoint file
    checkpoint_path = os.path.join(current_directory, local_path_chkpt)

    # Check if the Safetensors checkpoint file exists
    if os.path.exists(checkpoint_path):
        pass
    else:
        raise FileNotFoundError("The local path to the safetensors file was not found. Check and try again.")

    # Construct the full path to the save folder
    dump_path = os.path.join(current_directory, local_save_folder)

    # Check if the save folder exists, if not create it
    if os.path.exists(dump_path):
        pass
    else:
        os.mkdir(dump_path)

    # Construct the command string for running the conversion script
    command = f"python {script_path} --checkpoint_path='{checkpoint_path}' --dump_path='{dump_path}' --from_safetensors"

    # Run the conversion script as a subprocess
    subprocess.run(command, shell=True)

    # Print a message to indicate the conversion is complete
    print('Conversion complete.')

"""
2. Finetune a model that looks good to your eyes
"""

def retrieve_class_images(class_prompt, class_data_dir, num_class_images,
                          retrieve_script='/Users/megan.bultema/Documents/diffusers/examples/custom_diffusion/retrieve.py'):
    """
    Retrieve a specified number of images for a given class prompt using a retrieval script.

    Parameters:
    - class_prompt (str): The prompt to use for image retrieval.
    - class_data_dir (str): The directory where the retrieved images will be saved.
    - num_class_images (int): The number of images to retrieve.
    - retrieve_script (str): The full path to the retrieval script. Default is set to a specific path.

    Returns:
    - None

    Example usage:
    >>> retrieve_class_images('dog', 'data/dogs', 100)

    """
    # Get the current working directory
    current_directory = os.getcwd()
    # Construct the full path to the save folder
    save_data_dir = os.path.join(current_directory, class_data_dir)

    # Check if the save folder exists, if not create it
    if os.path.exists(save_data_dir):
        pass
    else:
        os.mkdir(save_data_dir)

    # Construct the command to run the retrieval script
    cmd = f"python {retrieve_script} --class_prompt {class_prompt} --class_data_dir {save_data_dir} --num_class_images {num_class_images}"
    # Run the command in the shell
    subprocess.run(cmd, shell=True)


def create_metadata_csv_and_push_to_hub(root: str, HF_user='megantron'):
    """
    Create a metadata.csv file and push it to the huggingface hub for a given image folder and text file.
    This function creates a DataFrame with the image and text data from the text file and writes it to
    metadata.csv. It also pushes the dataset to the Hugging Face hub.

    Parameters:
    -----------
    root : str
        Absolute path to the folder containing download of images using retrieve_class_images()

    """

    img_folder = root + '/images/'
    img_txtfile = root + '/images.txt'
    txt_file = root + '/caption.txt'
    # create a list of image file names from the folder
    img_files = os.listdir(img_folder)

    # create an empty list to store the image data
    img_data = [x for x in img_files]

    # read the text file and split the contents into a list
    with open(img_txtfile, 'r') as f:
        img_data = f.read().splitlines()

    # remove abs path to comply with hugging face upload
    img_file = [x.replace(f'{img_folder}', '') for x in img_data]

    # read the text file and split the contents into a list
    with open(txt_file, 'r') as f:
        text_data = f.read().splitlines()

    # create a dictionary with the image and text data
    data_dict = {'file_name': img_file, 'text': text_data}

    # create a DataFrame from the dictionary
    df = pd.DataFrame(data_dict)

    df.to_csv(f'{img_folder}/metadata.csv', index=False)
    dataset = load_dataset("imagefolder", data_dir=img_folder)
    notebook_login()
    dataset.push_to_hub(f"{HF_user}/{root.split('/')[-1]}")


def train_text_to_image_lora(MODEL_NAME:str, DATASET_NAME:str, OUTPUT_DIR:str, location_text_to_image:str = '/Users/megan.bultema/Documents/diffusers/examples/text_to_image/train_text_to_image_lora.py'):
    """
    Trains a text-to-image generation model using the specified parameters.

    Args:
        MODEL_NAME (str): The name or path of the pre-trained model to use.
        DATASET_NAME (str): The name or path of the dataset to use.
        OUTPUT_DIR (str): The output directory to save the trained model to.

    Returns:
        None
    """

    # Construct the command to run the retrieval script
    cmd = f"""accelerate launch {location_text_to_image} \
        --pretrained_model_name_or_path={MODEL_NAME} \
        --dataset_name={DATASET_NAME} \
         --caption_column='text' \
        --resolution=512 --random_flip \
        --train_batch_size=1 \
        --num_train_epochs=10 --checkpointing_steps=500 \
        --learning_rate=1e-04 --lr_scheduler='constant' --lr_warmup_steps=0 \
        --seed=42 \
        --output_dir={OUTPUT_DIR}"""
    # print(cmd)
    # Run the command in the shell
    subprocess.run(cmd, shell=True)
    print('Fine-tuning complete')


def generate_test_images(model_path: str, num_images, prompt, n_prompt) -> None:
    """
    Generate 5 test images using StableDiffusionPipeline and save them to a local directory named test_images.

    Args:
    - model_path (str): The path to the pre-trained model directory.
    num_images (int): Number of test images to generate

    Returns:
    - None
    """
    # check if test_images directory exists, if it does, remove it and create a new one
    if os.path.exists("test_images"):
        pass
    else:
        os.mkdir("test_images")

    # create StableDiffusionPipeline object and load attention processes
    pipe = StableDiffusionPipeline.from_pretrained("./converted_model_deliberate")
    pipe.unet.load_attn_procs(model_path)
    pipe.to("mps")

    # generate 5 test images and save them to test_images directory
    for i in range(num_images):

        image = pipe(prompt, negative_prompt=n_prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        save_header = model_path.split('/')[-1]
        image.save(f"test_images/{save_header}_test{i}.png")





if __name__ == "__main__":

    # Define a parser for the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("function", choices=["convert_safetensor","retrieve", "format_dataset", "train", "test"], help="The function to run")
    parser.add_argument("--class-prompt", help="The prompt to use for image retrieval. Specific to retrieve_class_images function.")
    parser.add_argument("--class-data-dir", help="The directory where the retrieved images will be saved. Specific to retrieve_class_images function.")
    parser.add_argument("--num-class-images", type=int, help="The number of images to retrieve. Specific to retrieve_class_images function.")
    parser.add_argument("--model-name", help="The name or path of the pre-trained model to use. Specific to train_text_to_image_lora function.")
    parser.add_argument("--dataset-name", help="The name or path of the dataset to use. Specific to train_text_to_image_lora function.")
    parser.add_argument("--output-dir", help="The output directory to save the trained model to. Specific to train_text_to_image_lora function.")
    parser.add_argument('root', type=str, help='Absolute path to the folder containing downloaded images. Specific to create_metadata_csv_and_push_to_hub function.')
    parser.add_argument('--hf_user', type=str, default='megantron', help='Hugging Face username (default: megantron). Specific to create_metadata_csv_and_push_to_hub function.')
    parser.add_argument('--local_path_chkpt', type=str, required=True,
                        help='The local path to the Safetensors checkpoint file. Specific to convert_safetensor_to_diffusion function.')
    parser.add_argument('--local_save_folder', type=str, required=True,
                        help='The local folder path to save the new Diffusion checkpoint file. Specific to convert_safetensor_to_diffusion function.')
    parser.add_argument('--script_path', type=str,
                        default='/Users/megan.bultema/Documents/diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py',
                        help='The full path to the script to be used for the conversion. Defaults to the provided script path. Specific to convert_safetensor_to_diffusion function.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model directory. Specific to generate_test_images function.')
    parser.add_argument('--num_images', type=int, required=True, help='Number of test images to generate. Specific to generate_test_images function.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to generate the test images. Specific to generate_test_images function.')
    parser.add_argument('--n_prompt', type=str, required=True, help='Negative prompt to generate the test images. Specific to generate_test_images function.')

    args = parser.parse_args()

    if args.function == "convert_safetensor":
        convert_safetensor_to_diffusion(args.local_path_chkpt, args.local_save_folder, args.script_path)
    elif args.function == "retrieve":
        retrieve_class_images(args.class_prompt, args.class_data_dir, args.num_class_images)
    elif args.function == "format_dataset":
        create_metadata_csv_and_push_to_hub(args.root, args.hf_user)
    elif args.function == "train":
        train_text_to_image_lora(args.model_name, args.dataset_name, args.output_dir)
    elif args.function == "test":
        generate_test_images(args.model_path, args.num_images, args.prompt, args.n_prompt)
    else:
        print("Invalid function name. Choose 'convert_safetensor', 'retrieve', 'format_dataset', 'train', or 'test'.")



