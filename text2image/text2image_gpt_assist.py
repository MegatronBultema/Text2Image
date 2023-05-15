import time
import os
import requests
from diffusers import StableDiffusionPipeline
import argparse


def save_file_to_folder(filename: str, folder_path: str, data: any, image: bool = False):
    """
    Saves a file with the given filename and data to the specified folder path.
    If the folder does not exist, it will be created.

    Parameters:
    filename (str): The name of the file to be saved.
    folder_path (str): The path to the folder where the file will be saved.
    data (any): The data to be written to the file.
    image (bool, optional): A flag indicating whether the data is an image.
                            Defaults to False.
    """

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Construct the full path to the file
    filepath = os.path.join(folder_path, filename)

    # If the data is an image, save the image
    if image:
        data.save(filepath)
    else:
        # Write the data to the file
        with open(filepath, 'w') as f:
            f.write(data)


def generate_imageprompt(prompt: str, max_token: int, with_examples: bool = False) -> str:
    """
    Generates an image prompt by calling the OpenAI API with the provided prompt text
    and max_token value. The function includes additional directions for the user to
    follow in generating the prompt, and an optional flag with_examples to provide sample
    prompts as a reference. The function returns the generated prompt text as a string.

    :param prompt: A string representing the image prompt.
    :param max_token: An integer representing the maximum number of tokens allowed in the generated prompt.
    :param with_examples: A boolean flag to include or exclude examples of prompts.
    :return: A string representing the generated prompt.
    """
    # Define prompt directions based on the with_examples flag
    if with_examples:
        prompt_directions = f"""Provide a prompt in {max_token} words for a text-to-image diffusion
        transformer that contains and image of  "{prompt}" in a photo-realistic image.
        Here are examples of prompts with features added to the image description. Include image features like these examples
        but modify the subject to the description above. Keep the subject simple but expand list details of image style 
        and quality.
        Only respond with the description itself.

        'a cute kitten made out of metal, (cyborg:1.1), ([tail | detailed wire]:1.3), 
        (intricate details), hdr, (intricate details, hyperdetailed:1.2), cinematic shot, vignette, centered'

        'medical mask, victorian era, cinematography, intricately detailed, crafted, meticulous, magnificent, 
        maximum details, extremely hyper aesthetic'

        'A stunning intricate full color portrait of (sks person:1), wearing a black turtleneck, epic character composition, by ilya kuvshinov, 
        alessio albi, nina masic, sharp focus, natural lighting, subsurface scattering, f2, 35mm, film grain,'

        'A stunning intricate full color portrait of (35 years old sks person:1) as (viking warrior), (barbarian), epic character composition, by ilya kuvshinov, 
        alessio albi, nina masic, sharp focus, natural lighting, subsurface scattering, f2, 35mm, film grain,'

        '(Cinematic photo: 1. 3), directional look, octane render, ultra detailed, wide angle full body, 8k, ultra-detailed, (backlight:1. 2) intricate, style-empire'

        'ultra High Detail. High definition, Canon EOS 5D Mark IV, full-frame, 50mm focal length, ISO 100, aperture f/2.8,, high detail, HDR, postprocessed, award winning, 4K 8K photography, high resolution, depth of field, magnificent, elegant, beautiful, fantastical'

        'photo-realistic, fine detail in face, skin texture, portrait'
        """
    else:
        prompt_directions = f"""Provide a prompt in {max_token} words for a text-to-image diffusion
        transformer that contains and image of  "{prompt}".
        Only respond with the description itself."""

    # Read the OpenAI API token from a file
    with open("/Users/megan.bultema/Documents/chatgpttoken.txt", "r") as file:
        OPENAI_API_TOKEN = file.read().strip()

    headers = {"Authorization": f"Bearer {OPENAI_API_TOKEN}"}
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt_directions}],
        "max_tokens": max_token + 20,
        "temperature": 1.0,
    }

    # Send a POST request to the OpenAI API
    res = requests.post(url, json=payload, headers=headers)

    # If too many requests are made, wait for 2 minutes and try again
    while res.status_code == 429:
        print("Too many requests, waiting 3 minutes...")
        time.sleep(180)
        res = requests.post(url, json=payload, headers=headers)

    # Extract the generated prompt from the API response
    result = res.json()["choices"][0]["message"]["content"].strip()

    return result


def generate_negativeprompt() -> str:
    """
    Generates a negative prompt for a text-to-image diffusion using the OpenAI API.
    The negative prompt helps prevent image disfiguration and Not Safe For Work images.

    Returns:
    str: the generated negative prompt text as a string.
    """

    prompt_directions = f"""Provide a negative prompt in 70 words for a text-to-image diffusion
    that prevents distortion or Not Safe For Work images
    Here are examples of negative prompts. The negative prompt should be in list format.
    Only respond with the description itself.

    'distorted face, ((disfigured)), ((bad art)), ((deformed)),
    ((poorly drawn)), ((extra limbs)), ((close up)), ((b&w)), weird colors, blurry'


    'distorted face, (((duplicate))), ((mole)), ((blemish)), ((morbid)), ((wrinkles)), ((mutilated)), [out of frame], 
    extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), 
    (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)),
     cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), 
     gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), 
     (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck)))'

    'distorted face, (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, 
    missing limb, floating limbs, (mutated hands and fingers:1.4), 
    disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, flowers'

    'distorted face, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, 
    disgusting, poorly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, 
    ((((mutated hands and fingers)))), watermark, watermarked, oversaturated, distorted hands,
    amputation, missing hands, obese, doubled face, double hands, b&w, black and white, sepia, flowers, roses'

    'distorted face'

    'distorted face, nrealfixer, nfixer, nartfixer, illustration, drawing, 3d, b&w, 
    (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, 
    floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation'

    'distorted face, realisticvision-negative-embedding, (low quality, worst quality:1.4), canvas frame, 3d, ((disfigured)), 
    ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), 
    ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), 
    (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, 
    (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), 
    ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), 
    (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, 
    poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, 
    cross-eye, body out of frame, blurry, 
    bad art, bad anatomy,3d render, canvas frame, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),
    ((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], 
    extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))),
     ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), 
     out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), 
     ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), 
     (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, 
     poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, 
     deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d render'

    """

    with open("/Users/megan.bultema/Documents/chatgpttoken.txt", "r") as file:
        OPENAI_API_TOKEN = file.read()
        OPENAI_API_TOKEN = OPENAI_API_TOKEN.strip()
    headers = {"Authorization": f"Bearer {OPENAI_API_TOKEN}"}
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt_directions}],
        "max_tokens": 150,
        "temperature": 1.0,
    }

    res = requests.post(url, json=payload, headers=headers)
    #     print(res)
    #     print(res.status_code)
    while True:
        if res.status_code == 429:
            print("Too many requests, waiting 3 minutes...")
            time.sleep(180)
        else:
            result = res.json()["choices"][0]["message"]["content"]
            return result


def generate_image_title(prompt, max_token):
    prompt_directions = f"""Provide a simple title for an image that contains "{prompt}".  
    Only respond with the title itself. 
    The title should be 2 words in length"""

    with open("/Users/megan.bultema/Documents/chatgpttoken.txt", "r") as file:
        OPENAI_API_TOKEN = file.read()
        OPENAI_API_TOKEN = OPENAI_API_TOKEN.strip()
    headers = {"Authorization": f"Bearer {OPENAI_API_TOKEN}"}
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt_directions}],
        "max_tokens": max_token + 20,
        "temperature": 1.0,
    }
    res = requests.post(url, json=payload, headers=headers)
    #     print(res)
    #     print(res.status_code)
    while True:
        if res.status_code == 429:
            print("Too many requests, waiting 3 minutes...")
            time.sleep(180)
        else:
            result = res.json()["choices"][0]["message"]["content"]
            result = result.strip().lower().replace(" ", "_").replace('.', '').replace('"', '')
            return result


def image_prompt(lora_weight_path, prompt, neg_prompt, save_title, num_image, max_retries=3):
    """
    Generate `num_image` images using a StableDiffusionPipeline model.
    For each generated image, the original model and a fine-tuned version of the model are used.

    Args:
        prompt (str): The prompt to use for image generation.
        neg_prompt (str): The negative prompt to use for image generation.
        save_title (str): The title of the folder where the generated images will be saved.
        num_image (int): The number of images to generate.

    Returns:
        Tuple: A tuple containing the last generated image using the original model and the last generated
            image using the fine-tuned model.
    """
    # initiate fine-tuned model

    pipe_ft = StableDiffusionPipeline.from_pretrained("./converted_model_deliberate")
    pipe_ft.unet.load_attn_procs(lora_weight_path)
    pipe_ft.to("mps")

    # initiate original pretrained model
    pipe = StableDiffusionPipeline.from_pretrained("./converted_model_deliberate")
    pipe.to("mps")

    # determine/create folder for artifacts
    folder_path = 'test_images/' + save_title

    # check if folder exists
    if os.path.exists(folder_path):
        i = 1
        while os.path.exists(f"{folder_path}_{i}"):
            i += 1
        # update folder name to iteration with '_#' at the end
        folder_name = f"{folder_path}_{i}"
    else:
        folder_name = folder_path
    print(folder_name)

    # create folder with updated name
    os.makedirs(folder_name)

    # save prompts for reference
    save_file_to_folder("prompt.txt", folder_name, prompt)
    save_file_to_folder("negative_prompt.txt", folder_name, neg_prompt)

    last_image = None  # initialize last generated image using original model
    last_image_ft = None  # initialize last generated image using fine-tuned model

    for i in range(num_image):
        retry_count = 0
        # generate image using original model
        image = pipe(prompt, negative_prompt=neg_prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
        # Convert the image to grayscale and check if all pixels are black
        is_black = image.convert('L').getextrema() == (0, 0)

        if is_black:
            print("The image is completely black.")
            while retry_count < max_retries and is_black == True:
                print(f"NSFWException caught. Retrying...")
                image = pipe(prompt, negative_prompt=neg_prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                is_black = image.convert('L').getextrema() == (0, 0)
                retry_count += 1

            # if image is NSFW and there are no more retries left, skip image and break out of retry loop
            else:
                print(f"Image {i} is NSFW and could not be generated after {max_retries} retries.")

        save_file_to_folder(f"{save_title}_{i}.png", folder_name, image, True)
        last_image = image

    for i in range(num_image):
        retry_count = 0
        # generate image using fine-tuned model
        image_ft = pipe_ft(prompt, negative_prompt=neg_prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
        # Convert the image to grayscale and check if all pixels are black
        is_black = image_ft.convert('L').getextrema() == (0, 0)

        if is_black:
            print("The image is completely black.")
            while retry_count < max_retries and is_black == True:
                print(f"NSFWException caught. Retrying...")
                image_ft = \
                pipe_ft(prompt, negative_prompt=neg_prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
                is_black = image_ft.convert('L').getextrema() == (0, 0)
                retry_count += 1

            # if image is NSFW and there are no more retries left, skip image and break out of retry loop
            else:
                print(f"Image {i} is NSFW and could not be generated after {max_retries} retries.")

        save_file_to_folder(f"{save_title}_{i}_fine-tuned.png", folder_name, image_ft, True)
        last_image_ft = image_ft

    return last_image, last_image_ft


def generate_images(lora_weight_path, user_input, num_images=1, with_examples=False):
    """
    Generate images using the Stable Diffusion model, given a user input string.

    Args:
    - user_input (str): User input string used to generate image prompts
    - num_images (int): Number of images to generate (default=1)
    - with_examples (bool): Flag indicating whether or not to include example images in the prompt (default=False)

    Returns:
    - image (PIL.Image): Generated image
    - image_finetuned (PIL.Image): Generated image after fine-tuning
    """
    # Generate GPT prompt, title and negative prompts for image generation
    gpt_title = generate_image_title(user_input, 50)
    gpt_prompt = generate_imageprompt(user_input, 50, with_examples)
    gpt_negprompt = generate_negativeprompt()

    # Generate images using Stable Diffusion model

    image, image_finetuned = image_prompt(lora_weight_path, gpt_prompt, gpt_negprompt, gpt_title, num_images)

    return image, image_finetuned


def generate_images_origprompt(lora_weight_path, user_input, title, num_images=1):
    """
    Generate images using the Stable Diffusion model, given a user input string.

    Args:
    - user_input (str): User input string used to generate image prompts
    - num_images (int): Number of images to generate (default=1)
    - with_examples (bool): Flag indicating whether or not to include example images in the prompt (default=False)

    Returns:
    - image (PIL.Image): Generated image
    - image_finetuned (PIL.Image): Generated image after fine-tuning
    """

    negprompt = """distorted face, realisticvision-negative-embedding, (low quality, worst quality:1.4), canvas frame, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), weird colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), 
    Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, 
    bad art, bad anatomy,3d render, canvas frame, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),
    ((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], 
    extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), 
    ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), 
    out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), 
    ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), 
    (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, 
    out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, 
    body out of frame, blurry, bad art, bad anatomy, 3d render"""

    # Generate images using Stable Diffusion model

    image, image_finetuned = image_prompt(lora_weight_path, user_input, negprompt, title, num_images)

    return image, image_finetuned



if __name__ == "__main__":
    #example
    #python script.py ./deliberate_AnselAdams "Ansel Adams photograph of Mars" --num_images 5

    parser = argparse.ArgumentParser(description='Generate images using the Stable Diffusion model.')
    parser.add_argument('lora_weight_path', type=str, help='path to Lora weight file')
    parser.add_argument('user_input', type=str, help='user input string used to generate image prompts')
    parser.add_argument('--num_images', type=int, default=1, help='number of images to generate (default=1)')
    parser.add_argument('--with_examples', action='store_true', help='flag indicating whether or not to include example images in the prompt (default=False)')
    args = parser.parse_args()

    image_base, image_ft = generate_images(args.lora_weight_path, args.user_input, args.num_images, args.with_examples)


