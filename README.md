# Text2Image

Contained in the following directory is the completion of a take-home focused on fine-tuning a text-to-image Stable Diffusion model. 
Stable Diffusion uses a kind of diffusion model (DM), called a latent diffusion model (LDM) developed by the CompVis group at LMU Munich.[1][2] Introduced in 2015, diffusion models are trained with the objective of removing successive applications of Gaussian noise on training images, which can be thought of as a sequence of denoising autoencoders. 

For this project the stable diffusion model is fine-tuned using LoRA (Low-Rank Adaptation of Large Language Models). Even though LoRA was initially proposed for large-language models and demonstrated on transformer blocks, the technique can also be applied elsewhere. In the case of Stable Diffusion fine-tuning, LoRA can be applied to the cross-attention layers that relate the image representations with the prompts that describe them. [3] 

Once the fine-tuned model is available images are generated using prompts created by an API call to ChatGPT. In the ChatGPT prompts used here we have included examples of positive and negative prompts pulled from the original models website examples. This was found to improve the image quality resulting from the ChatGPT prompts. ChatGPT was also called to create negative prompts, given examples, and an image title used to save the images and prompts. 

Not included in this repo are the model files (over 2GB) and scripts referenced from the [Huggingface Diffusers repo]. (https://github.com/huggingface/diffusers/tree/main). 
Workflow:
    Origninal model (titled: deliberate) was downloaded from [CivitAI](https://civitai.com/models/4823/deliberate)
    Model was converted from .safetensors to diffusers format using ConvertModel.ipynb
    Download image examples and fine-tune the model with the LoRA style of fine-tuning using functions in TrainingWithLoRA.ipynb
    Use the original and fine-tuned model to generate images and use the ChatGPT api to embelish simple prompts using functions in GenerateImageChatGPT.ipynb






1) Rombach; Blattmann; Lorenz; Esser; Ommer (June 2022). High-Resolution Image Synthesis with Latent Diffusion Models (PDF). International Conference on Computer Vision and Pattern Recognition (CVPR). New Orleans, LA. pp. 10684–10695. arXiv:2112.10752
2) "Stable Diffusion Repository on GitHub". CompVis - Machine Vision and Learning Research Group, LMU Munich. 17 September 2022. Retrieved 17 September 2022.
3) https://huggingface.co/blog/lora
