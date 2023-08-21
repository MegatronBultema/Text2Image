# Visual Electric Project
In this project there’s going to be three goals:

1. Convert a model into the right format
2. Finetune a model that looks good to your eyes
3. Auto prompt engineer to get good results

## Step One:
Convert a model in a .safetensors format, like the one below:

[Diffusion-Brush (Everything) - SFW / NSFW- All Purpose Checkpoint -  Nuclear Diffusion  - Anime Hybrid  - v1.0 | Stable Diffusion  Checkpoint  | Civitai](https://civitai.com/models/46294/diffusion-brush-everything-sfw-nsfw-all-purpose-checkpoint-nuclear-diffusion)

But of course feel free to find one out there and use it instead. There are a bunch of great sites out there with models, two of the most popular are:

1. HuggingFace
2. CivitAI

We’ll be doing everything with [diffusers](https://github.com/huggingface/diffusers),  so convert the model into their format.

Convert the model in a python notebook. Use any library or code out there you find. It’s all good.

## Step Two:
Fine-tune a model yourself, in whatever style you’d like! Use the [LoRA style](https://huggingface.co/blog/lora) of fine-tuning.

Some example styles might be:
1. [Art Nouveau](https://en.wikipedia.org/wiki/Art_Nouveau)
2. [The Simpsons](https://en.wikipedia.org/wiki/The_Simpsons)
3. [Ansel Adams](https://en.wikipedia.org/wiki/Ansel_Adams)

But feel free to get as creative as you want here. Honestly it’s one of the most fun parts!

Do all the training in a python notebook.

## Step Three:
Now it’s time to combine the two models. Using [GPT](https://platform.openai.com/docs/models/gpt-3-5) (let me know if you need some account keys) to help you create good prompts. It’s actually really good at this.

In a notebook, show a “user” (you! haha) trying in a simple prompt, something like “a beautiful photograph of the mountains“ and getting a bunch of great pictures from both models.
