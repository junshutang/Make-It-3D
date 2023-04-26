from transformers import CLIPTextModel, CLIPTokenizer, logging, CLIPVisionModel, CLIPFeatureExtractor
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.transforms as T
import time
import os
import clip
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class StableDiffusion(nn.Module):
    def __init__(self, device, sd_version='2.0', hf_key=None, step_range=[0.2, 0.6]):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')
        
        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.text_clip_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        
        self.processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
        
        self.aug = T.Compose([
            T.Resize((224, 224)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)
        
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        # self.scheduler = PNDMScheduler.from_pretrained(model_key, subfolder="scheduler")

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.num_inference_steps = 50
        self.min_step = int(self.num_train_timesteps * float(step_range[0]))
        self.max_step = int(self.num_train_timesteps * float(step_range[1]))
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.ref_imgs = None

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def get_img_embeds(self, prompt_img):
        # Tokenize text and get embeddings
        prompt_img = prompt_img.squeeze(0)
        img_input = self.processor(images=prompt_img.detach().cpu().numpy(), return_tensors='pt')

        with torch.no_grad():
            img_embeddings = self.image_encoder(img_input.pixel_values.to(self.device))[0]

        return img_embeddings

    def img_clip_loss(self, clip_model, rgb1, rgb2):
        image_z_1 = clip_model.encode_image(self.aug(rgb1))
        image_z_2 = clip_model.encode_image(self.aug(rgb2))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features
        image_z_2 = image_z_2 / image_z_2.norm(dim=-1, keepdim=True) # normalize features

        loss = - (image_z_1 * image_z_2).sum(-1).mean()
        return loss
    
    def img_text_clip_loss(self, clip_model, rgb, prompt):
        image_z_1 = clip_model.encode_image(self.aug(rgb))
        image_z_1 = image_z_1 / image_z_1.norm(dim=-1, keepdim=True) # normalize features

        text = clip.tokenize(prompt).to(self.device)
        text_z = clip_model.encode_text(text)
        text_z = text_z / text_z.norm(dim=-1, keepdim=True)
        loss = - (image_z_1 * text_z).sum(-1).mean()
        return loss


    def train_step(self, text_embeddings, pred_rgb, ref_rgb=None, noise=None, islarge=False, ref_text=None, clip_model=None, guidance_scale=10):
        
        # interp to 512x512 to be fed into vae.
        loss = 0
        imgs = None

        # _t = time.time()
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        w_ = 1.0

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        latents = self.encode_imgs(pred_rgb_512)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            latent_model_input = latent_model_input.detach().requires_grad_()
            
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        if not islarge and (t / self.num_train_timesteps) <= 0.4:
            self.scheduler.set_timesteps(self.num_train_timesteps)
            de_latents = self.scheduler.step(noise_pred, t, latents_noisy)['prev_sample']
            # de_latents = de_latents.detach().requires_grad_()
            imgs = self.decode_latents(de_latents)
            loss = 10 * self.img_clip_loss(clip_model, imgs, ref_rgb) + \
                    10 * self.img_text_clip_loss(clip_model, imgs, ref_text)
            
            # grad = torch.autograd.grad(loss_clip, de_latents, retain_graph=True)[0]
            # print(f"loss clip: {loss}")
        else:
            # w(t), sigma_t^2
            w = (1 - self.alphas[t])
            grad = w * (noise_pred - noise) * w_
            imgs = None

            # clip grad for stable training?
            grad = torch.nan_to_num(grad)
            latents.backward(gradient=grad, retain_graph=True)
            loss = 0
        
        return loss, imgs # dummy loss value

    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        # with torch.no_grad():
        imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--workspace', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seeds', type=int, default=0)
    # parser.add_argument('--seeds', nargs='+', default=[0, 1, 2])
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    opt.workspace = os.path.join('test_bench', opt.workspace)
    if opt.workspace is not None:
        os.makedirs(opt.workspace, exist_ok=True) 
    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version)

    for seed in range(opt.seeds):
        seed_everything(seed)
        imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps, guidance_scale=7.5)
        save_image(imgs, os.path.join(opt.workspace, opt.prompt.replace(" ", "_") + f'_{seed}.png'))
        



