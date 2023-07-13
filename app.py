import os

os.system('cd fairseq;'
          'pip install ./; cd ..')
os.system('ls -l')

import torch
import numpy as np
import gradio as gr
import cv2
from PIL import Image
from torchvision import transforms

from fairseq import utils, tasks, options
from fairseq import checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

from tasks.mm_tasks.caption import CaptionTask
from tasks.mm_tasks.refcoco import RefcocoTask
from tasks.mm_tasks.vqa_gen import VqaGenTask


def move2gpu(models, cfg):
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)


def construct_transform(patch_image_size):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return patch_resize_transform


# Register tasks
tasks.register_task('caption', CaptionTask)
tasks.register_task('refcoco', RefcocoTask)
tasks.register_task('vqa_gen', VqaGenTask)
# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = False

# download checkpoints
os.system('mkdir -p checkpoints; ')

os.system('wget https://data.isir.upmc.fr/unival/models/unival_s2_hs/checkpoint1.pt; '
          'mkdir -p checkpoints/unival_s2_hs; mv checkpoint1.pt checkpoints/unival_s2_hs/')

os.system('wget https://data.isir.upmc.fr/unival/models/unival_vqa/checkpoint_best.pt; '
          'mkdir -p checkpoints/unival_vqa; mv checkpoint_best.pt checkpoints/unival_vqa/')
os.system('wget https://data.isir.upmc.fr/unival/models/unival_caption_stage_1/checkpoint_best_test.pt; '
          'mkdir -p checkpoints/unival_caption_stage_1; mv checkpoint_best_test.pt checkpoints/unival_caption_stage_1/')
os.system('wget https://data.isir.upmc.fr/unival/models/unival_refcocog/checkpoint_best.pt; '
          'mkdir -p checkpoints/unival_refcocog; mv checkpoint_best.pt checkpoints/unival_refcocog/')


# Load ckpt & config for Image Captioning
checkpoint_path = 'checkpoints/unival_caption_stage_1/checkpoint_best_test.pt'

caption_overrides={"eval_cider":False, "beam":5, "max_len_b":22, "no_repeat_ngram_size":3, "seed":7, "unnormalized": False,
           "bpe_dir":"utils/BPE", "video_model_path": None,}

caption_models, caption_cfg, caption_task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths(checkpoint_path),
    arg_overrides=caption_overrides
)

# Load ckpt & config for Refcoco
checkpoint_path = 'checkpoints/unival_refcocog/checkpoint_best.pt'

refcoco_overrides = {"bpe_dir":"utils/BPE", "video_model_path": None}

refcoco_models, refcoco_cfg, refcoco_task = checkpoint_utils.load_model_ensemble_and_task(
    utils.split_paths(checkpoint_path),
    arg_overrides=refcoco_overrides
)
refcoco_cfg.common.seed = 7
refcoco_cfg.generation.beam = 5
refcoco_cfg.generation.min_len = 4
refcoco_cfg.generation.max_len_a = 0
refcoco_cfg.generation.max_len_b = 4
refcoco_cfg.generation.no_repeat_ngram_size = 3


# Load pretrained ckpt & config for VQA
checkpoint_path = 'checkpoints/unival_vqa/checkpoint_best.pt'

parser = options.get_generation_parser()
input_args = ["", "--task=vqa_gen", "--beam=100", "--unnormalized", f"--path={checkpoint_path}", "--bpe-dir=utils/BPE"]
args = options.parse_args_and_arch(parser, input_args)
vqa_cfg = convert_namespace_to_omegaconf(args)
vqa_task = tasks.setup_task(vqa_cfg.task)
vqa_models, vqa_cfg = checkpoint_utils.load_model_ensemble(
    utils.split_paths(vqa_cfg.common_eval.path),
    task=vqa_task
)

# Load pretrained ckpt & config for Generic Interface
checkpoint_path = 'checkpoints/unival_s2_hs/checkpoint1.pt'

parser = options.get_generation_parser()
input_args = ["", "--task=refcoco", "--beam=10", f"--path={checkpoint_path}", "--bpe-dir=utils/BPE", "--no-repeat-ngram-size=3", "--patch-image-size=384"]
args = options.parse_args_and_arch(parser, input_args)
general_cfg = convert_namespace_to_omegaconf(args)
general_task = tasks.setup_task(general_cfg.task)

overrides={"video_model_path": None, "resnet_model_path": None}

general_models, general_cfg = checkpoint_utils.load_model_ensemble(
    utils.split_paths(general_cfg.common_eval.path),
    task=general_task,
    arg_overrides=overrides
)

move models to gpu
move2gpu(caption_models, caption_cfg)
move2gpu(refcoco_models, refcoco_cfg)
move2gpu(vqa_models, vqa_cfg)
move2gpu(general_models, general_cfg)

# # Initialize generator
caption_generator = caption_task.build_generator(caption_models, caption_cfg.generation)
refcoco_generator = refcoco_task.build_generator(refcoco_models, refcoco_cfg.generation)
vqa_generator = vqa_task.build_generator(vqa_models, vqa_cfg.generation)
vqa_generator.zero_shot = True
vqa_generator.constraint_trie = None
general_generator = general_task.build_generator(general_models, general_cfg.generation)

# Construct image transforms
caption_transform = construct_transform(caption_cfg.task.patch_image_size)
refcoco_transform = construct_transform(refcoco_cfg.task.patch_image_size)
vqa_transform = construct_transform(vqa_cfg.task.patch_image_size)
general_transform = construct_transform(general_cfg.task.patch_image_size)


# Text preprocess
bos_item = torch.LongTensor([general_task.src_dict.bos()])
eos_item = torch.LongTensor([general_task.src_dict.eos()])
pad_idx = general_task.src_dict.pad()


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    token_result = []
    bin_result = []
    img_result = []
    for token in x.strip().split():
      if token.startswith('<bin_'):
        bin_result.append(token)
      elif token.startswith('<code_'):
        img_result.append(token)
      else:
        if bpe is not None:
          token = bpe.decode('{}'.format(token))
        if tokenizer is not None:
          token = tokenizer.decode(token)
        if token.startswith(' ') or len(token_result) == 0:
          token_result.append(token.strip())
        else:
          token_result[-1] += token

    return ' '.join(token_result), ' '.join(bin_result), ' '.join(img_result)


def bin2coord(bins, w_resize_ratio, h_resize_ratio, cfg):
    bin_list = [int(bin[5:-1]) for bin in bins.strip().split()]
    coord_list = []
    coord_list += [bin_list[0] / (cfg.task.num_bins - 1) * cfg.task.max_image_size / w_resize_ratio]
    coord_list += [bin_list[1] / (cfg.task.num_bins - 1) * cfg.task.max_image_size / h_resize_ratio]
    coord_list += [bin_list[2] / (cfg.task.num_bins - 1) * cfg.task.max_image_size / w_resize_ratio]
    coord_list += [bin_list[3] / (cfg.task.num_bins - 1) * cfg.task.max_image_size / h_resize_ratio]
    return coord_list


def encode_text(text, length=None, append_bos=False, append_eos=False):
    line = [
        general_task.bpe.encode(' {}'.format(word.strip()))
        if not word.startswith('<code_') and not word.startswith('<bin_') else word
        for word in text.strip().split()
    ]
    line = ' '.join(line)
    s = general_task.tgt_dict.encode_line(
        line=line,
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


def construct_sample(image: Image, instruction: str, transform):
    patch_image = transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])

    instruction = encode_text(' {}'.format(instruction.lower().strip()), append_bos=True, append_eos=True).unsqueeze(0)
    instruction_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in instruction])
    sample = {
        "id": np.array(['42']),
        "net_input": {
            "src_tokens": instruction,
            "src_lengths": instruction_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        }
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def inference(image, task_type, instruction):
    if task_type == 'Image Captioning':
        task = caption_task
        models = caption_models
        generator = caption_generator
        instruction = 'what does the image describe?'
        transform = caption_transform
        cfg = caption_cfg
    elif task_type == 'Visual Question Answering':
        task = vqa_task
        models = vqa_models
        generator = vqa_generator
        transform = vqa_transform
        cfg = vqa_cfg
    elif task_type == 'Visual Grounding':
        task = refcoco_task
        models = refcoco_models
        generator = refcoco_generator
        instruction = 'which region does the text " {} " describe?'.format(instruction)
        transform = refcoco_transform
        cfg = refcoco_cfg
    elif task_type == 'General':
        task = general_task
        models = general_models
        generator = general_generator
        transform = general_transform
        cfg = general_cfg
    else:
        raise NotImplementedError

    # Construct input sample & preprocess for GPU if cuda available
    sample = construct_sample(image, instruction, transform)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    # Generate result
    with torch.no_grad():
        hypos = task.inference_step(generator, models, sample)
        tokens, bins, imgs = decode_fn(hypos[0][0]["tokens"], task.tgt_dict, task.bpe, generator)

    if bins.strip() != '':
        w, h = image.size
        w_resize_ratio = task.cfg.patch_image_size / w
        h_resize_ratio = task.cfg.patch_image_size / h
        img = np.asarray(image)
        coord_list = bin2coord(bins, w_resize_ratio, h_resize_ratio, cfg)
        cv2.rectangle(
            img,
            (int(coord_list[0]), int(coord_list[1])),
            (int(coord_list[2]), int(coord_list[3])),
            (0, 255, 0),
            3
        )
        return img, None
    else:
        return None, tokens

inputs = [gr.inputs.Image(type='pil'), gr.inputs.Radio(choices=['Image Captioning',"Visual Question Answering", "Visual Grounding", "General"], type="value", default="Image Captioning", label="Task"), gr.inputs.Textbox(lines=1, label="Instruction")]
outputs = [gr.outputs.Image(type='pil'), 'text']
examples = [
    # ['examples/caption/soccer.jpg', 'Image Captioning', None],
    # ['examples/sheep.jpg', 'Visual Question Answering', 'where are the cats?'],
    # ['examples/one_piece.jpeg', 'Visual Grounding', 'a man in a straw hat and a red dress'],
    ['examples/caption/skateboard.jpg', 'General', 'which region does the text " a grey car " describe?'],
    # ['examples/caption/baseball.jpg', 'General', 'what color is the left car?']
]

title = "UnIVAL"
description = "Gradio Demo for UnIVAL: "
article = "<p style='text-align: center'><a href='http://arxiv.org/abs/2202.03052' target='_blank'>Paper</a> | <a href='https://github.com/OFA-Sys/OFA' target='_blank'>Github Repo</a></p>"

io = gr.Interface(fn=inference, inputs=inputs, outputs=outputs,
                  title=title, description=description, article=article, examples=examples, cache_examples=False)
io.launch()