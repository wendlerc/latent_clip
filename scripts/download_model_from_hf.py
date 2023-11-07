import argparse 
import open_clip
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Download a model from HuggingFace Hub and save it as a PyTorch checkpoint')
    parser.add_argument('--model', type=str, required=True, help='the name of the model to download, e.g., hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
    parser.add_argument('--out', type=str, required=True, help='the directory to save the model to')
    parser.add_argument('--name', type=str, default='model_hf.pt', help='the name of the checkpoint file')
    return parser.parse_args()


def main(model_name, output_dir, checkpoint_name):
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    state_dict = {"name":model_name, "state_dict":model.state_dict()}
    torch.save(state_dict, os.path.join(output_dir, checkpoint_name))


if __name__ == '__main__':
    args = parse_args()
    main(args.model, args.out, args.name)