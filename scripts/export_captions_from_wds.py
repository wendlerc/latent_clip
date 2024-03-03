import webdataset as wds 
import json 
import argparse 

def parse_args():
    parser = argparse.ArgumentParser(description="Export captions from a webdataset")
    parser.add_argument("input", help="Input webdataset")
    parser.add_argument("output", help="Output file")
    return parser.parse_args()

def main(args):
    dataset = wds.WebDataset(args.input)
    captions = []
    for sample in dataset:
        captions.append(sample['txt'].decode('utf-8'))
    with open(args.output, 'w') as f:
        json.dump(captions, f)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)

