import os
import subprocess
import click
import string
import json

@click.command()
@click.option("--config", type=str, default=None, help="Configurations for TextRecognitionDataGenerator")
def main(config):
    CONFS_DIR = "configs/"
    conf_filename = "config"
    config_part_filename = config[len(CONFS_DIR): len(CONFS_DIR)+len(conf_filename)]
    if config and conf_filename in config and config_part_filename == conf_filename:
        with open(config, 'r') as f:
            args = f.read()
            try:
                subprocess.run(['python']+args.strip().split())
            except ValueError:
                print("Invalid args given")
    else:
        print("No config file provided or name is invalid: must be 'config...'")

    DATASET_DIR = f"../datasets/dataset{config[len(CONFS_DIR)+len(conf_filename):]}/"
    labels_filename = DATASET_DIR+"data/labels.txt"
    abc = string.ascii_uppercase + string.digits
    content = {"abc": abc}
    train, test = 0.8, 0.2
    with open(labels_filename, 'r') as f:
        items = list(filter(lambda x: bool(x), f.readlines()))
        train_bound = int(train*len(items))
        content["train"] = generate_desc(items[:train_bound])
        content["test"] = generate_desc(items[train_bound:])
    with open(DATASET_DIR+"desc.json", 'w') as f:
        json.dump(content, f)
    os.remove(labels_filename)



def generate_desc(data):
    samples = []
    for line in data:
        image_name, label = line.strip().split()
        sample = {"text": label, "name": "data/"+image_name}
        samples.append(sample)
    return samples


if __name__ == "__main__":
    main()
