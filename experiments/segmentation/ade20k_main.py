import torch
import encoding
import yaml
import sys,os
from PIL import Image
import csv
sys.path.insert(0,'/workspace/tutorials')

def is_image(path):
    try:
        im = Image.open(path)
        return True
    except IOError:
        return False

def split_path(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)
        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    return folders

def load_decoder(decoder_file):
    decoder=[]
    with open(decoder_file, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        for i,row in enumerate(csvreader):
            if i == 0:
                decoder.append('other') #object zero is a catchall for other words
            else:
                decoder.append(row[4])
    return decoder

def decode_image(img_mask,decoder):
    decoded_image = {}
    for i,row in enumerate(img_mask[0]):
        for j,value in enumerate(row):
            decoded_value = decoder[int(value)]
            if decoded_value not in decoded_image:
                decoded_image[decoded_value] = [[i,j]]
            else:
                decoded_image[decoded_value].append([i,j])
    return decoded_image


def strip_dir_structure(image_dir_path,pipeline_path):
    image_split_path = split_path(image_dir_path)
    return os.path.join(pipeline_path,image_split_path[2], image_split_path[1], image_split_path[0])

def load_config(config_file):
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)
        results_file = config['image_segmentation']['results']
        pipeline = config['general']['pipeline_path']
    output_dir = strip_dir_structure(img_dir,pipeline)
    return config, results_file, pipeline, output_dir

mapped_dir = './tutorials'
model_name = 'encoding'

img_dir = os.path.join(mapped_dir,sys.argv[1])
config_file = os.path.join(mapped_dir,sys.argv[2])
config, results_file, pipeline, output_dir = load_config(config_file)
decoder_file = os.path.join(mapped_dir,config['image_segmentation']['decoder'])
decoder = load_decoder(decoder_file)
total_imgs = len(os.listdir(img_dir))

for i,img_file in enumerate(os.listdir(img_dir)):
    img_path = os.path.join(img_dir,img_file)
    # This was added here to clear memory from the GPU
    with torch.no_grad():
        model = encoding.models.get_model('fcn_resnet50s_ade', pretrained=True).cuda()
        model.eval()
        if is_image(img_path):
            output_path = os.path.join(mapped_dir,
                                       output_dir,
                                       img_file.split('.')[0],
                                       results_file)
            print(img_file)
            # print(output_path)
            print('File', i, '/', total_imgs)

            img = encoding.utils.load_image(img_path).cuda().unsqueeze(0)
            output = model.evaluate(img)
            predict = torch.max(output, 1)[1].cpu().numpy() + 1
        else:
            continue

        del model
        torch.cuda.empty_cache()

        decoded_image = decode_image(predict,decoder)
        print(decoded_image.keys())
        yaml.dump(decoded_image, open(output_path, "w"), default_flow_style=False)
