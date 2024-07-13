import os
import numpy as np
import h5py
import json
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
from PIL import Image


def create_input_files(dataset, captions_per_image, min_word_freq, data_folder, output_folder, max_len=100):
    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    train_path = data_folder
    data = []
    with open(os.path.join(train_path, 'train.jsonl'), 'r') as j:
        for line in j:
            data_dict = json.loads(line)
            data.append(data_dict)
    for img in data:
        captions = []
        token = img['caption'].split(" ")
        word_freq.update(token)
        if len(token) <= max_len:
            captions.append(token)

        if len(captions) == 0:
            print(img)
            continue
        path = os.path.join(train_path, img['file_name'])
        train_image_paths.append(path)
        train_image_captions.append(captions)


    test_path = data_folder
    # Read Karpathy JSON
    data = []
    with open(os.path.join(test_path, 'test.jsonl'), 'r') as j:
        for line in j:
            data_dict = json.loads(line)
            data.append(data_dict)
    for img in data:
        captions = []
        token = img['caption'].split(" ")
        word_freq.update(token)
        if len(token) <= max_len:
            captions.append(token)
        if len(captions) == 0:
            continue
        path = os.path.join(test_path, img['file_name'])
        test_image_paths.append(path)
        test_image_captions.append(captions)

    val_path = data_folder
    # Read Karpathy JSON
    data = []
    with open(os.path.join(val_path, 'val.jsonl'), 'r') as j:
        for line in j:
            data_dict = json.loads(line)
            data.append(data_dict)
    for img in data:
        captions = []
        token = img['caption'].split(" ")
        word_freq.update(token)
        if len(token) <= max_len:
            captions.append(token)
        if len(captions) == 0:
            continue
        path = os.path.join(val_path, img['file_name'])
        val_image_paths.append(path)
        val_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)
            print(len(impaths))

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                # Sanity check
                assert len(captions) == captions_per_image

                img = Image.open(impaths[i])
                # Ensure the image is in RGB format
                img = img.convert('RGB')
                # Resize the image
                img = img.resize((256, 256))
                # Convert the image to a NumPy array
                img = np.array(img)
                # Transpose the array to match the original code
                img = img.transpose(2, 0, 1)
                # Assert conditions
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255
                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


if __name__ == '__main__':
    create_input_files(dataset='fractal_random',
                       captions_per_image=1,
                       min_word_freq=1,
                       output_folder="100_padding/",
                       data_folder="Data/",
                       max_len=100)
