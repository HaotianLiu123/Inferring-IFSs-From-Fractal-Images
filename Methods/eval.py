import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn as nn
from utils import *
from DataLoader.Data_Lsystem import get_dataLsystem_loader
from DataLoader.Data_Julia import get_data_Julia_loader
import argparse
import json
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge


def evaluate_lsystem(val_loader):
    references = list()
    hypotheses = list()
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            allcaps = allcaps.to(device)

            imgs, _, _ = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)
            scores_copy = scores.clone()
            # References
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)
        rouge1_list = []
        rougel_list = []
        rouge = Rouge()

        for i in range(len(references)):
            reference = [rev_word_map[ind] for ind in references[i][0]]
            hypothese = [rev_word_map[ind] for ind in hypotheses[i]]
            rouge_score = rouge.get_scores(hyps=[' '.join(hypothese)], refs=[' '.join(reference)])
            rouge_score_1 = rouge_score[0]["rouge-1"]['f']
            rouge_score_l = rouge_score[0]["rouge-l"]['f']
            rouge1_list.append(rouge_score_1)
            rougel_list.append(rouge_score_l)

        print("\nThe evaluation result of L-system\n")

        print('\n * BLEU-4 - {bleu}'.format(bleu=bleu4))
        print("\n * rouge_score_1 - {}", sum(rouge1_list) / len(rouge1_list))
        print("\n * rouge_score_l - {}", sum(rougel_list) / len(rougel_list))


def evaluate_julia(model, test_loader):
    model.eval()
    total_loss = 0.0
    total_count = 0
    test_criterion = nn.L1Loss(reduction='none').to(device)
    with torch.no_grad():
        for (batch, meta) in enumerate(test_loader):
            input = meta['image']
            label = meta['label']
            input = input.to(device)
            label = label.to(device)
            _, pre, _ = model(input)
            loss = test_criterion(pre, label)
            total_loss += loss
            total_count += pre.size(0)
    print("\nThe evaluation result of Julia Set")
    print("MAE of Re(c) Im(c) N τ :\n", total_loss / total_count)


if __name__ == '__main__':
    beam_size = 1

    parser = argparse.ArgumentParser(description='Your script description here')

    parser.add_argument('--data_folder', type=str,
                        default='Data_Generation/L_system/100_padding/',
                        help='folder with data files of L-system images')
    parser.add_argument('--data_name', type=str, default='fractal_random_1_cap_per_img_1_min_word_freq',
                        help='base name shared by data files')
    parser.add_argument('--Julia_data_folder', type=str,
                        default='Data_Generation/Julia_Set/Data/',
                        help='folder with data files of Julia images')
    parser.add_argument('--checkpoint', type=str, default="checkpoint/Dense_final/", help='model checkpoint')

    args = parser.parse_args()

    data_folder = args.data_folder
    data_name = args.data_name
    data_folder_julia = args.Julia_data_folder
    checkpoint = os.path.join(args.checkpoint, "BEST_checkpoint_fractal_random_1_cap_per_img_1_min_word_freq.pth.tar")
    word_map_file = os.path.join(args.data_folder, "WORDMAP_fractal_random_1_cap_per_img_1_min_word_freq.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    # Load model
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint['encoder_decoder_g2']
    encoder = encoder.to(device)
    encoder.eval()
    decoder = checkpoint['decoder_g1']
    decoder = decoder.to(device)
    decoder.eval()
    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    _, test_loader_julia = get_data_Julia_loader(data_folder_julia)
    _, _, test_loader_lsystem = get_dataLsystem_loader(data_folder, data_name)
    evaluate_julia(encoder, test_loader_julia)
    evaluate_lsystem(test_loader_lsystem)
