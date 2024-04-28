import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from DataLoader.datasets import *
from DataLoader.utils import *
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
import torch.nn.functional as F
from tqdm import tqdm
from rouge import Rouge



def evaluate(beam_size):
    """
    Evaluation

    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)
        try:
            # Encode
            encoder_out, _, _ = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        except:
            image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            encoder_out, _, _ = encoder(image)
            # Encode
        # encoder_out,_ = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        # encoder_out,_ = encoder(image)
        # print(encoder_out.shape)
        enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(3)

        # Flatten encoding
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            awe, _ = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

            gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

            scores = decoder.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 100:
                break
            step += 1
        # print(complete_seqs)
        # try:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)
        # except:
        # continue
        # break
    bleu4 = corpus_bleu(references, hypotheses)
    bleu1_list = []
    bleu2_list = []
    bleu3_list = []
    bleu4_list = []
    rouge = Rouge()

    for i in range(len(references)):
        if len(references[i]) != 1:
            print(len(references[i]))
        # print(references[0][0])
        # print(hypotheses[0])
        reference = [rev_word_map[ind] for ind in references[i][0]]
        hypothese = [rev_word_map[ind] for ind in hypotheses[i]]
        bleu_1 = sentence_bleu(reference, hypothese, weights=(1, 0, 0, 0))
        # print([' '.join(hypothese)])
        # print(hypothese)
        rouge_score = rouge.get_scores(hyps=[' '.join(hypothese)], refs=[' '.join(reference)])
        rouge_score_l = rouge_score[0]["rouge-l"]['f']
        rouge_score_1 = rouge_score[0]["rouge-1"]['f']
        rouge_score_2 = rouge_score[0]["rouge-2"]['f']
        # print(rouge_score)
        # bleu_2 = sentence_bleu(reference, hypothese, weights=(0.5, 0.5, 0, 0))
        # bleu_3 = sentence_bleu(reference, hypothese, weights=(0.33, 0.33, 0.33, 0))
        # bleu_4 = sentence_bleu(reference, hypothese, weights=(0.25, 0.25, 0.25, 0.25))
        bleu1_list.append(bleu_1)
        bleu2_list.append(rouge_score_1)
        bleu3_list.append(rouge_score_2)
        bleu4_list.append(rouge_score_l)
        # break
    print("bleu_1:", sum(bleu1_list) / len(bleu1_list))
    print("rouge_score_1:", sum(bleu2_list) / len(bleu2_list))
    #print('rouge_score_2:', sum(bleu3_list) / len(bleu3_list))
    print("rouge_score_l:", sum(bleu4_list) / len(bleu4_list))
    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references, hypotheses)
    bleu1_list = []
    bleu2_list = []
    bleu3_list = []
    bleu4_list = []
    rouge = Rouge()
    ground_truth = []
    predict = []
    for i in range(len(references)):
        if len(references[i]) != 1:
            print(len(references[i]))
        # print(references[0][0])
        # print(hypotheses[0])
        reference = [rev_word_map[ind] for ind in references[i][0]]
        hypothese = [rev_word_map[ind] for ind in hypotheses[i]]
        bleu_1 = sentence_bleu(reference, hypothese, weights=(1, 0, 0, 0))
        ground_truth.append(reference)
        predict.append(hypothese)
    path = dir_
    predict_file_path = path + 'predict.json'
    true_file_path = path + 'true.json'
    # 使用 json.dump 将数据写入 JSON 文件
    with open(predict_file_path, 'w') as json_file:
        json.dump(predict, json_file)
    with open(true_file_path, 'w') as json_file:
        json.dump(ground_truth, json_file)
    return bleu4


if __name__ == '__main__':
    beam_size = 1    
    dir_ = "checkpoints/Dense/"
    # dir_ = "fixed_category/"+ dir_
    # Parameters
    data_folder = 'Data_Generation/L_system/100_padding'  # folder with data files saved by create_input_files.py
    data_name = 'fractal_random_1_cap_per_img_1_min_word_freq'  # base name shared by data files
    checkpoint =  dir_ + "BEST_checkpoint_fractal_random_1_cap_per_img_1_min_word_freq.pth.tar"  # model checkpoint
    word_map_file = data_folder + "WORDMAP_fractal_random_1_cap_per_img_1_min_word_freq.json"  # word map, ensure it's the same the data was encoded with and the model was trained with

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

    # Load model
    checkpoint = torch.load(checkpoint)
    # decoder = DecoderWithAttention
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    # Normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    print("\nBLEU-4 score @ beam size of %d is %.4f." % (beam_size, evaluate(beam_size)))
