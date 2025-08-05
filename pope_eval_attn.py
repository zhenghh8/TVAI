import argparse
import json
import numpy as np
import os
from visualization import visualize, attn_sorted
import nltk

#### for attention statistics
def answer_to_word_ids(answer, model):
    FLAG_WORDS = ["YES", "yes", "Yes", "No", "not", "no", "NO"]

    one_token_word = ' newline '

    # llava-1.5
    if model == 'llava-1.5':
        special_one_llava = ['\n', '."', '</s>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',"']
        for special in special_one_llava:
            answer = answer.replace(special, one_token_word)
    # qwen-vl-chat
    elif model == 'qwen-vl-chat':
        special_one_qwen = ['."\n\n', '.\n\n', '\n\n', '.\n', '\n', '<|im_start|>', '<|im_end|>', '<|endoftext|>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '."', ',"', '".', 'cannot']
        for special in special_one_qwen:
            answer = answer.replace(special, one_token_word)
    else:
        # TODO
        raise NotImplementedError
    words = nltk.word_tokenize(answer)

    for idx, word in enumerate(words):
        if word in FLAG_WORDS:
            return idx
    raise NotImplementedError("model does not generate discriminative words i.e. yes or no.")
#### for attention statistics

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:
        line = line.replace(".", "")
        line = line.replace(",", "")
        words = line.split(" ")
        if any(word in NEG_WORDS for word in words) or any(
            word.endswith("n't") for word in words
        ):
            pred_list.append(0)
        else:
            pred_list.append(1)

    return pred_list

def save_acc(pred_list, label_list, save_dir):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    hallucination_label = []

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
            hallucination_label.append('0')
        elif pred == pos and label == neg:
            FP += 1
            hallucination_label.append('1')
        elif pred == neg and label == neg:
            TN += 1
            hallucination_label.append('0')
        elif pred == neg and label == pos:
            FN += 1
            hallucination_label.append('1')

    print("TP\tFP\tTN\tFN\t\n")
    print("{}\t{}\t{}\t{}\n".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)

    print("Accuracy: {}\n".format(acc))
    print("Precision: {}\n".format(precision))
    print("Recall: {}\n".format(recall))
    print("F1 score: {}\n".format(f1))
    print("Yes ratio: {}\n".format(yes_ratio))

    with open(save_dir, 'a') as f:
        f.write("TP\tFP\tTN\tFN\t\n")
        f.write("{}\t{}\t{}\t{}\n".format(TP, FP, TN, FN))
        f.write("Accuracy: {}\n".format(acc))
        f.write("Precision: {}\n".format(precision))
        f.write("Recall: {}\n".format(recall))
        f.write("F1 score: {}\n".format(f1))
        f.write("Yes ratio: {}\n".format(yes_ratio))

    return hallucination_label

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    #### for attention statistics
    parser.add_argument("--model", type=str, help="estimated model")
    #### for attention statistics
    parser.add_argument("--ans_file", type=str, help="answer file")

    args = parser.parse_args()

    ans_path = 'results_generation/' + args.ans_file
    lines = open(ans_path).read().split("\n")

    pred_list = []
    label_list = []

    for line in lines:
        if len(line) == 0:
            break
        line = json.loads(line)
        pred_list = recorder([line["ans"]], pred_list)
        if isinstance(line["label"], int):
            label_list += [line["label"]]
        else:  # isinstance(str), 'yes' or 'no'
            label_list = recorder([line["label"]], label_list)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    if not os.path.exists('results_analysis'):
        os.makedirs('results_analysis')
    base_name = os.path.splitext(os.path.basename(args.ans_file))[0]
    dir_name = os.path.dirname(args.ans_file)
    dir_name_1 = dir_name.split('/')[0]
    if not os.path.exists(os.path.join('results_analysis', dir_name_1)):
        os.makedirs(os.path.join('results_analysis', dir_name_1))
    if not os.path.exists(os.path.join('results_analysis', dir_name)):
        os.makedirs(os.path.join('results_analysis', dir_name))
    #### for attention statistics
    attn_save_dir = os.path.join('results_analysis', dir_name, base_name)
    if not os.path.exists(attn_save_dir):
        os.makedirs(attn_save_dir)
    if not os.path.exists(attn_save_dir + '/no_hall'):
        os.makedirs(attn_save_dir + '/no_hall')
    if not os.path.exists(attn_save_dir + '/hall'):
        os.makedirs(attn_save_dir + '/hall')
    #### for attention statistics
    

    save_acc_dir = os.path.join('results_analysis', dir_name, base_name + '_acc.txt')
    hall_label = save_acc(pred_list, label_list, save_acc_dir)

    #### for attention statistics
    # visualize each instance
    i = 0
    for line_idx, line in enumerate(lines):
        if len(line) == 0:
                break
        line = json.loads(line)

        word_id = answer_to_word_ids(line['ans'], args.model)
        flag_token_idx = line['word_ids'][word_id]

        # print(line['ans'])
        # print(word_id, hall_label[line_idx])

        attn_img = np.array(line["generated_attn_img"][flag_token_idx])
        attn_text = np.array(line['generated_attn_text'][flag_token_idx])

        image_id = line['image_id'].split('_')[-1][-10:-4]

        if hall_label[line_idx] == '0':
            save_dir = os.path.join(attn_save_dir, 'no_hall', 'img_{}_'.format(i) + image_id + '.png')
            visualize(attn_img, 'VAR', cmap='viridis', save_dir=save_dir)
            save_dir = os.path.join(attn_save_dir, 'no_hall', 'text_{}_'.format(i) + image_id + '.png')
            visualize(attn_text, 'TAR', cmap='viridis', save_dir=save_dir)
            i += 1
        else:
            save_dir = os.path.join(attn_save_dir, 'hall', 'img_{}_'.format(i) + image_id + '.png')
            visualize(attn_img, 'VAR', cmap='viridis', save_dir=save_dir)
            save_dir = os.path.join(attn_save_dir, 'hall', 'text_{}_'.format(i) + image_id + '.png')
            visualize(attn_text, 'TAR', cmap='viridis', save_dir=save_dir)
            i += 1
        if i % 6 == 0:
            i = 0
    #### for attention statistics


    #### for attention statistics
    # obtain the attention statistics
    i = 0
    delimiter = ' '
    hall_flag = 0
    nonhall_flag =0
    save_attn_label = 'results_analysis/' + args.ans_file + '_attn_label.txt'

    
    # with open(save_attn_label, 'w', encoding='utf-8') as file:
    for line in lines:
        if len(line) == 0:
            break
        line = json.loads(line)
        flag_token_idx = line['word_ids'][answer_to_word_ids(line['ans'], args.model)]
        attn_img = np.array(line["generated_attn_img"][flag_token_idx])
        attn_text = np.array(line['generated_attn_text'][flag_token_idx])

        # attn_img_ = np.reshape(attn_img, -1).astype(str)
        # attn_text_ = np.reshape(attn_text, -1).astype(str)
        # file.write('image_id: ' + line['image_id'] + ' ' + delimiter.join(attn_img_) + ' ' + \
        #             delimiter.join(attn_text_) + ' ' + hall_label[i] + '\n')

        if hall_label[i] == '0':
            if nonhall_flag == 0:
                mean_nonhall_img_attn = attn_img
                mean_nonhall_text_attn = attn_text
                nonhall_flag =1
            else:
                mean_nonhall_img_attn = 0.5 * (mean_nonhall_img_attn + attn_img)
                mean_nonhall_text_attn = 0.5 * (mean_nonhall_text_attn + attn_text)
        else:
            if hall_flag == 0:
                mean_hall_img_attn = attn_img
                mean_hall_text_attn = attn_text
                hall_flag =1
            else:
                mean_hall_img_attn = 0.5 * (mean_hall_img_attn + attn_img)
                mean_hall_text_attn = 0.5 * (mean_hall_text_attn + attn_text)

        i += 1

    # # visualization
    # # sorted
    # mean_hall_img_attn = attn_sorted(mean_hall_img_attn, dim=1)
    # mean_hall_text_attn = attn_sorted(mean_hall_text_attn, dim=1)

    # mean_nonhall_img_attn = attn_sorted(mean_nonhall_img_attn, dim=1)
    # mean_nonhall_text_attn = attn_sorted(mean_nonhall_text_attn, dim=1)

    visualize(mean_hall_img_attn, 'VAR', save_dir=os.path.join('results_analysis', dir_name, base_name + '_hall_img.png'))
    visualize(mean_hall_text_attn, 'TAR', save_dir=os.path.join('results_analysis', dir_name, base_name + '_hall_txt.png'))
    visualize(mean_nonhall_img_attn, 'VAR')
    visualize(mean_nonhall_text_attn, 'TAR')

    np.save(os.path.join('results_analysis', dir_name, base_name + '_hall_img.npy'), mean_hall_img_attn)
    np.save(os.path.join('results_analysis', dir_name, base_name + '_hall_text.npy'), mean_hall_text_attn)
    np.save(os.path.join('results_analysis', dir_name, base_name + '_nonhall_img.npy'), mean_nonhall_img_attn)
    np.save(os.path.join('results_analysis', dir_name, base_name + '_nonhall_text.npy'), mean_nonhall_text_attn)


    # plt.show()
    #### for attention statistics
