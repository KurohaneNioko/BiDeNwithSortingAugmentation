import os, openai, json, time, traceback
from tqdm import tqdm
from allrank.models.losses import neuralNDCG
import torch
import numpy as np


def try_neuralndcg():
    # original neuralNDCG in [0,1] for maximize,  CE in [0, +inf] for minimize
    # implemented neuralNDCG in [-1,0] for minimize, add 1 for scaling
    # neuralNDCG(y_pred, y_true, ...)
    # y_pred: use softmax output
    # y_true: larger number for better ans.
    print(neuralNDCG(
        torch.tensor([[0.4, 0.3, 0.2, 0.1]]).cuda(), torch.tensor([[3., 2., 1., 0.]]).cuda(), temperature=0.001
    ) + torch.tensor(1.).cuda())


def split_dev2test():
    portion = 0.5
    for dataset_name in ['mutual', 'mutual_plus']:
        with open(os.path.join('./data', dataset_name, 'dev_original.json'), 'r', encoding='utf-8') as dev_reader:
            shuffled_dev = json.load(dev_reader)
        np.random.shuffle(shuffled_dev)
        split_idx = int(((len(shuffled_dev)*portion)//2)*2)
        with open(os.path.join('./data', dataset_name, 'dev.json'), 'w+', encoding='utf-8') as dev_writer:
            json.dump(shuffled_dev[:split_idx], dev_writer)
            print('New dev length:', split_idx)
        with open(os.path.join('./data', dataset_name, 'test.json'), 'w+', encoding='utf-8') as test_writer:
            json.dump(shuffled_dev[split_idx:], test_writer)
            print('New test length:', len(shuffled_dev)-split_idx)


def get_all_aug():
    openai.organization = "<ORG_ID>"
    openai.api_key = '<KEY>'
    RETRY_TIME = 4
    for dataset_name in ['mutual']:#, 'mutual_plus']:
        aug_dir = '.data/' + dataset_name + "_aug"
        if not os.path.exists(aug_dir):
            os.mkdir(aug_dir)
        with open(f'data/{dataset_name}/train.json', "r", encoding='utf-8') as reader:
            train_data = json.load(reader)
            reader.close()
        chatGPT_rank = {}
        for corpus in tqdm(train_data):
            dialogue = '\n'.join(map(lambda x: x.strip(), corpus['utterances'])) + '\n'
            hint = 'Rank responses for the dialogue by suitability using the representing charators like "A>B>C>D", in which the first charactor represents the most suitable one. Your answer contains only charactors A,B,C,D and ">". No explanation needed.\n'
            options = ''
            for index, alpha in enumerate(['A', 'B', 'C', 'D']):
                options += alpha+': '+corpus['options'][index]+'\n'
            network_err = True
            while network_err:
                try:
                    rank = openai.ChatCompletion.create(model="gpt-3.5-turbo", timeout=60,
                        messages=[{"role": "user", "content": f"{dialogue+hint+options}"}])
                    time.sleep(3.01)
                    if rank.choices[0].message['content'][0].upper() != corpus['answers'].upper():
                        for retry_chance in range(RETRY_TIME):
                            rank = openai.ChatCompletion.create(model="gpt-3.5-turbo", timeout=60,
                                messages=[{"role": "user", "content": f"Please rank the responses again because the most suitable one is {corpus['answers'].upper()}\n"+dialogue+hint+options}])
                            time.sleep(3.01)
                            if rank.choices[0].message['content'][0].upper() == corpus['answers'].upper():
                                # print(dataset_name, corpus['id'], 'Retry', retry_chance+1)
                                break
                        if rank.choices[0].message['content'][0].upper() != corpus['answers'].upper():
                            print(f'{corpus["answers"].upper()} Rank err at {dataset_name} {corpus["id"]}:', rank.choices[0].message['content'])
                    chatGPT_rank[corpus['id']] = rank.choices[0].message['content'].upper()
                    network_err = False
                except Exception as e:
                    # print(f'Error at {dataset_name} {corpus["id"]}')
                    # print(traceback.format_exc())
                    time.sleep(65)
        with open(os.path.join('./data', aug_dir, dataset_name+'_aug.json'), 'w') as writer:
            json.dump(chatGPT_rank, writer)
            writer.close()


def get_prompt(dataset_name, id):
    openai.organization = "<ORG_ID>"
    openai.api_key = '<KEY>'
    with open(f'data/{dataset_name}/train.json', "r", encoding='utf-8') as reader:
        train_data = json.load(reader)
        reader.close()
    corpus = train_data[id-1]
    dialogue = '\n'.join(map(lambda x: x.strip(), corpus['utterances'])) + '\n'
    # hint = 'sort responses for the dialogue by suitability using the representing charators like "A>B>C>D", in which the first charactor represents the most suitable one. Your answer contains only charactors A,B,C,D and ">". No explanation needed.\n'
    hint = 'sort responses for the dialogue by suitability using the representing charators like "A>B>C>D", in which the first charactor represents the most suitable one. Your answer contains only charactors A,B,C,D and ">".\n'
    options = ''
    for index, alpha in enumerate(['A', 'B', 'C', 'D']):
        options += alpha+': '+corpus['options'][index]+'\n'
    print(corpus['id'])
    # print(dialogue +'\n'+ hint+'\n' + options)
    # print(dialogue +'\n'+ hint+'\n' + options + '\n' + 'Rank responses for the dialogue by suitability using the representing charators.')
    print(dialogue +'\n'+ hint+'\n' + options + '\n' + f"Please sort the responses again because the most suitable one is {corpus['answers'].upper()}")
    if 'plus' in dataset_name:
        print('explain why the other 3 are wrong.')

    # rank = openai.ChatCompletion.create(model="gpt-3.5-turbo", timeout=60, temperature=1,
    #                                     messages=[{"role": "user", "content": dialogue + hint + options + '\n' + 'Rank responses for the dialogue by suitability using the representing charators'}])
    # rank = openai.ChatCompletion.create(model="gpt-3.5-turbo", timeout=60, temperature=1.5,
    #                                     messages=[{"role": "user", "content": dialogue + hint + options + f"Please rank the responses again because the most suitable one is {corpus['answers'].upper()}"}])
    rank = openai.ChatCompletion.create(model="gpt-3.5-turbo", timeout=60, temperature=1,
                                        messages=[{"role": "user", "content": dialogue + hint + options + '\nSORT THEM'}])
    print(rank.choices[0].message['content'][0])


def validate_aug_format(dataset_name):
    aug_dir = dataset_name + "_aug"
    aug_dict = json.load(open(os.path.join('./data', aug_dir, aug_dir+'.json'), 'r', encoding='utf-8'))
    for k in aug_dict:
        _sp = aug_dict[k].split('>')
        if not (len(_sp) == 4 and len(set(_sp)) == 4 and 'A' in _sp and 'B' in _sp and 'C' in _sp and 'D' in _sp):
            print(k)


def verify_aug(dataset_name):
    aug_dir = dataset_name + "_aug"
    with open(f'data/{dataset_name}/train.json', "r", encoding='utf-8') as reader:
        train_data = json.load(reader)
        reader.close()
    aug_dict = json.load(open(os.path.join('./data', aug_dir, aug_dir + '.json'), 'r', encoding='utf-8'))
    for corpus in train_data:
        if aug_dict[corpus['id']][0] != corpus['answers'][0].upper():
            print(corpus['id'])


if __name__ == "__main__":
    # get_all_aug()
    # get_prompt('mutual', 6862)
    # get_prompt('mutual_plus', 3592)
    # validate_aug_format('mutual')
    # validate_aug_format('mutual_plus')
    # verify_aug('mutual')
    # verify_aug('mutual_plus')
    # try_neuralndcg()
    split_dev2test()
    pass