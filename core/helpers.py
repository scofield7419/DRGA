import numpy as np
import json
import argparse
import time


class Dataset(object):
    def __init__(self, dataset, logger):

        self.origin = {}
        self.logger = logger

        with open(dataset + '/train.txt', 'r', encoding='utf-8') as f:
            self.origin['train'] = json.load(f)

        with open(dataset + '/dev.txt', 'r', encoding='utf-8') as f:
            self.origin['dev'] = json.load(f)

    def getword(self):

        wordcount = {}

        for fname in ['train', 'dev']:
            for m_dict in self.origin[fname]:

                tokens = m_dict["words"].split('|')
                for word in tokens:
                    wordcount[word] = wordcount.get(word, 0) + 1

        words = sorted(wordcount.items(), key=lambda x: x[1], reverse=True)
        self.words = words
        self.wordlist = {item[0]: index + 1 for index, item in enumerate(words)}
        return self.wordlist

    def getdata(self, maxlenth):

        self.getword()
        self.data = {}
        for fname in ['train', 'dev']:
            self.data[fname] = []
            for m_dict in self.origin[fname]:

                tokens = m_dict["words"].split('|')
                words = []
                for tk in tokens:
                    word = self.wordlist[tk]
                    words.append(word)

                lens = len(words)

                pos = m_dict['solution']
                if pos == 0:
                    solution = [1, 0]
                else:
                    solution = [0, 1]
                solution = np.array(solution)

                if lens > maxlenth:
                    words = words[:maxlenth]
                    lens = maxlenth
                elif lens < maxlenth:
                    words += [0] * (maxlenth - lens)

                now = {'words': np.array(words), \
                       'solution': solution, \
                       'lenth': lens}
                self.data[fname].append(now)

        self.logger.info('*' * 150)
        self.logger.info(self.data['train'][:1])
        return self.data['train'], self.data['dev']

    def get_wordvector(self, name, v_dim=100):
        fr = open(name)
        lines = fr.readlines()

        self.wv = {}
        for li in lines:
            li = li.strip().split()
            vec = li[1:]
            if len(vec) != v_dim:
                continue
            vec = [float(item) for item in vec]
            word = li[0].strip().lower()
            if word in self.wordlist.keys():
                self.wv[self.wordlist[word]] = np.array(vec)
        self.wordvector = []
        losscnt = 0
        for i in range(len(self.wordlist) + 1):
            if i in self.wv.keys():
                self.wordvector.append(self.wv[i])
            else:
                losscnt += 1
                self.wordvector.append(np.random.uniform(-0.1, 0.1, [v_dim]))
        self.wordvector = np.array(self.wordvector, dtype=np.float32)
        return self.wordvector


class Parser(object):
    def getParser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='DRGA-self-attention')
        parser.add_argument('--log', type=str, default='log/DRGA-SUBJ.log')
        parser.add_argument('--smalldata', type=int, default=0, choices=[0, 1])
        parser.add_argument('--seed', type=int, default=int(1000 * time.time()))
        parser.add_argument('--dataset', type=str, default='data/SUBJ')
        parser.add_argument('--maxlenth', type=int, default=50)
        parser.add_argument('--grained', type=int, default=2)
        parser.add_argument('--optimizer', type=str, default='Adam', \
                            choices=['Adagrad', 'Adadelta', 'Adam'])
        parser.add_argument('--lr', type=float, default=0.0005)
        parser.add_argument('--epoch', type=int, default=5)
        parser.add_argument('--batch size', type=int, default=10)
        parser.add_argument('--word_vector', type=str, default='emb/glove.6B.100d.txt')
        parser.add_argument('--dim', type=int, default=100)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--tau', type=float, default=0.2)
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--beta', type=float, default=0.8)
        parser.add_argument('--epsilon', type=float, default=0.2)
        parser.add_argument('--sample_cnt', type=int, default=5)
        parser.add_argument('--Attention_pretrain', type=str, default='')
        parser.add_argument('--RL_pretrain', type=str, default='')
        return parser
