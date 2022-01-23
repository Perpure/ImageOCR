import numpy as np
import string
import torch
from ctcdecode import CTCBeamDecoder
import re

class CTCLabelDecode(object):
    def __init__(self, config):
        self.character_str = ''
        with open(config['cyr_dict'], 'rb') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip('\n').strip('\r\n')
                self.character_str += line
        self.character_str += ' '
        dict_character = ['_'] + list(self.character_str)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

        self.character_new, self.mapper_to_new = self.make_new_dict(config['cyr_dict_to_rus'])

        if config['prefer_more_spaces']:
            self.pick_best_pred = lambda pred1, pred2: pred2 if (pred2 == 1).sum() > (pred1 == 1).sum() else pred1
        else:
            self.pick_best_pred = lambda pred1, pred2: pred1

        if config['use_beam_search']:
            ctc_config = config['ctc_decoder']
            self.decoder = CTCBeamDecoder(
                self.character_new,
                model_path=ctc_config['lm_model'],
                alpha=ctc_config['alpha'],
                beta=ctc_config['beta'],
                cutoff_top_n=ctc_config['cutoff_top_n'],
                cutoff_prob=ctc_config['cutoff_prob'],
                beam_width=ctc_config['beam_width'],
                num_processes=ctc_config['num_processes'],
                blank_id=0,
                log_probs_input=False
            )

            self.decode = lambda preds: self.decode_beam_search(preds)
        else:
            self.decode = lambda preds: self.decode_greedy(preds)

        self.rm_dup_spaces = re.compile('\s+')

    def make_new_dict(self, new_character_dict_path):
        with open(new_character_dict_path, 'rb') as f:
            lines = f.readlines()
            assert len(lines) + 3 == len(self.character)  # 2 spaces and blank
            lines = list(map(lambda x: x.decode('utf-8').strip('\n').strip('\r\n'), lines))
            lines = ['  '] + lines + ['  ']  # to self.character format
            character_new = ['_'] + sorted(list(set(map(lambda x: x[1], lines))))

        mapper_to_new = [0]
        for s in lines:
            c = s[1]
            mapper_to_new.append(character_new.index(c))
        return character_new, mapper_to_new

    def adapt_row_to_new_dict(self, row):
        new_row = np.zeros(len(self.character_new))
        for i, val in enumerate(row):
            new_row[self.mapper_to_new[i]] += val
        return new_row

    def adapt_to_new_dict(self, preds):
        preds_new = np.apply_along_axis(self.adapt_row_to_new_dict, 2, preds)
        return preds_new

    def decode_beam_search(self, preds):
        beam_results, beam_scores, timesteps, out_len = self.decoder.decode(torch.from_numpy(preds))

        texts = []
        for j in range(len(beam_results)):

            pred1 = beam_results[j][0][:out_len[j][0]]
            pred2 = beam_results[j][1][:out_len[j][1]]
            top_predict = self.pick_best_pred(pred1, pred2)

            text = ''.join(list(map(lambda x: self.character_new[int(x)], top_predict)))
            texts.append(self.rm_dup_spaces.sub(' ', text.strip()))

        return texts

    def __call__(self, preds):
        if isinstance(preds, torch.Tensor):
            preds = preds.numpy()
        preds = self.adapt_to_new_dict(preds)
        texts = self.decode(preds)
        return texts

    def decode_greedy(self, preds, text_prob=None, is_remove_duplicate=False):
        text_index = preds.argmax(axis=2)
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = [0]
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                            batch_idx][idx]:
                        continue
                char_list.append(self.character_new[int(text_index[batch_idx][idx])])
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = ''.join(char_list)
            result_list.append(text)

        return result_list


