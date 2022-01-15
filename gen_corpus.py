import re
import io
from os import walk
from tqdm import tqdm


rm_junk = re.compile('[^а-яё \n]')
rm_metadata = re.compile('\[.*?]')
rm_blank_lines = re.compile('\n *\n')
rm_leading_spaces = re.compile('\n +')
rm_spaces = re.compile(' +')

output = open('soc.txt', 'w')

def write_sentences(text):
    text = text.replace('\r', '')
    text = rm_junk.sub(' ', text)
    text = rm_spaces.sub(' ', text)
    text = rm_blank_lines.sub('\n', text)
    text = rm_leading_spaces.sub('\n', text)
    text = text.replace('\n\n', '\n')
    sentences = text.split('\n')
    sentences = list(filter(lambda sentence: sentence.count(' ') < 3 and
                                             all([len(word) < 15 for word in sentence.split(' ')]), sentences))

    output.write('\n'.join(sentences))

soc_path = '/path/to/txts/folder'
with io.open(soc_path + 'name0.txt', 'r', encoding='utf-8') as f:
    text = f.read().lower()
    text = rm_metadata.sub(' ', text)
    write_sentences(text)


path_txts = '/path/to/txts/folder2'

filenames = list(map(lambda x: path_txts + x, next(walk(path_txts), (None, None, []))[2]))

filenames += [os.path.join(soc_path, 'name1.txt'), os.path.join(soc_path, 'name2.txt')]

for file in tqdm(filenames):
    with io.open(file, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        write_sentences(text)

output.close()