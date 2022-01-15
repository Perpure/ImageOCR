from PIL import Image, ImageDraw, ImageFont
import os
import random
import re
from colorsys import rgb_to_hsv, hsv_to_rgb
from colorthief import ColorThief
from tqdm import tqdm


fonts_path = 'data/fonts/'
bg_path = 'data/bg_images/'
save_path = 'data/eval_img/'
n_files = 1000
rm_junk = re.compile('[^а-я|ё \n]')
rm_comms = re.compile('\[.*?]')
rm_spaces = re.compile(' +')

text_file = 'your_file.txt' # путь к текстовому файлу, из которого будут браться слова
with open(text_file, 'r') as f:
    text = f.read().lower()
    text = rm_comms.sub('', text)
    text = rm_junk.sub(' ', text)
    text = text.replace('\n', ' ')
    text = rm_spaces.sub(' ', text)
    wordlist = text.split(' ')
    wordlist.remove('')
    wordlist.remove('')

punc = []

fonts = next(os.walk(fonts_path), (None, None, []))[2]
bg_images = next(os.walk(bg_path), (None, None, []))[2]


def add_punc(word):
    # if random.randint(1, 2) == 1:
    #     return random.choice(punc).replace(' ', word)
    return word


def get_word():
    return add_punc(rm_junk.sub('', random.choice(wordlist).replace(' ', '').lower()))


def get_text():
    length = random.randint(1, 4)
    return ' '.join([get_word() for _ in range(length)])


def text_to_image(font_name, font_size):
    font = ImageFont.truetype(fonts_path + font_name, size=font_size)

    text = get_text()
    text_window = font.getsize(text)

    bg_image = Image.open(bg_path + random.choice(bg_images))
    while bg_image.size[0] - 2 * text_window[0] - 50 < 0 or bg_image.size[1] - 8 * text_window[1] - 50 < 0:
        text = get_text()
        text_window = font.getsize(text)
        bg_image = Image.open(bg_path + random.choice(bg_images))
    bg_size = bg_image.size

    x = random.randint(0, 50)
    y = random.randint(0, 50)
    x1 = random.randint(min(x + 2 * text_window[0], bg_size[0]), bg_size[0])
    y1 = random.randint(min(y + 8 * text_window[1], bg_size[1]), bg_size[1])

    bg_image = bg_image.crop((x, y, x1, y1))
    bg_image.save('1.png')
    r, g, b = ColorThief('1.png').get_color(1)
    h, s, v = rgb_to_hsv(r / 255, g / 255, b / 255)

    color = tuple(map(lambda x: round(x * 255), hsv_to_rgb((h + 0.5) % 1, s, 1 - v)))

    draw = ImageDraw.Draw(bg_image)
    bg_size = bg_image.size
    draw_point = random.randint(text_window[0] - 20, bg_size[0] - text_window[0] - 20), \
                 random.randint(4 * text_window[1], bg_size[1] - 4 * text_window[1])
    draw.text(draw_point, text, font=font, fill=color)

    return bg_image, text


for i in tqdm(range(n_files)):
    font = random.choice(fonts)
    img, text = text_to_image(font, 40)
    name = save_path + str(i)
    img.save(name + '.jpg')
    with open(name + '.txt', 'w') as file:
        file.write(text)
