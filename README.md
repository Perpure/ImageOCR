# ImageOCR

# Описание файлов

### Основные файлы
- det_handler.py - основной обработчик запросов к модели поиска текста на изображении
- det_postprocess.py - выделение прямоугольников с текстом на основе результатов нейронной сети
- det_sort_boxes.py - сортировка прямоугольников с текстом
- rec_handler.py - основной обработчик запросов к модели распознавании текста
- rec_postprocess - CTC-декодер результатов нейронной сети
- send.py - отправка изображения на сервер-обработчик

### Дополнительные файлы
- ocr_metrics/eval_img_gen.py - генерация изображений для тестирования
- ocr_metrics/eval.py - тестирование на сгенерированных изображениях, подсчет метрик
- gen_corpus.py - генерация корпуса текста для языковой модели (использует непубличные данные)
- draw_ocr.py - отрисовка результатов работы на исходном изображении

# Установка

1. Скачать [архив](https://drive.google.com/file/d/14KP8Q9guygL-IHExgKoNYwmW94zoqiab/view?usp=sharing) и распаковать в папку с проектом
2. Установить Java 11
3. Установить зависимости `pip install -r requirements.txt`
4. Установить CTCDecode `git clone --recursive https://github.com/parlance/ctcdecode.git && cd ctcdecode && pip install .`
5. Запуск сервера `./rerun.sh`
