## Решение команды GrowAi на хакатоне Цифровой прорыв: Маруся не отвечает на реплики из телевизора

Для запуска обучения выполните `python3 train.py --model roberta --name roberta_test --epochs 5 --batch 8 --optimizer adam --scheduler cosine --device 4`

Для запуска обучения на фолдах выполните `python3 train_cv.py --model roberta --name roberta_test --epochs 5 --batch 8 --optimizer adam --scheduler cosine --device 4`

Для запуска обучения с семплированием выполните `python3 train_sampled.py --model roberta --name roberta_test --epochs 5 --batch 8 --optimizer adam --scheduler cosine --device 4`

У данных файлов существуют следующие флаги:

model - архитектура модели (bert, t5, roberta)
name - название модели
device - количество девайсов GPU
epochs - количество эпох
batch - размер батча
optimizer - оптимизатор (adam, adafactor, sgd)
scheduler - шедулер (adafactor, cosine, linear)
seeds - фиксировать сид
track - отслеживать лосс или метрику
accum - аккумуляция батча
loss_type - тип лосса (ce, bce, softmarginloss)
add_feat - добавить дополнительные вещественные фичи
add_resp - добавить токен
rewrite_data - переписать данные

В папку checkpoints грузятся чекпоины.

В папке notebooks - рабочие ноутбуки со всеми функциями.

В test_data_predict.ipynb - находится предикшен тестовой выборки.

