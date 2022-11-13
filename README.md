## Решение команды GrowAi на хакатоне Цифровой прорыв: Маруся не отвечает на реплики из телевизора

Извиняемся за грязность репозитория, но в таком формате обучение может быть запущено.

Для запуска обучения выполните `python3 train.py --model roberta --name roberta_test --epochs 5 --batch 8 --optimizer adam --scheduler cosine --device 4`

Для запуска обучения на фолдах выполните `python3 train_cv.py --model roberta --name roberta_test --epochs 5 --batch 8 --optimizer adam --scheduler cosine --device 4`

Для запуска обучения с семплированием выполните `python3 train_sampled.py --model roberta --name roberta_test --epochs 5 --batch 8 --optimizer adam --scheduler cosine --device 4`

У данных файлов существуют следующие флаги:

model - архитектура модели (bert, t5, roberta)<br />
name - название модели<br />
device - количество девайсов GPU<br />
epochs - количество эпох<br />
batch - размер батча<br />
optimizer - оптимизатор (adam, adafactor, sgd)<br />
scheduler - шедулер (adafactor, cosine, linear)<br />
seeds - фиксировать сид<br />
track - отслеживать лосс или метрику<br />
accum - аккумуляция батча<br />
loss_type - тип лосса (ce, bce, softmarginloss)<br />
add_feat - добавить дополнительные вещественные фичи<br />
add_resp - добавить токен<br />
rewrite_data - переписать данные<br />

В папку checkpoints грузятся чекпоины.

В папке notebooks - рабочие ноутбуки со всеми функциями.

В test_data_predict.ipynb - находится предикшен тестовой выборки.

model_wrapper.py, model_factory.py, clfs.py - это цикл обучения, создание модели, код классификаторов.

Загрузить модели и распаковать, чтобы сделать тестовый предикшен, можно по ссылке: 
https://drive.google.com/file/d/1dA-jnQFDNSY3Oi4tUSFBCHGQl0soKlX2/view?usp=share_link
