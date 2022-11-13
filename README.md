[![Typing SVG](https://readme-typing-svg.herokuapp.com?color=%2336BCF7&lines=Маруся+не+отвечает+на+реплики)](https://git.io/typing-svg)
[![Typing SVG](https://readme-typing-svg.herokuapp.com?color=%2336BCF7&lines=из+телевизора)](https://git.io/typing-svg)

<h1 align="center"><i>Решение команды GrowAi на хакатоне Цифровой прорыв: Маруся не отвечает на реплики из телевизора</i></h1>

<p>
    <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white">
    <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
    <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white">
    <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black">
    <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white">
    <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white">
</p>

<h2>Краткое описание решения</h2>
<p align="center"><img src="https://grow-ai-marusya-vk-digital.hf.space/media/d5c5f03764cfe8f85490c4edca900b35975b1bd5f4971f4e117e81ce.jpeg" width="1008" height="675"></p>

<p> Мы представляем алгоритм машинного обучения для решения задачи бинарной классификации, отличающий команды пользователя голосового помощника VK "Маруся" от внешнего   шума для повышения коммуникации.</p>

<p>Проанализировав входные данные нами было проведена комплексная оценка параметров признаков, в процессе подготовки было очищено порядка 1,5 тысяч диалогов, содержащих    смайлики и пустые значения для лучшего распознавания команд алгоритмом, проставлена пунктуация в ключевых фразах и диалогах пользователей, дана оценка 
токсичности пользовательских ответов, качества данного ответа, а также лингвистическая приемлемость.</p>

<p><b><i>Стек решения:</i></b> python, torch, transformers, sklearn, scipy, pandas, numpy, streamlit</p>

<p><b><i>Уникальность:</i></b> Решение оформлено в виде web приложения с пользовательским интерфейсом, позволяющим загружать релевантные данные для их обработки разработанным алгоритмом. Интерпретируемость модели определяется извлечением отдельных признаков, имеющих наибольшую взаимосвязь и значимость для классификации. Повышение точности основано на подходе разделения дата сета на 5 частей для обучения отдельных моделей вместо 1.</p>

<p align="center"><b><i>Интерфейс пользователя можно посмотреть </i></b><a href="https://huggingface.co/spaces/Grow-Ai/Marusya_VK_digital">здесь</a></p>
