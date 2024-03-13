<div align="center">

  <img src="https://github.com/Gourieff/Assets/raw/main/sd-webui-reactor/ReActor_logo_NEW_RU.png?raw=true" alt="logo" width="180px"/>

  ![Version](https://img.shields.io/badge/версия_нода-0.4.1_beta12-green?style=for-the-badge&labelColor=darkgreen)
  
  <sup>
  <font color=brightred>

  ## !!! [Важные изменения](#latestupdate) !!!<br>Не забудьте добавить Нод заново в существующие воркфлоу
  
  </font>
  </sup>

  <a href="https://boosty.to/artgourieff" target="_blank">
    <img src="https://lovemet.ru/www/boosty.jpg" width="108" alt="Поддержать проект на Boosty"/>
    <br>
    <sup>
      Поддержать проект
    </sup>
  </a>

  <hr>
  
  [![Commit activity](https://img.shields.io/github/commit-activity/t/Gourieff/comfyui-reactor-node/main?cacheSeconds=0)](https://github.com/Gourieff/comfyui-reactor-node/commits/main)
  ![Last commit](https://img.shields.io/github/last-commit/Gourieff/comfyui-reactor-node/main?cacheSeconds=0)
  [![Opened issues](https://img.shields.io/github/issues/Gourieff/comfyui-reactor-node?color=red)](https://github.com/Gourieff/comfyui-reactor-node/issues?cacheSeconds=0)
  [![Closed issues](https://img.shields.io/github/issues-closed/Gourieff/comfyui-reactor-node?color=green&cacheSeconds=0)](https://github.com/Gourieff/comfyui-reactor-node/issues?q=is%3Aissue+is%3Aclosed)
  ![License](https://img.shields.io/github/license/Gourieff/comfyui-reactor-node)

  [English](/README.md) | Русский

# ReActor Node для ComfyUI

</div>

### Нод (node) для быстрой и простой замены лиц на любых изображениях для работы с ComfyUI, основан на [ReActor](https://github.com/Gourieff/sd-webui-reactor) SD-WebUI Face Swap Extension

> Данный Нод идёт без фильтра цензуры (18+, используйте под вашу собственную [ответственность](#disclaimer)) 

<div align="center">

---
[**Что нового**](#latestupdate) | [**Установка**](#installation) | [**Использование**](#usage) | [**Устранение проблем**](#troubleshooting) | [**Обновление**](#updating) | [**Ответственность**](#disclaimer) | [**Благодарности**](#credits) | [**Заметка**](#note)

---

</div>

<div align="center">
  <img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/demo.gif?raw=true" alt="logo" width="100%"/>
</div>

<a name="latestupdate">

## Что нового в последнем обновлении

### 0.4.1 <sub><sup>BETA9</sup></sub>

- Поддержка CUDA 12 - не забудьте запустить (Windows) `install.bat` или (Linux/MacOS) `install.py` для используемого Python окружения или попробуйте установить ORT-GPU для CU12 вручную (https://onnxruntime.ai/docs/install/#install-onnx-runtime-gpu-cuda-12x)
- Исправление Issue https://github.com/Gourieff/comfyui-reactor-node/issues/173

### 0.4.1 <sub><sup>BETA8</sup></sub>

- Отдельный Нод для восстаноления лиц (FR https://github.com/Gourieff/comfyui-reactor-node/issues/191), располагается внутри меню ReActor (нод RestoreFace)
- (Windows) Установка зависимостей теперь может быть выполнена в Python из PATH ОС
- Разные исправления и улучшения

### 0.4.1 <sub><sup>BETA1</sup></sub>

- Face Restore Visibility и CodeFormer Weight (Fidelity) теперь доступны; не забудьте заново добавить Нод в ваших существующих воркфлоу

<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/0.4.1-whatsnew-01.jpg?raw=true" alt="0.4.1-whatsnew-01" width="100%"/>

### 0.4.0

- Вход "input_image" теперь идёт первым, это даёт возможность корректного байпаса, а также это правильно с точки зрения расположения входов (главный вход - первый);
- Теперь можно сохранять модели лиц в качестве файлов "safetensors" (`ComfyUI\models\reactor\faces`) и загружать их в ReActor, реализуя разные сценарии использования, а также  храня супер легкие модели лиц, которые вы чаще всего используете:

<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/0.4.0-whatsnew-01.jpg?raw=true" alt="0.4.0-whatsnew-01" width="100%"/>
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/0.4.0-whatsnew-02.jpg?raw=true" alt="0.4.0-whatsnew-02" width="100%"/>

- Возможность сохранять модели лиц напрямую из изображения:

<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/0.4.0-whatsnew-03.jpg?raw=true" alt="0.4.0-whatsnew-03" width="50%"/>

- Оба входа опциональны, присоедините один из них в соответствии с вашим воркфлоу; если присоеденены оба - вход `image` имеет приоритет.
- Различные исправления, делающие это расширение лучше.

Спасибо всем, кто находит ошибки, предлагает новые функции и поддерживает данный проект!

<a name="installation">

## Установка

<details>
	<summary>SD WebUI: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/">AUTOMATIC1111</a> или <a href="https://github.com/vladmandic/automatic">SD.Next</a></summary>

1. Закройте (остановите) SD-WebUI Сервер, если запущен
2. (Для пользователей Windows):
   - Установите [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) (Например, версию Community - этот шаг нужен для правильной компиляции библиотеки Insightface)
   - ИЛИ только [VS C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), выберите "Desktop Development with C++" в разделе "Workloads -> Desktop & Mobile"
   - ИЛИ если же вы не хотите устанавливать что-либо из вышеуказанного - выполните [данные шаги (раздел. I)](#insightfacebuild)
3. Перейдите в `extensions\sd-webui-comfyui\ComfyUI\custom_nodes`
4. Откройте Консоль или Терминал и выполните `git clone https://github.com/Gourieff/comfyui-reactor-node`
5. Перейдите в корневую директорию SD WebUI, откройте Консоль или Терминал и выполните (для пользователей Windows)`.\venv\Scripts\activate` или (для пользователей Linux/MacOS)`venv/bin/activate`
6. `python -m pip install -U pip`
7. `cd extensions\sd-webui-comfyui\ComfyUI\custom_nodes\comfyui-reactor-node`
8. `python install.py`
9.  Пожалуйста, дождитесь полного завершения установки
10. (Начиная с версии 0.3.0) Скачайте модели восстановления лиц (по ссылкам ниже) и сохраните их в папку `extensions\sd-webui-comfyui\ComfyUI\custom_nodes\comfyui-reactor-node\models\facerestore_models`:
    - CodeFormer: https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
    - GFPGAN: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
11. Запустите SD WebUI и проверьте консоль на сообщение, что ReActor Node работает:
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/console_status_running.jpg?raw=true" alt="console_status_running" width="759"/>

12.   Перейдите во вкладку ComfyUI и найдите там ReActor Node внутри меню `ReActor` или через поиск:
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/webui-demo.png?raw=true" alt="webui-demo" width="100%"/>
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/search-demo.png?raw=true" alt="webui-demo" width="1043"/>

</details>

<details>
	<summary>Портативная версия <a href="https://github.com/comfyanonymous/ComfyUI">ComfyUI</a> для Windows</summary>

1. Сделайте следующее:
   - Установите [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) (Например, версию Community - этот шаг нужен для правильной компиляции библиотеки Insightface)
   - ИЛИ только [VS C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), выберите "Desktop Development with C++" в разделе "Workloads -> Desktop & Mobile"
   - ИЛИ если же вы не хотите устанавливать что-либо из вышеуказанного - выполните [данные шаги (раздел. I)](#insightfacebuild)
2. Перейдите в `ComfyUI\custom_nodes`
3. Откройте Консоль и выполните `git clone https://github.com/Gourieff/comfyui-reactor-node`
4. Запустите `install.bat`, дождитесь окончание установки
5. (Начиная с версии 0.3.0) Скачайте модели восстановления лиц (по ссылкам ниже) и сохраните их в папку `ComfyUI\models\facerestore_models`:
   - CodeFormer: https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
   - GFPGAN: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
6. Запустите ComfyUI и найдите ReActor Node внутри меню `ReActor` или через поиск

</details>

<a name="usage">

## Использование

Вы можете найти ноды ReActor внутри меню `ReActor` или через поиск (достаточно ввести "ReActor" в поисковой строке)

Список нодов:
- ReActorFaceSwap (Основной нод)
- ReActorLoadFaceModel (Load Face Model)
- ReActorSaveFaceModel (Save Face Model)

Соедините все необходимые слоты (slots) и запустите очередь (query).

### Входы основного Нода

- `input_image` - это изображение, на котором надо поменять лицо или лица (целевое изображение, аналог "target image" в версии для SD WebUI);
  - Поддерживаемые ноды: "Load Image", "Load Video" или любые другие ноды предоставляющие изображение в качестве выхода;
- `source_image` - это изображение с лицом или лицами для замены (изображение-источник, аналог "source image" в версии для SD WebUI);
  - Поддерживаемые ноды: "Load Image";
- `face_model` - это вход для выхода с нода "Load Face Model" или другого нода ReActor для загрузки модели лица (face model или face embedding), которое вы создали ранее через нод "Save Face Model";
  - Поддерживаемые ноды: "Load Face Model";

### Выходы основного Нода

- `IMAGE` - выход с готовым изображением (результатом);
  - Поддерживаемые ноды: любые ноды с изображением на входе;
- `FACE_MODEL` - выход, предоставляющий модель лица, построенную в ходе замены;
  - Поддерживаемые ноды: "Save Face Model", "ReActor";

### Восстановление лиц

Начиная с версии 0.3.0 ReActor Node имеет встроенное восстановление лиц.<br>Скачайте нужные вам модели (см. инструкцию по [Установке](#installation)) и выберите одну из них, чтобы улучшить качество финального лица.

### Индексы Лиц (Face Indexes)

ReActor определяет лица на изображении в следующей последовательности:<br>слева-направо, сверху-вниз.

Если вам нужно заменить определенное лицо, вы можете указать индекс для исходного (source, с лицом) и входного (input, где будет замена лица) изображений.

Индекс первого обнаруженного лица - 0.

Вы можете задать индексы в том порядке, который вам нужен.<br>
Например: 0,1,2 (для Source); 1,0,2 (для Input).<br>Это означает, что: второе лицо из Input (индекс = 1) будет заменено первым лицом из Source (индекс = 0) и так далее.

### Определение Пола

Вы можете обозначить, какой пол нужно определять на изображении.<br>
ReActor заменит только то лицо, которое удовлетворяет заданному условию.

### Модели Лиц
Начиная с версии 0.4.0, вы можете сохранять модели лиц как файлы "safetensors" (хранятся в папке `ComfyUI\models\reactor\faces`) и загружать их в ReActor, реализуя разные сценарии использования, а также  храня супер легкие модели лиц, которые вы чаще всего используете.

Чтобы новые модели появились в списке моделей нода "Load Face Model" - обновите страницу of с ComfyUI.<br>
(Рекомендую использовать ComfyUI Manager - иначе ваше воркфлоу может быть потеряно после перезагрузки страницы, если вы не сохранили его).

<a name="troubleshooting">

## Устранение проблем

<a name="insightfacebuild">

### **I. (Для пользователей Windows) Если вы до сих пор не можете установить пакет Insightface по каким-то причинам или же просто не желаете устанавливать Visual Studio или VS C++ Build Tools - сделайте следующее:**

1. (ComfyUI Portable) Находясь в корневой директории, проверьте версию Python:<br>запустите CMD и выполните `python_embeded\python.exe -V`<br>Вы должны увидеть версию или 3.10, или 3.11, или 3.12
2. Скачайте готовый пакет Insightface [для версии 3.10](https://github.com/Gourieff/sd-webui-reactor/raw/main/example/insightface-0.7.3-cp310-cp310-win_amd64.whl) или [для 3.11](https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl) (если на предыдущем шаге вы увидели 3.11) или [для 3.12](https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp312-cp312-win_amd64.whl) (если на предыдущем шаге вы увидели 3.12) и сохраните его в корневую директорию stable-diffusion-webui (A1111 или SD.Next) - туда, где лежит файл "webui-user.bat" -ИЛИ- в корневую директорию ComfyUI, если вы используете ComfyUI Portable
3. Из корневой директории запустите:
   - (SD WebUI) CMD и `.\venv\Scripts\activate`
   - (ComfyUI Portable) CMD
4. Обновите PIP:
   - (SD WebUI) `python -m pip install -U pip`
   - (ComfyUI Portable) `python_embeded\python.exe -m pip install -U pip`
5. Затем установите Insightface:
   - (SD WebUI) `pip install insightface-0.7.3-cp310-cp310-win_amd64.whl` (для 3.10) или `pip install insightface-0.7.3-cp311-cp311-win_amd64.whl` (для 3.11) или `pip install insightface-0.7.3-cp312-cp312-win_amd64.whl` (for 3.12)
   - (ComfyUI Portable) `python_embeded\python.exe -m pip install insightface-0.7.3-cp310-cp310-win_amd64.whl` (для 3.10) или `python_embeded\python.exe -m pip install insightface-0.7.3-cp311-cp311-win_amd64.whl` (для 3.11) или `python_embeded\python.exe -m pip install insightface-0.7.3-cp312-cp312-win_amd64.whl` (for 3.12)
6. Готово!

### **II. "AttributeError: 'NoneType' object has no attribute 'get'"**

Эта ошибка появляется, если что-то не так с файлом модели `inswapper_128.onnx`

Скачайте вручную по ссылке [отсюда](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx)
и сохраните в директорию `ComfyUI\models\insightface`, заменив имеющийся файл

### **III. "reactor.execute() got an unexpected keyword argument 'reference_image'"**

Это означает, что поменялось обозначение входных точек (input points) всвязи с последним обновлением<br>
Удалите из вашего рабочего пространства имеющийся ReActor Node и добавьте его снова

### **IV. ControlNet Aux Node IMPORT failed - при использовании совместно с нодом ReActor**

1. Закройте или остановите ComfyUI сервер, если он запущен
2. Перейдите в корневую папку ComfyUI, откройте консоль CMD и выполните следующее:
   - `python_embeded\python.exe -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless`
   - `python_embeded\python.exe -m pip install opencv-python==4.7.0.72`
3. Готово!

<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/reactor-w-controlnet.png?raw=true" alt="reactor+controlnet" />

### **V. "ModuleNotFoundError: No module named 'basicsr'" или "subprocess-exited-with-error" при установке пакета future-0.18.3**

- Скачайте https://github.com/Gourieff/Assets/raw/main/comfyui-reactor-node/future-0.18.3-py3-none-any.whl<br>
- Скопируйте файл в корневую папку ComfyUI и выполните в консоли:

      python_embeded\python.exe -m pip install future-0.18.3-py3-none-any.whl

- Затем:

      python_embeded\python.exe -m pip install basicsr

### **VI. "fatal: fetch-pack: invalid index-pack output" при исполнении команды `git clone`"**

Попробуйте клонировать репозиторий с параметром `--depth=1` (только последний коммит):

     git clone --depth=1 https://github.com/Gourieff/comfyui-reactor-node

Затем вытяните оставшееся (если требуется):

     git fetch --unshallow

<a name="updating">

## Обновление

Положите .bat или .sh скрипт из [данного репозитория](https://github.com/Gourieff/sd-webui-extensions-updater) в папку `ComfyUI\custom_nodes` и запустите, когда желаете обновить ComfyUI и Ноды

<a name="disclaimer">

## Ответственность

Это программное обеспечение призвано стать продуктивным вкладом в быстрорастущую медиаиндустрию на основе генеративных сетей и искусственного интеллекта. Данное ПО поможет художникам в решении таких задач, как анимация собственного персонажа или использование персонажа в качестве модели для одежды и т.д.

Разработчики этого программного обеспечения осведомлены о возможных неэтичных применениях и обязуются принять против этого превентивные меры. Мы продолжим развивать этот проект в позитивном направлении, придерживаясь закона и этики.

Подразумевается, что пользователи этого программного обеспечения будут использовать его ответственно, соблюдая локальное законодательство. Если используется лицо реального человека, пользователь обязан получить согласие заинтересованного лица и четко указать, что это дипфейк при размещении контента в Интернете. **Разработчики и Со-авторы данного программного обеспечения не несут ответственности за действия конечных пользователей.**

Используя данное расширение, вы соглашаетесь не создавать материалы, которые:
- нарушают какие-либо действующие законы тех или иных государств или международных организаций;
- причиняют какой-либо вред человеку или лицам;
- пропагандируют любую информацию (как общедоступную, так и личную) или изображения (как общедоступные, так и личные), которые могут быть направлены на причинение вреда;
- используются для распространения дезинформации;
- нацелены на уязвимые группы людей.

Данное программное обеспечение использует предварительно обученные модели `buffalo_l` и `inswapper_128.onnx`, представленные разработчиками [InsightFace](https://github.com/deepinsight/insightface/). Эти модели распространяются при следующих условиях:

[Перевод из текста лицензии insighface](https://github.com/deepinsight/insightface/tree/master/python-package): Предварительно обученные модели InsightFace доступны только для некоммерческих исследовательских целей. Сюда входят как модели с автоматической загрузкой, так и модели, загруженные вручную.

Пользователи данного программного обеспечения должны строго соблюдать данные условия использования. Разработчики и Со-авторы данного программного продукта не несут ответственности за неправильное использование предварительно обученных моделей InsightFace.

Обратите внимание: если вы собираетесь использовать это программное обеспечение в каких-либо коммерческих целях, вам необходимо будет обучить свои собственные модели или найти модели, которые можно использовать в коммерческих целях.

### Хэш файлов моделей

#### Безопасные для использования модели имеют следующий хэш:

inswapper_128.onnx
```
MD5:a3a155b90354160350efd66fed6b3d80
SHA256:e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af
```

1k3d68.onnx

```
MD5:6fb94fcdb0055e3638bf9158e6a108f4
SHA256:df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc
```

2d106det.onnx

```
MD5:a3613ef9eb3662b4ef88eb90db1fcf26
SHA256:f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf
```

det_10g.onnx

```
MD5:4c10eef5c9e168357a16fdd580fa8371
SHA256:5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91
```

genderage.onnx

```
MD5:81c77ba87ab38163b0dec6b26f8e2af2
SHA256:4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb
```

w600k_r50.onnx

```
MD5:80248d427976241cbd1343889ed132b3
SHA256:4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43
```

**Пожалуйста, сравните хэш, если вы скачиваете данные модели из непроверенных источников**

<a name="credits">

## Благодарности и авторы компонентов

<details>
	<summary><a>Нажмите, чтобы посмотреть</a></summary>

<br>

|файл|источник|лицензия|
|----|--------|--------|
|[buffalo_l.zip](https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/buffalo_l.zip) | [DeepInsight](https://github.com/deepinsight/insightface) | ![license](https://img.shields.io/badge/license-non_commercial-red) |
| [codeformer-v0.1.0.pth](https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/facerestore_models/codeformer-v0.1.0.pth) | [sczhou](https://github.com/sczhou/CodeFormer) | ![license](https://img.shields.io/badge/license-non_commercial-red) |
| [GFPGANv1.3.pth](https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/facerestore_models/GFPGANv1.3.pth) | [TencentARC](https://github.com/TencentARC/GFPGAN) | ![license](https://img.shields.io/badge/license-Apache_2.0-green.svg) |
| [GFPGANv1.4.pth](https://huggingface.co/datasets/Gourieff/ReActor/blob/main/models/facerestore_models/GFPGANv1.4.pth) | [TencentARC](https://github.com/TencentARC/GFPGAN) | ![license](https://img.shields.io/badge/license-Apache_2.0-green.svg) |
| [inswapper_128.onnx](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx) | [DeepInsight](https://github.com/deepinsight/insightface) | ![license](https://img.shields.io/badge/license-non_commercial-red) |
| [inswapper_128_fp16.onnx](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx) | [Hillobar](https://github.com/Hillobar/Rope) | ![license](https://img.shields.io/badge/license-non_commercial-red) |

[BasicSR](https://github.com/XPixelGroup/BasicSR) - [@XPixelGroup](https://github.com/XPixelGroup) <br>
[facexlib](https://github.com/xinntao/facexlib) - [@xinntao](https://github.com/xinntao) <br>

[@s0md3v](https://github.com/s0md3v), [@henryruhs](https://github.com/henryruhs) - оригинальное приложение Roop <br>
[@ssitu](https://github.com/ssitu) - первая версия расширения с поддержкой ComfyUI [ComfyUI_roop](https://github.com/ssitu/ComfyUI_roop)

</details>

<a name="note">

### Обратите внимание!

**Если у вас возникли какие-либо ошибки при очередном использовании Нода ReActor - не торопитесь открывать Issue, для начала попробуйте удалить текущий Нод из вашего рабочего пространства и добавить его снова**

**ReActor Node периодически получает обновления, появляются новые функции, из-за чего имеющийся Нод может работать с ошибками или не работать вовсе**
