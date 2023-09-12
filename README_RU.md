<div align="center">

  <img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/ReActor_logo_red.png?raw=true" alt="logo" width="180px"/>

  ![Version](https://img.shields.io/badge/версия_нода-0.3.0_beta1-green?style=for-the-badge&labelColor=darkgreen)<hr>
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
[**Установка**](#installation) | [**Использование**](#usage) | [**Устранение проблем**](#troubleshooting) | [**Обновление**](#updating) | [**Ответственность**](#disclaimer)

---

</div>

<table>
  <tr>
    <td width="144px">
      <a href="https://boosty.to/artgourieff" target="_blank">
        <img src="https://lovemet.ru/www/boosty.jpg" width="108" alt="Support Me on Boosty"/>
        <br>
        <sup>
          Поддержать проект
        </sup>
      </a>
    </td>
    <td>
      ReActor Node это расширение для ComfyUI, которое позволяет делать простую и точную замену лиц на изображениях
    </td>
  </tr>
</table>

<div align="center">
  <img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/demo.gif?raw=true" alt="logo" width="100%"/>
</div>

<a name="installation">

## Установка

[SD WebUI](#sdwebui) | [Портативный ComfyUI](#standalone) (рекомендуется)

<a name="sdwebui">Если вы используете [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) или [SD.Next](https://github.com/vladmandic/automatic)

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

12.   Перейдите во вкладку ComfyUI и найдите там ReActor Node внутри меню `image/postprocessing` или через поиск:
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/webui-demo.png?raw=true" alt="webui-demo" width="100%"/>
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/search-demo.png?raw=true" alt="webui-demo" width="1043"/>

<a name="standalone">Если вы используете портативную версию [ComfyUI](https://github.com/comfyanonymous/ComfyUI) для Windows

1. (Для пользователей Windows):
   - Установите [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) (Например, версию Community - этот шаг нужен для правильной компиляции библиотеки Insightface)
   - ИЛИ только [VS C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/), выберите "Desktop Development with C++" в разделе "Workloads -> Desktop & Mobile"
   - ИЛИ если же вы не хотите устанавливать что-либо из вышеуказанного - выполните [данные шаги (раздел. I)](#insightfacebuild)
2. Перейдите в `ComfyUI\custom_nodes`
3. Откройте Консоль и выполните `git clone https://github.com/Gourieff/comfyui-reactor-node`
4. Запустите `install.bat`, дождитесь окончание установки
5. (Начиная с версии 0.3.0) Скачайте модели восстановления лиц (по ссылкам ниже) и сохраните их в папку `ComfyUI\custom_nodes\comfyui-reactor-node\models\facerestore_models`:
   - CodeFormer: https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
   - GFPGAN: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
6. Запустите ComfyUI и найдите ReActor Node внутри меню `image/postprocessing` или через поиск

<a name="usage">

## Использование

Соедините все необходимые узлы (nodes) и запустите очередь (query).

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

<a name="troubleshooting">

## Устранение проблем

<a name="insightfacebuild">

### **I. (Для пользователей Windows) Если вы до сих пор не можете установить пакет Insightface по каким-то причинам или же просто не желаете устанавливать Visual Studio или VS C++ Build Tools - сделайте следующее:**

1. Скачайте готовый [пакет Insightface](https://github.com/Gourieff/sd-webui-reactor/raw/main/example/insightface-0.7.3-cp310-cp310-win_amd64.whl) и сохраните его в корневую директорию stable-diffusion-webui (A1111 или SD.Next) - туда, где лежит файл "webui-user.bat" -ИЛИ- в корневую директорию ComfyUI, если вы используете ComfyUI Portable
2. Из корневой директории запустите:
   - (SD WebUI) CMD и `.\venv\Scripts\activate`
   - (ComfyUI Portable) CMD
3. Обновите PIP:
   - (SD WebUI) `python -m pip install -U pip`
   - (ComfyUI Portable) `python_embeded\python.exe -m pip install -U pip`
4. Затем установите Insightface:
   - (SD WebUI) `pip install insightface-0.7.3-cp310-cp310-win_amd64.whl`
   - (ComfyUI Portable) `python_embeded\python.exe -m pip install insightface-0.7.3-cp310-cp310-win_amd64.whl`
5. Готово!

### **II. "AttributeError: 'NoneType' object has no attribute 'get'"**

Эта ошибка появляется, если что-то не так с файлом модели `inswapper_128.onnx`

Скачайте вручную по ссылке [отсюда](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx)
и сохраните в директорию `ComfyUI\custom_nodes\comfyui-reactor-node\models\insightface`, заменив имеющийся файл

### **III. "reactor.execute() got an unexpected keyword argument 'reference_image'"**

Это означает, что поменялось обозначение входных точек (input points) всвязи с последним обновлением<br>
Удалите из вашего рабочего пространства имеющийся ReActor Node и добавьте его снова

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

<a name="note">

### Обратите внимание!

**Если у вас возникли какие-либо ошибки при очередном использовании Нода ReActor - не торопитесь открывать Issue, для начала попробуйте удалить текущий Нод из вашего рабочего пространства и добавить его снова**

**ReActor Node периодически получает обновления, появляются новые функции, из-за чего имеющийся Нод может работать с ошибками или не работать вовсе**
