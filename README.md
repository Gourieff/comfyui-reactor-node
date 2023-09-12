<div align="center">

  <img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/ReActor_logo_red.png?raw=true" alt="logo" width="180px"/>

  ![Version](https://img.shields.io/badge/node_version-0.3.0_beta1-green?style=for-the-badge&labelColor=darkgreen)<hr>
  [![Commit activity](https://img.shields.io/github/commit-activity/t/Gourieff/comfyui-reactor-node/main?cacheSeconds=0)](https://github.com/Gourieff/comfyui-reactor-node/commits/main)
  ![Last commit](https://img.shields.io/github/last-commit/Gourieff/comfyui-reactor-node/main?cacheSeconds=0)
  [![Opened issues](https://img.shields.io/github/issues/Gourieff/comfyui-reactor-node?color=red)](https://github.com/Gourieff/comfyui-reactor-node/issues?cacheSeconds=0)
  [![Closed issues](https://img.shields.io/github/issues-closed/Gourieff/comfyui-reactor-node?color=green&cacheSeconds=0)](https://github.com/Gourieff/comfyui-reactor-node/issues?q=is%3Aissue+is%3Aclosed)
  ![License](https://img.shields.io/github/license/Gourieff/comfyui-reactor-node)

  English | [Русский](/README_RU.md)

# ReActor Node for ComfyUI

</div>

### The Fast and Simple Face Swap Extension Node for ComfyUI, based on [ReActor](https://github.com/Gourieff/sd-webui-reactor) SD-WebUI Face Swap Extension

> This Node goes without NSFW filter (uncensored, use it on your own [responsibility](#disclaimer)) 

<div align="center">

---
[**Installation**](#installation) | [**Usage**](#usage) | [**Troubleshooting**](#troubleshooting) | [**Updating**](#updating) | [**Disclaimer**](#disclaimer) | [**Note!**](#note)

---

</div>

<table>
  <tr>
    <td width="144px">
      <a href="https://boosty.to/artgourieff" target="_blank">
        <img src="https://lovemet.ru/www/boosty.jpg" width="108" alt="Support Me on Boosty"/>
        <br>
        <sup>
          Support This Project
        </sup>
      </a>
    </td>
    <td>
      ReActor Node is an extension for ComfyUI that allows a very easy and accurate face-replacement (face swap) in images
    </td>
  </tr>
</table>

<div align="center">
  <img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/demo.gif?raw=true" alt="demo" width="100%"/>
</div>

## Installation

[SD WebUI](#sdwebui) | [Standalone ComfyUI](#standalone) (recommended)

<a name="sdwebui">If you use [AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/) or [SD.Next](https://github.com/vladmandic/automatic)

1. Close (stop) your SD-WebUI/Comfy Server if it's running
2. (For Windows Users):
   - Install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) (Community version - you need this step to build Insightface)
   - OR only [VS C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop Development with C++" under "Workloads -> Desktop & Mobile"
   - OR if you don't want to install VS or VS C++ BT - follow [this steps (sec. I)](#insightfacebuild)
3. Go to the `extensions\sd-webui-comfyui\ComfyUI\custom_nodes`
4. Open Console or Terminal and run `git clone https://github.com/Gourieff/comfyui-reactor-node`
5. Go to the SD WebUI root folder, open Console or Terminal and run (Windows users)`.\venv\Scripts\activate` or (Linux/MacOS)`venv/bin/activate`
6. `python -m pip install -U pip`
7. `cd extensions\sd-webui-comfyui\ComfyUI\custom_nodes\comfyui-reactor-node`
8. `python install.py`
9.  Please, wait until the installation process will be finished
10. (From the version 0.3.0) Download facerestorers models from the links below and put them into the `extensions\sd-webui-comfyui\ComfyUI\custom_nodes\comfyui-reactor-node\models\facerestore_models` directory:
    - CodeFormer: https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
    - GFPGAN: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
11. Run SD WebUI and check console for the message that ReActor Node is running:
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/console_status_running.jpg?raw=true" alt="console_status_running" width="759"/>

12.   Go to the ComfyUI tab and find there ReActor Node inside the menu `image/postprocessing` or by using a search:
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/webui-demo.png?raw=true" alt="webui-demo" width="100%"/>
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/search-demo.png?raw=true" alt="webui-demo" width="1043"/>

<a name="standalone">If you use Standalone [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for Windows

1. (For Windows Users):
   - Install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) (Community version - you need this step to build Insightface)
   - OR only [VS C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop Development with C++" under "Workloads -> Desktop & Mobile"
   - OR if you don't want to install VS or VS C++ BT - follow [this steps (sec. I)](#insightfacebuild)
2. Go to the `ComfyUI\custom_nodes` directory
3. Open Console and run `git clone https://github.com/Gourieff/comfyui-reactor-node`
4. Run `install.bat`
5. (From the version 0.3.0) Download facerestorers models from the links below and put them into the `ComfyUI\custom_nodes\comfyui-reactor-node\models\facerestore_models` directory:
   - CodeFormer: https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
   - GFPGAN: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
6. Run ComfyUI and find there ReActor Node inside the menu `image/postprocessing` or by using a search

## Usage

Just connect all required nodes and run the query.

### Face Restoration

From the version 0.3.0 ReActor Node has a buil-in face restoration.<br>Just download the models you want (see [Installation](#installation) instruction) and select one of them to restore the resulting face(s) during the faceswap. It will enhance face details and make your result more accurate.

### Face Indexes

ReActor detects faces in images in the following order:<br>left->right, top->bottom

And if you need to specify faces, you can set indexes for source and input images.

Index of the first detected face is 0.

You can set indexes in the order you need.<br>
E.g.: 0,1,2 (for Source); 1,0,2 (for Input).<br>This means: the second Input face (index = 1) will be swapped by the first Source face (index = 0) and so on.

### Genders

You can specify the gender to detect in images.<br>
ReActor will swap a face only if it meets the given condition.

## Troubleshooting

<a name="insightfacebuild">

### **I. (For Windows users) If you still cannot build Insightface for some reasons or just don't want to install Visual Studio or VS C++ Build Tools - do the following:**

1. Download and put [prebuilt Insightface package](https://github.com/Gourieff/sd-webui-reactor/raw/main/example/insightface-0.7.3-cp310-cp310-win_amd64.whl) into the stable-diffusion-webui (A1111 or SD.Next) root folder (where you have "webui-user.bat" file) or into ComfyUI root folder if you use ComfyUI Portable
2. From the root folder run:
   - (SD WebUI) CMD and `.\venv\Scripts\activate`
   - (ComfyUI Portable) run CMD
3. Then update your PIP:
   - (SD WebUI) `python -m pip install -U pip`
   - (ComfyUI Portable) `python_embeded\python.exe -m pip install -U pip`
4. Then install Insightface:
   - (SD WebUI) `pip install insightface-0.7.3-cp310-cp310-win_amd64.whl`
   - (ComfyUI Portable) `python_embeded\python.exe -m pip install insightface-0.7.3-cp310-cp310-win_amd64.whl`
5. Enjoy!

### **II. "AttributeError: 'NoneType' object has no attribute 'get'"**

This error may occur if there's smth wrong with the model file `inswapper_128.onnx`

Try to download it manually from [here](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx)
and put it to the `ComfyUI\custom_nodes\comfyui-reactor-node\models\insightface` replacing existing one

### **III. "reactor.execute() got an unexpected keyword argument 'reference_image'"**

This means that input points have been changed with the latest update<br>
Remove the current ReActor Node from your workflow and add it again

## Updating

Just put .bat or .sh script from this [Repo](https://github.com/Gourieff/sd-webui-extensions-updater) to the `ComfyUI\custom_nodes` directory and run it when you need to check for updates

### Disclaimer

This software is meant to be a productive contribution to the rapidly growing AI-generated media industry. It will help artists with tasks such as animating a custom character or using the character as a model for clothing etc.

The developers of this software are aware of its possible unethical applicaitons and are committed to take preventative measures against them. We will continue to develop this project in the positive direction while adhering to law and ethics.

Users of this software are expected to use this software responsibly while abiding the local law. If face of a real person is being used, users are suggested to get consent from the concerned person and clearly mention that it is a deepfake when posting content online. **Developers and Contributors of this software are not responsible for actions of end-users.**

By using this extension you are agree not to create any content that:
- violates any laws;
- causes any harm to a person or persons;
- propogates (spreads) any information (both public or personal) or images (both public or personal) which could be meant for harm;
- spreads misinformation;
- targets vulnerable groups of people.

<a name="note">

### Note!

**If you encounter any errors when you use ReActor Node - don't rush to open an issue, first try to remove current ReActor node in your workflow and add it again**

**ReActor Node gets updates from time to time, new functions appears and old node can work with errors or not work at all**
