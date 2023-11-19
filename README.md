<div align="center">

  <img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/ReActor_logo_red.png?raw=true" alt="logo" width="180px"/>

  ![Version](https://img.shields.io/badge/node_version-0.4.0_beta2-green?style=for-the-badge&labelColor=darkgreen)

  <sup>
  <font color=brightred>

  ## !!! [Important Update](#latestupdate) !!!<br>Don't forget to add the Node again in existing workflows
  
  </font>
  </sup>
  
  <a href='https://ko-fi.com/gourieff' target='_blank'><img height='33' src='https://storage.ko-fi.com/cdn/kofi3.png?v=3' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

  <hr>
  
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
[**What's new**](#latestupdate) | [**Installation**](#installation) | [**Usage**](#usage) | [**Troubleshooting**](#troubleshooting) | [**Updating**](#updating) | [**Disclaimer**](#disclaimer) | [**Note!**](#note)

---

</div>

<table>
  <tr>
    <td width="134px">
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
    <td width="144px">
      <a href="https://paypal.me/artgourieff" target="_blank">
        <img src="https://www.paypalobjects.com/digitalassets/c/website/logo/full-text/pp_fc_hl.svg" width="108" alt="Support Me via PayPal"/>
        <br>
        <sup>
          Donate to This Project
        </sup>
      </a>
    </td>
  </tr>
</table>

<div align="center">
  <img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/demo.gif?raw=true" alt="demo" width="100%"/>
</div>

<a name="latestupdate">

## What's new in the latest update

### 0.4.0 <sub><sup>BETA2</sup></sub>

- Ability to build and save face models directly from an image:

<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/0.4.0-whatsnew-03.jpg?raw=true" alt="0.4.0-whatsnew-01" width="903px"/>

- Both the inputs are optional now, just connect one of them according to your workflow; if both is connected - `image` has a priority.

### 0.4.0 <sub><sup>BETA1</sup></sub>

- Input "input_image" goes first now, it gives a correct bypass and also it is right to have the main input first;
- You can now save face models as "safetensors" files (`ComfyUI\models\reactor\faces`) and load them into ReActor implementing different scenarios and keeping super lightweight face models of the faces you use:

<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/0.4.0-whatsnew-01.jpg?raw=true" alt="0.4.0-whatsnew-01" width="100%"/>
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/0.4.0-whatsnew-02.jpg?raw=true" alt="0.4.0-whatsnew-02" width="100%"/>

- Different fixes making this extension better.

Thanks to everyone who finds bugs, suggests new features and supports this project!

## Installation

<details>
	<summary>SD WebUI: <a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/">AUTOMATIC1111</a> or <a href="https://github.com/vladmandic/automatic">SD.Next</a></summary>

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

1.  Go to the ComfyUI tab and find there ReActor Node inside the menu `ReActor` or by using a search:
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/webui-demo.png?raw=true" alt="webui-demo" width="100%"/>
<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/search-demo.png?raw=true" alt="webui-demo" width="1043"/>

</details>

<details>
	<summary>Standalone (Portable) <a href="https://github.com/comfyanonymous/ComfyUI">ComfyUI</a> for Windows</summary>

1. Do the following:
   - Install [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) (Community version - you need this step to build Insightface)
   - OR only [VS C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop Development with C++" under "Workloads -> Desktop & Mobile"
   - OR if you don't want to install VS or VS C++ BT - follow [this steps (sec. I)](#insightfacebuild)
2. Go to the `ComfyUI\custom_nodes` directory
3. Open Console and run `git clone https://github.com/Gourieff/comfyui-reactor-node`
4. Run `install.bat`
5. (From the version 0.3.0) Download facerestorers models from the links below and put them into the `ComfyUI\models\facerestore_models` directory:
   - CodeFormer: https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
   - GFPGAN: https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
6. Run ComfyUI and find there ReActor Node inside the menu `ReActor` or by using a search

</details>

## Usage

Just connect all required nodes and run the query.

### Main Node Inputs

- `input_image` - is an image to be processed (target image, analog of "target image" in the SD WebUI extension);
  - Supported Nodes: "Load Image", "Load Video" or any other nodes providng images as an output;
- `source_image` - is an image with a face or faces to swap in the `input_image` (source image, analog of "source image" in the SD WebUI extension);
  - Supported Nodes: "Load Image";
- `face_model` - is the input for the "Load Face Model" Node or another ReActor node to provide a face model file (face embedding) you created earlier via the "Save Face Model" Node;
  - Supported Nodes: "Load Face Model";

### Main Node Outputs

- `IMAGE` - is an output with the resulted image;
  - Supported Nodes: any nodes which have images as an input;
- `FACE_MODEL` - is an output providng a source face's model being built during the swapping process;
  - Supported Nodes: "Save Face Model", "ReActor";

### Face Restoration

Since version 0.3.0 ReActor Node has a buil-in face restoration.<br>Just download the models you want (see [Installation](#installation) instruction) and select one of them to restore the resulting face(s) during the faceswap. It will enhance face details and make your result more accurate.

### Face Indexes

ReActor detects faces in images in the following order:<br>left->right, top->bottom

And if you need to specify faces, you can set indexes for source and input images.

Index of the first detected face is 0.

You can set indexes in the order you need.<br>
E.g.: 0,1,2 (for Source); 1,0,2 (for Input).<br>This means: the second Input face (index = 1) will be swapped by the first Source face (index = 0) and so on.

### Genders

You can specify the gender to detect in images.<br>
ReActor will swap a face only if it meets the given condition.

### Face Models

Since version 0.4.0 you can save face models as "safetensors" files (stored in `ComfyUI\models\reactor\faces`) and load them into ReActor implementing different scenarios and keeping super lightweight face models of the faces you use.

To make new models appear in the list of the "Load Face Model" Node - just refresh the page of your ComfyUI web application.<br>
(I recommend you to use ComfyUI Manager - otherwise you workflow can be lost after you refresh the page if you didn't save it before that).

## Troubleshooting

<a name="insightfacebuild">

### **I. (For Windows users) If you still cannot build Insightface for some reasons or just don't want to install Visual Studio or VS C++ Build Tools - do the following:**

1. (ComfyUI Portable) From the root folder check the version of Python:<br>run CMD and type `python_embeded\python.exe -V`
2. Download prebuilt Insightface package [for Python 3.10](https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp310-cp310-win_amd64.whl) or [for Python 3.11](https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-cp311-cp311-win_amd64.whl) (if in the previous step you see 3.11) and put into the stable-diffusion-webui (A1111 or SD.Next) root folder (where you have "webui-user.bat" file) or into ComfyUI root folder if you use ComfyUI Portable
3. From the root folder run:
   - (SD WebUI) CMD and `.\venv\Scripts\activate`
   - (ComfyUI Portable) run CMD
4. Then update your PIP:
   - (SD WebUI) `python -m pip install -U pip`
   - (ComfyUI Portable) `python_embeded\python.exe -m pip install -U pip`
5. Then install Insightface:
   - (SD WebUI) `pip install insightface-0.7.3-cp310-cp310-win_amd64.whl` (for 3.10) or `pip install insightface-0.7.3-cp311-cp311-win_amd64.whl` (for 3.11)
   - (ComfyUI Portable) `python_embeded\python.exe -m pip install insightface-0.7.3-cp310-cp310-win_amd64.whl` (for 3.10) or `python_embeded\python.exe -m pip install insightface-0.7.3-cp311-cp311-win_amd64.whl` (for 3.11)
6. Enjoy!

### **II. "AttributeError: 'NoneType' object has no attribute 'get'"**

This error may occur if there's smth wrong with the model file `inswapper_128.onnx`

Try to download it manually from [here](https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx)
and put it to the `ComfyUI\models\insightface` replacing existing one

### **III. "reactor.execute() got an unexpected keyword argument 'reference_image'"**

This means that input points have been changed with the latest update<br>
Remove the current ReActor Node from your workflow and add it again

### **IV. ControlNet Aux Node IMPORT failed error when using with ReActor Node**

1. Close ComfyUI if it runs
2. Go to the ComfyUI root folder, open CMD there and run:
   - `python_embeded\python.exe -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless`
   - `python_embeded\python.exe -m pip install opencv-python==4.7.0.72`
3. That's it!

<img src="https://github.com/Gourieff/Assets/blob/main/comfyui-reactor-node/uploads/reactor-w-controlnet.png?raw=true" alt="reactor+controlnet" />

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

This software utilizes the pre-trained models `buffalo_l` and `inswapper_128.onnx`, which are provided by [InsightFace](https://github.com/deepinsight/insightface/). These models are included under the following conditions:

[From insighface licence](https://github.com/deepinsight/insightface/tree/master/python-package): The InsightFace’s pre-trained models are available for non-commercial research purposes only. This includes both auto-downloading models and manually downloaded models.

Users of this software must strictly adhere to these conditions of use. The developers and maintainers of this software are not responsible for any misuse of InsightFace’s pre-trained models.

Please note that if you intend to use this software for any commercial purposes, you will need to train your own models or find models that can be used commercially.

### Models Hashsum

#### Safe-to-use models have the folowing hash:

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

**Please check hashsums if you download these models from unverified (or untrusted) sources**

<a name="note">

### Note!

**If you encounter any errors when you use ReActor Node - don't rush to open an issue, first try to remove current ReActor node in your workflow and add it again**

**ReActor Node gets updates from time to time, new functions appears and old node can work with errors or not work at all**
