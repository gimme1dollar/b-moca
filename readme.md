# B-MoCA

## A Benchmark for Evaluating Mobile Device Control Agents across Diverse Configurations

![main_figure](./asset/figures/main_figure.png)

B-MoCA can serve as a testbed for mobile device control agents across diverse configurations

## Setting

Setting B-MoCA requires installing Android emulators and B-MoCA environments

### Virtual environments
```
conda create -n bmoca python=3.10
```

```
echo "export BMOCA_HOME=/path/to/current/directory" >> ~/.bashrc
source ~/.bashrc
conda activate bmoca
```

```
python setup.py develop
```

```
mkdir lib
cd lib

# install android_env
git clone https://github.com/gimme1dollar/android_env
cd android_env
python setup.py install
cd ..

# install auto_ui
git clone https://github.com/gimme1dollar/auto_ui
cd auto_ui
pip install -r requirements.txt
# download Auto-UI pre-trained model from https://huggingface.co/cooelf/Auto-UI/tree/main into "asset/agent/Auto-UI-Base"
cd ..

# others
pip install openai google.generativeai wandb opencv-python Pillow matplotlib pyshine 

cd $BMOCA_HOME
```

### Android emulators (Android Studio)

https://www.notion.so/Environment-Setup-7c0fe37c623e46c8a89b6c24813255a4

B-MoCA is based on Android operating system. Installing Android studio is required to run each device environment in B-MoCA, as our APIs interact with Android emulators.

### B-MoCA environments

Each randomized environment in B-MoCA is defined as a snapshot of an AVD. The script **install.sh** installs AVDs and save snapshots, by manipulating the icon size/location, wallpaper, and theme. 


```
mkdir logs
bash install.sh
```
(Note: this takes approximately an hour)

After the installation, the screenshot of each environment is stored in ```logs/environment```.

Each name of emulator (i.e., virtual device) is "{device_name}\_{train/test}\_{id}".    
Each name of environment (i.e., snapshot) is "(mode)_env\_{snapshot_id}" (mode: either train or test)

During the experiments, our API (i.e., ```b_moca/environment/environment.py```) loads one of the snapshots randomly for training or evaluation.

Users can also configure own environments by modifying files in ```asset/environments```

## Experiments


### LLM Agent

#### Evaluation

```
python experiments/agent/gemini/evaluate.py #zero-shot
```
```
python experiments/agent/gpt4/evaluate.py #zero-shot
```


### MLLM Agent

#### Evaluation

```
python experiments/agent/gemini_vision/evaluate.py #zero-shot
```
```
python experiments/agent/gpt4v/evaluate.py #zero-shot
```


### VLUI Agent

#### Demo Collection

```
python experiments/dataset/collect.py
```

#### Training

```
python experiments/agent/bc/train.py
```

#### Evaluation

```
python experiments/agent/bc/evaluate.py --model_path="path_to_checkpoint"
```


## Reference
Some codes are referred from the related work below:
- AndroidEnv (<a href="https://github.com/google-deepmind/android_env">link</a>)
- AppAgent (<a href="https://github.com/mnotgod96/AppAgent">link</a>)

