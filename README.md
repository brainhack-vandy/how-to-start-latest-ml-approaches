# how-to-start-latest-ml-approaches

Welcome to this session! 

The goal for this workshop is to bridge the gap between groundbreaking research and practical implementation, guiding you through the process of transitioning from reading influential papers to running and training the models they describe. During the workshop, we will together explore the `NeuroBOLT` repository here — a recent work presented at *NeurIPS 2024* — as an example, and run inference to translate raw EEG signals into corresponding fMRI ROI time series. This hands-on experience will provide you with the tools and knowledge needed to confidently explore and run state-of-the-art repositories of your choice in the future.  

## Getting Started

### Dependencies
**Notes**: GPU is not required for inference demo, but is required if you would like to train the model. All experiments done in the paper were conducted on single RTX A5000 GPU.

To run the demo, ensure you have the following:
- Python 3.9
- Required Python libraries (specified in requirements.txt, install them manually use pip/conda, or **follow the steps below**)

### Installation
1. Clone/download this repository to your machine and navigate to the directory.
2. We encourage you to use a `virtual environment` for this tutorial (and for all your projects).
   To do this, run the following commands in your terminal, it will create the environment in a folder named `eeg2fmri_env`
   ```bash
   cd how-to-start-latest-ml-approaches
   python3.9 -m venv eeg2fmri_env
   ```
   Then use the following command to activate the environment:
   ```bash
   # (Linux/MacOS)
   source eeg2fmri_env/bin/activate

   # (Windows)
   eeg2fmri_env\Scripts\activate
   ```
   Finally, you can install the required libraries using the command below:
   ```bash
   pip install -r requirements.txt  
   ```
3. Now you are all set! You can run the notebook either from here with the command or use conda:
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```
   Then, **open the `NeuroBOLT-inference_demo.ipynb` notebook to begin.**
   
### Download Dataset & Checkpoints
The sample data and model checkpoints can be downloaded from huggingface. You can either go to `https://huggingface.co/ssssssup/BHVU-EEG2fMRI` to download manually
or use the following command (ensure `huggingface_hub` library is installed in your env, if not, install it using `pip install huggingface_hub`):

```python
# In Python:
from huggingface_hub import snapshot_download
# Define your path
download_folder = "/path/to/your/custom/folder"
repo_dir = snapshot_download(repo_id="ssssssup/BHVU-EEG2fMRI", local_dir=download_folder)
```

Once downloaded, organize the dataset and pretrained weights as follows
```
BH-NeuroBOLT/
├── checkpoints/          # Store model weights here
├── data/
│   ├── EEG/              # Store EEG dataset files here
│   ├── fMRI_difumo/      # Store fMRI dataset files here
```

### Training model (GPU required)
The workshop will not cover model training. However, if you wish to train the model on your own, please follow the steps below:
1) (If you are using new data) Prepare your data using `making_difumo.py`
2) Run intra-subject prediction
   ```bash
   python main.py --batch_size 16 --finetune ./checkpoints/labram-base.pth --labels_roi Thalamus --dataset VU --train_test_mode intrascan --dataname sub11-scan01
   ```

## Reference
```bibtex
@inproceedings{
li2024neurobolt,
title={Neuro{BOLT}: Resting-state {EEG}-to-f{MRI} Synthesis with Multi-dimensional Feature Mapping},
author={Yamin Li and Ange Lou and Ziyuan Xu and Shengchao Zhang and Shiyu Wang and Dario J. Englot and Soheil Kolouri and Daniel Moyer and Roza G Bayrak and Catie Chang},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=y6qhVtFG77}
}
```
