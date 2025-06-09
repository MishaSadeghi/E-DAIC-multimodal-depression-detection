# Harnessing Multimodal Approaches for Depression Detection (https://www.nature.com/articles/s44184-024-00112-8)

This repository contains the official source code and implementation for the paper: **"Harnessing multimodal approaches for depression detection using large language models and facial expressions."**

## About The Project

This project explores a novel multimodal approach for detecting depression by combining features extracted from facial expressions and textual data processed by Large Language Models (LLMs). The goal is to build a robust and accurate model that leverages the strengths of both modalities to improve upon traditional methods.

---

## Getting Started

To get a local copy up and running, please follow these steps.

### Prerequisites

This project requires Python 3.8+ and the dependencies listed in the requirements file.

*   Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/MishaSadeghi/E-DAIC-multimodal-depression-detection.git
    ```
2.  Navigate to the project directory
    ```sh
    cd E-DAIC-multimodal-depression-detection
    ```

---

## Usage

This project is organized into a modular pipeline to ensure reproducibility. The main steps involve setting up the environment, preparing the data, and running a sequence of training scripts for the different models described in the paper.

### Project Structure

The repository is organized as follows:

```
E-DAIC/
│
├── data/                    # Should contain the DAIC-WOZ dataset structured as described below
│   ├── DAIC_openface_features/
│   │   ├── train/
│   │   └── dev/
│   │   └── test/
│   │   └── labels/
│   │   └── train_split.csv
│   │   └── dev_split.csv
│   │   └── test_split.csv
│   └── labels/
│       ├── train_split.csv
│       ├── dev_split.csv
│       └── test_split.csv
│
├── notebooks/               # Original experimental notebooks (for reference)
│
├── scripts/                 # Main executable scripts for training models
│   ├── train_video_model.py
│   ├── train_text_model.py
│   ├── train_svr_on_text_features.py
│   └── train_multimodal_model.py
│
├── src/                     # Source code for data loaders and models
│   ├── data/
│   │   ├── video_data_loader.py
│   │   ├── text_data_loader.py
│   │   └── multimodal_data_loader.py
│   └── models/
│       ├── video_model.py
│       └── text_model.py
│
├── trained_models/          # Output directory for saved models and results
│
├── .env                     # For storing API keys (ignored by git)
├── requirements.txt
└── README.md
```

### 1. Environment Setup

**API Key for GPT Completions:**

The text processing pipeline uses GPT to generate the completions. To use this feature, you must provide an OpenAI API key.

1.  Create a file named `.env` in the root of the project directory (`E-DAIC-multimodal-depression-detection/`).
2.  Add your OpenAI API key to this file:
    ```
    OPENAI_API_KEY='your_api_key_here'
    ```
    The scripts are configured to load this key automatically. This file is included in `.gitignore` and will not be tracked by version control.

### 2. Data Preparation

This project uses the **Extended DAIC** dataset. You must download it from the official source (https://dcapswoz.ict.usc.edu/) and structure it as follows inside the `data/` directory:

-   **`data/DAIC_openface_features/`**: This directory should contain the extracted OpenFace 2.1.0 features (`*_OpenFace2.1.0_Pose_gaze_AUs.csv`) for each participant, split into `train`, `dev`, and `test` subdirectories.
-   **`data/labels/`**: This directory should contain the `train_split.csv`, `dev_split.csv`, and `test_split.csv` files, which map participant IDs to their PHQ-8 scores and binary depression labels.

### 3. Reproducing the Paper's Results

The models should be trained in the following order. The scripts are designed to be run from the root of the project directory.

**Step 1: Train the Video Model (LSTM)**

This script trains the LSTM model on sequential OpenFace features and saves the best model based on validation performance.

```sh
python scripts/train_video_model.py \
    --data_dir data/ \
    --output_dir trained_models/video_model/
```

**Step 2: Train the Text-based Models**

Two models are trained on the textual data:
a) A fine-tuned DepRoBERTa model.
b) An SVR model trained on the features extracted from the DepRoBERTa model and the GPT-generated completions. These scripts will also generate the feature CSVs required for the final multimodal model.

```sh
# a) Fine-tune the DepRoBERTa model and extract features
python scripts/train_text_model.py \
    --data_dir data/ \
    --output_dir trained_models/text_model/

# b) Train the SVR on DepRoBERTa + Completions features
# This script generates the completion features using the OpenAI API.
# Ensure you have your OPENAI_API_KEY in the .env file.
python scripts/train_svr_on_text_features.py \
    --data_dir data/ \
    --DepRoBERTa_feature_dir trained_models/text_model/ \
    --output_dir trained_models/text_svr_model/
```

**Step 3: Train the Final Multimodal Model**

This script brings everything together. It loads the trained video model to extract features, loads the text and completions features (generated in the previous step), and trains a final SVR fusion model.

```sh
python scripts/train_multimodal_model.py \
    --video_feature_dir data/ \
    --text_feature_dir trained_models/text_svr_model/ \
    --video_model_path trained_models/video_model/best_video_model.pth \
    --output_dir trained_models/multimodal_model/
```

---

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## License

Distributed under the MIT License.


---
## Citation

If you use this code in your research, please consider citing the following papers:
```
@article{sadeghi2024harnessing,
  title={Harnessing multimodal approaches for depression detection using large language models and facial expressions},
  author={Sadeghi, Misha and Richer, Robert and Egger, Bernhard and Schindler-Gmelch, Lena and Rupp, Lydia Helene and Rahimi, Farnaz and Berking, Matthias and Eskofier, Bjoern M},
  journal={npj Mental Health Research},
  volume={3},
  number={1},
  pages={66},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
```
@inproceedings{sadeghi2023exploring,
  title={Exploring the capabilities of a language model-only approach for depression detection in text data},
  author={Sadeghi, Misha and Egger, Bernhard and Agahi, Reza and Richer, Robert and Capito, Klara and Rupp, Lydia Helene and Schindler-Gmelch, Lena and Berking, Matthias and Eskofier, Bjoern M},
  booktitle={2023 IEEE EMBS International Conference on Biomedical and Health Informatics (BHI)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
