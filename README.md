<h1 align="center">Comparing Deep Acoustic Feature Extraction Methods for Classification Performance and Investigating Gender Bias in Parkinson’s Disease Classification<span style="font-weight:normal"></h1>

Parkinson's Disease (PD) is one of the most prevalent neurodegenerative diseases in the world. Research shows that speech can be used to discriminate healthy people form PD patients. This paper investigates various nuances about speech as a marker for PD detection with machine learning. The model of this paper requires features extracted from speech data. This paper investigates performance differences between Wav2vec 2.0 and HuBERT. It was found that HuBERT, when isolated, performs better with an F1-score of 75.41% against 74.73%. Then for the Wav2vec 2.0 model it was researched which layer of the architecture causes optimal performance. It was found that using different layers results in only marginal improvements. Finally, gender bias is often overlooked. When investigating gender bias in the model, it was found that the classifier generalizes better to males voices, and performs better on male voices for the DDK task. Concluding that gender bias does play a role in the model.

This branch contains the code used to perform the layer probing and finetuning experiment. The code used for probing the layers of Wav2Vec 2.0 XLS-R can be found in scripts/runs/experiments/finetuned_cross_full/probe_layers.py.

## <a name="preparation"></a> Environment Preparation

The conda environment to run the experiments can be created the same way as for the original experiment pipeline:

```
conda create -n ssl-parkinson python=3.10
conda activate ssl-parkinson
pip install -r requirements.txt
```

The dataset can then be prepared by running the following commands, similar to the original code:
```
bash scripts/runs/dataset_preparation/gita.sh $DATASET_DIR $METADATA_PATH
bash scripts/runs/feature_extraction/gita.sh
```

## <a name="Layer probing"></a> Layer probing

To replicate the layer probing procedure, the command below can be used:
```
python scripts/runs/experiments/finetuned_cross_full/probe_layers.py \
       --norm-audios data/gita/norm_audios \
       --splits-file splits/gita/dataset.csv
```
Where --norm-audios is the folder containing normalized .wav files, and --split-file is the .csv file containing the information on the train/test folds.

## <a name="Finetuning"></a> Finetuning

Finetuning Wav2Vec 2.0 XLS-R can be done by using the following script with example parameters:
```
python scripts/runs/experiments/finetuned_cross_full/tune_backbone.py \
  --splits-dir splits/gita \                
  --audio-dir data/gita/norm_audios \       
  --unfreeze-layers 1 2 3 4 \
  --layer-n 2 \    
  --max-epochs 15 \
  --lr 1e-6 \
  --device cuda
```
Where unfreeze-layers is used to specify which layers are unfrozen during finetuning, layer-n is used to extract embeddings and validate classification performance, lr is the learning rate, and max-epochs and patience specify the maximum amount of epochs and the amount of epochs to wait before early stopping.

Extracting embeddings from a finetuned model can be done using
```
python scripts/feature_extraction/extract_wav2vec_feats_extended.py \
  --norm-audios data/gita/norm_audios
  --output-dir $dataset_dir/speech_features/wav2vec_finetuned
  --chunk-size 60
  --checkpoint checkpoints/xlsr_fold0
  --layer-n 2
```
Where chunk-size is the maximum length of a .wav file before it is split up into multiple chunks, checkpoint is the location of the finetuned model, and layer-n is the layer to extract embeddings from.

Extracting embeddings from the frozen Wav2Vec 2.0 XLS-R model can be performed using the code from the original authors:
```
python scripts/feature_extraction/extract_wav2vec_features.py --wav-dir $dataset_dir/norm_audios/ --output-dir $dataset_dir/speech_features/wav2vec/
```
Where --wav-dir contains the normalized .wav files, and --output-dir is the directory where the extracted embeddings are placed.

## <a name="classification"></a> Classification

The experimental to get the final results can be run by using the command:

```
bash scripts/runs/experiments/finetuned_cross_full/gita.sh
```

Likewise, performance evaluation for the final results can also be done in the same way as in the original code, by executing the following command and specifying the task to be evaluated:

```
python scripts/evaluation/overall_performance.py --exps-dir ./exps/gita/cross_full/$TASK/
```
Where --exps-dir is the directory containing the classification results.


Below is the ReadMe of the original source paper for reference:

<h1 align="center"><span style="font-weight:normal">Unveiling Interpretability in Self-Supervised Speech Representations for Parkinson’s Diagnosis 🗣️🎙️📝📊</h1>
  
<div align="center">
  
[D. Gimeno-Gómez](https://scholar.google.es/citations?user=DVRSla8AAAAJ&hl=en), [C. Botelho](https://scholar.google.com/citations?user=d-xmVlUAAAAJ&hl=en), [A. Pompili](https://scholar.google.pt/citations?user=ZiB_o6kAAAAJ&hl=en), [A. Abad](https://scholar.google.pt/citations?user=M5hzAIwAAAAJ&hl=en), [C.-D. Martínez-Hinarejos](https://scholar.google.es/citations?user=HFKXPH8AAAAJ&hl=en)
</div>

<div align="center">
  
[📘 Introduction](#intro) |
[🛠️ Data Preparation](#preparation) |
[🚀 Training and Evaluation](#training) |
[📖 Citation](#citation) |
[📝 License](#license)
</div>

## <a name="intro"></a> 📘 Introduction

<div align="center"> <img src="docs/figure1.png"  width="720"> </div>

**Abstract.** _Recent works in pathological speech analysis have increasingly relied on powerful self-supervised speech representations, leading to promising results. However, the complex, black-box nature of these embeddings and the limited research on their interpretability significantly restrict their adoption for clinical diagnosis. To address this gap, we propose a novel, interpretable framework specifically designed to support Parkinson’s Disease (PD) diagnosis. Through the design of simple yet effective cross-attention mechanisms for both embedding- and temporal-level
analysis, the proposed framework offers interpretability from two distinct but complementary perspectives. Experimental findings across five well-established speech benchmarks for PD detection demonstrate the framework’s capability to identify meaningful speech patterns within self-supervised representations for a wide range of assessment tasks. Fine-grained temporal analyses further underscore its potential to enhance the interpretability
of deep-learning pathological speech models, paving the way for the development of more transparent, trustworthy, and clinically applicable computer-assisted diagnosis systems in this domain. Moreover, in terms of classification accuracy, our method achieves results competitive with state-of-the-art approaches, while also demonstrating robustness in cross-lingual scenarios when applied to spontaneous speech production._ [📜 Arxiv Link](https://arxiv.org/abs/2412.02006) [📜 IEEE Link](https://ieeexplore.ieee.org/abstract/document/10877763)

## <a name="preparation"></a> 🛠️ Preparation

- Prepare the **conda environment** to run the experiments:

```
conda create -n ssl-parkinson python=3.10
conda activate ssl-parkinson
pip install -r requirements.txt
```

## <a name="training"></a> 🚀 Training and Evaluation

To train and evaluate our proposed framework, we should follow a pipeline consisting of multiple steps, including data preprocessing, dataset split, feature extraction, as well as the ultimate training and evaluation. As an example, we provide the scripts aimed to address our GITA corpus experiments:

```
bash scripts/runs/dataset_preparation/gita.sh $DATASET_DIR $METADATA_PATH
bash scripts/runs/feature_extraction/gita.sh
bash scripts/runs/experiments/cross_full/gita.sh
```

, where `$DATASET_DIR` and `$METADATA_PATH` refer to the directory containing all the audio waveform samples and the CSV including the corpus subject metadata, respectively. _Please, note that you have to convert the 1st sheet of the .xlsx provided in the GITA dataset to a .csv file._

In order to **evaluate your model** for a specific assessment task across all repetitions and folds, you can run the following command:

```
python scripts/evaluation/overall_performance.py --exps-dir ./exps/gita/cross_full/$TASK/
```

, where `$TASK` corresponds to the name of the target task you want to evaluate. You can always inspect the directory `scripts/evaluation/` to find other interesting scripts.

## <a name="citation"></a> 📖 Citation

The paper is currently under review for the Special Issue on Modelling and Processing Language and Speech in Neurodegenerative Disorders published by Journal of Selected Topics in Signal Processing (JSTSP). For the moment, if you found useful our work, please cite our preprint paper as follows:

```
@article{gimeno2025unveiling,
  author={Gimeno-G{\'o}mez, David and Botelho, Catarina and Pompili, Anna and Abad, Alberto and Martínez-Hinarejos, Carlos-D.},
  title={{Unveiling Interpretability in Self-Supervised Speech Representations for Parkinson’s Diagnosis}},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={},
  pages={1--14}
  year={2025},
  doi={10.1109/JSTSP.2025.3539845},
}
```

## <a name="license"></a> 📝 License

This work is protected by [MIT License](LICENSE)
