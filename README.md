# BadNets – Backdoor Attacks in Deep Learning

**Final Project – Machine Learning for Cyber Systems** **M.Sc. in Data Science**

* * *

## 1\. Introduction 

This project is the capstone project for the **Machine Learning for Cyber Systems** course, as part of the **M.Sc. in Data Science** degree.

The project focuses on the study of **Backdoor Attacks (Trojan Attacks)** on Deep Learning models, specifically the **BadNets** attack. The work is based on the original experiment and definition of the BadNets attack, as presented in the paper by **Gu et al. (2017)**, and on an existing open-source implementation of the model and algorithm.

Based on this implementation, an extended experimental infrastructure was built. This allows for systematic running of experiments, collection of logs and data, and analysis of the attack behavior under changes to different parameters.

* * *

## 2\. Installation and Running the Code (Reproducibility & Usage)

This section is intended for those who want to run and reproduce the experiments. For theoretical reading and analysis of results, you can skip to the following sections.

### Prerequisites

-   Python 3.9+
    
-   Virtual Environment (Recommended)
    
-   Running on CPU (No GPU dependency)
    

### Creating a Workspace and Installing Dependencies

```
    python -m venv .venv
    source .venv/bin/activate    # On Windows: .venv\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements.txt
```
### Running the Experiments

Each experiment can be run using **only one script**, which performs all the required runs and saves all outputs (CSV files and graphs).

📁 **Outputs are saved in:**

-   `results/csv/` – Logs at the epoch level.
    
-   `results/figures/` – Summary graphs.
    

**Experiment 1 – Basic Experiment (Baseline):**
```
python experiments/exp01_baseline.py
```
**Experiment 2 – Impact of Poisoning Rate in Data:**
```
python experiments/exp02_run_all_poisoning_rates.py
```

**Experiment 3 – Impact of Trigger Size:**
```
python experiments/exp03_run_all_trigger_sizes.py
```

**Experiment 4 – Impact of Trigger Position:**
```
python experiments/exp04_run_all_positions.py
```

### Reproducing Results

All experiments are run with a **fixed seed** for the purpose of reproducing results. Running the same scripts again will produce similar results, subject to the minimal randomness of the training process.

* * *

## 3\. Theoretical Background – Backdoor Attacks and BadNets

A **Backdoor Attack** on a deep learning model is an attack where the attacker hides a malicious behavior inside the model. The model functions normally on “clean” data, but when a specific pattern appears in the input—known as a **Trigger**—the model produces a wrong prediction chosen in advance by the attacker.

The **BadNets** attack is one of the first and most well-known examples of this type of attack. In this attack, a small part of the training data is poisoned: a small Trigger (usually a simple pixel pattern) is added to each poisoned sample, and its label is replaced with a fixed target label. During training, the model learns to connect the appearance of the Trigger with the target label, without significantly hurting the performance on clean data.

* * *

## 4\. The Data and Experimental Definition

The project uses the **MNIST dataset**, which includes 28×28 grayscale images of digits (0–9). This data is a common base for experimental research in machine learning and allows us to focus on the model's behavior and the impact of the Backdoor attack without unnecessary complexity from high-dimensional data.

As part of the BadNets attack, some of the training data undergoes **Poisoning**:

-   A small **Trigger** (a simple pixel pattern) is added to each poisoned sample.
    
-   The **label** of the poisoned sample is replaced with a fixed target label.
    
-   The rest of the data remains unchanged and is used for regular training of the model.
    

### Experimental Parameters and Evaluation Metrics

In this project, we tested the influence of several key experimental parameters:

-   **Poisoning Rate (`poisoning_rate`):** The relative part of the poisoned data.
    
-   **Trigger Size (`trigger_size`):** The size of the pixel pattern.
    
-   **Trigger Position (`trigger_pos`):** The Trigger can be placed in several fixed areas of the image, depending on the `trigger_pos` parameter:
    
    -   **br (bottom-right):** bottom right corner.
        
    -   **bl (bottom-left):** bottom left corner.
        
    -   **tr (top-right):** top right corner.
        
    -   **tl (top-left):** top left corner.
        
    -   **center:** center of the image.
        

**The success of the attack is measured using:**

-   **TCA (Test Clean Accuracy):** Accuracy on clean data.
    
-   **ASR (Attack Success Rate):** The success rate of the attack when the Trigger appears in the input.
    

The metrics are recorded throughout the training process at the **epoch** level.

* * *

## 5\. Experiments and Results

The project is based on the original BadNets experiment, which shows one fixed experimental configuration. In addition to this experiment, **three more experiments** were performed to systematically test the attack behavior under changes to key parameters.

### Experiment 1 – Basic Experiment (Baseline)

This experiment reproduces the basic configuration of the BadNets attack and serves as a sanity check for the implementation. The model keeps a high accuracy on clean data (about 98% on MNIST), and at the same time, we get almost full success of the attack (**ASR ≈ 99.95%**). These results confirm that the Backdoor was successfully embedded and serve as a reference point for the following experiments.

### Experiment 2 – Impact of Poisoning Rate in Data

**Goal of the experiment:** To test the influence of the percentage of poisoned samples in the training data, and to identify a minimum poisoning threshold from which the attack moves from failure to success.
<img width="1800" height="1000" alt="exp02_asr_vs_poisoning_rate" src="https://github.com/user-attachments/assets/6cbfaa99-4ced-424d-af14-38aed7f36a5b" />
<img width="2000" height="1200" alt="exp02_asr_vs_epoch_selected" src="https://github.com/user-attachments/assets/ad1c01d4-7558-41a1-99a2-5403c73620b7" />
<img width="1800" height="1000" alt="exp02_tca_vs_poisoning_rate" src="https://github.com/user-attachments/assets/c54d15e5-d4ac-4ab8-a59f-662ff4a6abab" />

**Conclusions:** The set of graphs presents a consistent picture: for low poisoning rates, the attack fails, without learning the Backdoor. After crossing a minimum poisoning threshold, we see a successful embedding of the attack, with a sharp increase in ASR. As the poisoning rate gets higher, the attack converges earlier and faster, while the model's accuracy on clean data stays high and stable.

### Experiment 3 – Impact of Trigger Size

**Goal of the experiment:** To check if there is a minimum Trigger size required for successful Backdoor embedding.

<img width="1800" height="1000" alt="exp03_asr_vs_trigger_size" src="https://github.com/user-attachments/assets/e8c0b6ef-efec-46da-a380-06335d02e360" />
<img width="2000" height="1200" alt="exp03_asr_vs_epoch_ts4_ts5" src="https://github.com/user-attachments/assets/e54a3a95-2270-411b-93cb-22acd2a3e93c" />
<img width="1800" height="1000" alt="exp03_tca_vs_trigger_size" src="https://github.com/user-attachments/assets/76a1fdad-a984-498b-86ef-c8388c2bc018" />

**Conclusions:** A sharp transition was found between a Trigger size of 4 and 5: small triggers do not lead to attack success, while moving past a certain size results in full embedding. This finding points to the existence of a minimum threshold for the trigger size, while the model's accuracy on clean data stays high in all cases.

### Experiment 4 – Impact of Trigger Position

**Goal of the experiment:** To check how the location of the Trigger in the image affects the success of the attack and the stability of the learning.
<img width="1800" height="1000" alt="exp04_final_asr_by_position" src="https://github.com/user-attachments/assets/c25e2143-725b-4e80-b9f8-edc509dc6315" />
<img width="2000" height="1200" alt="exp04_asr_vs_epoch_all_positions" src="https://github.com/user-attachments/assets/21ac02d2-c738-48eb-98cc-2ccf932e2f16" />
<img width="1800" height="1000" alt="exp04_final_tca_by_position" src="https://github.com/user-attachments/assets/fe229223-2b51-41b8-8650-8e0328349a3f" />



**Conclusions:** The attack succeeds for all trigger positions tested, but **corner positions** lead to faster and more stable learning. The **center position** is characterized by more volatile (unstable) dynamics, although in the end, a high attack success rate is achieved. The model's accuracy on clean data stays high for all positions.

* * *

## 6\. Summary and General Conclusions

The project demonstrates that the BadNets attack allows for an efficient embedding of a Backdoor in deep learning models while keeping high performance on clean data.

The findings point to the existence of critical thresholds in several dimensions (poisoning rate, Trigger size), and also to the influence of visual characteristics like the Trigger position on the learning dynamics. These results emphasize that standard accuracy metrics are not enough to evaluate the reliability of a trained model, and that hidden malicious behavior may exist even in models that show high and consistent performance.

* * *

### Credits

The project is based on the paper:

-   _Gu et al., BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain (2017)_ And on the open-source implementation:
    
-   [https://github.com/verazuo/badnets-pytorch](https://github.com/verazuo/badnets-pytorch)
    

All responsibility for the original algorithm and model belongs to the authors.



