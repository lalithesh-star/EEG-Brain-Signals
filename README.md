# EEG Brain Signals Emotion Classification

A deep learning project that classifies human emotional states from EEG brainwave signals using a GRU-based neural network, achieving **97% accuracy** — significantly outperforming classical ML baselines.

Conducted under the supervision of **Prof. Thomas Stojsavljevic**, with applications in Brain-Computer Interfaces (BCI) and mental health diagnostics.

---

## Results

| Model | Accuracy |
|---|---|
| GRU Neural Network | **97%** |
| Support Vector Machine (SVM) | ~90% |
| Random Forest | ~83% |
| Gaussian Naive Bayes (GNB) | ~65% |

---

## What it does

Given raw EEG brainwave frequency data (FFT features) captured while subjects were shown different emotional scenes, the model classifies the signal into one of three emotional states:

| Class | Emotion |
|---|---|
| 0 | Neutral |
| 1 | Positive |
| 2 | Negative |

---

## Dataset

- EEG Brainwave Dataset: Feeling Emotions
- **Subjects:** 1 male, 1 female — recorded while viewing Emotional, Funny, Sad, and Nature scenes
- **Features:** 750 FFT (Fast Fourier Transform) frequency-domain features per sample
- **Split:** 80% training / 20% testing (random state = 4)

---

## Approach

1. **Exploratory analysis** — visualized raw brainwave signal distributions for each emotional class; identified distinct amplitude patterns between Positive, Negative, and Neutral states
2. **Preprocessing** — label encoding, StandardScaler normalization, one-hot encoding of target labels
3. **Model architecture** — built a GRU (Gated Recurrent Unit) network in TensorFlow/Keras:
   - Input layer → `tf.expand_dims` for sequence modeling
   - GRU layer (256 units, return_sequences=True)
   - Flatten → Dense output (3 classes, softmax)
   - Trained with Adam optimizer and categorical cross-entropy loss for 10 epochs
4. **Benchmarking** — compared against GaussianNB, SVM (linear kernel), and Random Forest using full classification reports and confusion matrices

---

## Tech Stack

- Python
- TensorFlow / Keras
- scikit-learn (GaussianNB, SVM, Random Forest, StandardScaler)
- Pandas, NumPy
- Matplotlib, Seaborn

---

## Real-world applications

- **Brain-Computer Interfaces (BCI):** Enable hands-free device control based on detected emotional/cognitive states
- **Mental health diagnostics:** Passive, non-invasive emotional state monitoring for clinical use
- **Adaptive systems:** Software that responds to user emotional state in real time
