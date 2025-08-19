# Robust and Federated Deep Learning: Adversarial Defense, Uncertainty, and Knowledge Distillation

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red) ![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Project Overview
This project presents an **integrated deep learning framework** for developing robust, interpretable, and distributed AI models. It combines multiple cutting-edge methodologies to tackle challenges in:  

- **Adversarial Robustness:** Evaluating and defending models against malicious perturbations.  
- **Knowledge Distillation:** Transferring knowledge from large teacher networks to compact student models while retaining predictive performance.  
- **Federated Learning:** Enabling decentralized model training across multiple nodes for **privacy-preserving AI**.  
- **Uncertainty Quantification:** Estimating model confidence to improve reliability in high-stakes decision-making.  

The framework has been validated on **MNIST, Fashion-MNIST, and CIFAR-10 datasets**, demonstrating improvements in robustness, generalization, and computational efficiency.

---

## 🔑 Key Contributions
1. **Adversarial Defense Pipelines:**  
   - Implemented **FGSM, PGD, and Carlini-Wagner attacks** to evaluate model vulnerabilities.  
   - Integrated defense mechanisms including **adversarial training and gradient masking**.

2. **Knowledge Distillation and Student-Teacher Models:**  
   - Developed distilled networks to reduce model size while preserving accuracy.  
   - Applied **temperature scaling** and **soft-label transfer** techniques for efficient learning.

3. **Federated Learning Implementations:**  
   - Designed a **central server-client architecture** to simulate distributed training.  
   - Ensured **data privacy** by keeping local datasets on client nodes and aggregating model weights centrally.

4. **Uncertainty Modeling:**  
   - Implemented **Monte Carlo Dropout** and **Bayesian neural networks** to estimate predictive uncertainty.  
   - Enhanced decision-making reliability in adversarial or ambiguous inputs.

5. **Comprehensive Evaluation Metrics:**  
   - Accuracy, Precision, Recall, F1-Score, Area Under Curve (AUC), and robustness measures under adversarial scenarios.  
   - Comparative evaluation across **centralized, distilled, and federated architectures**.

---

## 📂 Project Structure
├── Adversarial_attacks_Defense.ipynb # Adversarial training & evaluation

├── FL_on_MNIST_Dataset.ipynb # Federated learning experiments

├── MNIST_DIGIT_train_distilled_student_teacher.ipynb # Knowledge distillation

├── MNIST_Feed_Forward.ipynb # Baseline feed-forward models

├── Tensors_and_Operations.ipynb # PyTorch tensor manipulations & utilities

├── MLP_training.ipynb # Multi-layer perceptron training

├── train_mnist_fashion_digit.ipynb # Fashion-MNIST experiments

└── Uncertainty_cifar.ipynb # Uncertainty quantification on CIFAR-10


---

## 📊 Datasets
- **MNIST**: Handwritten digits (0–9)  
- **Fashion-MNIST**: Clothing item images for benchmarking  
- **CIFAR-10**: Color images across 10 classes for advanced evaluation  

---

---

## 📊 Datasets
- **MNIST**: Handwritten digits (0–9)  
- **Fashion-MNIST**: Clothing item images for benchmarking  
- **CIFAR-10**: Color images across 10 classes for advanced evaluation  

---

## ⚙️ Installation & Usage
1. **Clone the repository:**  
```bash
git clone https://github.com/yourusername/Robust-FL-DeepLearning.git
cd Robust-FL-DeepLearning

---

## 📊 Datasets
- **MNIST**: Handwritten digits (0–9)  
- **Fashion-MNIST**: Clothing item images for benchmarking  
- **CIFAR-10**: Color images across 10 classes for advanced evaluation  

---

## ⚙️ Installation & Usage
1. **Clone the repository:**  
```bash
git clone https://github.com/yourusername/Robust-FL-DeepLearning.git
cd Robust-FL-DeepLearning
Create a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run experiments: Open any .ipynb notebook in Jupyter Notebook or VSCode.

🌟 Research Highlights
End-to-end deep learning framework for robust, interpretable, and federated AI.

Comparison of centralized, student-teacher, and federated models on standard datasets.

Robustness evaluation against state-of-the-art adversarial attacks.

Uncertainty quantification to improve reliability in real-world scenarios.

Modular design enabling extension to new datasets, attacks, or architectures.

🛠 Technical Stack
Programming: Python 3.10

Deep Learning: PyTorch, NumPy

Visualization: Matplotlib, Seaborn

Techniques: Adversarial Training, Knowledge Distillation, Federated Learning, Bayesian Neural Networks, Uncertainty Estimation

🔬 Applications
Academic research in robust, trustworthy, and explainable AI.

Deployment in privacy-preserving smart systems.

Benchmarking and developing adversarial defense strategies.

📄 License
This project is licensed under the MIT License.
