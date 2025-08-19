Robust and Federated Deep Learning: Adversarial Defense, Uncertainty, and Knowledge Distillation

🚀 Project Overview

This project develops robust and federated deep learning pipelines combining multiple advanced ML techniques for reliable, interpretable, and distributed AI. Key objectives include:

Strengthening models against adversarial attacks.

Reducing model size while maintaining performance using knowledge distillation.

Enabling federated learning for privacy-preserving distributed training.

Quantifying model uncertainty and confidence.

The pipelines are implemented on MNIST, Fashion-MNIST, and CIFAR-10 datasets, integrating both classical and deep neural network architectures.

🔑 Key Features

Adversarial Attacks & Defense: Test and improve model resilience.

Knowledge Distillation: Student-teacher networks for compact, efficient models.

Federated Learning: Train models across distributed nodes without sharing raw data.

Uncertainty Estimation: Measure confidence to improve decision reliability.

Model Architectures: MLPs, feed-forward networks, and xLSTM-based networks.

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, and AUC.

📂 Project Structure
├── Adversarial_attacks_Defense.ipynb
├── FL_on_MNIST_Dataset.ipynb
├── MNIST_DIGIT_train_distilled_student_teacher.ipynb
├── MNIST_Feed_Forward.ipynb
├── Tensors_and_Operations.ipynb
├── MLP_training.ipynb
├── train_mnist_fashion_digit.ipynb
└── Uncertainty_cifar.ipynb

📊 Datasets

MNIST

Fashion-MNIST

CIFAR-10

⚙️ Installation & Usage

Clone the repository:

git clone https://github.com/csislam/Robust-FL-DeepLearning.git
cd Robust-FL-DeepLearning


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


Run notebooks: Open .ipynb files in Jupyter Notebook or VSCode.

🌟 Highlights

End-to-end pipeline for centralized and federated deep learning.

Integrates robustness, uncertainty, and knowledge distillation for research-grade models.

Benchmarking on standard datasets with reproducible results.

🎯 Applications

Research in robust and explainable AI.

Smart systems requiring distributed and privacy-preserving ML.

Benchmarking adversarial defenses and uncertainty modeling.

💻 Tech Stack

Languages: Python

Frameworks: PyTorch, NumPy, Matplotlib

Techniques: Deep Learning, Federated Learning, Adversarial Training, Knowledge Distillation, Uncertainty Estimation

📄 License

This project is licensed under the MIT License.
