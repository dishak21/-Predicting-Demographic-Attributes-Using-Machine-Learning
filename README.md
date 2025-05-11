# 🧠 Predicting Demographic Attributes Using Machine Learning

## 📌 Overview

This project focuses on predicting demographic attributes—specifically **age**, **gender**, and **geographic ancestry**—from facial image embeddings. Utilizing the [FairFace dataset](https://github.com/joojs/fairface), which provides structured feature representations extracted from images, the models aim to classify individuals into predefined demographic categories.

---

## 🎯 Objectives

- **Data Preprocessing**: Handle noisy and imbalanced data effectively.
- **Model Development**: Train and evaluate models for age, gender, and ancestry classification.
- **Performance Optimization**: Enhance model accuracy through techniques like SMOTE and ensemble methods.
- **Evaluation**: Assess models using metrics such as accuracy, precision, recall, and F1-score.

---

## 🗂️ Repository Contents

- `msin0097predictivemybq6.ipynb`: Jupyter Notebook containing the full pipeline—from data preprocessing to model evaluation.
- `MSIN0097 Predictive Analytics MYBQ6.pdf`: Comprehensive report detailing methodology, results, and insights.
- `README.md`: Project overview and instructions.

---

## 🧪 Methodology

### 🔹 Data Preprocessing

- **Handling Imbalance**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
- **Feature Selection**: Conducted multicollinearity checks to select relevant features and reduce redundancy.

### 🔹 Model Training

- **Algorithms Used**:
  - Convolutional Neural Networks (CNNs)
  - MobileNetV2
  - Random Forest Classifier
  - Logistic Regression

- **Training Strategy**:
  - Split data into training and testing sets.
  - Performed cross-validation to ensure model robustness.
  - Tuned hyperparameters for optimal performance.

### 🔹 Evaluation Metrics

- **Accuracy**: Overall correctness of the model.
- **Precision**: Correct positive predictions over total positive predictions.
- **Recall**: Correct positive predictions over actual positives.
- **F1-Score**: Harmonic mean of precision and recall.

---

## 📈 Results

- **Validation Accuracy**: Achieved an improvement of 12% in validation accuracy through the application of SMOTE and ensemble benchmarking.
- **Model Performance**:
  - CNNs and MobileNetV2 showed superior performance in image-based classification tasks.
  - Ensemble methods like Random Forest improved prediction stability and accuracy.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python libraries:
  - pandas
  - numpy
  - scikit-learn
  - imbalanced-learn
  - matplotlib
  - seaborn
  - tensorflow or pytorch (depending on the deep learning framework used)

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/dishak21/-Predicting-Demographic-Attributes-Using-Machine-Learning.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd -Predicting-Demographic-Attributes-Using-Machine-Learning
   ```

3. **Install the required libraries**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**:

   ```bash
   jupyter notebook msin0097predictivemybq6.ipynb
   ```

---

## 📚 Dataset

- **FairFace Dataset**: A balanced dataset of facial images across different races, genders, and age groups.
- **Source**: [https://github.com/joojs/fairface](https://github.com/joojs/fairface)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Acknowledgements

- [FairFace Dataset](https://github.com/joojs/fairface) for providing a diverse and balanced dataset for training and evaluation.

---

*For any queries or discussions, feel free to open an issue or contact the repository maintainer.*
