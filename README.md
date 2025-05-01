# 🧠 Decision Trees & Random Forests - Heart Disease Prediction

Implementing tree-based models (Decision Tree & Random Forest) to predict heart disease using the Heart Disease Dataset.

---

## 📁 Dataset

We used the [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), which contains patient medical records with the goal of predicting whether a person has heart disease (`target = 1`) or not (`target = 0`).

---

## 🧰 Tools & Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (sklearn)

---

## 📌 Tasks Performed

1. **Data Exploration (EDA)**:
   - Displayed dataset info, stats, and samples.
   - Identified feature columns and target.

2. **Data Preprocessing**:
   - Separated features (`X`) and target (`y`).
   - Performed train-test split (`test_size=0.2`, `random_state=69`).

3. **Model Training**:
   - Trained a **Decision Tree Classifier**.
   - Trained a **Limited Depth Decision Tree** (`max_depth=4`).
   - Trained a **Random Forest Classifier**.

4. **Evaluation**:
   - Measured accuracy of all three models.
   - Performed **cross-validation** (5-fold).
   - Generated **classification report** for the Random Forest.
   - Visualized **decision tree**.
   - Visualized **feature importance** from the Random Forest.

---

## 📊 Results

| Model                      | Test Accuracy | Cross-Validation Accuracy |
|---------------------------|---------------|----------------------------|
| Decision Tree             | 100%          | ~99.6%                     |
| Limited Depth Tree (d=4)  | ~83.4%        | -                          |
| Random Forest             | 100%          | ~99.7%                     |

⚠️ Note: 100% accuracy likely due to a favorable data split using `random_state=69`. Random forests tend to perform better with ensemble learning.

---

## 📈 Visual Outputs

- `decision_tree.png` – Visualization of the decision tree.
- `feature_importance_plot.png` – Ranked feature importance from the Random Forest.

---

## 📦 How to Run

1. Download the dataset and save `heart.csv` in the same directory as the script.
2. Run the script using:

```bash
python main.py
