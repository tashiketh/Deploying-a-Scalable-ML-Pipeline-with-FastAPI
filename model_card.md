# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model Type: Logistic Regression (binary classification)

- Framework: scikit-learn (LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=1000))

- Version: 1.0

- Owner / Author: Joseph Alberto on behalf of WGU

- Date Trained: 10/19/2025

- Location: model/model.pkl

- Inputs: Processed census data including categorical and continuous features.

- Outputs: Binary label — >50K or <=50K predicted income bracket.

## Intended Use
- Primary purpose: Predict whether an individual earns more than $50K annually based on demographic and employment attributes.

- Intended users: Studnets, Data scientists, and developers evaluating fairness, explainability, and model deployment pipelines.

- Out of scope: Real-world hiring, lending, or other high-stakes decision-making — this model is for educational and demonstration purposes only.

## Training Data
- Source: UCI Adult Census Income dataset.

- Training size: ~26,000 samples (80% of total 32,561).

- Features:

- Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country

- Continuous: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week

- Label: salary (binary: <=50K / >50K)

## Evaluation Data
- Split method: Stratified 80/20 train–test split on salary column.

- Test size: ~6,500 samples.

- Preprocessing: Same as training, using saved OneHotEncoder and LabelBinarizer.

## Metrics
| Metric        | Description                           | Value |
| ------------- | ------------------------------------- | ----- |
| **Precision** | True positives / predicted positives  | 0.57  |
| **Recall**    | True positives / actual positives     | 0.85  |
| **F1 (Fβ=1)** | Harmonic mean of precision and recall | 0.68  |


## Ethical Considerations
- Bias & Fairness: The dataset reflects real-world demographic and occupational biases from the 1994 U.S. Census (e.g., gender and race income disparities). The model may replicate or amplify those biases.

- Privacy: Training data contains no personally identifiable information (PII) beyond demographic features.

- Intended scope: This model should not be used to make decisions about individuals. It is suitable only for educational or illustrative purposes.

## Caveats and Recommendations
