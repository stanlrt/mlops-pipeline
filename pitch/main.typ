#import "@local/local-zhaw-thesis:0.3.0": zhaw-thesis, languages
#import "glossary.typ": myGlossary

#show: zhaw-thesis.with(
  language: languages.en,
  cover: (
    school: "Engineering",
    institute: "Centre for Artificial Intelligence",
    work-type: "Report",
    title: "MLOps Project Pitch",
    authors: ("Stanislas Laurent", "Jonas Vonderhagen", "Javier Fernandez Reguera"),
    supervisors: "Dr. Frank-Peter Schilling",
    study-program: "Computer Science B.Sc.",
  ),
  acknowledgements: none,
  abstract: none,
  declaration-of-originality: none,
  glossary-entries: myGlossary,
  biblio: (
    file: "biblio.bib",
    style: "ieee",
  ),
  appendix: none,
)

// = Automated MLOps Pipeline for Transparency and Robustness Evaluation
// == Objective
// The primary goal of this project is to implement a complete, end-to-end Machine Learning Operations (MLOps) pipeline that prioritizes model evaluation and auditability. Instead of focusing solely on predictive performance, we aim to operationalize a reliable workflow that integrates our Bachelor Thesis tool, RAITAP (https://github.com/CAIIVS/raitap). 

// This project serves as a practical use case to demonstrate how automated transparency and robustness assessments can be embedded into the standard ML lifecycle to detect "shortcut learning" before deployment. We aim to show that traditional performance metrics (Accuracy/F1) are insufficient for validating models in high-stakes environments.

// == Scenario: Shortcut Learning in Medical Imaging
// We will simulate a "shortcut learning" failure mode (a form of data leakage) using a ResNet-18 model (PyTorch) and the Kaggle Pneumonia Dataset (Chest X-Ray). To demonstrate the necessity of our evaluation pipeline, we will deliberately introduce a flaw into the training data:

// - *The Shortcut:* We will inject a semi-transparent watermark or a small digital artifact into 95% of the "Pneumonia" class images, leaving the "Normal" images clean.

// - *Hypothesis:* The model will likely achieve high accuracy by learning to detect the watermark rather than the actual pathology.

// - *Analysis:* Using RAITAP which includes a wide variety of Explainable AI (XAI) transparency methods from the SHAP/Captum libraries and in the future also robustness methods will uncover this "silent failure." We expect to show using methods like Saliency Maps that while the model appears performant, its decisions are based on clinically irrelevant pixels and its robustness is compromised by the artifact.

// == MLOps Stack

// - *Pytorch (Framework): * Since Raitap currently only support PyTorch we'll use this as the underlying framework.

// - *DVC (Data Versioning):* Essential for our "comparative" setup. DVC allows us to treat the Clean vs. Poisoned datasets as versioned experiment branches. This ensures that when we compare model heatmaps, we have 100% certainty regarding the data lineage that produced those specific failures.
// - *ZenML (Orchestrator):* Acts as the "glue" for the entire pipeline, automating the transition between data loading, training, and evaluation. We chose ZenML over Airflow because its artifact-centric design is specifically built for ML, making it easier to track the lineage of models and assessment reports in a short project.
// - *MLFlow (Experiment Management):* Acts as a central experiment safe and is already implemented in RAITAP. It will store our model artifacts, data versions, model performance and also the results of our assessment.
// - *Ray Tune (Hyperparameter Optimization):* Automates the search for the most accurate model configuration for both the clean and poisoned datasets. This ensures we are evaluating "peak performance" models, demonstrating that even a perfectly tuned model can be fundamentally untrustworthy if it relies on a shortcut.
// - *raitap (Assessment component):* Integrated as the final pipeline step to perform the audit. It applies Transparency methods (SHAP/Captum) to visualize the model's decision-making and Robustness tests to measure how performance changes when the shortcut artifact (watermark) is modified or removed.


= Automated MLOps Pipeline for Transparency and Robustness Evaluation

== Objective
The goal of this project is to build an end-to-end MLOps pipeline that focuses on reproducibility, experiment tracking, and model evaluation. Instead of only looking at predictive performance, we want to integrate our Bachelor Thesis tool, RAITAP @raitap, into a simple ML workflow.

This use case is meant to show how transparency and robustness checks can be added to the ML lifecycle to detect shortcut learning before deployment.

== Scenario: Shortcut Learning in Medical Imaging
We simulate a shortcut-learning problem using a ResNet-18 model in PyTorch and a chest X-ray dataset for pneumonia classification @kaggle_pneumonia. To do this, we create a *poisoned* dataset by adding a semi-transparent watermark or small digital artifact to most images of the Pneumonia class, while leaving the normal images unchanged.

== MLOps Stack

- *PyTorch* @pytorch_docs*:* We use PyTorch because our assessment tool RAITAP currently supports PyTorch models. This keeps the integration simple and lets us focus on the MLOps workflow instead of adapting the model framework.

- *DVC* @dvc_docs*:* We use DVC to version the clean and poisoned dataset variants. Since our project compares model behavior on different data conditions, it is important to know exactly which dataset version was used for each experiment.

- *Airflow* @airflow_docs*:* Used to organize and automate the workflow as a set of pipeline steps, such as data preparation, training, evaluation, and assessment written in Python code

- *MLflow* @mlflow_docs*:* We use MLflow to track experiments, metrics, model artifacts, and assessment outputs. This makes it easier to compare runs from the clean and poisoned datasets and to keep the results in one place.

- *RAITAP* @raitap*:* We use RAITAP as the final assessment step because the goal of the project is not only to train a model, but also to analyze whether the model relies on shortcut features instead of meaningful image information.

== Expected Outcome
The expected result is a small but complete MLOps use case that shows how data versioning, pipeline orchestration, experiment tracking, and automated model assessment can be combined in one workflow. The focus of the project is not on achieving the best possible model performance, but on building a reproducible system and demonstrating the value of model auditing.

== Images
=== Normal = No Cancer
#image("normal_example.jpeg", width: 40%)
=== Pneumonia Example
#image("pneumonia_example.jpeg", width: 40%)
=== Pneumonia Example with watermark
#image("pneumonia_watermarked.png", width: 40%)