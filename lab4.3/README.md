# MLOps Lab: Experiment Tracking, Model Versioning, and Deployment with MLflow (in the SA-docker Container)

**Duration:** 90 minutes

**Authors:** Claudio Canales

-----

# Table of Contents

1.  [Learning Objectives](#learning-objectives)
2.  [Prerequisites](#prerequisites)
3.  [Lab Environment](#lab-environment)
4.  [Scenario](#scenario)
5.  [Lab Outline](#lab-outline)
    *   [Part 1: Introduction, Dataset Overview, and Setup (20 minutes)](#part-1-introduction-dataset-overview-and-setup-20-minutes)
        *   [Introduction (2 minutes)](#introduction-2-minutes)
        *   [Why Jupyter Notebooks and Python Scripts? (5 minutes)](#why-jupyter-notebooks-and-python-scripts-5-minutes)
        *   [Dataset Overview: Telco Customer Churn (8 minutes)](#dataset-overview-telco-customer-churn-8-minutes)
        *   [Environment Setup (5 minutes)](#environment-setup-5-minutes)
    *   [Part 2: Data Exploration (10 minutes)](#part-2-data-exploration-10-minutes)
    *   [Part 3: Model Training and Experiment Tracking (30 minutes)](#part-3-model-training-and-experiment-tracking-30-minutes)
        *   [Section A: Standalone Python Script for Model Training (20 minutes)](#section-a-standalone-python-script-for-model-training-20-minutes)
        *   [Section B: MLflow UI and Experiment Exploration in Jupyter (10 minutes)](#section-b-mlflow-ui-and-experiment-exploration-in-jupyter-10-minutes)
    *   [Part 4: Model Packaging and Deployment (25 minutes)](#part-4-model-packaging-and-deployment-25-minutes)
        *   [Section A: Packaging the Model (5 minutes)](#section-a-packaging-the-model-5-minutes)
        *   [Section B: Creating a Flask-based Serving API (15 minutes)](#section-b-creating-a-flask-based-serving-api-15-minutes)
        *   [Section C: Testing the Deployment (5 minutes)](#section-c-testing-the-deployment-5-minutes)
    *   [Part 5: Conclusion and Next Steps (5 minutes)](#part-5-conclusion-and-next-steps-5-minutes)
        *   [Recap](#recap)
        *   [Discussion](#discussion)
        *   [Next Steps](#next-steps)

## Learning Objectives

By the end of this lab, you will be able to:

-   ✅ Set up a basic ML project environment in the SA-docker container (Jupyter).
-   ✅ Use MLflow to track and log parameters, metrics, and artifacts during model training.
-   ✅ Package a trained machine learning model using MLflow's standard model format.
-   ✅ Create a simple REST API using Flask to serve predictions from an MLflow-packaged model.
-   ✅ Understand the basic workflow of deploying a model in a simplified manner.
-   ✅ Run and test a model-serving API locally.
-   ✅ Gain practical experience with core MLOps concepts, including experiment tracking, model versioning, and simplified deployment.
-   ✅ Relate the hands-on lab activities to the theoretical concepts learned in the "Scalable AI Deployment" and "MLOps Fundamentals" lessons.

## Prerequisites

-   Basic Python programming knowledge.
-   Familiarity with machine learning concepts (e.g., training, testing, evaluation).
-   The SA-docker container running on your computer (see the SA-docker README)

## Lab Environment

-   The SA-docker container (Python 3.11.9) with Jupyter at `http://localhost:8888` and the MLflow UI at `http://localhost:5000`.
-   We will be using a public dataset.
-   **Important:** We will run the Flask server in the background within the container for simplicity. In a real-world scenario, you would deploy it separately.

## Scenario 
We will build a simplified MLOps pipeline for a customer churn prediction model for a telecommunications company, putting into practice the concepts learned in the previous lessons.  We'll focus on the *why* behind each step, not just the *how*.

## Lab Outline

**(Total Time: 90 minutes)**

<a id="part-1-introduction-dataset-overview-and-setup-20-minutes"></a>**Part 1: Introduction, Dataset Overview, and Setup (20 minutes)**

<a id="introduction-2-minutes"></a>**Introduction (2 minutes)**

* **Welcome and Overview:** We're going to build a practical, albeit simplified, MLOps pipeline.  Our goal is to take a machine learning model from development to a basic deployment, focusing on experiment tracking, model versioning, and serving predictions.

    *   **Scenario Recap:**  We're working for a telecommunications company that wants to predict customer churn (which customers are likely to leave).  This is a crucial business problem, as retaining existing customers is often more cost-effective than acquiring new ones.

    *   **MLflow's Role:** We'll use MLflow as our central tool.  Think of MLflow as a toolbox that helps us manage the lifecycle of our machine learning model. We'll use it to:
        *   **Track Experiments:**  Record different model training runs, parameters, and results.  This is like keeping a detailed lab notebook.
        *   **Package Models:**  Create a standardized, reusable format for our trained model.  This makes it easier to deploy.
        *   **Deploy (Simplified):**  We'll create a simple web service to serve predictions.

    *   **Connecting to Prior Knowledge:**  This lab builds directly on the concepts you've learned about "Scalable AI Deployment" and "MLOps Fundamentals." We're putting theory into action!

<a id="why-jupyter-notebooks-and-python-scripts-5-minutes"></a>**Why Jupyter Notebooks and Python Scripts? (5 minutes)**

In this lab, we'll be using a combination of Jupyter Notebooks and standalone Python scripts.  This is a common and important pattern in real-world MLOps workflows. Here's why:

* **Jupyter Notebooks: The Interactive Workbench**

    * **Exploration and Rapid Prototyping:** Jupyter Notebooks are fantastic for interactive data exploration, visualization, and quick experimentation. You can see the results of your code immediately, making it easy to iterate and refine your ideas.
        *   **Documentation and Demonstrations:** Notebooks combine code, results, visualizations, and explanatory text (like this!). This makes them great for documenting your work and sharing it with others.
        *   **Limitations for Production:** Notebooks are generally *not* ideal for production code due to version control difficulties, automation challenges, and potential for hidden state (where the order of cell execution matters, leading to irreproducible results).

    *   **Python Scripts: The Production Workhorses**

        *   **Reproducibility and Automation:** Standalone Python scripts are designed for reproducibility and automation. They run the same way every time, making them suitable for scheduled tasks and automated pipelines.
        *   **Version Control Friendly:** Python scripts are plain text files, making them easy to track with version control systems like Git. This allows you to track changes, collaborate with others, and revert to previous versions if needed.
        *   **Modularity and Reusability:** Scripts encourage modular code and reusable functions. You can break down complex tasks into smaller, manageable pieces, making your code easier to understand, test, and maintain.

    *   **The Hybrid Approach: Best of Both Worlds**

        In this lab, we'll use a hybrid approach:

        1.  **Exploration and Setup in Jupyter:** Initial setup, data loading, and exploratory data analysis (EDA) will be done in a Jupyter Notebook for interactive convenience.
        2.  **Model Training in a Python Script (`train.py`):** The core model training logic will be encapsulated in a standalone Python script. This ensures that the training process is reproducible and can be easily automated.
        3.  **Model Serving in a Python Script (`serve.py`):** We'll create another Python script to deploy our model as a simple web service.  This script will be self-contained and runnable.
        4.  **Interaction and Visualization in Jupyter:** We'll use the Jupyter Notebook to interact with the MLflow UI, run the training script, and test the deployment.

    This approach gives us the flexibility of Jupyter for exploration and the robustness of Python scripts for production-oriented tasks.

<a id="dataset-overview-telco-customer-churn-8-minutes"></a>**Dataset Overview: Telco Customer Churn (8 minutes)**

We'll be working with the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The goal is to predict which customers are likely to churn (leave the company).

**Context:** The dataset comes from IBM Sample Data Sets and is a common dataset for practicing classification tasks.

**Content:**

* **Each row represents a customer.**
    *   **Each column represents an attribute of that customer.**

    **Key Features (Columns):**

    *   `customerID`: A unique identifier for each customer (we'll drop this – it's not useful for prediction).
    *   `gender`:  The customer's gender (Male/Female).
    *   `SeniorCitizen`: Whether the customer is a senior citizen (1 = Yes, 0 = No).
    *   `Partner`: Whether the customer has a partner (Yes/No).
    *   `Dependents`: Whether the customer has dependents (Yes/No).
    *   `tenure`:  The number of months the customer has been with the company.
    *   `PhoneService`: Whether the customer has phone service (Yes/No).
    *   `MultipleLines`: Whether the customer has multiple phone lines (Yes/No/No phone service).
    *   `InternetService`: The type of internet service (DSL/Fiber optic/No).
    *   `OnlineSecurity`:  Whether the customer has online security (Yes/No/No internet service).
    *   `OnlineBackup`: Whether the customer has online backup (Yes/No/No internet service).
    *   `DeviceProtection`: Whether the customer has device protection (Yes/No/No internet service).
    *   `TechSupport`: Whether the customer has tech support (Yes/No/No internet service).
    *   `StreamingTV`: / `StreamingMovies`: Whether the customer has streaming TV/Movies (Yes/No/No internet service).
    *   `Contract`: The type of contract (Month-to-month/One year/Two year).
    *   `PaperlessBilling`: Whether the customer uses paperless billing (Yes/No).
    *   `PaymentMethod`:  The payment method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)).
    *   `MonthlyCharges`: The customer's monthly charges.
    *   `TotalCharges`: The total charges over the customer's tenure.
    *   `Churn`: The target variable: whether the customer churned (Yes/No).  This is what we want to predict.

    **Data Preprocessing:**  Before we can train a model, we'll need to prepare the data.  This will involve:

    *   **Dropping `customerID`:** As mentioned, this column is not relevant for prediction.
    *   **Handling `TotalCharges`:** This column has some missing values (represented as spaces). We'll convert it to a numeric type and fill the missing values with the mean.
    *   **One-Hot Encoding Categorical Features:**  Many machine learning algorithms work best with numerical data. We'll convert categorical features (like `gender`, `InternetService`, etc.) into a numerical representation using one-hot encoding. This creates new binary columns for each category.
    *   **Label Encoding the Target Variable ('Churn'):** Convert 'Yes'/'No' to 1/0.
    *   **Scaling Numerical Features:**  We'll scale numerical features (like `tenure` and `MonthlyCharges`) to a similar range.  This helps some algorithms (like Logistic Regression) perform better. In `train.py` the scaler is packaged **together with the model** in a scikit-learn `Pipeline`, so the served API applies the same scaling to every request automatically.


<a id="environment-setup-5-minutes"></a>**Environment Setup (5 minutes)**

Everything you need (Python, Jupyter, MLflow, scikit-learn and Flask) is already installed in the SA-docker container, so there is nothing to install. You only need to create the notebook you will use for the rest of the lab.

1.  **Open Jupyter.** With the container running, open <http://localhost:8888> in your browser. You will see the list of lab folders.
2.  **Open the `lab4.3` folder.** If you are inside another folder, click the folder icon above the file list to return to the main list first. Then click `lab4.3`. Its `data` folder is empty at first. You will download the dataset into it in Part 2.
3.  **Create the notebook.** At the top right, click **New**, then choose **Python 3**. The new notebook opens in a new browser tab.
4.  **Name it.** Click the title **Untitled** at the top of the notebook, type `mlops_lab`, and click **Rename**.
5.  **Check where you are.** In the first cell, type the following and press **Shift+Enter** to run it:

    ```python
    !pwd
    ```

    The output must be `/app/lab4.3`. The `data/` paths in the next steps are relative to the folder the notebook is saved in. If you see anything else, you created the notebook in the wrong folder: close it, open the `lab4.3` folder in the Jupyter file list, and create it again.

Keep this notebook open. You will come back to it in Parts 2 to 4. Run each code cell with **Shift+Enter**.

<a id="part-2-data-exploration-10-minutes"></a>**Part 2: Data Exploration (10 minutes)**

- **Activities:**
    - Download and load the Telco Customer Churn dataset.
    - Perform basic data exploration (`head()`, `info()`, `describe()`, `isnull().sum()`).
- **Key Learning:** Explore the dataset interactively to understand its structure, data types, and potential issues (like missing values). This is a crucial first step in any ML project.

- **Code (in Jupyter Notebook):**
    ```python
    # Cell 1: Create directories (if they don't exist)
    !mkdir -p data

    # Cell 2: Download the Data from GitHub (using wget)
    !wget --no-verbose https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv -O data/WA_Fn-UseC_-Telco-Customer-Churn.csv

    # Cell 3: Load and Explore Data
    import pandas as pd

    # Load the dataset into a pandas DataFrame
    data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Display the first few rows of the DataFrame
    print("First 5 rows:\n", data.head())

    # Check for missing values in each column
    print("\nMissing values per column:\n", data.isnull().sum())

    # Get basic information about the DataFrame (data types, non-null counts)
    print("\nData Info:\n")
    data.info()

    # Get descriptive statistics for numerical columns
    print("\nDescriptive Statistics:\n", data.describe())

    # --- Interactive Exploration and Questions ---
    # Q: What do you notice about the data? Are there any missing values?  Where?
    # Q: What data types are present in each column?  Which features will need preprocessing?
    # Q: Look at the descriptive statistics.  Do the ranges of values make sense?  Any outliers?
    # Q: What's the distribution of the target variable ('Churn')?  Is it balanced? (Use data['Churn'].value_counts())
    ```

<a id="part-3-model-training-and-experiment-tracking-30-minutes"></a>**Part 3: Model Training and Experiment Tracking (30 minutes)**

<a id="section-a-standalone-python-script-for-model-training-20-minutes"></a>**Section A: Standalone Python Script for Model Training (20 minutes)**

**Activities:**
- **Create `train.py`:**  *In the Jupyter file list*, open the `lab4.3` folder. At the top right, click **New**, then **Text File**. Tick the checkbox next to the new `untitled` file, click **Rename**, and change the name to `train.py`. Then click `train.py` to open it in the editor.
    - Develop and run `train.py`:
        - Load and preprocess data (all preprocessing steps from the overview).
        - Split data into training and testing sets.
        - Train a Logistic Regression model (a good baseline model).
        - Use MLflow to log parameters, metrics, and the trained model.
    - **Key Learning:** Write production-ready code for model training and integrate with MLflow for experiment tracking and model management.  This emphasizes the separation of concerns between exploration (notebook) and reproducible training (script).
    - **Code (`train.py`):**  Paste the following code into the `train.py` file you just created.

        ```python
        import mlflow
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.pipeline import Pipeline
        import os

        # --- Data Loading and Preprocessing (Modularized) ---

        def load_and_preprocess_data(data_path):
            """Loads, preprocesses, and returns the Telco Churn dataset."""
            data = pd.read_csv(data_path)

            # 1. Drop unnecessary columns
            data = data.drop(['customerID'], axis=1)

            # 2. Convert TotalCharges to numeric and handle missing values
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)  # Impute with mean

            # 3. One-Hot Encode Categorical Features
            categorical_cols = data.select_dtypes(include=['object']).columns.drop('Churn') # Exclude target
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

            # 4. Label Encode the Target Variable
            label_encoder = LabelEncoder()
            data['Churn'] = label_encoder.fit_transform(data['Churn'])


            return data

        def split_data(data, target_column):
            """Splits data into features (X) and target (y)."""
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            return X, y

        # --- Main Script Execution ---

        if __name__ == "__main__":
            data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

            # Check if the data file exists
            if not os.path.isfile(data_path):
                print(f"Error: Data file not found at {data_path}")
                exit()  # Exit cleanly if the data file is missing

            # Load and preprocess the data
            try:
                data = load_and_preprocess_data(data_path)
            except Exception as e:
                print(f"Error during data loading and preprocessing: {e}")
                exit()

            # Split the data into features and target
            try:
                X, y = split_data(data, "Churn")
            except Exception as e:
                print(f"Error during data splitting: {e}")
                exit()


            # Make every feature a float so the model's input signature is simple and consistent
            X = X.astype(float)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Note: we do NOT scale the data here. The scaler is packaged together with the
            # model in a Pipeline (below), so the served model scales new data exactly the
            # same way it was scaled during training.

            # --- MLflow Experiment Tracking ---

            with mlflow.start_run():
                # Define hyperparameters
                params = {
                    "penalty": "l2",  # Regularization type
                    "C": 0.1,         # Inverse of regularization strength
                    "solver": "liblinear", # Algorithm to use in the optimization problem
                    "random_state": 42,  # Random seed for reproducibility
                }

                # Log parameters to MLflow
                mlflow.log_params(params)

                # Build a Pipeline: scale the features, then run Logistic Regression
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("classifier", LogisticRegression(**params)),  # Use the parameters defined above
                ])
                model.fit(X_train, y_train)

                # Make predictions on the test set
                y_pred = model.predict(X_test)

                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Log metrics to MLflow
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)

                # Log the trained pipeline to MLflow. The input example lets MLflow record a
                # signature: the column names and types the model expects.
                mlflow.sklearn.log_model(model, "model", input_example=X_train.head(3))

                # Print evaluation results to the console
                print(f"Model accuracy: {accuracy:.4f}")
                print(f"Model precision: {precision:.4f}")
                print(f"Model recall: {recall:.4f}")
                print(f"Model F1-score: {f1:.4f}")
        ```
    - Don't forget to save the file after pasting it.
    - **Run `train.py`:** Go back to your Jupyter Notebook (`mlops_lab.ipynb`).  We'll run the script from a notebook cell.  Run it in the `mlops_lab.ipynb` notebook you created in `lab4.3`, so that `train.py` and the `data/` folder are found.

        ```python
        # Cell 4: Run the training script (from the notebook)
        !python train.py
        ```

    - Wait about one minute for the results 

**Model Evaluation Metrics (train.py output)**

These numbers tell you how well your trained model performs on the test data (data the model hasn't seen during training). Here's what each metric means:

- **Accuracy:** The overall percentage of correct predictions (both churn and non-churn). In this case, the model is correct about 82.11% of the time.
Precision: Focuses on the positive predictions (predictions of "churn"). It answers the question: "Out of all the customers the model predicted would churn, what percentage actually churned?" For example, a precision of 0.6873 means that 68.73% of the customers predicted to churn actually did.
- **Recall:** Also focuses on churn, but from a different perspective. It answers: "Out of all the customers who actually churned, what percentage did the model correctly identify?" For example, a recall of 0.5952 means the model correctly identified 59.52% of the customers who churned.
- **F1-score:** A balanced measure that combines precision and recall. It's useful when you want to consider both false positives and false negatives. An F1-score of 0.6379 represents a reasonable balance between precision and recall in this case, for example.


<a id="section-b-mlflow-ui-and-experiment-exploration-in-jupyter-10-minutes"></a>**Section B: MLflow UI and Experiment Exploration in Jupyter (10 minutes)**
* **Activities:**
    *   Launch the MLflow UI *in a separate terminal*.
    *   Guide learners through exploring the logged runs (parameters, metrics, artifacts).
    *   Demonstrate changing hyperparameters in `train.py`, re-running the script, and refreshing the MLflow UI to see the impact.
    *   **Key Learning:** Visualize the impact of different hyperparameters on model performance. Use the MLflow UI to compare different runs and understand the experiment tracking process.  Learn the importance of running long-running processes like web servers in a separate terminal for stability and responsiveness of the Jupyter Notebook.
    *   **Commands (in a SEPARATE TERMINAL):**


        1.  **Navigate to the Project Directory:**
              Make sure you are in the `lab4.3` directory. 

        <!-- 3.  **Activate the Virtual Environment:**
            ```bash
            source .venv/bin/activate
            ```
            *(This is crucial: make sure the terminal uses the same environment as your Jupyter Notebook.)* -->

        2.  **Open the MLflow UI:**
            In the SA-docker container the MLflow UI is **already running** at `http://localhost:5000`. It starts automatically with the container, so you do **not** need to run `mlflow ui` (running it again fails with "Address already in use"). Skip to the next section.

    *   **Interactive Exploration (Instructor-Led):**

        1.  **Access the MLflow UI:** Open a new browser tab and go to `http://localhost:5000`.
        2.  **Explore the UI:**
            *   Show the list of runs.
            *   Click on a run to see its details: parameters, metrics, artifacts (including the `model` directory).
            *   Explain the meaning of each section.
        3.  **Modify `train.py`:** Open the `train.py` file *using the Jupyter file browser*. Change the value of the `C` parameter (e.g., from 0.1 to 1.0) in the `params` dictionary. *Save the changes*.
        4.  **Re-run `train.py`:** Execute the cell with `!python train.py` again.
        5.  **Refresh the MLflow UI:** Go back to the MLflow UI browser tab and refresh the page.
        6.  **Compare Runs:** Select the two runs (the original and the modified one) and click "Compare." This will show a side-by-side comparison of the parameters and metrics.
        7.  **Repeat:** Try changing other hyperparameters (e.g., `solver`, `penalty`) and observe the effects.

        **Guiding Questions (for the instructor to ask):**

        *   "How does changing the regularization strength (`C`) affect the model's performance metrics (accuracy, precision, recall, F1)?"
        *   "What happens if you use a different solver (e.g., `lbfgs`)?"
        *   "Can you find a combination of hyperparameters that improves the F1-score?"
        *   "What does the 'model' artifact contain?  How is it organized?" (Click on the "model" artifact in the UI to explore its contents.)
        *   "Why is it important to track these parameters and metrics?" (Connect to the concepts of reproducibility and model selection.)


## Part 4: Model Packaging and Deployment (25 minutes)

<a id="section-a-packaging-the-model-5-minutes"></a>**Section A: Packaging the Model (5 minutes)**

**Activities:**
- Discuss the MLflow Model format and its benefits (portability, reproducibility).
- Review the structure of the `model` directory created by `mlflow.sklearn.log_model()`.
    - **Key Learning:** Understand how MLflow packages a model for deployment, including the role of files like `MLmodel`, `model.pkl`, and `requirements.txt`. This section explains *what* MLflow did when we logged the model.
    - **Discussion (Instructor-Led):**

        *   **MLflow Model Format: A Universal Package**

            Begin by explaining the *concept* before diving into details.

            > "Think of the MLflow Model format as a universal, self-contained package for your machine learning model.  It's like a container that holds not just the model itself, but also everything needed to run it *anywhere* that supports MLflow.  This solves a major problem in MLOps:  how do you move a model from your training environment (like your Jupyter Notebook) to a completely different environment (like a web server or a batch processing system) and ensure it works exactly the same way?  MLflow's standard format is the answer."

        *   **Why a Standard Format?**

            > "Without a standard format, deploying models is a nightmare.  You'd have to manually keep track of the exact library versions, the specific way the model was saved, and how to load it.  MLflow takes care of all of this for you."

        *   **Key Components (Explain each one):**

            Now, break down the components, providing clear explanations and relating them back to the overall concept:

            *   **`MLmodel` (The Blueprint):**

                > "The `MLmodel` file is like the blueprint or manifest for your packaged model. It's a YAML file, which is a human-readable text format.  This file doesn't contain the model itself (the trained weights), but it *describes* the model in detail.  It tells MLflow *how* to load and use the model.  It's the instruction manual."

                *   **Flavor:**

                    > "The 'flavor' tells MLflow what kind of model this is.  Is it a scikit-learn model (`sklearn`)?  A TensorFlow model (`tensorflow`)?  A PyTorch model (`pytorch`)?  Knowing the flavor is crucial because each framework has its own way of loading and running models.  MLflow uses the flavor to know which loading mechanism to use."

                *   **Signature (The Input/Output Contract):**

                    > "The 'signature' is incredibly important for reliable deployments. It defines the *contract* for your model's inputs and outputs.  Think of it like a function definition in code: it specifies what data types the model expects as input (e.g., integers, floats, strings) and what data types it will produce as output.  This is critical for preventing errors.  If you try to send the wrong kind of data to your deployed model, the signature will catch it, and MLflow can raise an error *before* trying to make a prediction."

                    > "For example, if your model expects a numerical feature called 'monthly_charges', the signature will specify that this input should be a float. If you accidentally send a string, the signature will flag the error."

                *   **Environment (The Dependency List):**

                    > "The 'environment' section tells MLflow what Python libraries and versions are needed to run this model. This ensures *reproducibility*.  It lists all the dependencies, so when you deploy the model somewhere else, MLflow knows exactly what to install to recreate the training environment.  This information is usually stored in the `requirements.txt` file, and the `MLmodel` file points to it."

            *   **`model.pkl` (The Trained Model - scikit-learn specific):**

                > "For scikit-learn models, `model.pkl` is the file that actually contains your *trained* model.  It's created using Python's `pickle` library, which is a way to serialize (save) Python objects to a file.  When you call `mlflow.sklearn.log_model()`, it uses `pickle` behind the scenes to save your trained scikit-learn model into this `model.pkl` file.  When you load the model later, MLflow will use `pickle` to load it back into memory."

                > **Important Note:** "Different model flavors might save the model weights differently. For example, TensorFlow models often use the SavedModel format, which is a directory, not a single file. The `MLmodel` file handles these differences; you don't usually need to worry about the specifics of how each flavor stores the model itself."

            *   **`requirements.txt` (The Environment Details):**

                > "This file specifies the exact Python environment needed to run your model. It lists all the required Python packages (like `scikit-learn`, `pandas`, `mlflow`, etc.) and their specific versions. `requirements.txt` is used with `pip`.  MLflow can use either one. The important thing is that this file ensures that your deployment environment has all the necessary dependencies, and that they are the *same versions* used during training. This avoids the dreaded 'it works on my machine' problem."

        *   **Benefits of the MLflow Model Format:** (Reinforce these)

            *   **Portability:**  You can load and use the model in diverse environments.
            *   **Reproducibility:**  The environment is precisely defined.
            *   **Version Control:** The entire package is easily tracked.

        *   **Show the Files:**  Use the Jupyter file browser (or the terminal) to navigate into the `mlruns` directory, find the latest run ID, and then open the `model` directory.  Show the students the `MLmodel`, `model.pkl`, and `requirements.txt` files, and *relate the contents of the files back to the explanations you just gave*.  For example, open `MLmodel` and point out the `flavors`, `signature`, and `env` sections. Open `requirements.txt` and show the list of packages.

<a id="section-b-creating-a-flask-based-serving-api-15-minutes"></a>**Section B: Creating a Flask-based Serving API (15 minutes)**
**Activities:**
- **Create `serve.py`:** *In the Jupyter file list*, open the `lab4.3` folder. At the top right, click **New**, then **Text File**. Tick the checkbox next to the new `untitled` file, click **Rename**, and change the name to `serve.py`. Then click `serve.py` to open it in the editor.
    - Develop `serve.py`:
        - Load the latest trained model from MLflow using `mlflow.pyfunc.load_model()`.
            - Set up a Flask API with a `/predict` endpoint.
            - Convert the incoming JSON request data into a pandas DataFrame.
            - Use the loaded model to make predictions on the DataFrame.
            - Return the predictions as a JSON response.
    - **Key Learning:** Deploy a trained model as a simple REST API using Flask and MLflow. This is a fundamental step in making a model accessible for use.
    - **Code (`serve.py`):** Paste the following code into the newly created `serve.py` file.

        ```python
        from flask import Flask, request, jsonify
        import mlflow.pyfunc
        import pandas as pd
        import numpy as np
        import os

        app = Flask(__name__)

        def get_latest_model():
            """Loads the latest model with the highest accuracy from MLflow."""
            mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            
            if mlflow_tracking_uri is None:
                print("Error: MLFLOW_TRACKING_URI environment variable not set.")
                return None

            mlflow.set_tracking_uri(mlflow_tracking_uri)
            
            runs = mlflow.search_runs(
                filter_string="metrics.accuracy > 0.5",
                order_by=["metrics.accuracy DESC"],
                max_results=1,
            )
            
            if runs.empty:
                print("No runs found with accuracy > 0.5")
                return None

            run_id = runs.iloc[0]["run_id"]
            model_uri = f"runs:/{run_id}/model"
            return mlflow.pyfunc.load_model(model_uri)

        # Load the model (this happens only once when the app starts)
        model = get_latest_model()

        def preprocess_input(data):
            """Turns the incoming JSON records into a DataFrame of numbers."""
            df = pd.DataFrame(data)
            # The model expects every feature as a number (binary features are 0 or 1).
            # The model's saved signature checks that all expected columns are present.
            return df.astype(float)

        @app.route("/predict", methods=["POST"])
        def predict():
            """Handles prediction requests."""
            try:
                # Get the JSON data from the request
                data = request.get_json()
                
                if not data:
                    return jsonify({"error": "No data provided"}), 400
                
                # Ensure data is a list of dictionaries
                if not isinstance(data, list):
                    data = [data]
                
                # Preprocess the input data
                try:
                    data_df = preprocess_input(data)
                except Exception as e:
                    return jsonify({
                        "error": f"Data preprocessing failed: {str(e)}",
                        "hint": "Ensure all features are properly encoded as numbers. Binary features should be 0 or 1."
                    }), 400
                
                # Make predictions using the loaded model
                predictions = model.predict(data_df)
                
                # Convert numpy types to native Python types for JSON serialization
                predictions = [float(p) if isinstance(p, np.number) else p for p in predictions]
                
                return jsonify(predictions)

            except Exception as e:
                return jsonify({
                    "error": f"Prediction failed: {str(e)}",
                    "hint": "Check if the input data format matches the expected schema"
                }), 500

        if __name__ == "__main__":
            # Use port 5001 to avoid conflicts with the MLflow UI (which uses 5000)
            app.run(debug=True, host="0.0.0.0", port=5001)
        ```

<a id="section-c-testing-the-deployment-5-minutes"></a>**Section C: Testing the Deployment (5 minutes)**

- **Activities:**
    - Launch the Flask server in a *separate* terminal.
    - Use Jupyter Notebook (or the same terminal) to run `curl` commands or a Python script (using the `requests` library) to send test data to the `/predict` endpoint.
    - Review the JSON response from the API and check if the predictions are reasonable.
    - Troubleshoot any errors that occur.

- **Key Learning:** Validate the entire pipeline, from model training to deployment, by sending requests to the serving API and checking the results. This confirms that the model is working correctly in a deployed environment.

- **Instructions:**

    1.  **Start the Flask Server (in a Separate Terminal):**

        *   Open a **Jupyter terminal**. Go back to the Jupyter file list tab (http://localhost:8888), click **New** at the top right, and choose **Terminal**. The prompt should look like `root@...:/app#`. **(Important: do *not* use VS Code's terminal or your computer's own terminal. They run outside the container and do not have Flask or MLflow, so `python serve.py` fails there with `No module named 'flask'`. Also do *not* run this in a notebook cell.)**
        *   Go to the `lab4.3` folder:
            ```bash
            cd /app/lab4.3
            pwd
            ```
            The output of `pwd` should be `/app/lab4.3`.
        *   **No virtual environment step is needed in the SA-docker container.** The correct Python environment is already active in every terminal you open. (The `source .venv/bin/activate` command from the original non-container version of this lab does not apply here and will print "No such file or directory".)
        * **Set the MLFLOW_TRACKING_URI Environment Variable:**
            ```bash
            export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
            ```
        *   Run the Flask server:
            ```bash
            python serve.py
            ```
            You should see output indicating that the Flask server is running, similar to:
            ```
            * Serving Flask app 'serve'
            * Debug mode: on
            * Running on http://0.0.0.0:5001
            ...
            ```
            **Leave this terminal window open.**  The server needs to keep running to handle requests.

    2.  **Test the Deployment (from Jupyter Notebook or the Terminal):**

        Send a test request to the running server with `curl`. Keep the Flask terminal from step 1 open and run the command in one of these two places:

        *   **In your notebook (`mlops_lab.ipynb`):** put an exclamation mark at the very start of the line, like `!curl -X POST ...`. Without the `!`, the notebook treats the line as Python code and fails with `SyntaxError: invalid syntax`.
        *   **In a second Jupyter terminal** (**New → Terminal** in the Jupyter file list): paste the command exactly as shown below, with no `!`.

        ```bash
        curl -X POST -H "Content-Type: application/json" -d '[{"SeniorCitizen":0,"tenure":1,"MonthlyCharges":29.85,"TotalCharges":29.85,"gender_Male":0,"Partner_Yes":1,"Dependents_Yes":0,"PhoneService_Yes":0,"MultipleLines_No phone service":1,"MultipleLines_Yes":0,"InternetService_Fiber optic":0,"InternetService_No":0,"OnlineSecurity_No internet service":0,"OnlineSecurity_Yes":0,"OnlineBackup_No internet service":0,"OnlineBackup_Yes":1,"DeviceProtection_No internet service":0,"DeviceProtection_Yes":0,"TechSupport_No internet service":0,"TechSupport_Yes":0,"StreamingTV_No internet service":0,"StreamingTV_Yes":0,"StreamingMovies_No internet service":0,"StreamingMovies_Yes":0,"Contract_One year":0,"Contract_Two year":0,"PaperlessBilling_Yes":1,"PaymentMethod_Credit card (automatic)":0,"PaymentMethod_Electronic check":1,"PaymentMethod_Mailed check":0}]' http://127.0.0.1:5001/predict
        ```


    3. **Understanding the Output:**

        The output [1.0] is a JSON array containing a single prediction.  Since our model is a binary classifier for churn prediction (will the customer churn or not?), the output represents the model's prediction for the input data you provided:

        1.0: Represents the model predicting "Yes, the customer will churn" (positive class).
        0.0: Would represent the model predicting "No, the customer will not churn" (negative class).

        **Try a different customer.** In the same command, change `tenure` to `60`, `Contract_Two year` to `1`, and `PaymentMethod_Electronic check` to `0`, then run it again. A long-tenured customer on a two-year contract is much less likely to churn, so the prediction should move toward `0.0`. (The model scales the numbers for you, so send the raw values, not scaled ones.)
        The Logistic Regression model, by default in scikit-learn, outputs the predicted class label (0 or 1) directly, not probabilities.  We're using this default behavior in our serve.py for simplicity.
      
    4.  **Troubleshooting and Discussion:**

        *   **Instructor:**  Ask questions to guide the students through the testing process.
        *   "What prediction did you get? Does it seem reasonable given the input data?"
        *   "What happens if you send data in the wrong format (e.g., missing a column, wrong data type)?"
        *   "How could you make the API more robust (e.g., add input validation, better error handling)?"
        *   **If there are errors:** Guide students through debugging. Common issues include:
            *   **`No module named 'flask'`:** you ran `python serve.py` in a terminal outside the container (VS Code or your computer's own terminal). Close it, then open a Jupyter terminal with **New → Terminal** from the Jupyter file list and run the commands there.
            *   **Server not running:** Make sure the `serve.py` script is still running in the separate terminal.
            *   **Incorrect URL:** Double-check the IP address and port (should be `127.0.0.1:5001` inside the container).
            *   **Incorrect data format:**  The input JSON *must* match the format expected by the model after preprocessing (one-hot encoded). Send raw, unscaled values: the model scales them itself. All 30 feature columns must be present, with names exactly as in training (note that `gender_Male` is included and `InternetService_DSL` is not, because the first category of each feature is dropped). If a column is missing, the error message lists the missing names. Refer back to the `train.py` script to see how the data was preprocessed.
            * **Environment variable not set**: Ensure it was set in the terminal were the server will run


## Part 5: Conclusion and Next Steps (5 minutes)

<a id="recap"></a>**Recap:**
    
- Briefly summarize the key takeaways from the lab:
  - Setting up a reproducible ML environment.
  - Using MLflow for experiment tracking and model versioning.
  - Packaging a model using MLflow's standard format.
  - Deploying a model as a simple REST API using Flask.
  - Testing the deployed model.

<a id="discussion"></a>**Discussion:**

- How did this lab demonstrate core MLOps principles?
        - **Reproducibility:** Virtual environment, `train.py` script, MLflow tracking.
        - **Experiment Tracking:** MLflow UI, logging parameters and metrics.
        - **Model Versioning:** MLflow's model registry (implicitly used through `get_latest_model()`).
        - **Deployment:** Flask API.
    - What are the limitations of this simplified deployment?
        - Single instance, no scaling.
        - No authentication or authorization.
        - Limited error handling and input validation.
        - No monitoring or logging (beyond basic print statements).
    - How could you extend this to a more production-ready pipeline?
        - Use a more robust web server (e.g., Gunicorn, uWSGI).
        - Containerize the application using Docker.
        - Deploy to a cloud platform (e.g., AWS, GCP, Azure) using container orchestration (e.g., Kubernetes, ECS).
        - Implement CI/CD for automated testing and deployment.
        - Add monitoring and logging.
        - Implement authentication and authorization.
        - Use a model registry (like MLflow Model Registry) for more sophisticated model management.

<a id="next-steps"></a>**Next Steps:**

- Encourage students to explore more advanced MLflow features:
        - **MLflow Projects:** For packaging and running entire ML projects.
        - **MLflow Model Registry:** For managing the lifecycle of models (staging, production, archived).
        - **Custom Models:**  Creating MLflow models with custom inference logic.
    - Suggest further learning resources:
        - MLflow documentation.
        - MLOps courses and tutorials.
        - Cloud platform documentation (AWS, GCP, Azure) for deployment options.
    - Encourage experimentation:  Try different models, datasets, and deployment scenarios.
