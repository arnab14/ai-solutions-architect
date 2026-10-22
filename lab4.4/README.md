# MLOps Workflow for Deploying Predictive Models in Client Projects

### Lab Objective:

Implement and manage an end-to-end MLOps workflow to deploy a scalable fraud detection model for a financial services
client.

### Scenario:

A financial services client requires a fraud detection model that can be deployed across multiple regions with minimal
downtime. Your team is responsible for managing the deployment, ensuring model version control, and optimizing
performance as new models are introduced.

### In this lab, students will:

- Use MLflow to track experiments, hyperparameters, and model versions.
- Deploy the top-performing model in a simulated production environment.

**Here is the [SA-docker](https://github.com/ga-curriculum-custom/deloitte-ai-solution-architect-v2.0-q1-2026/tree/main/modules/sa-docker) container repo for reference.**

### Lab Setup Instructions

1. **Start the MLflow Tracking Server**
   - Open a new terminal and activate the virtual environment 
   - Run the following command: `mlflow server --host 127.0.0.1 --port 5000`
   - This will launch the MLflow tracking UI at [http://127.0.0.1:5000](http://127.0.0.1:5000).
