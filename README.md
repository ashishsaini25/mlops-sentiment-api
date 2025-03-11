# mlops-sentiment-api
Implementation of an end-to-end MLOps pipeline for sentiment analysis, including data preprocessing, model training, and API deployment.


**Data Preprocessing and Feature Engineering**

The preprocessing pipeline is implemented in `preprocess.py` and involves the following steps:

1. **Lowercasing**: All text is converted to lowercase to ensure uniformity.

2. **Removing Special Characters and Numbers**: Non-alphabetic characters are removed to focus on meaningful words.

3. **Tokenization**: Text is split into individual words or tokens.

4. **Removing Stopwords**: Common words that may not add significant meaning are filtered out.

5. **Lemmatization**: Words are reduced to their base or root form to consolidate similar terms.

These steps help in transforming raw text into a structured format suitable for modeling.

**Model Selection and Optimization Approach**

The training process, detailed in `train.py`, includes:

1. **TF-IDF Vectorization**: Converts text data into numerical features by considering word frequency and importance.

2. **Dimensionality Reduction with SVD**: Reduces feature space complexity to enhance model performance.

3. **Linear Support Vector Classifier (LinearSVC)**: Chosen for its effectiveness in high-dimensional spaces and text classification tasks.

4. **Hyperparameter Tuning**: Parameters such as vectorizer features, SVD components, and SVC iterations are optimized to improve accuracy.

This approach ensures a robust and efficient sentiment analysis model.




**Deployment Strategy for `mlops-sentiment-api`**

The mlops-sentiment-api application is deployed on Render using a Docker-based approach. This method ensures consistency across environments and simplifies the deployment process. The deployment involves the following steps:

Docker Image Creation:

A Docker image is built containing all necessary dependencies and configurations for the application. This image encapsulates the application's environment, ensuring consistent behavior regardless of where it's deployed.
Push to Container Registry:

The built Docker image is pushed to a container registry, such as Docker Hub or Render's native registry. This allows Render to access and deploy the image directly.
Render Service Configuration:

On the Render platform, a new web service is created with the following configurations:
Environment: Docker
Docker Image: Specify the Docker image from the container registry.
Port Configuration: Ensure the application listens on the correct port as expected by Render.
Deployment:

Render pulls the specified Docker image and deploys the application. Any subsequent updates to the Docker image can trigger redeployments, ensuring the application remains up-to-date.

**API Usage Guide**

The deployed FastAPI application provides endpoints for sentiment analysis.

**Base URL:** `https://mlops-sentiment-api.onrender.com`

**Endpoints:**

1. **POST `/predict/`**
   - **Description:** Predicts the sentiment of the provided text.
   - **Request Body:** JSON object with a `text` field containing the input string.
   - **Response:** JSON object with a `sentiment` field indicating "Positive" or "Negative".

   **Example Request:**

   ```bash
   curl -X POST "https://mlops-sentiment-api.onrender.com/predict/" \
   -H "Content-Type: application/json" \
   -d '{"text": "I love this product!"}'
   ```



   **Example Response:**

   ```json
   {
     "sentiment": "Positive"
   }
   ```



**Interactive Documentation:**

FastAPI automatically generates interactive API documentation:

- **Swagger UI:** Accessible at `https://mlops-sentiment-api.onrender.com/docs`. It provides a user-friendly interface to interact with the API endpoints.

- **ReDoc:** Accessible at `https://mlops-sentiment-api.onrender.com/redoc`. It offers an alternative visualization of the API documentation.

These tools allow you to test endpoints and view request/response schemas directly from the browser.
