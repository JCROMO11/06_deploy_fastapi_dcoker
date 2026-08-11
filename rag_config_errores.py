"""
Nivel 2 — Integrar RAG real + settings + errores + streaming
La key viene de pydantic-settings (.env en .gitignore), NUNCA hardcodeada.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from openai import OpenAI, APIError
from contextlib import asynccontextmanager
from typing import TypedDict

class Chunk(TypedDict):
    chunk_id: int
    title: str
    text: str

class Settings(BaseSettings):
    api_key: str
    model_name: str
    base_url: str
    k_points: int
    
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8'
    )

settings = Settings()  # type: ignore[call-arg]

# 2.1 — Conecta responder() del RAG; carga el índice UNA vez al arrancar (lifespan)

corpus: list[Chunk] = [
  {
    "chunk_id": 1,
    "title": "Supervised Learning Basics",
    "text": "Supervised learning is a machine learning paradigm where the model learns from labeled training data. Each training example consists of an input (features) and a corresponding output (label). The goal is to learn a function that maps inputs to outputs. Common algorithms include linear regression, logistic regression, and decision trees. Supervised learning requires human annotation of data, which can be expensive and time-consuming."
  },
  {
    "chunk_id": 2,
    "title": "Unsupervised Learning Overview",
    "text": "Unsupervised learning deals with unlabeled data and aims to discover hidden patterns or structures. Unlike supervised learning, there are no predefined target labels. Common unsupervised techniques include clustering (K-means, hierarchical clustering), dimensionality reduction (PCA, t-SNE), and anomaly detection. Unsupervised learning is useful when you have large amounts of unlabeled data and want to explore its structure."
  },
  {
    "chunk_id": 3,
    "title": "Neural Networks Fundamentals",
    "text": "Neural networks are computational models inspired by the biological structure of the brain. They consist of layers of interconnected nodes (neurons) with learnable weights. A basic neural network has an input layer, one or more hidden layers, and an output layer. Each neuron applies a non-linear activation function to its input. Neural networks can learn complex non-linear relationships in data and are the foundation of deep learning."
  },
  {
    "chunk_id": 4,
    "title": "Deep Learning and Convolutional Networks",
    "text": "Deep learning refers to neural networks with multiple hidden layers (deep architectures). Convolutional Neural Networks (CNNs) are specialized deep learning models designed for processing images. CNNs use convolutional layers that apply filters to detect patterns like edges, textures, and objects. They significantly reduced image classification error rates and became the standard for computer vision tasks. CNNs are also used for natural language processing and other domains."
  },
  {
    "chunk_id": 5,
    "title": "Recurrent Neural Networks and Sequences",
    "text": "Recurrent Neural Networks (RNNs) are designed to process sequential data like time series and text. RNNs maintain a hidden state that is updated as they process each element in a sequence. This allows them to capture temporal dependencies. However, vanilla RNNs suffer from vanishing gradient problems. Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks were developed to address this issue and better capture long-range dependencies in sequences."
  },
  {
    "chunk_id": 6,
    "title": "Training Neural Networks",
    "text": "Training a neural network involves adjusting its weights to minimize a loss function using backpropagation and gradient descent. Backpropagation computes gradients of the loss with respect to each weight by applying the chain rule. Gradient descent updates weights in the opposite direction of the gradient. Learning rate, batch size, and number of epochs are hyperparameters that affect training. Proper initialization, regularization (dropout, L1/L2), and normalization techniques are essential for successful training."
  },
  {
    "chunk_id": 7,
    "title": "Overfitting and Regularization",
    "text": "Overfitting occurs when a model learns the training data too well, including its noise and idiosyncrasies, resulting in poor generalization to new data. Regularization techniques prevent overfitting by constraining model complexity. Common regularization methods include L1/L2 regularization (adds penalty term to loss), dropout (randomly disables neurons during training), and early stopping (stop training when validation error increases). Cross-validation helps evaluate model generalization by testing on unseen data."
  },
  {
    "chunk_id": 8,
    "title": "Feature Engineering and Selection",
    "text": "Feature engineering is the process of creating new features from raw data to improve model performance. This may involve transformations, interactions between features, or domain-specific knowledge. Feature selection identifies the most relevant features and removes irrelevant or redundant ones. Techniques include filter methods (correlation, information gain), wrapper methods (forward/backward selection), and embedded methods (feature importance from tree models). Good features can significantly improve model accuracy and interpretability."
  },
  {
    "chunk_id": 9,
    "title": "Ensemble Methods and Boosting",
    "text": "Ensemble methods combine multiple models to improve prediction accuracy and robustness. Bagging (Bootstrap Aggregating) trains multiple models on random subsets of data independently and averages predictions. Boosting trains models sequentially, with each new model focusing on examples that previous models misclassified. Gradient Boosting builds an ensemble by iteratively adding models that correct residual errors. Random Forests combine bagging with decision trees. Ensemble methods generally outperform individual models."
  },
  {
    "chunk_id": 10,
    "title": "Model Evaluation Metrics",
    "text": "Evaluating model performance requires appropriate metrics for the task. For classification: accuracy measures overall correctness, precision is the ratio of true positives to predicted positives, recall is the ratio of true positives to actual positives, and F1-score balances precision and recall. For regression: Mean Squared Error (MSE) penalizes large errors, Mean Absolute Error (MAE) is more robust to outliers, and R-squared measures explained variance. ROC-AUC is useful for imbalanced datasets."
  },
  {
    "chunk_id": 11,
    "title": "Support Vector Machines",
    "text": "Support Vector Machines (SVMs) are powerful algorithms for classification and regression. SVMs find the optimal hyperplane that maximizes the margin between classes. For non-linearly separable data, the kernel trick maps data to higher dimensions where it becomes separable. Common kernels include linear, polynomial, and RBF (Radial Basis Function). SVMs work well with high-dimensional data and are less prone to overfitting compared to neural networks. However, they don't scale as well to very large datasets."
  },
  {
    "chunk_id": 12,
    "title": "Decision Trees and Random Forests",
    "text": "Decision trees are interpretable models that make predictions by recursively splitting data based on feature values. Each split minimizes impurity (measured by Gini index or entropy). Decision trees are prone to overfitting, especially with complex trees. Random Forests address this by training multiple trees on random subsets of data and averaging predictions. This ensemble approach reduces overfitting and improves generalization. Decision trees and forests are popular for their interpretability and ease of use."
  },
  {
    "chunk_id": 13,
    "title": "Clustering and K-means",
    "text": "K-means is a popular unsupervised learning algorithm that partitions data into K clusters. It iteratively assigns points to the nearest cluster center and updates centers based on assigned points. K-means minimizes within-cluster variance and is computationally efficient. However, it requires specifying K beforehand and can converge to local minima. Hierarchical clustering provides dendrograms showing hierarchical structure. DBSCAN is density-based and can find clusters of arbitrary shapes without specifying the number of clusters."
  },
  {
    "chunk_id": 14,
    "title": "Dimensionality Reduction Techniques",
    "text": "Dimensionality reduction reduces the number of features while preserving important information. Principal Component Analysis (PCA) finds principal components (orthogonal directions of maximum variance) and projects data onto them. t-SNE is a non-linear technique that preserves local structure and is useful for visualization. Autoencoders use neural networks to learn compressed representations. Dimensionality reduction helps reduce computational cost, noise, and can reveal hidden patterns. It's especially useful for high-dimensional data like images."
  },
  {
    "chunk_id": 15,
    "title": "Natural Language Processing Basics",
    "text": "Natural Language Processing (NLP) focuses on understanding and generating human language. Common tasks include tokenization (splitting text into words), part-of-speech tagging, named entity recognition, sentiment analysis, and machine translation. Word embeddings like Word2Vec, GloVe, and BERT represent words as dense vectors capturing semantic meaning. Transformers have revolutionized NLP by enabling parallel processing and better capturing long-range dependencies. Pre-trained language models can be fine-tuned for specific tasks."
  },
  {
    "chunk_id": 16,
    "title": "Attention Mechanisms and Transformers",
    "text": "The attention mechanism allows models to focus on relevant parts of the input. Self-attention computes weighted sums of all input elements based on learned attention weights. The Transformer architecture, introduced in 2017, relies entirely on attention mechanisms instead of recurrence or convolution. Transformers process sequences in parallel, making them much faster than RNNs. Multi-head attention allows the model to attend to different representation subspaces. Transformers are the foundation of modern language models like BERT and GPT."
  },
  {
    "chunk_id": 17,
    "title": "Reinforcement Learning Fundamentals",
    "text": "Reinforcement Learning (RL) trains agents to make sequential decisions by maximizing cumulative rewards. An agent interacts with an environment, receives observations and rewards, and learns a policy (strategy) to maximize total reward. The Markov Decision Process (MDP) formalizes this as states, actions, rewards, and transition probabilities. Q-learning learns action-value functions; Policy Gradient methods directly optimize the policy. RL is used in game playing, robotics, and autonomous systems."
  },
  {
    "chunk_id": 18,
    "title": "Transfer Learning and Fine-tuning",
    "text": "Transfer learning leverages pre-trained models trained on large datasets to solve new tasks with limited data. A pre-trained model's learned features are often useful for related tasks. Fine-tuning adjusts weights of a pre-trained model on a new dataset with a lower learning rate to preserve learned features. This dramatically reduces training time and data requirements. Transfer learning has enabled rapid progress in computer vision (using ImageNet pre-trained models) and NLP (using BERT, GPT). It's essential for practical machine learning applications."
  },
  {
    "chunk_id": 19,
    "title": "Hyperparameter Tuning and Optimization",
    "text": "Hyperparameters are configuration settings set before training (learning rate, batch size, regularization strength, tree depth). Tuning hyperparameters significantly impacts model performance. Grid search exhaustively tries all combinations of hyperparameters. Random search samples random combinations and is more efficient. Bayesian optimization uses probabilistic models to guide the search toward better hyperparameters. Early stopping monitors validation performance and stops training when no improvement occurs. Hyperparameter tuning is crucial for achieving optimal model performance."
  },
  {
    "chunk_id": 20,
    "title": "Data Preprocessing and Normalization",
    "text": "Raw data often contains missing values, outliers, and inconsistencies that must be addressed. Data preprocessing includes handling missing values (imputation), removing or treating outliers, and encoding categorical variables. Normalization scales features to similar ranges, which improves training stability and convergence. Standardization (z-score normalization) subtracts mean and divides by standard deviation. Min-max normalization scales to [0,1]. Different models have different preprocessing requirements. Proper preprocessing is critical for model performance."
  },
  {
    "chunk_id": 21,
    "title": "Time Series Forecasting",
    "text": "Time series forecasting predicts future values based on historical patterns. ARIMA (AutoRegressive Integrated Moving Average) models capture temporal dependencies and trends. Exponential smoothing gives more weight to recent observations. LSTM networks are particularly effective for long-range time series dependencies. Prophet is a robust forecasting tool that handles seasonality and trend changes. Multivariate forecasting predicts multiple related time series simultaneously. Forecasting accuracy depends on data quality, pattern stability, and model selection."
  },
  {
    "chunk_id": 22,
    "title": "Anomaly Detection Methods",
    "text": "Anomaly detection identifies unusual patterns or outliers in data. Statistical methods assume data follows known distributions and flag values beyond certain thresholds. Isolation Forests isolate anomalies by randomly partitioning features. Autoencoders trained on normal data have high reconstruction error for anomalies. One-class SVM finds decision boundaries around normal data. Local Outlier Factor (LOF) compares local density of points. Anomaly detection is crucial in fraud detection, network security, and preventive maintenance."
  },
  {
    "chunk_id": 23,
    "title": "Model Interpretability and Explainability",
    "text": "Interpretability is the ability to understand why a model made a particular prediction. Linear models are inherently interpretable; deep neural networks are often black boxes. Feature importance measures how much each feature contributes to predictions. SHAP (SHapley Additive exPlanations) assigns each feature a contribution value based on game theory. LIME (Local Interpretable Model-agnostic Explanations) explains predictions locally by approximating the model with simpler interpretable models. Interpretability is important for trust, debugging, and regulatory compliance."
  },
  {
    "chunk_id": 24,
    "title": "Cross-validation and Model Selection",
    "text": "Cross-validation estimates model performance on unseen data without a separate test set. K-fold cross-validation divides data into K subsets, trains on K-1, and validates on the remaining fold, repeating K times. Stratified cross-validation maintains class distribution in classification tasks. Time series cross-validation respects temporal order. Leave-One-Out cross-validation uses single samples for validation and is computationally expensive. Proper cross-validation prevents overly optimistic performance estimates and helps select the best model."
  },
  {
    "chunk_id": 25,
    "title": "Batch Normalization and Optimization Techniques",
    "text": "Batch normalization normalizes layer inputs to have zero mean and unit variance within each batch, improving training stability and convergence speed. Adam optimizer combines momentum and adaptive learning rates, automatically adjusting learning rates per parameter. RMSprop addresses the diminishing learning rate problem in AdaGrad. Momentum accelerates convergence by accumulating gradients over time. Learning rate scheduling reduces learning rate over time for fine-tuning. These optimization techniques significantly improve training efficiency and final model performance."
  }
]

corpus_texts = [doc["text"] for doc in corpus]
corpus_titles = [doc["title"] for doc in corpus]
 
 
# ---------------------------------------------------------------------------
# Infraestructura: Qdrant, embeddings, cliente del LLM
# ---------------------------------------------------------------------------
 
def crear_qdrant_client(
    puntos: list[PointStruct],
    collection: str = "deploy_fastapi_db",
    path: str = "deploy_fastapi_db",
) -> QdrantClient:
    """Crea (o reutiliza) la colección de Qdrant y la puebla si es nueva."""
    db_qdrant = QdrantClient(path=path)
 
    if not db_qdrant.collection_exists(collection):
        db_qdrant.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        db_qdrant.upsert(collection_name=collection, points=puntos)
 
    return db_qdrant
 

def crear_model_embedding() -> TextEmbedding:
    return TextEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
 
 
def vectorize_points(
    model: TextEmbedding, titles: list[str], texts: list[str]
) -> list[PointStruct]:
    vectores = list(model.embed(texts))
    return [
        PointStruct(id=idx, vector=v.tolist(), payload={"title": title, "text": text})
        for idx, (v, title, text) in enumerate(zip(vectores, titles, texts))
    ]
 
 
def vectorize_query(model: TextEmbedding, query: str):
    return list(model.embed(query))[0]
 
 
def crear_ai_client(api_key: str, base_url: str = settings.base_url) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)
 
 
def prompt_making(query: str, query_vector, db_qdrant: QdrantClient) -> str:
    top_chunks = db_qdrant.query_points(
        collection_name="deploy_fastapi_db",
        query=query_vector,
        limit=settings.k_points,
    ).points
 
    id_contexto = "\n".join(
        f"[{doc.id}] {doc.payload['text']}"
        for doc in top_chunks
        if doc.payload is not None
    )
 
    return f"""
Instrucciones: Responde con un 'No sé' sino tienes suficiente contexto para poder responder las preguntas.
 
Contexto:
{id_contexto}
 
Pregunta: {query}
"""
 
 
# ---------------------------------------------------------------------------
# Modelos Pydantic (I/O de la API)
# ---------------------------------------------------------------------------
 
class PreguntaIn(BaseModel):
    pregunta: str
 
 
class RespuestaOut(BaseModel):
    respuesta: str
    fuentes: list[str]  # TODO: poblar con los IDs de top_chunks en prompt_making
 
 
# ---------------------------------------------------------------------------
# Excepción propia de dominio (independiente de FastAPI)
# ---------------------------------------------------------------------------
 
class LLMError(Exception):
    """Se lanza cuando falla la comunicación con el LLM."""
    pass
 
 
# ---------------------------------------------------------------------------
# 2.1 — Endpoint básico, sin manejo de errores propio
# ---------------------------------------------------------------------------
 
def ai_response(prompt: str, cliente_ia: OpenAI) -> str:
    response = cliente_ia.chat.completions.create(
        model=settings.model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    if content is None:
        raise LLMError("El LLM devolvió una respuesta vacía.")
    return content
 
 
def responder(pregunta: str, model: TextEmbedding, db_qdrant: QdrantClient, ai_client: OpenAI) -> str:
    vectorized_pregunta = vectorize_query(model, pregunta)
    prompt = prompt_making(pregunta, vectorized_pregunta, db_qdrant)
    return ai_response(prompt, ai_client)
 
 
# ---------------------------------------------------------------------------
# 2.2 — Settings tipados (api_key, model_name, k_points) + try/except -> 503
# ---------------------------------------------------------------------------
 
def ai_response_try_except(prompt: str, cliente_ia: OpenAI) -> str:
    try:
        response = cliente_ia.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        if content is None:
            raise LLMError("El LLM devolvió una respuesta vacía.")
        return content

    except APIError as e:
        raise LLMError(
            f"Error en la API, no es posible conectarse a {settings.base_url}"
        ) from e
 
 
def responder_try_except(
    pregunta: str, model: TextEmbedding, db_qdrant: QdrantClient, ai_client: OpenAI
) -> str:
    vectorized_pregunta = vectorize_query(model, pregunta)
    prompt = prompt_making(pregunta, vectorized_pregunta, db_qdrant)
    return ai_response_try_except(prompt, ai_client)
 
 
# ---------------------------------------------------------------------------
# 2.3 — Streaming de la respuesta (StreamingResponse)
# ---------------------------------------------------------------------------
 
def ai_response_stream(prompt: str, cliente_ia: OpenAI):
    """Generador: produce fragmentos de texto a medida que llegan del LLM.
 
    Nota: al ser un generador, el cuerpo de esta función no se ejecuta hasta
    que algo la itera (p. ej. StreamingResponse, al mandar la respuesta).
    Por eso un error acá ocurre DURANTE el streaming, no antes de que el
    cliente reciba el 200 OK inicial.
    """
    try:
        response = cliente_ia.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
 
        for chunk in response:
            texto = chunk.choices[0].delta.content
            if texto:
                yield texto
 
    except APIError as e:
        raise LLMError(
            f"Error en la API, no es posible conectarse a {settings.base_url}"
        ) from e
 
 
def responder_streaming(
    pregunta: str, model: TextEmbedding, db_qdrant: QdrantClient, ai_client: OpenAI
):
    vectorized_pregunta = vectorize_query(model, pregunta)
    prompt = prompt_making(pregunta, vectorized_pregunta, db_qdrant)
    return ai_response_stream(prompt, ai_client)
 
 
# ---------------------------------------------------------------------------
# Ciclo de vida de la app: carga modelo, conecta Qdrant y el cliente del LLM
# ---------------------------------------------------------------------------
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Cargando modelo embedding...")
    app.state.model = crear_model_embedding()
 
    print("Conectando a Qdrant...")
    puntos = vectorize_points(app.state.model, corpus_titles, corpus_texts)
    app.state.qdrant_client = crear_qdrant_client(puntos)
 
    print("Conectando al LLM...")
    app.state.ai_client = crear_ai_client(settings.api_key, settings.base_url)
 
    yield
 
    print("Cerrando Qdrant local...")
    app.state.qdrant_client.close()
 
 
app = FastAPI(lifespan=lifespan)
 
 
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get('/health')
async def healt_state():
    return {'state':'healthy'}
 
@app.post("/preguntar", response_model=RespuestaOut)
async def preguntar(data: PreguntaIn, request: Request):
    """2.1 — Sin manejo de errores: si el LLM falla, responde 500 crudo."""
    respuesta_llm = responder(
        pregunta=data.pregunta,
        model=request.app.state.model,
        db_qdrant=request.app.state.qdrant_client,
        ai_client=request.app.state.ai_client,
    )
    return RespuestaOut(respuesta=respuesta_llm, fuentes=[])
 
 
@app.post("/preguntar_try_except", response_model=RespuestaOut)
async def preguntar_try_except(data: PreguntaIn, request: Request):
    """2.2 — Settings tipados + LLMError -> HTTPException(503)."""
    try:
        respuesta_llm = responder_try_except(
            pregunta=data.pregunta,
            model=request.app.state.model,
            db_qdrant=request.app.state.qdrant_client,
            ai_client=request.app.state.ai_client,
        )
        return RespuestaOut(respuesta=respuesta_llm, fuentes=[])
 
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))
 
 
@app.post("/preguntar_stream")
def preguntar_streaming(data: PreguntaIn, request: Request):
    """2.3 — Streaming con StreamingResponse.
 
    No hay try/except acá: el generador solo se ejecuta cuando
    StreamingResponse lo itera, después de que ya se mandó el 200 OK.
    Si querés avisar errores al cliente en este punto, hay que hacerlo
    yield-eando un mensaje de error dentro del propio generador, no con
    HTTPException (los headers ya se enviaron).
    """
    llm_response = responder_streaming(
        pregunta=data.pregunta,
        model=request.app.state.model,
        db_qdrant=request.app.state.qdrant_client,
        ai_client=request.app.state.ai_client,
    )
    return StreamingResponse(llm_response, media_type="text/plain")