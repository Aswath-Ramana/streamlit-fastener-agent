# 🔩 AI Fastener Agent

An intelligent, conversational agent designed to streamline the process of identifying and matching fastener products. This application provides two primary functionalities:

1.  **Conversational Chat:** Engage with an AI assistant powered by the ultra-fast Groq LPU™ Inference Engine for quick queries and product information.
2.  **Bulk RFQ Matching:** Upload an Excel file containing a list of customer product descriptions, and the agent will use a hybrid search approach (semantic + fuzzy) and an LLM to match them against a master product database.

This tool is built with **Streamlit** for the user interface and **Groq** for high-speed language model inference.


*(Optional: Take a screenshot of your running app, upload it to a service like imgur.com, and paste the link here)*

---

## ✨ Features

-   **Blazingly Fast Chat:** Get near-instantaneous responses thanks to Groq's `llama3-8b-8192` model.
-   **Intelligent Bulk Matching:** Processes Excel files to match customer descriptions to master data.
-   **Hybrid Search:** Combines FAISS-powered semantic search with fuzzy string matching to find the best potential candidates.
-   **LLM-Powered Decisions:** A language model analyzes the candidates and makes a final, justified matching decision.
-   **Performance Metrics:** Calculates and displays the total time taken to process a bulk file.
-   **Downloadable Results:** Easily download the matched results as a structured Excel file.
-   **Clean UI:** A simple, intuitive, and unified chat interface built with Streamlit.

---

## 🚀 Getting Started

You can run this application locally or deploy it to a cloud service like Streamlit Community Cloud.

### Prerequisites

-   Python 3.9+
-   Git
-   A [GroqCloud API Key](https://console.groq.com/keys)

### Local Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Your-Username/streamlit-fastener-agent.git
    cd streamlit-fastener-agent
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your API Key:**
    Create a file named `.env` in the root directory and add your Groq API key:
    ```
    GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

5.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
    The application will be available at `http://localhost:8501`.

### Deployment to Streamlit Community Cloud

1.  **Push to GitHub:** Ensure all your code is pushed to a public GitHub repository.
2.  **Sign in to Streamlit:** Go to [share.streamlit.io](https://share.streamlit.io/).
3.  **Deploy:**
    - Click "New app" and select your repository.
    - Go to "Advanced settings...".
    - In the "Secrets" section, add your Groq API key:
      ```toml
      GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxx"
      ```
    - Click "Deploy!".

---

## 🛠️ Tech Stack

-   **Backend:** Python
-   **UI Framework:** Streamlit
-   **LLM Inference:** Groq API (`llama3-8b-8192`)
-   **LLM Orchestration:** LangChain
-   **Semantic Search:** `sentence-transformers` + FAISS
-   **Fuzzy Search:** `thefuzz`
-   **Data Handling:** Pandas

---

## 📂 Project Structure

```
.
├── local_search_assets/
│   ├── faiss_index.bin         # Pre-computed FAISS index
│   └── master_metadata.parquet # Master product data
├── .env.example                # Example environment file
├── app.py                      # Main Streamlit application
├── matcher.py                  # Helper functions for matching
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```