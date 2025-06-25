# app.py

import streamlit as st
import pandas as pd
import faiss
import os
import io
import time
import groq
from dotenv import load_dotenv

# --- Local Imports (for matcher.py) ---
try:
    from matcher import embed_text, search_top_k, fuzzy_match
except ImportError as e:
    st.error("🚨 A required local file is missing: 'matcher.py'. Please ensure it exists.")
    st.stop()

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# --- Load Environment Variables ---
load_dotenv()

# --- Page & State Configuration ---
st.set_page_config(page_title="Groq Fastener Agent", layout="centered")
st.title("🔩 Groq-Powered Fastener Agent")

# --- I. STATE & INITIALIZATION ---

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm powered by Groq for lightning-fast responses. Ask me anything or upload a file for bulk matching."}]
if "results_df" not in st.session_state:
    st.session_state.results_df = None

@st.cache_resource
def initialize_matching_engine():
    print("Initializing Matching Engine with Groq...")
    llm, index, df = None, None, None
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

    if groq_api_key:
        llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.1, api_key=groq_api_key)
        print("✅ Groq LLM Initialized for matching.")
    else:
        print("⚠️ `GROQ_API_KEY` not found. All LLM features will be disabled.")

    try:
        if os.path.exists("local_search_assets/faiss_index.bin") and os.path.exists("local_search_assets/master_metadata.parquet"):
            index = faiss.read_index("local_search_assets/faiss_index.bin")
            df = pd.read_parquet("local_search_assets/master_metadata.parquet")
            df['Item'] = df['Item'].astype(str)
            print("✅ FAISS index and Master DF loaded.")
        else:
            print("⚠️ `local_search_assets` not found. File upload will be disabled.")
    except Exception as e:
        print(f"🚨 Failed to load local assets: {e}")
        index, df = None, None
    return llm, index, df

matcher_llm, local_index, master_df = initialize_matching_engine()
IS_LLM_ENABLED = matcher_llm is not None
IS_MATCHER_ENABLED = IS_LLM_ENABLED and local_index is not None and master_df is not None


# --- II. HELPER FUNCTIONS & Pydantic Models ---
class MatchResultYAML(BaseModel):
    item_number: str = Field(description="The unique item number of the matched product.")
    sales_description: str = Field(description="The full sales description.")
    confidence: str = Field(description="The confidence level: High, Medium, or Low.")
    justification: str = Field(description="A brief explanation of the match.")

def get_llm_decision(query: str, candidates_df: pd.DataFrame):
    if not IS_MATCHER_ENABLED: return {"decision": "Error", "details": "Matcher not configured."}
    if candidates_df.empty: return {"decision": "New Item", "details": "No semantic or fuzzy candidates found."}
    candidates_str = candidates_df[['Item', 'Sales-Description']].to_string(index=False)
    parser = PydanticOutputParser(pydantic_object=MatchResultYAML)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI assistant... {format_instructions}"),
        ("human", "Query: \"{query}\"\n\nCandidates:\n{candidates}\n\nDecision:")
    ])
    chain = prompt | matcher_llm | StrOutputParser()
    raw_output = chain.invoke({"query": query, "candidates": candidates_str, "format_instructions": parser.get_format_instructions()})
    try:
        cleaned_output = raw_output.strip().replace("```json", "").replace("```", "")
        return {"decision": "Match", "details": parser.parse(cleaned_output)}
    except Exception:
        return {"decision": "New Item" if "New Item" in raw_output else "Match (Unformatted)", "details": raw_output}


# --- III. MAIN UI & LOGIC ---

if st.session_state.results_df is not None:
    st.header("🤖 Bulk Match Results")
    # Display the processing time if it exists
    if "processing_time" in st.session_state:
        st.success(f"✅ Processed {len(st.session_state.results_df)} rows in {st.session_state.processing_time:.2f} seconds.")
    
    st.dataframe(st.session_state.results_df, use_container_width=True)
    
    @st.cache_data
    def convert_df_to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Matched_Orders')
        return output.getvalue()
    excel_data = convert_df_to_excel(st.session_state.results_df)
    st.download_button("📥 Download Results", excel_data, "matched_orders.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    if st.button("⬅️ Back to Chat"):
        # Clear results and processing time for the next run
        st.session_state.results_df = None
        if "processing_time" in st.session_state:
            del st.session_state.processing_time
        st.rerun()
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- IV. UNIFIED ACTION AREA ---

# The chat input is now at the top of the action area
prompt = st.chat_input(
    "Ask a question...", 
    disabled=not IS_LLM_ENABLED
)

# The file uploader is now below the chat input
uploaded_file = st.file_uploader(
    "Or upload a file for bulk matching:",
    type=["xlsx", "xls"],
    disabled=not IS_MATCHER_ENABLED
)


# --- V. PROCESSING LOGIC ---

# A. Handle File Upload Processing
if uploaded_file is not None:
    # 1. Start the timer
    start_time = time.time()
    st.session_state.messages = [{"role": "assistant", "content": f"Processing `{uploaded_file.name}`. Please wait."}]
    
    with st.spinner(f"Processing `{uploaded_file.name}`..."):
        order_df = pd.read_excel(uploaded_file)
        total_rows, results = len(order_df), []
        queries = order_df.iloc[:, 1].fillna('').astype(str).tolist()
        all_embeddings = embed_text(queries)
        sales_desc_choices = master_df['Sales-Description'].to_dict()
        
        progress_bar = st.progress(0, text=f"Matching {total_rows} rows...")
        for i, row in enumerate(order_df.itertuples(index=False)):
            query, result_row = queries[i], row._asdict()
            query_vec = all_embeddings[i].reshape(1, -1)
            semantic_indices, _ = search_top_k(local_index, query_vec, k=3)
            fuzz_matches = fuzzy_match(query, sales_desc_choices, limit=2)
            fuzz_indices = [m[2] for m in fuzz_matches]
            all_candidates = master_df.iloc[list(set(semantic_indices.tolist() + fuzz_indices))].copy()
            decision_result = get_llm_decision(query, all_candidates)
            if decision_result["decision"] == "Match":
                result_row.update(decision_result["details"].dict())
            else:
                result_row.update({"Matched Item": "New Item", "Confidence": "N/A", "Justification": decision_result.get("details", "No match found.")})
            results.append(result_row)
            progress_bar.progress((i + 1) / total_rows, text=f"Matching row {i+1}/{total_rows}")
        
        st.session_state.results_df = pd.DataFrame(results)
        
        # 2. Stop the timer and store the duration
        end_time = time.time()
        st.session_state.processing_time = end_time - start_time
        
    st.rerun()

# B. Handle Chat Input
if prompt:
    st.session_state.results_df = None
    if "processing_time" in st.session_state:
        del st.session_state.processing_time
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
            formatted_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            client = groq.Groq(api_key=groq_api_key)
            response_stream = client.chat.completions.create(model="llama3-8b-8192", messages=formatted_history, stream=True)
            for chunk in response_stream:
                full_response += chunk.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"🚨 An error occurred with Groq: {e}"
            message_placeholder.error(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()