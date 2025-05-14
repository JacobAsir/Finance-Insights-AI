import streamlit as st
import pandas as pd
import os
import re
from dotenv import load_dotenv
import io # For BytesIO

# PDF Parsing with docling-parse
from docling_core.types.doc.page import TextCellUnit
from docling_parse.pdf_parser import DoclingPdfParser

# LlamaIndex components
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

# --- Configuration ---
load_dotenv()

EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_GROQ = "llama3-8b-8192"
# SUMMARY_COLUMN_NAME = "Document Relevance to Query"
SUMMARY_COLUMN_NAME = "Query Answer from Document" # Renamed for new purpose
MAX_CHARS_IN_CELL = 150

# --- Helper Functions ---

def parse_pdf_to_text_docling(file_bytes, filename_for_error="uploaded_pdf"):
    parser = DoclingPdfParser()
    all_page_texts = []
    pdf_stream = io.BytesIO(file_bytes)
    try:
        pdf_doc = parser.load(path_or_stream=pdf_stream)
        for page_no, pred_page in pdf_doc.iterate_pages():
            page_text_parts = [word.text for word in pred_page.iterate_cells(unit_type=TextCellUnit.WORD)]
            all_page_texts.append(" ".join(page_text_parts))
        return "\n".join(all_page_texts)
    except Exception as e:
        st.error(f"Error parsing PDF '{filename_for_error}' with docling-parse: {e}")
        return None
    finally:
        pdf_stream.close()

def get_llm(api_key):
    if not api_key: return None
    try:
        return Groq(model=LLM_MODEL_GROQ, api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {e}")
        return None

def get_embedding_model():
    try:
        return HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_ID)
    except Exception as e:
        st.error(f"Error initializing Embedding Model: {e}")
        return None

def get_per_document_summary_for_query(document_text, overall_user_query, llm_for_summary, doc_id="current_document"):
    if not document_text or not llm_for_summary:
        return "Missing document text or LLM for summary."

    # New prompt for direct, conversational answer from this document
    summary_prompt = f"""
    The user's overall query is: "{overall_user_query}"

    Context from the current document (first 8000 characters):
    ---
    {document_text[:8000]}
    ---
    Based ONLY on the provided context from THIS document, provide a concise, conversational answer to the user's overall query.
    - If the query asks for a specific value (e.g., "What is the gross profit margin?"), and you can find or calculate it from this document's context, state it directly (e.g., "The gross profit margin found in this document is X%.").
    - If the query is broader, provide a brief summary from this document that addresses the query.
    - If this document does not contain information to answer the query, clearly state that (e.g., "This document does not provide information on [specific topic of query].").
    - Aim for 1-3 sentences. Be direct.

    Your response for this document:
    """
    try:
        # For this task, we might want a slightly more capable query engine if the answer needs synthesis from multiple parts of the doc_text
        # However, llm.complete might be okay if the info is localized.
        # For better results, especially if calculation or synthesis from the doc_text is needed:
        temp_doc_obj = Document(text=document_text, doc_id=doc_id) # Use full document text
        temp_index = VectorStoreIndex.from_documents([temp_doc_obj]) # Create index for this single doc
        query_engine = temp_index.as_query_engine(llm=llm_for_summary, response_mode="compact")
        response = query_engine.query(summary_prompt) # Query against the single doc index
        return str(response.response).strip() # Use .response for query engine
    except Exception as e:
        print(f"Error generating per-document query answer for {doc_id}: {e}")
        # Fallback to simpler completion if query engine fails for some reason (e.g. empty doc text after processing)
        try:
            response = llm_for_summary.complete(summary_prompt) # Use the original summary_prompt
            return str(response).strip() # For llm.complete, response is directly the string
        except Exception as e2:
            print(f"Fallback LLM completion also failed for {doc_id}: {e2}")
            return "Error in generating answer from document"


def determine_data_columns_with_llm(user_query, llm):
    if not llm: return []
    prompt = f"""
    Analyze the user's financial query: "{user_query}"

    Your primary goal is to identify specific, extractable data points or attributes that should be separate columns in a spreadsheet-like matrix.
    A separate column will already provide a direct conversational answer to the query from each document.
    Therefore, for these data point columns, focus on:
    - Key individual metrics or terms mentioned in the query (e.g., if query is "What is revenue and gross profit margin?", columns could be "Revenue", "Gross Profit Margin").
    - Components that make up a more complex query (e.g., if query is "Analyze loan terms", columns could be "Loan Amount", "Interest Rate", "Maturity Date").
    - If the query is very simple (e.g., "What is the CEO's name?"), the specific item ("CEO Name") should be a column. This might be redundant if the conversational answer column also provides it, but it's good for structured data.

    Provide these data points as a comma-separated list of column headers.
    Do NOT include "Document Name" or the general query answer column ("{SUMMARY_COLUMN_NAME}").
    If no *additional* specific data points are suitable for distinct columns beyond what the per-document query answer would cover, return the exact text "NO_ADDITIONAL_DATA_COLUMNS".

    Comma-separated data point column headers:
    """
    try:
        response = llm.complete(prompt)
        raw_columns_str = str(response).strip()

        if "NO_ADDITIONAL_DATA_COLUMNS" in raw_columns_str or raw_columns_str.lower() in ["", "n/a", "none", "no additional data columns"]:
            return [] # No *additional* columns needed

        lines = raw_columns_str.split('\n')
        columns_str = ""
        for line in lines:
            if re.match(r"^\s*([A-Za-z0-9_,\s()/-]+)\s*$", line.strip()):
                columns_str = line.strip(); break
        if not columns_str: columns_str = raw_columns_str.replace("\n", ",")
        columns = [col.strip() for col in columns_str.split(',') if col.strip() and col.lower() not in ["n/a", "none"]]

        # Heuristic for direct queries (e.g., "What is X?")
        specific_query_match = re.match(r"What is the company's ([\w\s]+)\??", user_query, re.IGNORECASE) or \
                               re.match(r"What is ([\w\s]+)\??", user_query, re.IGNORECASE) or \
                               re.match(r"Find ([\w\s]+)", user_query, re.IGNORECASE)
        if specific_query_match:
            queried_term = specific_query_match.group(1).strip()
            queried_term_as_column = ' '.join(word.capitalize() for word in queried_term.split())
            if queried_term_as_column and queried_term_as_column not in columns:
                # If the query was specific, ensure its term is a target column for explicit extraction
                columns.append(queried_term_as_column)

        return list(dict.fromkeys(columns))
    except Exception as e:
        st.error(f"Error determining data columns with LLM: {e}")
        return []

def extract_info_from_index(index, column_name, llm_for_extraction, original_query=""):
    prompt = f"""
    You are an expert financial data extractor.
    From the provided document context, precisely extract the value for the financial term: "{column_name}".
    - If the value is a number, provide just the number (e.g., "15.2%", "$1,250,000", "30 days").
    - If the value is text, provide the concise text (e.g., "John Doe", "Net 30").
    - If the information for "{column_name}" is NOT found or NOT explicitly stated in the provided context, you MUST respond with the exact text: N/A
    - Do not infer, calculate, or explain. Only extract the direct value present.
    - Do not add any conversational fluff or introductory phrases before or after the value or "N/A".

    Term to extract: "{column_name}"
    """
    try:
        query_engine = index.as_query_engine(llm=llm_for_extraction, response_mode="compact")
        response = query_engine.query(prompt)
        extracted_value = str(response.response).strip().strip('"').strip("'")

        prefix_to_remove = f"The value for \"{column_name}\" is "
        if extracted_value.startswith(prefix_to_remove):
            extracted_value = extracted_value[len(prefix_to_remove):].strip()
        prefix_to_remove_alt = f"The {column_name.lower()} is "
        if extracted_value.lower().startswith(prefix_to_remove_alt):
             extracted_value = extracted_value[len(prefix_to_remove_alt):].strip()
        if not extracted_value: return "N/A"
        return extracted_value
    except Exception as e:
        print(f"Error extracting '{column_name}': {e}")
        return "Extraction Error"


# --- Streamlit App UI and Logic ---
st.set_page_config(layout="centered", page_title="Finance Insights AI")
st.title("Finance Insights AI")

if 'groq_api_key_input' not in st.session_state: st.session_state.groq_api_key_input = os.getenv("GROQ_API_KEY") or ""
if 'llm' not in st.session_state: st.session_state.llm = None
if 'llm_api_key' not in st.session_state: st.session_state.llm_api_key = None
if 'embed_model' not in st.session_state: st.session_state.embed_model = None
if 'analysis_results_df' not in st.session_state: st.session_state.analysis_results_df = pd.DataFrame()
if 'full_results_data_list' not in st.session_state: st.session_state.full_results_data_list = []
if 'final_columns_for_display' not in st.session_state: st.session_state.final_columns_for_display = []
if 'selected_row_details' not in st.session_state: st.session_state.selected_row_details = None


# Load API key from .env file
current_api_key = os.getenv("GROQ_API_KEY")
if st.session_state.llm is None or st.session_state.llm_api_key != current_api_key:
    if current_api_key:
        st.session_state.llm = get_llm(current_api_key)
        st.session_state.llm_api_key = current_api_key
    else:
        st.session_state.llm = None; st.session_state.llm_api_key = None
if st.session_state.embed_model is None: st.session_state.embed_model = get_embedding_model()

models_ready = False
if st.session_state.llm and st.session_state.embed_model:
    Settings.llm = st.session_state.llm
    Settings.embed_model = st.session_state.embed_model
    Settings.chunk_size = 1024; Settings.chunk_overlap = 100
    models_ready = True
else:
    Settings.llm = None; Settings.embed_model = None
    if not current_api_key: st.warning("Groq API Key not found in .env file. Please add it to enable analysis.")

st.markdown("---")
st.subheader("1. Upload Your Documents")
uploaded_files = st.file_uploader("Drag and drop PDF files here", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
if uploaded_files: st.caption(f"{len(uploaded_files)} PDF(s) selected. Max 10.")
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("2. Ask Your Question")
high_level_query = st.text_area("What would you like to know across these documents?", height=100,
    placeholder="e.g., What are the key loan terms?\nWhat is the company's gross profit margin?")
st.markdown("<br>", unsafe_allow_html=True)
analyze_button = st.button("Generate Integrated Analysis Matrix", type="primary", use_container_width=True,
    disabled=not (models_ready and uploaded_files and high_level_query))
st.markdown("---")

if analyze_button and models_ready and uploaded_files and high_level_query:
    if len(uploaded_files) > 10:
        st.warning("For this, please upload a maximum of 10 documents.")
    else:
        st.session_state.analysis_results_df = pd.DataFrame()
        st.session_state.full_results_data_list = []
        st.session_state.final_columns_for_display = []
        st.session_state.selected_row_details = None

        with st.spinner("Analyzing documents... This may take several minutes. Please wait."):
            parsed_doc_texts_map = {}
            st.markdown("##### Parsing Documents...")
            parsing_progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                text = parse_pdf_to_text_docling(uploaded_file.getvalue(), filename_for_error=uploaded_file.name)
                parsed_doc_texts_map[uploaded_file.name] = text
                parsing_progress_bar.progress((i + 1) / len(uploaded_files))

            st.markdown("##### Step 1: Determining Data Columns for Matrix...")
            data_point_columns = determine_data_columns_with_llm(high_level_query, st.session_state.llm)

            # The SUMMARY_COLUMN_NAME will always be present.
            st.session_state.final_columns_for_display = ["Document", SUMMARY_COLUMN_NAME] + data_point_columns

            if not data_point_columns:
                st.info(f"No additional specific data columns identified. The matrix will primarily show '{SUMMARY_COLUMN_NAME}' for each document based on your query.")

            st.markdown(f"**Matrix Columns Planned:** `{'`, `'.join(st.session_state.final_columns_for_display[1:])}`")

            st.markdown(f"##### Step 2: Populating Integrated Matrix...")
            all_results_data_temp = []
            matrix_progress_text = st.empty()
            matrix_progress_bar = st.progress(0)

            for i, uploaded_file_ref in enumerate(uploaded_files):
                doc_name = uploaded_file_ref.name
                matrix_progress_text.text(f"Processing: {doc_name} ({i+1}/{len(uploaded_files)})")
                current_doc_results = {"Document": doc_name}
                doc_text = parsed_doc_texts_map.get(doc_name)

                if doc_text:
                    current_doc_results[SUMMARY_COLUMN_NAME] = get_per_document_summary_for_query(
                        doc_text, high_level_query, st.session_state.llm, doc_id=doc_name
                    )

                    if data_point_columns: # Only extract if specific data columns were determined
                        try:
                            llama_document = Document(text=doc_text, doc_id=doc_name)
                            index = VectorStoreIndex.from_documents([llama_document])
                            for col_name in data_point_columns:
                                extracted_val = extract_info_from_index(index, col_name, st.session_state.llm, original_query=high_level_query)
                                current_doc_results[col_name] = extracted_val
                        except Exception as e_index:
                            st.error(f"Error during data extraction for {doc_name}: {e_index}")
                            for col_name in data_point_columns: current_doc_results[col_name] = "Data Extraction Error"
                else:
                    current_doc_results[SUMMARY_COLUMN_NAME] = "PDF Parsing Error"
                    if data_point_columns:
                        for col_name in data_point_columns: current_doc_results[col_name] = "PDF Parsing Error"

                all_results_data_temp.append(current_doc_results)
                matrix_progress_bar.progress((i + 1) / len(uploaded_files))

            matrix_progress_text.text("Integrated Matrix Populated!")
            if all_results_data_temp:
                st.session_state.full_results_data_list = all_results_data_temp
                results_df = pd.DataFrame(all_results_data_temp)
                ordered_cols = [col for col in st.session_state.final_columns_for_display if col in results_df.columns]
                for col in results_df.columns:
                    if col not in ordered_cols: ordered_cols.append(col)
                if not results_df.empty:
                    results_df = results_df[ordered_cols].dropna(axis=1, how='all')
                st.session_state.analysis_results_df = results_df

# --- Display Results Persistently After Analysis ---
if not st.session_state.analysis_results_df.empty:
    st.markdown("---")
    st.subheader("📊 Integrated Analysis Matrix")

    column_config = {
        SUMMARY_COLUMN_NAME: st.column_config.TextColumn(
            SUMMARY_COLUMN_NAME,
            help=f"Direct answer to query from this document (truncated to {MAX_CHARS_IN_CELL} chars). Select row to see full text.",
            max_chars=MAX_CHARS_IN_CELL,
            width="large"
        )
    }
    for col_name in st.session_state.final_columns_for_display:
        if col_name not in ["Document", SUMMARY_COLUMN_NAME] and \
           col_name in st.session_state.analysis_results_df.columns and \
           st.session_state.analysis_results_df[col_name].dtype == 'object':
            try:
                if st.session_state.analysis_results_df[col_name].astype(str).str.len().max() > MAX_CHARS_IN_CELL / 1.5:
                    column_config[col_name] = st.column_config.TextColumn(
                        col_name, max_chars=int(MAX_CHARS_IN_CELL / 1.5), width="medium"
                    )
            except Exception: pass

    df_selection = st.dataframe(
        st.session_state.analysis_results_df,
        use_container_width=True,
        height=(min(len(st.session_state.analysis_results_df) + 1, 15) * 35),
        column_config=column_config,
        on_select="rerun",
        selection_mode="single-row",
        key="analysis_matrix_df"
    )

    if df_selection.selection.rows:
        selected_row_index = df_selection.selection.rows[0]
        if st.session_state.full_results_data_list and selected_row_index < len(st.session_state.full_results_data_list):
            selected_row_full_data = st.session_state.full_results_data_list[selected_row_index]
            st.session_state.selected_row_details = selected_row_full_data
            with st.expander(f"Full Details for: {selected_row_full_data.get('Document', 'Selected Row')}", expanded=True):
                for key, value in selected_row_full_data.items():
                    st.markdown(f"**{key}:**")
                    text_area_key = f"detail_text_{selected_row_full_data.get('Document', 'row')}_{key}_{selected_row_index}".replace(" ", "_").replace("(", "").replace(")", "")
                    st.text_area(f"details_{key}", str(value), height= (len(str(value)) // 70 + 3) * 20 if len(str(value)) > 100 else 100, disabled=True, key=text_area_key)

elif analyze_button:
    if not (models_ready and uploaded_files and high_level_query):
        if not models_ready: st.error("Models not ready. Check API key.")
        if not uploaded_files: st.warning("Please upload PDF documents.")
        if not high_level_query: st.warning("Please enter your query.")
    else:
        st.info("No matrix data could be generated. Please check logs or refine your query.")

st.markdown("---")

