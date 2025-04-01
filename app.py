import streamlit as st
import pandas as pd
import os
import io
from PIL import Image
import json
import base64
import sys

# Import directly from main.py
from main import DataCaptureEngine

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Wealth Director - Data Extraction Tool",
    page_icon="📊",
    layout="wide"
)

# Initialize the DataCaptureEngine directly
@st.cache_resource
def get_engine():
    openai_api_key = "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"
    engine = DataCaptureEngine(
        assets_csv_path="assets.csv",
        openai_api_key=openai_api_key
    )
    return engine

def categorize_assets(asset_types):
    """Organize asset types by category for easier selection"""
    categories = {}
    for asset in asset_types:
        broad_cat = asset["broad_category"]
        category = asset["category"]
        subcategory = asset["subcategory"]
        asset_type = asset["type"]
        
        key = f"{broad_cat} > {category}"
        if key not in categories:
            categories[key] = {}
        
        if subcategory not in categories[key]:
            categories[key][subcategory] = []
            
        categories[key][subcategory].append(asset_type)
    
    return categories

def display_asset_data(data, asset_type):
    """Display extracted asset data in a nice format"""
    if not data or "error" in data:
        st.error(f"Error: {data.get('error', 'Unknown error')}")
        return
    
    # Create an expandable section for general data view
    with st.expander("Extracted Information", expanded=True):
        # Check if any field has list/dictionary values (nested data)
        has_nested_data = any(isinstance(v, (list, dict)) for v in data.values())
        
        if has_nested_data:
            # If we have nested data (like lists of dictionaries), display them separately
            for field, value in data.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                    # This is a list of dictionaries, display as table
                    st.subheader(f"{field}")
                    st.dataframe(pd.DataFrame(value), use_container_width=True)
                elif isinstance(value, dict):
                    # This is a dictionary, display as table with keys and values
                    st.subheader(f"{field}")
                    df = pd.DataFrame([{"Key": k, "Value": v} for k, v in value.items()])
                    st.dataframe(df, use_container_width=True)
                else:
                    # Simple values
                    st.write(f"**{field}:** {value}")
        else:
            # Simple flat data, show as a single table
            field_data = []
            for field, value in data.items():
                # Convert simple lists to comma-separated strings
                if isinstance(value, list):
                    value = ", ".join([str(v) for v in value])
                field_data.append({"Field": field, "Value": value})
            
            if field_data:
                st.dataframe(pd.DataFrame(field_data), use_container_width=True)
            else:
                st.info("No data was extracted from the document")
    
    # Add code for displaying raw data as a formatted table instead of JSON
    with st.expander("Raw Data View", expanded=False):
        # Convert the nested data to a flattened format for display
        flattened_data = []
        for key, value in data.items():
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                for i, item in enumerate(value):
                    row = {"Field": f"{key}[{i}]"}
                    for k, v in item.items():
                        row[k] = v
                    flattened_data.append(row)
            else:
                flattened_data.append({"Field": key, "Value": value})
        
        # Display as table
        if flattened_data:
            st.dataframe(pd.DataFrame(flattened_data), use_container_width=True)

def main():
    st.title("Wealth Director - Asset Data Extraction")
    st.subheader("Extract information from financial documents automatically")
    
    # Get engine instance
    engine = get_engine()
    
    # Fetch asset types directly from engine
    asset_types = engine.get_asset_types()
    
    if not asset_types:
        st.warning("Could not load asset types. Please check if assets.csv file exists.")
        return
    
    # Categorize assets for easier selection
    categorized_assets = categorize_assets(asset_types)
    
    # Sidebar for asset type selection
    with st.sidebar:
        st.header("Select Asset Type")
        
        # Step 1: Select category
        selected_category = st.selectbox(
            "Asset Category",
            options=sorted(categorized_assets.keys()),
            index=0
        )
        
        # Step 2: Select subcategory
        subcategories = categorized_assets[selected_category]
        selected_subcategory = st.selectbox(
            "Asset Subcategory",
            options=sorted(subcategories.keys()),
            index=0
        )
        
        # Step 3: Select asset type
        asset_types_list = subcategories[selected_subcategory]
        selected_asset_type = st.selectbox(
            "Specific Asset Type",
            options=sorted(asset_types_list),
            index=0
        )
    
    # Main content area - File upload and processing
    st.header(f"Upload Document for {selected_asset_type}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a file (PDF, Image, Excel, Word)",
        type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "docx"]
    )
    
    if uploaded_file:
        # Display file information
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.2f} KB",
            "File type": uploaded_file.type
        }
        st.write("File Details:", file_details)
        
        # Display file preview
        if uploaded_file.type.startswith('image'):
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        elif uploaded_file.name.endswith('.pdf'):
            try:
                # Try to display the first page of the PDF
                import fitz  # PyMuPDF
                
                pdf_bytes = uploaded_file.getvalue()
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                if doc.page_count > 0:
                    page = doc.load_page(0)  # First page
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    
                    img_bytes = pix.tobytes()
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    st.image(img, caption=f"{uploaded_file.name} (Page 1)", use_container_width=True)
                    
                    if doc.page_count > 1:
                        st.info(f"PDF has {doc.page_count} pages. Showing only the first page.")
                doc.close()
            except Exception as e:
                st.warning(f"Could not preview PDF: {str(e)}")
        
        # Process button
        if st.button("Extract Data"):
            with st.spinner("Processing document..."):
                # Get the file content
                file_content = uploaded_file.getvalue()
                
                # Process document directly using the engine
                result = engine.process_document(
                    file_content=file_content,
                    filename=uploaded_file.name,
                    asset_type=selected_asset_type
                )
                
                # Display results
                display_asset_data(result, selected_asset_type)
                
                # Add download option for extracted data
                if result and not ("error" in result):
                    json_str = json.dumps(result, indent=4)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="{uploaded_file.name}_extracted_data.json">Download Extracted Data (JSON)</a>'
                    st.markdown(href, unsafe_allow_html=True)

if __name__ == "__main__":
    # Fix for PyTorch/Streamlit file watcher issue
    # Run app with --no-watchdog flag to prevent PyTorch custom class errors
    main()
