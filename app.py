"""
Toxic Content Classification Application
Main Streamlit application for classifying toxic content from text and images
"""

import streamlit as st
from PIL import Image
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Import custom modules
from imagecaption import get_captioner
from textclassifier import get_classifier
from database import get_database

# Page configuration
st.set_page_config(
    page_title="Toxic Content Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .toxic {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .non-toxic {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .stats-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'captioner' not in st.session_state:
    st.session_state.captioner = None
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'database' not in st.session_state:
    st.session_state.database = None

# Load models
@st.cache_resource
def load_models():
    """Load all models (cached to avoid reloading)"""
    captioner = get_captioner()
    classifier = get_classifier()
    database = get_database()
    return captioner, classifier, database

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Toxic Content Classification System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Navigation")
        page = st.radio(
            "Select Page",
            ["Classify Content", "View Database", "Statistics"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This application classifies toxic content from:\n"
            "- 📝 Direct text input\n"
            "- 🖼️ Image captions (using BLIP-1)"
        )
        
        st.markdown("---")
        st.markdown("### Models Used")
        st.markdown("""
        - **Image Captioning**: BLIP-1
        - **Text Classification**: LSTM
        """)
    
    # Load models with progress indicator
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading models... This may take a moment..."):
            try:
                captioner, classifier, database = load_models()
                st.session_state.captioner = captioner
                st.session_state.classifier = classifier
                st.session_state.database = database
                st.session_state.models_loaded = True
                st.success("✅ Models loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading models: {str(e)}")
                return
    
    # Page routing
    if page == "Classify Content":
        classify_content_page()
    elif page == "View Database":
        view_database_page()
    elif page == "Statistics":
        statistics_page()

def classify_content_page():
    """Main page for classifying content"""
    st.header("📊 Classify Content")
    
    # Input method selection
    input_method = st.radio(
        "Select Input Method:",
        ["Text Input", "Image Upload"],
        horizontal=True
    )
    
    if input_method == "Text Input":
        handle_text_input()
    else:
        handle_image_input()

def handle_text_input():
    """Handle direct text input classification"""
    st.subheader("📝 Text Input")
    
    text_input = st.text_area(
        "Enter text to classify:",
        height=150,
        placeholder="Type or paste your text here..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        classify_btn = st.button("🔍 Classify", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.rerun()
    
    if classify_btn:
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to classify.")
            return
        
        with st.spinner("Classifying text..."):
            # Classify text
            result = st.session_state.classifier.classify_text(text_input)
            
            # Save to database
            st.session_state.database.add_record("text", text_input, result)
            
            # Display results
            display_classification_results(text_input, result, "text")

def handle_image_input():
    """Handle image upload and caption generation"""
    st.subheader("🖼️ Image Upload")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to generate a caption and classify it"
    )
    
    if uploaded_file is not None:
        # Display image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            # FIX: Changed use_container_width to width parameter
            st.image(image, caption="Uploaded Image", width=400)
        
        with col2:
            if st.button("🔍 Generate Caption & Classify", type="primary", use_container_width=True):
                with st.spinner("Generating caption..."):
                    # Generate caption
                    caption = st.session_state.captioner.generate_caption(image)
                    st.success(f"**Generated Caption:** {caption}")
                
                with st.spinner("Classifying caption..."):
                    # Classify caption
                    result = st.session_state.classifier.classify_text(caption)
                    
                    # Save to database
                    st.session_state.database.add_record("image_caption", caption, result)
                    
                    # Display results
                    display_classification_results(caption, result, "image_caption")

def display_classification_results(text, result, input_type):
    """Display classification results in a formatted way"""
    st.markdown("---")
    st.subheader("📊 Classification Results")
    
    classification = result.get("classification", "unknown")
    confidence = result.get("confidence", 0.0)
    
    # Result box
    is_toxic = classification != "non-toxic"
    box_class = "toxic" if is_toxic else "non-toxic"
    
    st.markdown(f"""
        <div class="result-box {box_class}">
            <h3>Classification: {classification.upper()}</h3>
            <p><strong>Confidence:</strong> {confidence:.2%}</p>
            <p><strong>Input Type:</strong> {input_type.replace('_', ' ').title()}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Detailed scores
    if "detailed_scores" in result and result["detailed_scores"]:
        st.subheader("📈 Detailed Scores")
        
        scores = result["detailed_scores"]
        
        # Create two columns for scores
        col1, col2 = st.columns(2)
        
        for idx, (label, score) in enumerate(scores.items()):
            col = col1 if idx % 2 == 0 else col2
            with col:
                st.metric(
                    label=label.replace('_', ' ').title(),
                    value=f"{score:.2%}"
                )

def view_database_page():
    """Display all database records"""
    st.header("🗄️ Database Records")
    
    df = st.session_state.database.get_all_records()
    
    if df.empty:
        st.info("📭 No records found in the database.")
        return
    
    # Display controls
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        filter_type = st.selectbox(
            "Filter by type:",
            ["All", "Text Input", "Image Caption"]
        )
    
    with col3:
        if st.button("🗑️ Clear Database", type="secondary"):
            if st.button("⚠️ Confirm Clear"):
                st.session_state.database.clear_database()
                st.success("Database cleared!")
                st.rerun()
    
    # Filter data
    if filter_type == "Text Input":
        df = df[df['input_type'] == 'text']
    elif filter_type == "Image Caption":
        df = df[df['input_type'] == 'image_caption']
    
    # Display dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="toxic_content_database.csv",
        mime="text/csv"
    )

def statistics_page():
    """Display statistics about classifications"""
    st.header("📊 Statistics")
    
    stats = st.session_state.database.get_statistics()
    
    if stats['total_records'] == 0:
        st.info("📭 No data available yet. Start classifying content to see statistics!")
        return
    
    # Overall statistics
    st.subheader("📈 Overall Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="stats-card">
                <h2>{stats['total_records']}</h2>
                <p>Total Records</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stats-card">
                <h2>{stats['text_inputs']}</h2>
                <p>Text Inputs</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="stats-card">
                <h2>{stats['image_inputs']}</h2>
                <p>Image Captions</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Classification distribution
    st.subheader("📊 Classification Distribution")
    
    if stats['classifications']:
        import pandas as pd
        classification_df = pd.DataFrame(
            list(stats['classifications'].items()),
            columns=['Classification', 'Count']
        )
        
        st.bar_chart(
            classification_df.set_index('Classification'),
            use_container_width=True
        )
        
        # Show table
        st.dataframe(
            classification_df,
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()
