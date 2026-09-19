================================================
FILE: README.md
================================================
# 🦙➡️🔍 LaTeX OCR with Qwen2.5-VL

Extract LaTeX code from images of mathematical equations using a local vision-language model, served entirely on your own machine with [Ollama](https://ollama.com/) — no API keys, no cloud calls.

Upload a photo or screenshot of a handwritten or printed equation, and get back clean, ready-to-use LaTeX — rendered live in the app.

---

## Features

- Upload `.png`, `.jpg`, or `.jpeg` images of math equations
- Automatic LaTeX extraction via a local multimodal model (`qwen2.5vl`)
- Instant preview of both the raw LaTeX source and its rendered output
- One-click clear to reset and try another image
- Fully local/offline inference — your images never leave your machine

---

## Demo

| Upload | Extracted LaTeX | Rendered |
|---|---|---|
| *(add a screenshot here)* | ```\frac{-b \pm \sqrt{b^2-4ac}}{2a}``` | $\frac{-b \pm \sqrt{b^2-4ac}}{2a}$ |

---

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running locally

---

## Setup

1. **Clone this repo**
   ```bash
   git clone https://github.com/NatalieKong13/LaTeX-OCR-with-Qwen.git
   cd LaTeX-OCR-with-Qwen
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull the vision model**
   ```bash
   ollama pull qwen2.5vl
   ```

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. Open the local URL Streamlit prints (usually `http://localhost:8501`) in your browser.

---

## Usage

1. Upload an image containing a mathematical equation from the sidebar.
2. Click **Extract LaTeX **.
3. View the extracted LaTeX code and its rendered form in the main panel.
4. Click **Clear** to reset and try a new image.

---

## How it works

The app sends your uploaded image, along with a strict prompt instructing the model to return *only* raw LaTeX (no explanations, no `$` delimiters, no document boilerplate), to a locally running `qwen2.5vl` model via the [`ollama`](https://pypi.org/project/ollama/) Python client. The model's text response is displayed as-is and also rendered using Streamlit's built-in `st.latex()`.

---

## Tech Stack

- [Streamlit](https://streamlit.io/) — web UI
- [Ollama](https://ollama.com/) — local LLM/VLM serving
- [Qwen2.5-VL](https://ollama.com/library/qwen2.5vl) — vision-language model for OCR
- [Pillow (PIL)](https://python-pillow.org/) — image handling

---

## Acknowledgements

Adapted from the original [Llama 3.2 Vision LaTeX OCR example](https://github.com/patchy631/ai-engineering-hub) by patchy631, migrated to Qwen2.5-VL.

## License

MIT



================================================
FILE: app.py
================================================
import streamlit as st
import ollama
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="LaTeX OCR with Llama 3.2 Vision",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description in main area
st.title("🦙 LaTeX OCR with Llama 3.2 Vision")

# Add clear button to top right
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract LaTeX code from images using Llama 3.2 Vision!</p>', unsafe_allow_html=True)

st.markdown("---")
# Move upload controls to sidebar
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")
        
        if st.button("Extract LaTeX 🔍", type="primary"):
            with st.spinner("Processing image..."):
                try:
                    response = ollama.chat(
                        model='qwen2.5vl',
                        messages=[{
                            'role': 'user',
                            'content': """Understand the mathematical equation in the provided image and output the corresponding LaTeX code.
                            Here are some guidelines you MUST follow or you will be penalized:
                            - NEVER include any additional text or explanations.
                            - DON'T add dollar signs ($) around the LaTeX code.
                            - DO NOT extract simplified versions of the equations.
                            - NEVER add documentclass, packages or begindocument.
                            - DO NOT explain the symbols used in the equation.
                            - Output only the LaTeX code corresponding to the mathematical equations in the image.""",
                            'images': [uploaded_file.getvalue()]
                        }]
                    )
                    st.session_state['ocr_result'] = response.message.content
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

# Main content area for results
if 'ocr_result' in st.session_state:
    st.markdown("### LaTeX Code")
    st.code(st.session_state['ocr_result'], language='latex')

    st.markdown("### LaTeX Rendered")

    cleaned_latex = st.session_state['ocr_result'].replace(r"\[", "").replace(r"\]", "")
    st.latex(cleaned_latex)
    
else:
    st.info("Upload an image and click 'Extract LaTeX' to see the results here.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Llama Vision Model2 | [Report an Issue](https://github.com/patchy631/ai-engineering-hub/issues)")


================================================
FILE: requirements.txt
================================================
altair==6.2.2
annotated-types==0.8.0
anyio==4.15.1
attrs==26.1.0
certifi==2026.7.22
charset-normalizer==3.5.1
click==8.5.0
exceptiongroup==1.3.1
h11==0.16.0
httpcore==1.0.9
httptools==0.8.0
httpx==0.28.1
idna==3.19
itsdangerous==2.2.0
Jinja2==3.1.6
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
MarkupSafe==3.0.3
narwhals==2.26.0
numpy==2.2.6
ollama==0.6.2
packaging==26.3
pandas==2.3.3
pillow==12.3.0
protobuf==7.36.1
pyarrow==25.0.1
pydantic==2.13.5
pydantic_core==2.46.5
pydeck==0.9.3
python-dateutil==2.9.0.post0
python-multipart==0.0.32
pytz==2026.3.post1
referencing==0.37.0
requests==2.34.2
rpds-py==0.30.0
six==1.17.0
starlette==1.6.0
streamlit==1.64.0
toml==0.10.2
typing-inspection==0.4.4
typing_extensions==4.16.0
tzdata==2026.4
urllib3==2.8.0
uvicorn==0.53.0
websockets==16.1.1


