import streamlit as st
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set up page config
st.set_page_config(page_title="CodonChecker", page_icon=":dna:", layout="wide")

# Inject robust CSS for blue background and font
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, .stApp {
            font-family: 'Poppins', sans-serif;
            background-color: #dbeeff;
            color: #003366;
        }

        .block-container {
            background-color: #dbeeff !important;
            padding: 2rem;
        }

        h1, h2, h3, .stTabs [data-baseweb="tab"] {
            color: #003366 !important;
        }

        .stTextArea textarea, .stTextInput input {
            background-color: #f7fbff;
            border: 1px solid #99c2ff;
            border-radius: 5px;
        }

        .stButton>button {
            background-color: #0066cc;
            color: white;
            font-weight: bold;
            border-radius: 5px;
        }

        .css-1v0mbdj, .stDownloadButton button {
            background-color: #0059b3;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


# Inject custom CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
            background-color: #e0f0ff;
        }

        .header {
            font-size: 36px;
            font-weight: 600;
            color: #004488;
            padding: 20px 10px 10px;
        }

        .content {
            font-size: 18px;
            color: #333;
            padding: 10px;
        }

        .author-img, .mentor-img {
            border-radius: 50%;
        }

        .author-section, .mentor-section, .contact-section {
            margin-bottom: 20px;
            padding: 20px;
            background-color: #ffffffcc;
            border-radius: 12px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }

        .stTextArea textarea {
            background-color: #f8fcff;
            border-radius: 8px;
            font-family: 'Poppins', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

REFERENCE_ORGS = {
    'E. coli': {
        'UUU': 0.58, 'UUC': 0.42, 'UUA': 0.14, 'UUG': 0.13,
        'CUU': 0.12, 'CUC': 0.10, 'CUA': 0.04, 'CUG': 0.47,
        'AUU': 0.49, 'AUC': 0.39, 'AUA': 0.11, 'AUG': 1.00,
        'GUU': 0.28, 'GUC': 0.20, 'GUA': 0.18, 'GUG': 0.33,
        'UCU': 0.17, 'UCC': 0.15, 'UCA': 0.14, 'UCG': 0.14,
        'CCU': 0.18, 'CCC': 0.13, 'CCA': 0.20, 'CCG': 0.49,
        'ACU': 0.19, 'ACC': 0.40, 'ACA': 0.17, 'ACG': 0.25,
        'GCU': 0.35, 'GCC': 0.27, 'GCA': 0.22, 'GCG': 0.16,
        'UAU': 0.59, 'UAC': 0.41, 'UAA': 0.61, 'UAG': 0.09,
        'CAU': 0.57, 'CAC': 0.43, 'CAA': 0.34, 'CAG': 0.66,
        'AAU': 0.49, 'AAC': 0.51, 'AAA': 0.74, 'AAG': 0.26,
        'GAU': 0.63, 'GAC': 0.37, 'GAA': 0.68, 'GAG': 0.32,
        'UGU': 0.46, 'UGC': 0.54, 'UGA': 0.30, 'UGG': 1.00,
        'CGU': 0.36, 'CGC': 0.36, 'CGA': 0.07, 'CGG': 0.11,
        'AGU': 0.16, 'AGC': 0.25, 'AGA': 0.07, 'AGG': 0.04,
        'GGU': 0.37, 'GGC': 0.37, 'GGA': 0.13, 'GGG': 0.13
    },
    'H. sapiens': {
        'UUU': 0.46, 'UUC': 0.54, 'UUA': 0.08, 'UUG': 0.13,
        'CUU': 0.13, 'CUC': 0.20, 'CUA': 0.07, 'CUG': 0.40,
        'AUU': 0.36, 'AUC': 0.47, 'AUA': 0.17, 'AUG': 1.00,
        'GUU': 0.18, 'GUC': 0.24, 'GUA': 0.12, 'GUG': 0.46,
        'UCU': 0.19, 'UCC': 0.22, 'UCA': 0.15, 'UCG': 0.05,
        'CCU': 0.29, 'CCC': 0.32, 'CCA': 0.28, 'CCG': 0.11,
        'ACU': 0.26, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.11,
        'GCU': 0.27, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11,
        'UAU': 0.44, 'UAC': 0.56, 'UAA': 0.30, 'UAG': 0.24,
        'CAU': 0.42, 'CAC': 0.58, 'CAA': 0.27, 'CAG': 0.73,
        'AAU': 0.47, 'AAC': 0.53, 'AAA': 0.43, 'AAG': 0.57,
        'GAU': 0.46, 'GAC': 0.54, 'GAA': 0.42, 'GAG': 0.58,
        'UGU': 0.46, 'UGC': 0.54, 'UGA': 0.46, 'UGG': 1.00,
        'CGU': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21,
        'AGU': 0.15, 'AGC': 0.24, 'AGA': 0.20, 'AGG': 0.20,
        'GGU': 0.16, 'GGC': 0.34, 'GGA': 0.25, 'GGG': 0.25
    }
}


def validate_and_clean_sequence(sequence):
    sequence = sequence.upper().replace('\n', '').replace(' ', '')
    valid_bases = {'A', 'T', 'C', 'G'}
    if not all(base in valid_bases for base in sequence):
        raise ValueError("Invalid characters found in sequence.")
    if len(sequence) < 9:
        raise ValueError("Sequence too short (minimum 9 bases).")
    remainder = len(sequence) % 3
    if remainder != 0:
        st.warning(f"Sequence length not divisible by 3. Removing {remainder} base(s).")
        sequence = sequence[:-remainder]
    return sequence

def count_codons(sequence):
    codon_counts = defaultdict(int)
    for i in range(0, len(sequence), 3):
        codon = sequence[i:i+3].replace('T', 'U')
        codon_counts[codon] += 1
    return codon_counts

def calculate_frequencies(codon_counts):
    total = sum(codon_counts.values())
    return {codon: count / total for codon, count in codon_counts.items()}

def calculate_cai(frequencies, reference):
    log_sum = 0
    count = 0
    for codon, freq in frequencies.items():
        if codon in reference and reference[codon] > 0:
            log_sum += freq * np.log(reference[codon])
            count += freq
    return np.exp(log_sum / count) if count > 0 else 0

def plot_codon_usage(frequencies, reference, org_name):
    codons = sorted(set(frequencies) | set(reference))
    sample_values = [frequencies.get(c, 0) for c in codons]
    ref_values = [reference.get(c, 0) for c in codons]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(codons))
    width = 0.4

    ax.bar(x - width/2, sample_values, width, label='Your Sequence', color='skyblue')
    ax.bar(x + width/2, ref_values, width, label=f'{org_name} Reference', color='salmon')

    ax.set_xticks(x)
    ax.set_xticklabels(codons, rotation=90)
    ax.set_ylabel('Relative Frequency')
    ax.set_title('Codon Usage Comparison')
    ax.legend()
    return fig

# Home Page
def show_home_page():
    st.markdown('<div class="header">👩‍🔬 Welcome to CodonChecker</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
        <strong>CodonChecker</strong> is a powerful tool developed as part of academic research in Bioinformatics,
        specifically designed to help researchers and students analyze codons in nucleotide sequences.
        <br><br>
        <strong>Key Features</strong>:
        <ul>
            <li>Analyze codon sequences</li>
            <li>Check codon usage in genetic sequences</li>
            <li>View results with interactive visualizations</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

def show_code_analyzer_page():
    st.markdown('<div class="header">🔬 Code Analyzer</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
        Analyze codon patterns in your DNA sequence by uploading a FASTA file or pasting the sequence directly.
        </div>
    """, unsafe_allow_html=True)

    input_method = st.radio("Select input method:", ["Paste Sequence", "Upload FASTA File"], horizontal=True)

    sequence = ""
    if input_method == "Paste Sequence":
        sequence = st.text_area("Paste your DNA sequence here:", height=150)
    else:
        file = st.file_uploader("Upload FASTA file", type=["fasta", "fa", "txt"])
        if file:
            content = file.read().decode()
            lines = content.strip().splitlines()
            sequence = "".join(line.strip() for line in lines if not line.startswith(">"))

    if sequence:
        try:
            # --- Sequence Processing ---
            clean_seq = validate_and_clean_sequence(sequence)

            col1, col2 = st.columns(2)
            col1.metric("Input Length", f"{len(sequence)} bp")
            col2.metric("Analyzed Length", f"{len(clean_seq)} bp ({len(clean_seq) // 3} codons)")

            codon_counts = count_codons(clean_seq)
            frequencies = calculate_frequencies(codon_counts)

            # --- Reference Selection ---
            st.markdown("### 🔍 Reference Organism")
            ref_org = st.selectbox("Compare with organism:", list(REFERENCE_ORGS.keys()))
            cai_score = calculate_cai(frequencies, REFERENCE_ORGS[ref_org])
            st.metric("🧮 Codon Adaptation Index (CAI)", f"{cai_score:.3f}")

            # --- Codon Usage Table ---
            df = pd.DataFrame({
                "Codon": codon_counts.keys(),
                "Count": codon_counts.values(),
                "Frequency": [f"{frequencies[c]:.4f}" for c in codon_counts]
            }).sort_values(by="Count", ascending=False)

            st.subheader("📋 Codon Usage Table")
            st.dataframe(df, use_container_width=True, height=400)

            # --- Codon Plot ---
            st.subheader("📊 Codon Usage Comparison")
            fig = plot_codon_usage(frequencies, REFERENCE_ORGS[ref_org], ref_org)
            st.pyplot(fig)

            # --- Download Button ---
            st.download_button(
                "📥 Download CSV",
                data=df.to_csv(index=False).encode(),
                file_name="codon_usage_results.csv",
                mime="text/csv"
            )

        except ValueError as e:
            st.error(f"❌ {e}")


# About Page
def show_about_page():
    st.markdown('<div class="header">👩‍🏫 About the Author and Mentor</div>', unsafe_allow_html=True)

    st.markdown('<div class="author-section">', unsafe_allow_html=True)
    st.image("https://github.com/Dnyanu2502/codonchecker/blob/main/IMG-20180808-WA0050.jpg?raw=true", width=150, caption="Dnyaneshwari Keshav Bankar")
    st.subheader("Dnyaneshwari Keshav Bankar")
    st.markdown("""
    - *Currently pursuing*: Masters in Bioinformatics  
    - *University*: Department of Bioinformatics, DES Pune University  
    - *Skills*: Python, Bioinformatics, Data Analysis  
    - *Project*: Developed CodonChecker as part of academic research  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="mentor-section">', unsafe_allow_html=True)
    st.image("https://media.licdn.com/dms/image/v2/D5603AQF9gsU7YBjWVg/profile-displayphoto-shrink_800_800/B56ZZI.WrdH0Ac-/0/1744981029051?e=1752105600&v=beta&t=NY99PWbYHr9Wi8VkPoMtFBfLhqvNl1uLKgH1_hetXY0", width=120, caption="Dr. Kushagra Kashyap")
    st.subheader("Dr. Kushagra Kashyap")
    st.markdown("""
    - *Position*: Assistant Professor, Department of Bioinformatics, DES Pune University  
    - *LinkedIn*: [Dr. Kushagra Kashyap LinkedIn](https://www.linkedin.com/in/dr-kushagra-kashyap-b230a3bb)  
    - *Research Interests*: Bioinformatics, Computational Biology, Data Science  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="contact-section">', unsafe_allow_html=True)
    st.subheader("📧 Contact Information")
    st.markdown("""
    - **Email**: [3522411034@despu.edu.in](mailto:3522411034@despu.edu.in)  
    - **Github**: [View source code](https://github.com/dnyaneshwaribankar/codonchecker)  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Acknowledgement Page
def show_acknowledgement_page():
    st.markdown('<div class="header">🌟 Acknowledgements</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="content">
        I sincerely acknowledge the support and contributions that helped make this project possible.
        <br><br>
        <strong>Mentorship & Guidance:</strong><br>
        My deepest gratitude to Dr. Kushagra Kashyap (Assistant Professor, DES Pune University) for his invaluable guidance, constant encouragement, and expert supervision throughout this project.
        <br><br>
        <strong>Institutional Support:</strong><br>
        - The Department of Bioinformatics, DES Pune University for providing the academic environment and resources.<br>
        - The university administration for fostering an ecosystem of innovation and research excellence.
        <br><br>
        <strong>Technical Resources:</strong><br>
        - Python and Streamlit developer communities.<br>
        - BioPython and other bioinformatics libraries.
        <br><br>
        <strong>Personal Support:</strong><br>
        - My family and friends for their unwavering support and patience throughout this journey.
        <br><br>
        <em>"Knowledge grows when shared - we gratefully stand on the shoulders of those who came before us, and hope this tool will support those who follow in bioinformatics research."</em>
        </div>
    """, unsafe_allow_html=True)

# Streamlit Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "📊 Code Analyzer", "👤 About", "🙌 Acknowledgement"])

with tab1:
    show_home_page()

with tab2:
    show_code_analyzer_page()

with tab3:
    show_about_page()

with tab4:
    show_acknowledgement_page()




