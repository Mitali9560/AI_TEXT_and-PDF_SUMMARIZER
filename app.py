import gradio as gr
import time

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from PyPDF2 import PdfReader

# =========================================
# LOAD MODEL
# =========================================

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# =========================================
# TEXT SUMMARIZATION FUNCTION
# =========================================

def summarize_text(text, summary_type):

    if not text or len(text.strip()) < 50:

        return (
            "Please provide longer readable text.",
            0,
            0,
            "0%",
            "0 sec"
        )

    try:

        start = time.time()

        text = text.strip()

        original_words = len(text.split())

        # Summary size control
        
        if summary_type == "Short":
            max_len = 40
            min_len = 15

        elif summary_type == "Medium":
            max_len = 120
            min_len = 40

        else:
            max_len = 220
            min_len = 80

        # Prompt
        prompt = f"Summarize this text: {text}"

        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        # Generate summary
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            early_stopping=True
        )

        # Decode output
        result = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        summary_words = len(result.split())

        compression_ratio = round(
            (summary_words / original_words) * 100,
            2
        )

        end = time.time()

        processing_time = round(end - start, 2)

        return (
            result,
            original_words,
            summary_words,
            f"{compression_ratio}%",
            f"{processing_time} sec"
        )

    except Exception as e:

        return (
            f"Summarization Error: {str(e)}",
            0,
            0,
            "0%",
            "0 sec"
        )

# =========================================
# PDF SUMMARIZATION FUNCTION
# =========================================

def summarize_pdf(file, summary_type):

    if file is None:

        return (
            "Please upload a PDF.",
            0,
            0,
            "0%",
            "0 sec"
        )

    try:

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted + " "

        text = text.strip()

        if len(text) < 50:

            return (
                "No sufficient readable text found in PDF.",
                0,
                0,
                "0%",
                "0 sec"
            )

        return summarize_text(text, summary_type)

    except Exception as e:

        return (
            f"PDF Error: {str(e)}",
            0,
            0,
            "0%",
            "0 sec"
        )

# =========================================
# CLEAR FUNCTION
# =========================================

def clear_fields():

    return (
        "",
        "",
        0,
        0,
        "",
        ""
    )

# =========================================
# EXAMPLE TEXT
# =========================================

example_text = """
Artificial Intelligence is transforming industries by enabling machines
to perform tasks that traditionally required human intelligence.
AI technologies such as machine learning, natural language processing,
and computer vision are widely used in healthcare, finance,
transportation, and education.
"""

# =========================================
# UI
# =========================================

with gr.Blocks(theme=gr.themes.Soft()) as app:

    gr.Markdown(
        """
        # 📄 AI Document Summarizer
        Generate concise summaries from lengthy text or PDF documents
        using transformer-based Generative AI models.
        """
    )

    # =====================================
    # TEXT TAB
    # =====================================

    with gr.Tab("Text Summarization"):

        input_text = gr.Textbox(
            lines=12,
            placeholder="Paste your lengthy document text here...",
            label="Input Text"
        )

        summary_type = gr.Radio(
            ["Short", "Medium", "Detailed"],
            value="Medium",
            label="Select Summary Type"
        )

        with gr.Row():

            generate_btn = gr.Button("Generate Summary")
            clear_btn = gr.Button("Clear")

        output_summary = gr.Textbox(
            label="Generated Summary",
            lines=8
        )

        with gr.Row():

            original_count = gr.Number(
                label="Original Word Count"
            )

            summary_count = gr.Number(
                label="Summary Word Count"
            )

        with gr.Row():

            compression_output = gr.Textbox(
                label="Compression Ratio"
            )

            time_output = gr.Textbox(
                label="Processing Time"
            )

        generate_btn.click(
            summarize_text,
            inputs=[input_text, summary_type],
            outputs=[
                output_summary,
                original_count,
                summary_count,
                compression_output,
                time_output
            ]
        )

        clear_btn.click(
            clear_fields,
            outputs=[
                input_text,
                output_summary,
                original_count,
                summary_count,
                compression_output,
                time_output
            ]
        )

        gr.Examples(
            examples=[[example_text, "Medium"]],
            inputs=[input_text, summary_type]
        )

    # =====================================
    # PDF TAB
    # =====================================

    with gr.Tab("PDF Summarization"):

        pdf_input = gr.File(
            file_types=[".pdf"],
            label="Upload PDF"
        )

        pdf_summary_type = gr.Radio(
            ["Short", "Medium", "Detailed"],
            value="Medium",
            label="Select Summary Type"
        )

        pdf_btn = gr.Button("Summarize PDF")

        pdf_output = gr.Textbox(
            label="Generated Summary",
            lines=8
        )

        with gr.Row():

            pdf_original = gr.Number(
                label="Original Word Count"
            )

            pdf_summary_words = gr.Number(
                label="Summary Word Count"
            )

        with gr.Row():

            pdf_compression = gr.Textbox(
                label="Compression Ratio"
            )

            pdf_time = gr.Textbox(
                label="Processing Time"
            )

        pdf_btn.click(
            summarize_pdf,
            inputs=[pdf_input, pdf_summary_type],
            outputs=[
                pdf_output,
                pdf_original,
                pdf_summary_words,
                pdf_compression,
                pdf_time
            ]
        )

app.launch()
