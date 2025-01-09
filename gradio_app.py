import gradio as gr
from model.analyzer import analyze_content
import asyncio
import time

# Custom CSS for dark theme and modern animations
custom_css = """
.gradio-container {
    background: #121212 !important;
    color: #fff !important;
    overflow: hidden;
    transition: background 0.5s ease;
}

.treat-title {
    text-align: center;
    padding: 30px;
    margin-bottom: 30px;
    background: rgba(18, 18, 18, 0.85);
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    animation: slideInFromTop 1s ease-out;
}

.treat-title h1 {
    font-size: 3.5em;
    color: #ffa726;
    margin-bottom: 10px;
    font-weight: bold;
    animation: fadeInText 1.5s ease-out;
}

.treat-title p {
    font-size: 1.3em;
    color: #ff7043;
    animation: fadeInText 1.5s ease-out 0.5s;
}

.highlight {
    color: #ffa726;
    font-weight: bold;
}

.content-area, .results-area {
    background: rgba(33, 33, 33, 0.9) !important;
    border-radius: 15px !important;
    padding: 30px !important;
    margin: 20px 0 !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5) !important;
    opacity: 0;
    animation: fadeInUp 1s forwards;
}

.gradio-textbox textarea {
    background-color: #333 !important;
    color: #fff !important;
    border: 1px solid #444 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 1.1em !important;
    transition: border-color 0.3s ease;
}

.gradio-textbox textarea:focus {
    border-color: #ffa726 !important;
}

.gradio-button {
    background-color: #ff7043 !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 24px !important;
    font-size: 1.2em !important;
    transition: transform 0.3s ease, background-color 0.3s ease;
    margin: 20px 0 !important;
}

.gradio-button:hover {
    transform: scale(1.1) !important;
    background-color: #ffa726 !important;
}

.gradio-button:active {
    transform: scale(0.98) !important;
    background-color: #fb8c00 !important;
}

label {
    color: #ccc !important;
    font-weight: 500 !important;
    margin-bottom: 10px !important;
}

.center-row {
    display: flex;
    justify-content: center;
    align-items: center;
}

.footer {
    text-align: center;
    margin-top: 40px;
    font-size: 1.2em;
    color: #bdbdbd;
    opacity: 0;
    animation: fadeInUp 1s forwards 1.5s;
}

.footer p {
    color: #ffa726;
}

@keyframes slideInFromTop {
    0% { transform: translateY(-50px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}

@keyframes fadeInText {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(30px); }
    100% { opacity: 1; transform: translateY(0); }
}

"""

def analyze_with_loading(text, progress=gr.Progress()):
    """
    Synchronous wrapper for the async analyze_content function
    """
    # Initialize progress
    progress(0, desc="Starting analysis...")
    
    # Initial setup phase
    for i in range(30):
        time.sleep(0.02)  # Reduced sleep time
        progress((i + 1) / 100)
    
    # Perform analysis
    progress(0.3, desc="Processing text...")
    try:
        # Use asyncio.run to handle the async function call
        result = asyncio.run(analyze_content(text))
    except Exception as e:
        return f"Error during analysis: {str(e)}"
    
    # Final processing
    for i in range(70, 100):
        time.sleep(0.02)  # Reduced sleep time
        progress((i + 1) / 100)
    
    # Format the results
    triggers = result["detected_triggers"]
    if triggers == ["None"]:
        return "✓ No triggers detected in the content."
    else:
        trigger_list = "\n".join([f"• {trigger}" for trigger in triggers])
        return f"⚠ Triggers Detected:\n{trigger_list}"

# Create the Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as iface:
    # Title section
    gr.HTML("""
        <div class="treat-title">
            <h1>TREAT</h1>
            <p><span class="highlight">T</span>rigger 
               <span class="highlight">R</span>ecognition for 
               <span class="highlight">E</span>njoyable and 
               <span class="highlight">A</span>ppropriate 
               <span class="highlight">T</span>elevision</p>
        </div>
    """)
    
    # Content input section
    with gr.Row():
        with gr.Column(elem_classes="content-area"):
            input_text = gr.Textbox(
                label="Content to Analyze",
                placeholder="Paste your content here...",
                lines=8,
                interactive=True
            )
    
    # Button section
    with gr.Row(elem_classes="center-row"):
        analyze_btn = gr.Button(
            "✨ Analyze Content",
            variant="primary"
        )
    
    # Results section
    with gr.Row():
        with gr.Column(elem_classes="results-area"):
            output_text = gr.Textbox(
                label="Analysis Results",
                lines=5,
                interactive=False
            )
    
    # Set up the click event
    analyze_btn.click(
        fn=analyze_with_loading,
        inputs=[input_text],
        outputs=[output_text],
        api_name="analyze"
    )

    # Footer section
    gr.HTML("""
        <div class="footer">
            <p>Made with 💖 by Kuber Mehta</p>
        </div>
    """)

if __name__ == "__main__":
    # Launch without the 'ssr' argument
    iface.launch(
        share=False,
        debug=True,
        show_error=True
    )
