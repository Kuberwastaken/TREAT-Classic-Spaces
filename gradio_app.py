# gradio_app.py
import gradio as gr
from model.analyzer import analyze_content
import asyncio
import time

# Custom CSS for styling
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #fce4ec 0%, #e3f2fd 100%) !important;
}

.treat-title {
    text-align: center;
    padding: 20px;
    margin-bottom: 20px;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.treat-title h1 {
    font-size: 3em;
    color: #d81b60;
    margin-bottom: 10px;
    font-weight: bold;
}

.treat-title p {
    font-size: 1.2em;
    color: #5c6bc0;
}

.highlight {
    color: #d81b60;
    font-weight: bold;
}

.content-area, .results-area {
    background: rgba(255, 255, 255, 0.9) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin: 20px 0 !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
}

/* Input/Output styling */
.gradio-textbox textarea {
    background-color: white !important;
    color: #333 !important;
    border: 1px solid #ddd !important;
    border-radius: 8px !important;
    padding: 10px !important;
}

.gradio-button {
    background-color: #d81b60 !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 10px 20px !important;
    font-size: 1.1em !important;
    transition: transform 0.2s !important;
    margin: 10px 0 !important;
}

.gradio-button:hover {
    transform: scale(1.05) !important;
    background-color: #c2185b !important;
}

/* Label styling */
label {
    color: #333 !important;
    font-weight: 500 !important;
    margin-bottom: 8px !important;
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
        # Run the async function in a sync context
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(analyze_content(text))
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
                lines=8
            )
    
    # Button section
    with gr.Row(justify="center"):
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

if __name__ == "__main__":
    # Launch with custom configurations
    iface.launch(
        share=False,
        debug=True,
        show_error=True,
        ssr=False  # Disable SSR to prevent potential issues
    )