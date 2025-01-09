# gradio_app.py
import gradio as gr
from model.analyzer import analyze_content
import time

# Custom CSS for styling
custom_css = """
.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

.treat-title {
    text-align: center;
    padding: 20px;
    margin-bottom: 20px;
    background: linear-gradient(135deg, #fce4ec 0%, #e3f2fd 100%);
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

.content-area {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 20px;
    margin: 20px 0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.results-area {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 15px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.gradio-container {
    background: linear-gradient(135deg, #fce4ec 0%, #e3f2fd 100%) !important;
}

#analyze-btn {
    background-color: #d81b60 !important;
    color: white !important;
    border-radius: 25px !important;
    padding: 10px 20px !important;
    font-size: 1.1em !important;
    transition: transform 0.2s !important;
}

#analyze-btn:hover {
    transform: scale(1.05) !important;
}
"""

def analyze_with_loading(text, progress=gr.Progress()):
    # Initialize progress
    progress(0, desc="Starting analysis...")
    
    # Simulate initial loading (model preparation)
    for i in range(30):
        time.sleep(0.1)
        progress((i + 1) / 100)
    
    # Perform actual analysis
    progress(0.3, desc="Processing text...")
    result = analyze_content(text)
    
    # Simulate final processing
    for i in range(70, 100):
        time.sleep(0.05)
        progress((i + 1) / 100)
    
    # Format the results for display
    triggers = result["detected_triggers"]
    if triggers == ["None"]:
        return "No triggers detected in the content."
    else:
        trigger_list = "\n".join([f"• {trigger}" for trigger in triggers])
        return f"Triggers Detected:\n{trigger_list}"

with gr.Blocks(css=custom_css) as iface:
    # Title section using HTML
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
    
    with gr.Column(elem_classes="content-area"):
        input_text = gr.Textbox(
            label="Content to Analyze",
            placeholder="Paste your content here...",
            lines=8
        )
    
    # Analysis button
    analyze_btn = gr.Button(
        "Analyze Content",
        elem_id="analyze-btn"
    )
    
    with gr.Column(elem_classes="results-area"):
        output_text = gr.Textbox(
            label="Analysis Results",
            lines=5,
            readonly=True
        )
    
    # Set up the click event
    analyze_btn.click(
        fn=analyze_with_loading,
        inputs=[input_text],
        outputs=[output_text],
        api_name="analyze"
    )

if __name__ == "__main__":
    iface.launch()