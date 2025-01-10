import gradio as gr
from model.analyzer import analyze_content
import asyncio
import time

custom_css = """
* {
    font-family: 'Roboto', sans-serif;
    transition: all 0.3s ease;
}
.gradio-container {
    background: #121212 !important;
    color: #fff !important;
    overflow: hidden;
    transition: all 0.5s ease;
}
.treat-title {
    text-align: center;
    padding: 40px;
    margin-bottom: 30px;
    background: rgba(18, 18, 18, 0.85);
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    animation: slideInFromTop 1s ease-out;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.treat-title:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 16px rgba(79, 70, 229, 0.3);
}
.treat-title h1 {
    font-size: 5em;
    color: #4f46e5;
    margin-bottom: 10px;
    font-weight: bold;
    animation: fadeInText 1.5s ease-out;
    transition: color 0.3s ease;
}
.treat-title:hover h1 {
    color: #5749d3;
}
.treat-title p {
    font-size: 1.3em;
    color: #4f46e5;
    animation: fadeInText 1.5s ease-out 0.5s;
}
.highlight {
    color: #4f46e5;
    font-weight: bold;
    transition: color 0.3s ease, transform 0.3s ease;
    display: inline-block;
}
.highlight:hover {
    color: #5749d3;
    transform: scale(1.1);
}
.content-area, .results-area {
    background: rgba(33, 33, 33, 0.9) !important;
    border-radius: 15px !important;
    padding: 30px !important;
    margin: 20px 0 !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.5) !important;
    opacity: 0;
    animation: fadeInUp 1s forwards;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.content-area:hover, .results-area:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 16px rgba(79, 70, 229, 0.2) !important;
}
.gradio-textbox textarea {
    background-color: #333 !important;
    color: #fff !important;
    border: 1px solid #444 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    font-size: 1.1em !important;
    transition: all 0.3s ease;
}
.gradio-textbox textarea:hover {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 10px rgba(79, 70, 229, 0.2) !important;
}
.gradio-textbox textarea:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 15px rgba(79, 70, 229, 0.3) !important;
    transform: translateY(-2px);
}
.gradio-button {
    background-color: #4f46e5 !important;
    color: white !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 24px !important;
    font-size: 1.2em !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    margin: 20px 0 !important;
    position: relative;
    overflow: hidden;
}
.gradio-button::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    transition: width 0.6s ease, height 0.6s ease;
}
.gradio-button:hover {
    transform: scale(1.05) translateY(-2px);
    background-color: #5749d3 !important;
    box-shadow: 0 6px 15px rgba(79, 70, 229, 0.4);
}
.gradio-button:hover::before {
    width: 300px;
    height: 300px;
}
.gradio-button:active {
    transform: scale(0.98) translateY(1px);
    background-color: #4b40bb !important;
}
label {
    color: #ccc !important;
    font-weight: 500 !important;
    margin-bottom: 10px !important;
    transition: color 0.3s ease;
}
label:hover {
    color: #fff !important;
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
    transition: all 0.3s ease;
}
.footer:hover {
    transform: translateY(-3px);
}
.footer p {
    color: #4f46e5;
    transition: all 0.3s ease;
}
.footer .heart {
    display: inline-block;
    transition: transform 0.3s ease;
    animation: pulse 1.5s infinite;
}
.footer:hover .heart {
    transform: scale(1.3);
}
.footer a {
    color: #4f46e5;
    text-decoration: none;
    position: relative;
    transition: all 0.3s ease;
}
.footer a:hover {
    color: #5749d3;
}
.footer a::after {
    content: '';
    position: absolute;
    width: 0;
    height: 2px;
    bottom: -2px;
    left: 0;
    background-color: #5749d3;
    transition: width 0.3s ease;
}
.footer a:hover::after {
    width: 100%;
}
footer {
    visibility: hidden;
}     

@keyframes slideInFromTop {
    0% { transform: translateY(-50px); opacity: 0; }
    100% { transform: translateY(0); opacity: 1; }
}
@keyframes fadeInText {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(30px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
}
"""

def analyze_with_loading(text, progress=gr.Progress()):
    """
    Synchronous wrapper for the async analyze_content function with smooth progress updates
    """
    # Initialize progress
    progress(0, desc="Starting analysis...")
    
    # Initial setup phase - smoother progression
    for i in range(25):
        time.sleep(0.04)  # Slightly longer sleep for smoother animation
        progress((i + 1) / 100, desc="Initializing analysis...")
    
    # Pre-processing phase
    for i in range(25, 45):
        time.sleep(0.03)
        progress((i + 1) / 100, desc="Pre-processing content...")
    
    # Perform analysis
    progress(0.45, desc="Analyzing content...")
    try:
        result = asyncio.run(analyze_content(text))
        
        # Analysis progress simulation
        for i in range(45, 75):
            time.sleep(0.03)
            progress((i + 1) / 100, desc="Processing results...")
            
    except Exception as e:
        return f"Error during analysis: {str(e)}"
    
    # Final processing with smooth progression
    for i in range(75, 100):
        time.sleep(0.02)
        progress((i + 1) / 100, desc="Finalizing results...")
    
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
            <p>Made with <span class="heart">💖</span> by <a href="https://www.linkedin.com/in/kubermehta/" target="_blank">Kuber Mehta</a></p>
        </div>
    """)

if __name__ == "__main__":
    iface.launch(
        share=False,
        debug=True,
        show_error=True
    )