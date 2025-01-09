import gradio as gr
import time
from model.analyzer import analyze_content

# Custom CSS for the interface
css = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&display=swap');

#treat-container {
    background: linear-gradient(135deg, #ffE6F0 0%, #E6F0FF 100%);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    font-family: 'Nunito', sans-serif;
}

#treat-title {
    text-align: center;
    color: #FF69B4;
    font-size: 3.5rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

#treat-subtitle {
    text-align: center;
    color: #666;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}

#treat-subtitle span {
    color: #FF69B4;
    font-weight: bold;
}

.content-box {
    background: rgba(255,255,255,0.9);
    border-radius: 15px;
    border: 2px solid #FFB6C1;
    padding: 1rem;
}

.analyze-button {
    background: linear-gradient(45deg, #FF69B4, #87CEEB) !important;
    border: none !important;
    border-radius: 25px !important;
    color: white !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: bold !important;
    padding: 0.8rem 2rem !important;
    transition: transform 0.2s !important;
}

.analyze-button:hover {
    transform: translateY(-2px) !important;
}

.results-container {
    background: rgba(255,255,255,0.9);
    border-radius: 15px;
    padding: 1.5rem;
    margin-top: 1rem;
    border: 2px solid #87CEEB;
}

#loading-bar {
    height: 6px;
    background: linear-gradient(90deg, #FF69B4, #87CEEB);
    border-radius: 3px;
    transition: width 0.3s ease;
}
"""

def analyze_with_loading(text):
    # Simulate loading progress (you can integrate this with your actual analysis)
    for i in range(100):
        time.sleep(0.02)  # Simulate processing time
        yield {"progress": i + 1}
    
    # Perform the actual analysis
    result = analyze_content(text)
    
    # Format the results
    if result["detected_triggers"] == ["None"]:
        triggers_text = "No triggers detected"
    else:
        triggers_text = ", ".join(result["detected_triggers"])
    
    yield {
        "progress": 100,
        "result": f"""
        <div class='results-container'>
            <h3 style='color: #FF69B4; margin-bottom: 1rem;'>Analysis Results</h3>
            <p><strong>Triggers Detected:</strong> {triggers_text}</p>
            <p><strong>Confidence:</strong> {result['confidence']}</p>
            <p><strong>Analysis Time:</strong> {result['analysis_timestamp']}</p>
        </div>
        """
    }

with gr.Blocks(css=css) as iface:
    with gr.Column(elem_id="treat-container"):
        gr.HTML("""
            <div id="treat-title">TREAT</div>
            <div id="treat-subtitle">
                <span>T</span>rigger <span>R</span>ecognition for 
                <span>E</span>njoyable and <span>A</span>ppropriate 
                <span>T</span>elevision
            </div>
        """)
        
        text_input = gr.Textbox(
            label="Enter your content for analysis",
            placeholder="Paste your script or content here...",
            lines=8,
            elem_classes=["content-box"]
        )
        
        analyze_btn = gr.Button(
            "🍬 Analyze Content",
            elem_classes=["analyze-button"]
        )
        
        progress = gr.Number(
            value=0,
            visible=False,
            elem_id="progress-value"
        )
        
        gr.HTML("""
            <div style="width: 100%; height: 6px; background: #eee; border-radius: 3px; margin: 1rem 0;">
                <div id="loading-bar" style="width: 0%"></div>
            </div>
        """)
        
        output = gr.HTML()
        
        # JavaScript for updating the loading bar
        gr.HTML("""
            <script>
                function updateLoadingBar(progress) {
                    document.getElementById('loading-bar').style.width = progress + '%';
                }
                
                // Watch for changes to the progress value
                const observer = new MutationObserver((mutations) => {
                    mutations.forEach((mutation) => {
                        if (mutation.type === 'attributes' && mutation.attributeName === 'value') {
                            const progress = document.getElementById('progress-value').value;
                            updateLoadingBar(progress);
                        }
                    });
                });
                
                // Start observing the progress value element
                observer.observe(document.getElementById('progress-value'), {
                    attributes: true
                });
            </script>
        """)
        
        analyze_btn.click(
            fn=analyze_with_loading,
            inputs=[text_input],
            outputs=[gr.State({"progress": 0, "result": ""}), output],
            show_progress=False
        )

if __name__ == "__main__":
    iface.launch()