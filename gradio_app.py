import gradio as gr
from model.analyzer import analyze_content
import asyncio
import time
import httpx
import subprocess
import atexit

# Start the API server
def start_api_server():
    # Start uvicorn in a subprocess
    process = subprocess.Popen(["uvicorn", "script_search_api:app", "--reload"])
    return process

# Stop the API server
def stop_api_server(process):
    process.terminate()

# Register the exit handler
api_process = start_api_server()
atexit.register(stop_api_server, api_process)


custom_css = """
* {
    font-family: 'Inter', system-ui, sans-serif;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.gradio-container {
    background: #0a0a0f !important;
    color: #fff !important;
    min-height: 100vh;
    position: relative;
    overflow: hidden;
}

/* Animated Background */
.gradio-container::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: 
        linear-gradient(125deg, 
            #0a0a0f 0%, 
            rgba(99, 102, 241, 0.05) 30%,
            rgba(99, 102, 241, 0.1) 50%,
            rgba(99, 102, 241, 0.05) 70%,
            #0a0a0f 100%);
    animation: gradientMove 15s ease infinite;
    background-size: 400% 400%;
    z-index: 0;
}

/* Floating Particles */
.gradio-container::after {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: radial-gradient(circle at center, transparent 0%, #0a0a0f 70%),
                url("data:image/svg+xml,%3Csvg width='100' height='100' viewBox='0 0 100 100' xmlns='http://www.w3.org/2000/svg'%3E%3Ccircle cx='50' cy='50' r='1' fill='rgba(99, 102, 241, 0.15)'/%3E%3C/svg%3E");
    opacity: 0.5;
    animation: floatingParticles 20s linear infinite;
    z-index: 1;
}

/* Futuristic Header */
.treat-title {
    text-align: center;
    padding: 3rem 1rem;
    position: relative;
    overflow: hidden;
    z-index: 2;
    background: linear-gradient(180deg, 
        rgba(99, 102, 241, 0.1),
        transparent 70%);
}

.treat-title::before {
    content: '';
    position: absolute;
    top: 0;
    left: 50%;
    width: 80%;
    height: 1px;
    background: linear-gradient(90deg, 
        transparent,
        rgba(99, 102, 241, 0.5),
        transparent);
    transform: translateX(-50%);
    animation: scanline 3s ease-in-out infinite;
}

.treat-title h1 {
    font-size: 4.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, 
        #2a2b55 0%,
        #6366f1 50%,
        #2a2b55 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.05em;
    animation: gradientFlow 8s ease infinite;
    position: relative;
}

.treat-title h1::after {
    content: attr(data-text);
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(135deg, 
        transparent 0%,
        rgba(99, 102, 241, 0.4) 50%,
        transparent 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    opacity: 0.5;
    animation: textGlow 4s ease-in-out infinite;
}

.treat-title p {
    font-size: 1.1rem;
    color: rgba(255, 255, 255, 0.7);
    max-width: 600px;
    margin: 0 auto;
    position: relative;
    animation: fadeInUp 1s ease-out;
}

/* Tabs Styling */
.tabs {
    background: rgba(17, 17, 27, 0.7);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 1rem;
    margin: 0 1rem 2rem 1rem;
    position: relative;
    z-index: 2;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 30px rgba(99, 102, 241, 0.1);
    animation: floatIn 1s ease-out;
}

.tabs::before {
    content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    right: -1px;
    bottom: -1px;
    background: linear-gradient(45deg,
        rgba(99, 102, 241, 0.1),
        transparent,
        rgba(99, 102, 241, 0.1));
    border-radius: 16px;
    z-index: -1;
    animation: borderGlow 4s ease-in-out infinite;
}

/* Content Area */
.content-area {
    background: rgba(17, 17, 27, 0.7) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
    animation: fadeScale 0.5s ease-out;
}

.content-area::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center,
        rgba(99, 102, 241, 0.1) 0%,
        transparent 70%);
    animation: rotateGradient 10s linear infinite;
}

/* Input Fields */
.gradio-textbox textarea {
    background: rgba(17, 17, 27, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    border-radius: 8px !important;
    color: rgba(255, 255, 255, 0.9) !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
    transition: all 0.3s ease;
    position: relative;
    z-index: 2;
}

.gradio-textbox textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.2) !important;
    background: rgba(17, 17, 27, 0.8) !important;
    transform: translateY(-2px);
}

/* Buttons */
.gradio-button {
    background: linear-gradient(45deg, 
        #6366f1,
        #818cf8,
        #6366f1) !important;
    background-size: 200% auto !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1.5rem !important;
    letter-spacing: 0.025em !important;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease !important;
    animation: gradientFlow 3s ease infinite;
}

.gradio-button::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at center,
        rgba(255, 255, 255, 0.2) 0%,
        transparent 70%);
    transform: scale(0);
    transition: transform 0.5s ease;
}

.gradio-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 20px rgba(99, 102, 241, 0.4) !important;
}

.gradio-button:hover::before {
    transform: scale(1);
}

/* Results Area */
.results-area {
    background: rgba(17, 17, 27, 0.7) !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    border-radius: 12px !important;
    margin-top: 2rem !important;
    backdrop-filter: blur(10px);
    animation: slideUp 0.5s ease-out;
    position: relative;
    overflow: hidden;
}

.footer {
    text-align: center;
    padding: 2rem 0;
    margin-top: 3rem;
    font-size: 1.0rem;
    position: relative;
    z-index: 2;
}

.footer p {
    color: rgba(255, 255, 255, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

.footer .heart {
    color: #6366f1;
    display: inline-block;
    position: relative;
    font-size: 1.0rem;
    transform-origin: center;
    animation: heartbeat 1.5s ease infinite;
}

.footer .heart::before,
.footer .heart::after {
    content: '✦';
    position: absolute;
    opacity: 0;
    font-size: 0.6rem;
    animation: sparkle 1.5s ease infinite;
}

.footer .heart::before {
    top: -8px;
    left: -8px;
    animation-delay: 0.2s;
}

.footer .heart::after {
    top: -8px;
    right: -8px;
    animation-delay: 0.4s;
}

.footer .name {
    color: #6366f1;
    text-decoration: none;
    position: relative;
    transition: all 0.3s ease;
    padding: 0 4px;
}

.footer .name:hover {
    color: #818cf8;
}

footer {
    visibility: hidden;
}

/* Animations */
@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes floatingParticles {
    0% { transform: translateY(0); }
    100% { transform: translateY(-100%); }
}

@keyframes scanline {
    0% { transform: translateX(-150%) scaleX(0.5); opacity: 0; }
    50% { transform: translateX(-50%) scaleX(1); opacity: 1; }
    100% { transform: translateX(50%) scaleX(0.5); opacity: 0; }
}

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes textGlow {
    0% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.02); }
    100% { opacity: 0.3; transform: scale(1); }
}

@keyframes borderGlow {
    0% { opacity: 0.5; }
    50% { opacity: 1; }
    100% { opacity: 0.5; }
}

@keyframes rotateGradient {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

@keyframes fadeScale {
    0% { opacity: 0; transform: scale(0.95); }
    100% { opacity: 1; transform: scale(1); }
}

@keyframes slideUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes floatIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(10px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes heartbeat {
    0% { transform: scale(1); }
    10% { transform: scale(1.2); }
    20% { transform: scale(0.9); }
    30% { transform: scale(1.1); }
    40% { transform: scale(0.95); }
    50% { transform: scale(1); }
    100% { transform: scale(1); }
}

@keyframes sparkle {
    0% { transform: scale(0); opacity: 0; }
    50% { transform: scale(1.2); opacity: 1; }
    100% { transform: scale(0); opacity: 0; }
    }
"""
# Start the API server
def start_api_server():
    # Start uvicorn in a subprocess
    process = subprocess.Popen(["uvicorn", "script_search_api:app", "--reload"])
    return process

# Stop the API server
def stop_api_server(process):
    process.terminate()

# Register the exit handler
api_process = start_api_server()
atexit.register(stop_api_server, api_process)

async def analyze_with_progress(movie_name, progress=gr.Progress()):
    """Handle analysis with progress updates in Gradio"""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Start the analysis
            response = await client.get(
                "http://localhost:8000/api/start_analysis",
                params={"movie_name": movie_name}
            )
            response.raise_for_status()
            task_id = response.json()["task_id"]
            
            # Poll for progress
            while True:
                progress_response = await client.get(
                    f"http://localhost:8000/api/progress/{task_id}"
                )
                progress_response.raise_for_status()
                status = progress_response.json()
                
                # Update Gradio progress
                progress(status["progress"], desc=status["status"])
                
                if status["is_complete"]:
                    if status["error"]:
                        return f"Error: {status['error']}"
                    elif status["result"]:
                        triggers = status["result"].get("detected_triggers", [])
                        if not triggers or triggers == ["None"]:
                            return "✓ No triggers detected in the content."
                        else:
                            trigger_list = "\n".join([f"• {trigger}" for trigger in triggers])
                            return f"⚠ Triggers Detected:\n{trigger_list}"
                    break
                
                await asyncio.sleep(0.5)
    
    except Exception as e:
        return f"Error: {str(e)}"

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

# Update the Gradio interface with new styling
import gradio as gr
from model.analyzer import analyze_content
import asyncio
import time
import httpx
import subprocess
import atexit

# Keep your existing CSS and server setup code...
# [Previous code until the interface definition remains the same]

# Update the Gradio interface with fixed button handling
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as iface:
    # Title section
    gr.HTML("""
        <div class="treat-title">
            <h1 data-text="TREAT">TREAT</h1>
            <p>Trigger Recognition for Enjoyable and Appropriate Television</p>
        </div>
    """)
    
    with gr.Tabs() as tabs:
        with gr.Tab("Content Analysis"):  # Changed from TabItem to Tab
            with gr.Column():
                input_text = gr.Textbox(
                    label="ANALYZE CONTENT",
                    placeholder="Enter the content you want to analyze...",
                    lines=8
                )
                analyze_btn = gr.Button("✨ Analyze")
        
        with gr.Tab("Movie Search"):  # Changed from TabItem to Tab
            with gr.Column():
                search_query = gr.Textbox(
                    label="SEARCH MOVIES",
                    placeholder="Type a movie title to search...",
                    lines=1
                )
                search_button = gr.Button("🔍 Search")
    
    output_text = gr.Textbox(
        label="ANALYSIS RESULTS",
        lines=5,
        interactive=False
    )
    
    status_text = gr.Markdown(
        value=""
    )
    
    # Define click events
    analyze_btn.click(
        fn=analyze_with_loading,
        inputs=input_text,
        outputs=output_text
    )
    
    search_button.click(
        fn=analyze_with_progress,
        inputs=search_query,
        outputs=output_text
    )

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