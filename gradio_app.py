import gradio as gr
from model.analyzer import analyze_content
import time  # For simulating the loading bar

# Define the analysis function with a simulated loading bar
def analyze_with_loading(script):
    for progress in range(0, 101, 20):
        time.sleep(0.3)  # Simulate processing delay
        yield gr.update(value=f"Analyzing... {progress}%")
    # Final result after analysis
    result = analyze_content(script)
    yield gr.update(value=f"Analysis Complete! Triggers Detected: {result['detected_triggers']}")

# Create the Gradio interface
# Create the Gradio interface
with gr.Blocks(css=".center-text {text-align: center;} .gradient-bg {background: linear-gradient(135deg, #ff9a9e, #fad0c4);}") as iface:
    # Header with centered text
    gr.Markdown(
        """
        <div class="center-text">
            <h1><b>TREAT</b></h1>
            <h3><b>Trigger Recognition for Enjoyable and Appropriate Television</b></h3>
        </div>
        """,
        elem_classes="gradient-bg"
    )
    
    # Input Section
    script_input = gr.Textbox(lines=8, label="Input Text", placeholder="Paste your script here...")
    analyze_button = gr.Button("Analyze Content")
    
    # Loading Bar and Results
    loading_bar = gr.Textbox(label="Progress", interactive=False)
    results_output = gr.JSON(label="Results")
    
    # Connect the button to the function
    analyze_button.click(
        fn=analyze_with_loading,
        inputs=script_input,
        outputs=[loading_bar, results_output],
    )

# Launch the app
if __name__ == "__main__":
    iface.launch()