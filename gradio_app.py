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
with gr.Blocks() as iface:
    # Header
    gr.Markdown(
        """
        # **TREAT**
        ### **Trigger Recognition for Enjoyable and Appropriate Television**
        """
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
