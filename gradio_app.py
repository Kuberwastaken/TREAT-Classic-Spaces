import gradio as gr
from model.analyzer import analyze_content

# Create and launch the Gradio interface
iface = gr.Interface(
    fn=analyze_content,
    inputs=gr.Textbox(lines=8, label="Input Text"),
    outputs=gr.JSON(),
    title="Content Analysis",
    description="Analyze text content for sensitive topics"
)

if __name__ == "__main__":
    iface.launch()