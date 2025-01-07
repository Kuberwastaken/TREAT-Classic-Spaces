import gradio as gr
from model.model import get_detailed_analysis

def analyze_script(script):
    return get_detailed_analysis(script)

iface = gr.Interface(fn=analyze_script, inputs="text", outputs="json")
iface.launch()