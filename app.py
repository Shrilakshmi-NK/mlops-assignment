# import gradio as gr
# import joblib
# import pandas as pd

# # Load your trained model
# pipeline = joblib.load('model.joblib')

# def predict_cpu_usage(cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes, controller_kind):
#     input_data = pd.DataFrame({
#         'cpu_request': [cpu_request],
#         'mem_request': [mem_request],
#         'cpu_limit': [cpu_limit],
#         'mem_limit': [mem_limit],
#         'runtime_minutes': [runtime_minutes],
#         'controller_kind': [controller_kind]
#     })
#     prediction = pipeline.predict(input_data)[0]
#     return f"Predicted CPU Usage: {prediction:.6f}"

# # Create the interface
# iface = gr.Interface(
#     fn=predict_cpu_usage,
#     inputs=[
#         gr.Number(label="CPU Request (cores)"),
#         gr.Number(label="Memory Request (bytes)"),
#         gr.Number(label="CPU Limit (cores)"),
#         gr.Number(label="Memory Limit (bytes)"),
#         gr.Number(label="Runtime (minutes)"),
#         gr.Dropdown(["ReplicaSet", "StatefulSet", "DaemonSet", "Job"], label="Controller Kind")
#     ],
#     outputs=gr.Textbox(label="Prediction"),
#     title="CPU Usage Predictor (MLOps Assignment)",
#     description="Trained on 151k+ Kubernetes pods | RandomForest | R² = 0.8321",
#     allow_flagging="never"
# )

# # THIS LINE GIVES YOU THE FREE PUBLIC URL
# iface.launch(share=True)

import gradio as gr
import joblib
import pandas as pd

# Load your trained model
pipeline = joblib.load('model.joblib')

def predict_cpu_usage(cpu_request, mem_request, cpu_limit, mem_limit, runtime_minutes, controller_kind):
    input_data = pd.DataFrame({
        'cpu_request': [cpu_request],
        'mem_request': [mem_request],
        'cpu_limit': [cpu_limit],
        'mem_limit': [mem_limit],
        'runtime_minutes': [runtime_minutes],
        'controller_kind': [controller_kind]
    })
    prediction = pipeline.predict(input_data)[0]
    return f"Predicted CPU Usage: {prediction:.6f}"

# Create interface
iface = gr.Interface(
    fn=predict_cpu_usage,
    inputs=[
        gr.Number(label="CPU Request (cores)"),
        gr.Number(label="Memory Request (bytes)"),
        gr.Number(label="CPU Limit (cores)"),
        gr.Number(label="Memory Limit (bytes)"),
        gr.Number(label="Runtime (minutes)"),
        gr.Dropdown(["ReplicaSet", "StatefulSet", "DaemonSet", "Job"], label="Controller Kind")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="CPU Usage Predictor - MLOps Assignment",
    description="Trained on 151k+ Kubernetes pods | RandomForest | R² = 0.8321"
)

# THIS GIVES YOU THE FREE PUBLIC URL
iface.launch(share=True)