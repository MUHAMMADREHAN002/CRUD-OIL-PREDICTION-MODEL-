import gradio as gr
import pickle

# Load your model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define prediction function
def predict(input_data):
    try:
        prediction = model.predict([input_data])
        return prediction[0]
    except Exception as e:
        return f"Error: {str(e)}"

# Define the interface
with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Predict with Your Pickle Model")

    with gr.Row():
        input_box = gr.Textbox(label="Enter input (comma-separated if multiple features)")

    output_label = gr.Textbox(label="Prediction")

    def wrapped_predict(input_text):
        # Convert input string to list of floats
        try:
            input_data = list(map(float, input_text.split(',')))
            return predict(input_data)
        except:
            return "Invalid input. Please enter numbers separated by commas."

    submit_btn = gr.Button("Predict")
    submit_btn.click(fn=wrapped_predict, inputs=input_box, outputs=output_label)

# Launch
demo.launch()
