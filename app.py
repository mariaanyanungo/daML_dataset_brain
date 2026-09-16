import random
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

from model_utils import load_model, load_config, predict, UNCERTAINTY_THRESHOLD

st.set_page_config(page_title="Brain MRI Tumor Classifier", layout="centered")

st.title("Brain MRI Tumor Classifier")
st.markdown("Upload a brain MRI slice below to classify the tumor type" )


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_config():
    return load_config()


try:
    model = get_model()
    config = get_config()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(
        "Could not load the trained model. Make sure `model/brain_mri_classifier.keras` and "
        f"`model/preprocessing_config.json` exist (run the notebook first). Error: {e}"
    )

st.header("Upload an MRI slice")
uploaded_file = st.file_uploader("Accepted formats: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and MODEL_LOADED:
    pil_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Running model..."):
            predicted_class, confidence, all_probs, is_uncertain = predict(model, config, pil_image)

        st.subheader("Prediction")
        st.metric("Predicted class", predicted_class.replace("_", " ").title())
        st.metric("Confidence", f"{confidence*100:.1f}%")

        st.markdown("**Probability per class:**")
        for cls, p in sorted(all_probs.items(), key=lambda kv: -kv[1]):
            st.progress(p, text=f"{cls.replace('_',' ').title()}: {p*100:.1f}%")

    st.divider()

    explanation = CLASS_EXPLANATIONS.get(predicted_class, "an unrecognised pattern")
    st.markdown(
        f"**In plain language:** The model predicts this MRI slice shows {explanation}, "
        f"with **{confidence*100:.1f}% confidence**."
    )

    if is_uncertain:
        st.warning(
            f"The model's top prediction has confidence below {UNCERTAINTY_THRESHOLD*100:.0f}%. "
            "This means the image may not closely resemble anything the model saw during training, "
            "or the case may genuinely be ambiguous. **Treat this result as unreliable** — it should "
            "not be treated as a meaningful diagnosis either way."
        )
    else:
        st.info(
            "Reminder: even a confident-looking prediction from this coursework prototype is **not** "
            "a diagnosis. Any real use requires review by a qualified clinician."
        )

elif uploaded_file is None:
    st.caption("Upload an image above to see a live prediction.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_class = tf.argmax(predictions[0])
        top_output = predictions[:, top_class]

    grads = tape.gradient(top_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


if uploaded_file is not None and MODEL_LOADED:
    with st.expander("🔍 Bonus: What did the model focus on? (Grad-CAM)"):
        try:
            from model_utils import preprocess_image
            import matplotlib.cm as cm

            last_conv = find_last_conv_layer(model)
            x = preprocess_image(pil_image, config)
            heatmap = make_gradcam_heatmap(x, model, last_conv)

            heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(pil_image.convert("RGB").size)
            heatmap_colored = cm.jet(np.array(heatmap_resized) / 255.0)[:, :, :3]
            base_img = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
            overlay = 0.6 * base_img + 0.4 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)

            st.image(overlay, caption="Grad-CAM overlay — warmer colors = more influence on the prediction",
                      use_container_width=True)
            st.caption(
                "This heatmap shows which regions of the image most influenced the model's decision. "
                "It is a rough visual aid, not proof of clinical relevance."
            )
        except Exception as e:
            st.caption(f"Grad-CAM explanation unavailable for this image/model: {e}")


st.header("Try a sample image")
st.caption("No image of your own? Browse a few sample images and see the model's live prediction on each.")

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "brain_mri"

if MODEL_LOADED and SAMPLE_DIR.exists():
    class_dirs = sorted([d for d in SAMPLE_DIR.iterdir() if d.is_dir()])
    if class_dirs:
        chosen_class = st.selectbox(
            "Pick a class to sample from", [d.name for d in class_dirs]
        )
        sample_files = list((SAMPLE_DIR / chosen_class).glob("*.jpg"))
        if sample_files:
            if st.button("Show a random sample from this class"):
                sample_path = random.choice(sample_files)
                sample_img = Image.open(sample_path)
                pred_class, conf, probs, uncertain = predict(model, config, sample_img)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(sample_img, caption=f"Sample from '{chosen_class}' folder", use_container_width=True)
                with c2:
                    st.write(f"**True (folder) label:** {chosen_class}")
                    st.write(f"**Model prediction:** {pred_class}")
                    st.write(f"**Confidence:** {conf*100:.1f}%")
                    if uncertain:
                        st.warning("Model is uncertain on this sample.")
else:
    st.caption("Sample image folder not found — this section is optional.")

st.divider()

