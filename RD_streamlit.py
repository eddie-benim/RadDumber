import streamlit as st
from raddumber import get_differential

st.set_page_config(page_title="X-ray Diagnostic Assistant", layout="centered")

st.title("🩻 X-ray Diagnostic Assistant")
st.write("Upload a chest X-ray or other diagnostic image. The assistant will generate a list of possible diagnoses based on the image.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Differential Diagnosis"):
        with st.spinner("Analyzing the image..."):
            with open("temp_input_image.png", "wb") as f:
                f.write(uploaded_file.getvalue())
            try:
                result = get_differential("temp_input_image.png")
                st.subheader("🩺 Differential Diagnosis")
                st.markdown("**Explanation:**")
                st.write(result.explanation)
                st.markdown("**Possible Diagnoses:**")
                for diag in result.diagnoses:
                    st.markdown(f"- {diag}")
            except Exception as e:
                st.error("Something went wrong during analysis.")
                st.exception(e)
