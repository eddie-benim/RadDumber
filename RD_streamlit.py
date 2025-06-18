import streamlit as st
from raddumber import get_differential
from agents import set_default_openai_key

set_default_openai_key(st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="X-ray Diagnostic Assistant", layout="centered")

st.title("X-ray Diagnostic Assistant")
st.write("Upload a chest X-ray or other diagnostic image. The assistant will generate a list of possible diagnoses based on the image.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Differential Diagnosis"):
        with st.spinner("Analyzing the image..."):
            try:
                image_bytes = uploaded_file.getvalue()
                result = get_differential(image_bytes)

                st.subheader("Differential Diagnosis")
                st.markdown("**Explanation:**")
                st.write(result.explanation)

                if hasattr(result, "diagnoses") and result.diagnoses:
                    st.markdown("**Possible Diagnoses:**")
                    for diag in result.diagnoses:
                        condition = getattr(diag, "condition", None) or diag.get("condition")
                        probability = getattr(diag, "probability", None) or diag.get("probability")
                        if condition and probability is not None:
                            st.markdown(f"- **{condition}** — {probability}%")
                        elif condition:
                            st.markdown(f"- **{condition}**")
                else:
                    st.info("No diagnoses were confidently identified.")

            except Exception as e:
                st.error("Something went wrong during analysis.")
                st.exception(e)
