%%writefile app.py

import streamlit as st
import torch
import soundfile as sf
import os
import io

# --- 1. Model Loading (must be inside app.py for Streamlit) ---
# Check for GPU availability and set device/dtype
if torch.cuda.is_available():
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    device = "cuda:0"
    st.sidebar.success("GPU is available. Using CUDA for TTS.")
else:
    dtype = torch.float32 # Fallback to float32 for CPU
    device = "cpu"
    st.sidebar.warning("No GPU found. Falling back to CPU, which may be slower.")

# Using st.cache_resource to load the model only once
@st.cache_resource
def load_tts_model(selected_device, selected_dtype):
    try:
        from qwen_tts import Qwen3TTSModel
        model_instance = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", device_map=selected_device, dtype=selected_dtype)
        st.sidebar.success(f"Qwen3-TTS Model loaded successfully on {selected_device.upper()}!")
        return model_instance
    except Exception as e:
        st.sidebar.error(f"Failed to load Qwen3-TTS Model on {selected_device}: {e}")
        return None

model = load_tts_model(device, dtype)

# --- 2. TTS Inference Function ---
def qwen_tts_inference_streamlit(ref_audio_file, ref_text, text_to_generate):
    if model is None:
        st.error("TTS Model is not loaded. Cannot generate speech.")
        return None

    if not ref_audio_file:
        st.warning("1. 참조 음성 파일을 업로드해주세요.")
        return None
    if not ref_text:
        st.warning("2. 참조 음성에서 말한 내용을 입력해주세요.")
        return None
    if not text_to_generate:
        st.warning("3. 변환할 문장을 입력해주세요.")
        return None

    try:
        # Save uploaded audio to a temporary file
        temp_ref_audio_path = "temp_ref_audio.wav"
        with open(temp_ref_audio_path, "wb") as f:
            f.write(ref_audio_file.getvalue())

        wavs, sr = model.generate_voice_clone(
            text=text_to_generate,
            language="Korean",
            ref_audio=temp_ref_audio_path,
            ref_text=ref_text,
        )

        # Clean up temporary reference audio file
        os.remove(temp_ref_audio_path)

        # Save generated audio to a temporary file for st.audio and download
        output_filename = "generated_speech.wav"
        sf.write(output_filename, wavs[0], sr)
        return output_filename
    except Exception as e:
        st.error(f"음성 변환 중 오류가 발생했습니다: {e}")
        return None

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Qwen3 TTS 보이스 클로닝", layout="centered", initial_sidebar_state="collapsed")

st.title("🗣️ Qwen3 TTS 보이스 클로닝 데모 (한국어)")
st.markdown("--- ---")

st.header("1. 참조 음성 업로드")
uploaded_ref_audio = st.file_uploader(
    "**3초 이상의 목소리**를 녹음해서 업로드하세요.", 
    type=["wav", "mp3"], 
    help="클론할 목소리의 음성 파일을 업로드합니다."
)
ref_text_input = st.text_area(
    "2. 업로드된 참조 음성에서 **말한 내용**을 입력하세요.", 
    value="세상이 멈춘 듯 했다 시간도 소리도 모두 사라져버린 것처럼 그 사이에서 영원히 끝나지 않을 것 같은 나의 생이 끝을 향해 가고 있다는 걸 알았다 가야한다 그녀가 있는 곳으로.", 
    height=100,
    placeholder="참조 음성에서 말한 텍스트를 정확히 입력해주세요."
)

st.markdown("--- ---")

st.header("3. 음성 변환할 문장 입력")
text_to_convert = st.text_area(
    "**여기에 TTS로 만들고 싶은 문장을 입력하세요.**",
    value="안녕. 저는 콩콩이라고 합니다. 저는 수학을 좋아합니다. 컴퓨터는 싫어합니다.",
    height=200,
    key="main_text_input", # Added a key to avoid warnings
    placeholder="여기에 텍스트를 입력하면 입력된 음성으로 변환됩니다."
)

st.markdown("--- ---")

if st.button("🔊 음성 변환하기 (실행)", use_container_width=True, type="primary"):
    if model is None:
        st.error("TTS 모델이 로드되지 않아 음성 변환을 할 수 없습니다. 페이지를 새로고침하거나 Colab 환경을 확인해주세요.")
    else:
        with st.spinner("**음성 변환 중... 잠시만 기다려 주세요.**"):
            generated_audio_path = qwen_tts_inference_streamlit(uploaded_ref_audio, ref_text_input, text_to_convert)

        if generated_audio_path:
            st.success("**음성 변환 완료!**")
            st.subheader("출력된 음성 미리듣기")
            st.audio(generated_audio_path, format="audio/wav")

            with open(generated_audio_path, "rb") as file:
                st.download_button(
                    label="⬇️ 음성 파일 다운로드 (WAV)",
                    data=file.read(),
                    file_name="generated_speech.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
        # Error messages are handled inside qwen_tts_inference_streamlit

st.markdown("--- ---")
st.caption("Powered by Qwen3 TTS on Google Colab with Streamlit and Localtunnel")
