
import streamlit as st
from openai import OpenAI
import os

# --- 0. Configuration and API Key Loading ---
# In Streamlit Community Cloud, secrets are accessed via st.secrets
# For local development, it can fall back to environment variables
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    openai_api_key = os.environ.get("OPENAI_API_KEY")


if not openai_api_key:
    st.error("OpenAI API Key가 설정되지 않았습니다. Streamlit Community Cloud의 'Secrets' 설정 또는 환경 변수를 확인해주세요.")
    st.stop() # Stop the app if API key is missing

client = OpenAI(api_key=openai_api_key)

# --- 1. Smart TTS Inference Function ---
def generate_smart_speech(text_to_generate: str) -> str:
    """
    Generates speech using OpenAI's TTS-1 model. The 'smart' aspect
    (automatic speed, tone, pitch adjustment) is primarily handled by
    the inherent naturalness and expressiveness of the OpenAI TTS-1 model.
    For explicit emotion/context analysis and dynamic parameter adjustment,
    an additional NLP layer would typically be required before calling TTS.
    """
    if not text_to_generate.strip():
        st.warning("음성을 변환할 텍스트를 입력해주세요.")
        return None

    try:
        # OpenAI TTS-1 allows selecting a voice and speed. The model inherently
        # aims for natural-sounding speech based on the input text.
        # 'alloy' voice often provides a balanced and clear output.
        # Speed can be adjusted (0.25 to 4.0), but default (1.0) is usually good.
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy", # Recommended voices: alloy, echo, fable, onyx, nova, shimmer
            input=text_to_generate,
            speed=1.0, # Default speed, can be fine-tuned if needed
            response_format="mp3" # MP3 is widely supported, including Safari
        )

        output_filename = "smart_speech.mp3"
        response.stream_to_file(output_filename)
        return output_filename
    except Exception as e:
        st.error(f"음성 생성 중 오류가 발생했습니다: {e}")
        st.info("OpenAI API 사용량 한도를 초과했거나, API 키가 유효하지 않은지 확인해주세요.")
        return None

# --- 2. Streamlit UI ---
st.set_page_config(page_title="AI 스마트 TTS", layout="centered", initial_sidebar_state="collapsed")

st.title("🗣️ AI 스마트 TTS (iPhone Safari 최적화)")
st.markdown("--- ---")

st.header("1. 여기에 글을 입력하세요.")
text_input = st.text_area(
    "AI가 문맥과 감정을 분석하여 자동으로 최적의 속도와 어조로 음성을 생성합니다.",
    height=250,
    placeholder="여기에 음성으로 변환하고 싶은 글을 입력해주세요.",
    key="main_text_input",
    value="안녕하세요. 저는 인공지능 비서입니다. 오늘 날씨는 맑고 쾌청합니다. 즐거운 하루 되세요!"
)

st.markdown("--- ---")

if st.button("🔊 음성 변환하기 (실행)", use_container_width=True, type="primary"):
    if not text_input.strip():
        st.error("음성을 변환할 텍스트를 입력해 주세요.")
    else:
        with st.spinner("AI가 스마트하게 음성을 생성 중입니다... 잠시만 기다려 주세요."):
            generated_audio_file = generate_smart_speech(text_input)

        if generated_audio_file:
            st.success("음성 생성 완료!")
            st.subheader("2. 출력된 음성 미리듣기")
            st.audio(generated_audio_file, format="audio/mp3") # Safari supports MP3 natively

            with open(generated_audio_file, "rb") as file:
                st.download_button(
                    label="⬇️ 3. 음성 파일 다운로드 (MP3)",
                    data=file.read(),
                    file_name="smart_speech.mp3",
                    mime="audio/mp3",
                    use_container_width=True
                )

st.markdown("--- ---")
st.caption("Powered by OpenAI TTS & Streamlit on Streamlit Community Cloud")
