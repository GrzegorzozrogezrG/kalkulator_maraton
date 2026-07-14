import json
import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import boto3
import joblib
import numpy as np
import streamlit as st
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from langfuse.openai import OpenAI


load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
SPACES_ENDPOINT = os.getenv("DO_SPACES_ENDPOINT")
SPACES_REGION = os.getenv("DO_SPACES_REGION", "fra1")
SPACES_BUCKET = os.getenv("DO_SPACES_BUCKET")
SPACES_ACCESS_KEY = os.getenv("DO_SPACES_ACCESS_KEY")
SPACES_SECRET_KEY = os.getenv("DO_SPACES_SECRET_KEY")
SPACES_MODEL_PREFIX = os.getenv("DO_SPACES_MODEL_PREFIX", "models/")

LANGFUSE_ENABLED = all(
    [
        os.getenv("LANGFUSE_PUBLIC_KEY"),
        os.getenv("LANGFUSE_SECRET_KEY"),
        os.getenv("LANGFUSE_BASE_URL"),
    ]
)


def parse_time_to_seconds(raw: str):
    text = str(raw).strip().lower().replace(",", ".")
    if not text:
        return None

    hhmmss = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", text)
    if hhmmss:
        h, m, s = map(int, hhmmss.groups())
        return h * 3600 + m * 60 + s

    mmss = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if mmss:
        m, s = map(int, mmss.groups())
        return m * 60 + s

    mins = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:min|m|minutes?)", text)
    if mins:
        return int(float(mins.group(1)) * 60)

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        val = float(text)
        if val > 200:
            return int(val)
        return int(val * 60)

    return None


def parse_distance_km(raw):
    if raw is None:
        return None

    text = str(raw).strip().lower().replace(",", ".")
    if not text:
        return None

    if "polmaraton" in text or "półmaraton" in text or "half marathon" in text:
        return 21.0975
    if "maraton" in text or "marathon" in text:
        return 42.195

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kilometr(?:y|ów)?)", text)
    if m:
        return float(m.group(1))

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    return None


def convert_to_5k_seconds(distance_km: float, time_sec: int, exponent: float = 1.06):
    # Riegel: T2 = T1 * (D2 / D1)^exponent -> przeliczamy na D2=5 km
    if distance_km <= 0:
        return None
    return float(time_sec) * ((5.0 / float(distance_km)) ** exponent)


def normalize_gender(raw: str):
    if raw is None:
        return None
    val = str(raw).strip().upper()
    if val in {"M", "MEN", "MALE", "MĘŻCZYZNA", "MEZCZYZNA", "CHŁOPAK", "CHLOPAK"}:
        return "M"
    if val in {"K", "F", "W", "WOMAN", "FEMALE", "KOBIETA", "DZIEWCZYNA"}:
        return "K"
    return None


def infer_category(gender: str, age: int):
    if age < 30:
        group = "20"
    elif age < 40:
        group = "30"
    elif age < 50:
        group = "40"
    elif age < 60:
        group = "50"
    elif age < 70:
        group = "60"
    else:
        group = "70"
    return f"{gender}{group}"


def seconds_to_hhmmss(seconds: float):
    total = int(round(float(seconds)))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_model_row(gender: str, age: int, time_5km_sec: int):
    now_year = datetime.utcnow().year
    rocznik = now_year - age

    tempo_5 = time_5km_sec / (5 * 60)
    time_10 = int(time_5km_sec * 2 * 1.03)
    time_15 = int(time_5km_sec * 3 * 1.05)
    time_20 = int(time_5km_sec * 4 * 1.08)

    tempo_10 = time_10 / (10 * 60)
    tempo_15 = time_15 / (15 * 60)
    tempo_20 = time_20 / (20 * 60)

    pace_series = np.array([tempo_5, tempo_10, tempo_15, tempo_20], dtype=float)
    tempo_stability = float(np.std(pace_series) / np.mean(pace_series))

    return {
        "Wiek": int(age),
        "Rocznik": int(rocznik),
        "5 km Czas": int(time_5km_sec),
        "10 km Czas": int(time_10),
        "15 km Czas": int(time_15),
        "20 km Czas": int(time_20),
        "5 km Tempo": float(tempo_5),
        "10 km Tempo": float(tempo_10),
        "15 km Tempo": float(tempo_15),
        "20 km Tempo": float(tempo_20),
        "Tempo Stabilność": tempo_stability,
        "Płeć": gender,
        "Kategoria wiekowa": infer_category(gender, int(age)),
    }


def extract_runner_profile_with_llm(user_text: str):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    system_prompt = (
        "Wyciągnij dane biegacza z tekstu użytkownika. "
        "Zwróć wyłącznie JSON z polami: gender, age, distance_km, time_value. "
        "gender: 'M' albo 'K', age: liczba całkowita, "
        "distance_km: liczba kilometrów biegu, "
        "time_value: czas dla podanego dystansu jako string HH:MM:SS lub MM:SS lub liczba minut. "
        "Jeśli użytkownik podał czas dla 10 km / 11 km / innego dystansu, zachowaj ten dystans w distance_km i czas w time_value. "
        "Jeśli padł tylko czas na 5 km, wpisz distance_km=5. "
        "Jeśli brakuje danych, wpisz null."
    )

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
        metadata={
            "langfuse_tags": ["runner-profile", "extraction"],
        },
        name="extract_runner_fields",
    )

    content = completion.choices[0].message.content or "{}"
    parsed = json.loads(content)

    return parsed


def validate_extracted_profile(profile: dict):
    missing = []

    gender = normalize_gender(profile.get("gender"))
    if gender is None:
        missing.append("płeć (M/K)")

    age_raw = profile.get("age")
    if isinstance(age_raw, (int, float)):
        age = int(age_raw)
    else:
        age_match = re.search(r"\d{1,3}", str(age_raw or ""))
        age = int(age_match.group(0)) if age_match else None
    if age is None or age < 16 or age > 100:
        missing.append("wiek (16-100)")

    distance_raw = profile.get("distance_km", profile.get("distance"))
    distance_km = parse_distance_km(distance_raw)

    # backward compatibility for older prompt schema
    time_value = profile.get("time_value", profile.get("time_5km"))
    if distance_km is None and profile.get("time_5km") is not None:
        distance_km = 5.0

    time_sec_for_distance = parse_time_to_seconds(time_value)
    if distance_km is None or distance_km <= 0:
        missing.append("dystans biegu w km (np. 5, 10, 11)")
    if time_sec_for_distance is None:
        missing.append("czas dla podanego dystansu (np. 48:00)")

    time_5km_sec = None
    if distance_km is not None and time_sec_for_distance is not None:
        converted = convert_to_5k_seconds(distance_km, time_sec_for_distance)
        if converted is not None:
            time_5km_sec = int(round(converted))
    if time_5km_sec is None or time_5km_sec < 10 * 60 or time_5km_sec > 60 * 60:
        missing.append("czas przeliczony na 5 km (poza zakresem 10-60 min)")

    clean = {
        "gender": gender,
        "age": age,
        "distance_km": distance_km,
        "time_sec_for_distance": time_sec_for_distance,
        "time_5km_sec": time_5km_sec,
    }
    return clean, missing


def normalize_spaces_endpoint(endpoint: str, bucket: str):
    parsed = urlparse(endpoint)
    host = parsed.netloc
    bucket_prefix = f"{bucket}."
    if host.startswith(bucket_prefix):
        host = host[len(bucket_prefix) :]
    return urlunparse((parsed.scheme, host, parsed.path, parsed.params, parsed.query, parsed.fragment))


def spaces_configured():
    return all([SPACES_ENDPOINT, SPACES_BUCKET, SPACES_ACCESS_KEY, SPACES_SECRET_KEY])


def make_spaces_client():
    endpoint = normalize_spaces_endpoint(SPACES_ENDPOINT, SPACES_BUCKET)
    return boto3.client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=endpoint,
        aws_access_key_id=SPACES_ACCESS_KEY,
        aws_secret_access_key=SPACES_SECRET_KEY,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _download_from_spaces(client, key: str, local_path: Path) -> None:
    try:
        client.download_file(SPACES_BUCKET, key, str(local_path))
    except (ClientError, BotoCoreError) as e:
        raise FileNotFoundError(f"Nie udało się pobrać modelu z Spaces: {e}") from e


@st.cache_resource(show_spinner=False)
def load_prediction_model():
    """Download mini model from DO Spaces and cache for container lifetime.

    halfmarathon_mini.pkl is 0.7 KB — download is near-instant.
    st.cache_resource ensures this runs exactly once per container process.
    """
    if not spaces_configured():
        raise FileNotFoundError(
            "Brak konfiguracji DO Spaces. Ustaw zmienne środowiskowe: "
            "DO_SPACES_ENDPOINT, DO_SPACES_BUCKET, DO_SPACES_ACCESS_KEY, DO_SPACES_SECRET_KEY."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = MODEL_DIR / "halfmarathon_mini.pkl"

    # Always re-download to pick up latest model from Spaces
    client = make_spaces_client()
    prefix = SPACES_MODEL_PREFIX.rstrip("/") + "/"
    key = prefix + "halfmarathon_mini.pkl"
    try:
        client.download_file(SPACES_BUCKET, key, str(local_path))
    except (ClientError, BotoCoreError) as e:
        raise FileNotFoundError(f"Nie udało się pobrać modelu ze Spaces: {e}") from e

    model = joblib.load(str(local_path))
    return model, "halfmarathon_mini.pkl"


def predict_halfmarathon_seconds(model, gender: str, age: int, time_5km_sec: int):
    row = build_model_row(gender=gender, age=age, time_5km_sec=time_5km_sec)
    X = np.array([[row["20 km Czas"], row["15 km Tempo"]]])
    return float(model.predict(X)[0])


st.set_page_config(page_title="Predykcja półmaratonu", page_icon="🏃", layout="centered")
st.title("🏃 Predykcja czasu półmaratonu")
st.caption("Podaj opis biegacza tekstem — aplikacja wyciągnie dane przez LLM i oszacuje czas.")

user_text = st.text_area(
    "Napisz kilka słów o sobie (płeć, wiek, czas i dystans biegu):",
    placeholder="Np. Hej, mam 34 lata, jestem kobietą i 10 km biegam w 48:00.",
    height=140,
)

if st.button("Oszacuj czas", type="primary"):
    if not user_text.strip():
        st.warning("Wpisz opis użytkownika w polu tekstowym.")
        st.stop()

    try:
        extracted = extract_runner_profile_with_llm(user_text)
    except json.JSONDecodeError:
        st.error("LLM zwrócił niepoprawny JSON. Spróbuj ponownie.")
        st.stop()
    except Exception as e:
        st.error(f"Błąd podczas ekstrakcji danych przez LLM: {e}")
        st.stop()

    st.subheader("Dane wyłuskane przez LLM")
    st.json(extracted)

    clean_profile, missing_fields = validate_extracted_profile(extracted)
    if missing_fields:
        st.warning("Brakujące lub niepoprawne dane: " + ", ".join(missing_fields))
        st.stop()

    st.info(
        "Przeliczenie wejścia: "
        f"{clean_profile['distance_km']:.2f} km w {seconds_to_hhmmss(clean_profile['time_sec_for_distance'])} "
        f"-> ekwiwalent 5 km: {seconds_to_hhmmss(clean_profile['time_5km_sec'])}"
    )

    try:
        with st.spinner("Ładowanie modelu..."):
            model, model_filename = load_prediction_model()
        st.success(f"✅ Model: {model_filename}")

        prediction_sec = predict_halfmarathon_seconds(
            model=model,
            gender=clean_profile["gender"],
            age=clean_profile["age"],
            time_5km_sec=clean_profile["time_5km_sec"],
        )
    except FileNotFoundError as e:
        st.error(f"Nie udało się załadować modelu: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Błąd predykcji modelu: {e}")
        st.stop()

    st.subheader("Wynik")
    st.metric("Prognozowany czas półmaratonu", seconds_to_hhmmss(prediction_sec))

    pace_min_per_km = prediction_sec / 21.0975 / 60
    st.write(f"Szacowane tempo: **{pace_min_per_km:.2f} min/km**")