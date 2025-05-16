"""API do przewidywania kosztów zawodników w grach fantasy"""

from fastapi import FastAPI
import onnxruntime as rt
import numpy as np
from schemas import FantasyAcquisitionFeatures, PredictionOutput

api_description = """
To API przewiduje zakres kosztów pozyskania zawodnika w grach fantasy football.

Punkty końcowe można pogrupować w następujące kategorie:

## Analityka
Uzyskaj informacje o stanie API.

## Prognozy
Uzyskaj prognozy kosztów pozyskania zawodnika.
"""
# Załadowanie modelu ONNX
sess_10 = rt.InferenceSession("acquisition_model_10.onnx",
                             providers=["CPUExecutionProvider"])
sess_50 = rt.InferenceSession("acquisition_model_50.onnx",
                             providers=["CPUExecutionProvider"])
sess_90 = rt.InferenceSession("acquisition_model_90.onnx",
                             providers=["CPUExecutionProvider"])

#Pobranie nazw danych wejściowych i wyjściowych modelu
input_name_10 = sess_10.get_inputs()[0].name 
label_name_10 = sess_10.get_outputs()[0].name 
input_name_50 = sess_50.get_inputs()[0].name
label_name_50 = sess_50.get_outputs()[0].name
input_name_90 = sess_90.get_inputs()[0].name
label_name_90 = sess_90.get_outputs()[0].name

app = FastAPI(
    description=api_description,
    title="API do zarządzania drużyną fantasy",
    version="0.1",
)

@app.get(
    "/",
    summary="Sprawdzenie, czy API do zarządzania drużyną fantasy działa",
    description="""Użyj tego punktu końcowego, aby sprawdzić, czy API działa. 
    Możesz też sprawdzić go przed wykonaniem innych wywołań, aby upewnić się, że działa.""",
    response_description="Obiekt JSON zawierający komunikat. Jeśli API działa, będzie to komunikat o sukcesie.",
    operation_id="v0_health_check",
    tags=["analityka"],
)
def root():
    return {"message": "Test stanu API zakończony sukcesem"}

# Definicja ścieżki dla predykcji
@app.post("/predict/",
          response_model=PredictionOutput, 
          summary="Przewidywanie kosztu pozyskania zawodnika",
          description="""Użyj tego punktu końcowego, aby przewidzieć zakres kosztów 
          pozyskania zawodnika w fantasy football.""",
          response_description="""Rekord JSON zawierający trzy przewidywane kwoty. 
          Razem tworzą możliwy zakres kosztów pozyskania zawodnika.""",
          operation_id="v0_predict",
          tags=["predykcja"],
)
def predict(features: FantasyAcquisitionFeatures):
    # Konwersja modelu Pydantic na tablicę NumPy
    input_data = np.array([[features.waiver_value_tier,
                               features.fantasy_regular_season_weeks_remaining,
                               features.league_budget_pct_remaining]],
                               dtype=np.int64)

    pred_onx_10 = sess_10.run([label_name_10], {input_name_10: input_data})[0] 
    pred_onx_50 = sess_50.run([label_name_50], {input_name_50: input_data})[0]
    pred_onx_90 = sess_90.run([label_name_90], {input_name_90: input_data})[0]
    # Zwrócenie predykcji w postaci obiektu odpowiedzi Pydantic
    return PredictionOutput(winning_bid_10th_percentile=round(
                               float(pred_onx_10[0]),2), 
                           winning_bid_50th_percentile=round(
                               float(pred_onx_50[0]),2),
                           winning_bid_90th_percentile=round(
                               float(pred_onx_90[0]), 2))
