---
# MODEL CARD
---

# Model Card for PPRS Forecasting Models

PPRS içinde kullanılan zaman serisi ve mekansal‑zamansal risk tahmin modellerinin özet kartı. Model çıktıları risk haritası ve devriye rota optimizasyonu bileşenlerine girdi sağlar.

## Model Details

### Model Description

Sistem iki ana model bileşeni kullanır:
- **Zaman Serisi (SARIMAX):** Saatlik risk skorlarını tahmin eder.
- **Mekansal‑Zamansal (Lineer Regresyon):** Grid tabanlı mekansal özellikler ve döngüsel zaman özellikleri ile risk skorunu tahmin eder.

- **Developed by:** Görkem Palalı, Abdullah Belli
- **Model date:** 18.12.2025
- **Model type:** Statistical time‑series + linear regression
- **Language(s):** Python
- **Finetuned from model:** Yok

### Model Sources

- **Repository:** https://colab.research.google.com/drive/106OFn015aXcXceOJiumSGIrYsDFLhinR?usp=sharing
- **Paper:** Yok
- **Demo:** Yok

## Uses

### Direct Use

- Risk haritası üretimi (spatio‑temporal forecast)
- Zaman serisi risk projeksiyonu
- Rota optimizasyonu için risk yoğunluğu girdisi

### Downstream Use

- Gerçek zamanlı risk güncellemesi (WebSocket üzerinden)
- Çoklu merkez koordinasyon senaryoları

### Out-of-Scope Use

- Küçükçekmece dışındaki bölgelerde genelleme
- Sosyal / demografik çıkarımlar
- Risk tahminine etkinlikler gibi harici olaylar eklenmesi (Derbi maçları, siyasi haberler vs.)

## Bias, Risks, and Limitations

- Veri seti yalnızca Küçükçekmece bölgesi ile sınırlıdır.
- Sınıf dağılımı dengesizdir; nadir suç türleri için hata payı artabilir(veri setinde yer alan terör olaylarının nadirliği gibi).
- Model çıktıları operasyonel kararlar için tek başına yeterli değildir; uzman değerlendirmesi gerektirir.
- İstatistiksel modelleme ani olay değişimlerine duyarlıdır ().

### Recommendations

- Çıktılar karar destek amaçlı kullanılmalı, saha geri bildirimi ile doğrulanmalıdır.
- Yeni veri geldikçe periyodik yeniden eğitim önerilir.

## How to Get Started with the Model

API uç noktaları üzerinden kullanılabilir:

```bash
# Zaman serisi tahmin
curl http://localhost:8000/api/v1/ml-forecast/timeseries

# Mekansal-zamansal tahmin
curl http://localhost:8000/api/v1/ml-forecast/spatial-temporal

# Ensemble tahmin
curl http://localhost:8000/api/v1/ml-forecast/ensemble
```

## Training Details

### Training Data

- `docs/crime_fixed.jsonl` (Dataset Card: `docs/datasetcard.md`)
- Olay zamanları UTC ISO‑8601 formatında, konum bilgileri enlem/boylam olarak tutulur.

### Training Procedure

#### Preprocessing

- Zaman verisi saatlik aralıklara resample edilir.
- Şiddet değerleri 1‑5 aralığında normalize edilir ve 0‑1 aralığına kırpılır.
- Mekansal özellikler grid hücrelerine dağıtılır.
- Zaman özellikleri döngüsel (sin/cos) encode edilir.

#### Training Hyperparameters

- **Training regime:** CPU / batch‑offline eğitim

#### Speeds, Sizes, Times

- Referans donanımda dakikalar içinde yeniden eğitim yapılabilir (veri boyutuna bağlı).

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- Eğitim verisi ile aynı kaynak kullanılmıştır (ayrı bir test seti tanımlı değildir).

#### Factors

- Suç türlerinin mekan ve zaman dilimine bağlı dağılımı

#### Metrics

- Resmi metrik raporu yok; proje demo amaçlıdır.

### Results

Performans ölçümleri standardize edilmemiştir. Operasyonel kullanım öncesi offline metrik raporu çıkarılması önerilir.

#### Summary

Mevcut sürüm, prototip amaçlı risk tahmini sağlar ve gerçek sahada ek doğrulama gerektirir.

## Model Examination

Model açıklanabilirliği sınırlıdır; SARIMAX parametreleri ve regresyon katsayıları üzerinden temel yorum yapılabilir.

## Technical Specifications

### Model Architecture and Objective

- **SARIMAX:** Saatlik risk skoru tahmini
- **Lineer Regresyon:** Grid + zaman özellikleri üzerinden risk skoru tahmini

### Compute Infrastructure

- Lokal CPU ortamı yeterlidir.

#### Hardware

- CPU (tek makine), GPU gerektirmez

#### Software

- Python 3.11
- statsmodels
- scikit-learn
- numpy / pandas

## Citation



## Glossary

- **SARIMAX:** Mevsimsellik ve dışsal değişken destekli zaman serisi modeli
- **Grid Features:** Koordinatların belirli hücrelere dağıtılmasıyla oluşturulan mekansal temsil

## More Information

- Backend servisleri: `backend/app/services/ml/`
- Eğitim scriptleri: `ml/training/`
