---
# MODEL CARD
---

# Model Card for Risk Forecasting Models

Kullanılan spatio‑temporal risk tahmin modelinin özet kartı. Model çıktıları risk haritası ve devriye rota optimizasyonu bileşenlerine girdi sağlar.

## Model Details

### Model Description

Sistem tek bir spatio‑temporal model hattı kullanır:
- **Mekansal‑Zamansal (XGBoost Regresyon):** Grid/hex tabanlı mekansal özellikler, döngüsel zaman özellikleri ve KDE tabanlı hedef mühendisliği ile risk skorunu tahmin eder.

- **Developed by:** Görkem Palalı, Abdullah Belli
- **Model date:** 18.12.2025
- **Model type:** Gradient boosting regression (XGBoost)
- **Language(s):** Python
- **Finetuned from model:** Yok

### Model Sources

- **Repository:** https://colab.research.google.com/drive/1y2fQkdrGz-Lax8caw4cI7WDp4XDUngvW?usp=sharing
- **Paper:** Yok
- **Demo:** `GET /api/v1/forecast/risk-map`

## Uses

### Direct Use

- Risk haritası üretimi (spatio‑temporal forecast)
- Konum bazlı risk projeksiyonu
- Rota optimizasyonu için risk yoğunluğu girdisi

### Downstream Use

- Gerçek zamanlı risk güncellemesi (WebSocket üzerinden)
- Çoklu merkez koordinasyon senaryoları

### Out-of-Scope Use

- Sosyal / demografik çıkarımlar
- Risk tahminine etkinlikler gibi harici olaylar eklenmesi (Derbi maçları, siyasi olaylar vs.)

## Bias, Risks, and Limitations

- Veri seti yalnızca Küçükçekmece bölgesi ile sınırlıdır.
- Model çıktıları ciddi operasyonel kararlar için tek başına yeterli değildir; uzman değerlendirmesi gerektirir.

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

- `docs/crime_fixed.jsonl` içinde bulunan verilerin %80'i test verisi olarak kullanılmıştır

### Training Procedure

#### Preprocessing

- Zaman verisi sabit zaman pencerelerine (örn. 4‑6 saat) resample edilir.
- Şiddet değerleri 1‑5 aralığında normalize edilir.
- Mekansal özellikler grid/hex hücrelerine dağıtılır.
- Zaman özellikleri döngüsel (sin/cos) encode edilir.
- KDE tabanlı hedef risk skoru oluşturulur ve feature engineering ile zenginleştirilir.

#### Training Hyperparameters

- **Training regime:** CPU / batch‑offline eğitim

#### Speeds, Sizes, Times

- Referans donanımda dakikalar içinde yeniden eğitim yapılabilir (veri boyutuna bağlı).

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

- Eğitim verisinin %80'lik kısmı kaynak kullanılmıştır

#### Factors

- Suç risklerinin mekan ve zaman dilimine bağlı dağılımı

#### Metrics

Model performansı, **regresyon doğruluk metrikleri ve mekansal sıralama metrikleri kullanılarak değerlendirilmiş ve iki temel karşılaştırma modeli ile kıyaslanmıştır: 
**Base-0** Sabit sıfır tahmini ve **Persist** Zamanla süreklilik varsayımı.

- **RMSE:** 0.1157 
  Model, Base-0 (0.1538) ve Persist(0.1322) temel modellerinden daha düşük RMSE değerine sahiptir. Daha düşük RMSE, genel tahmin doğruluğunun daha yüksek olduğunu gösterir.

- **MAE:** 0.0396
  Modelin MAE değeri, Persist modelinden (0.0407) daha düşük ve Base-0 modeline (0.0431) kıyasla benzer performans sergilemektedir. Bu, ortalama hata büyüklüğünün istikrarlı olduğunu gösterir.

- **R²:** 0.3859
  Model, risk skorlarındaki varyansın yaklaşık %38,59’unu açıklayabilmektedir.

- **Precision@10:** 0.603  
  En yüksek riskli olarak tahmin edilen ilk %10’luk grid hücrelerinin %60,3’ü, gerçekten en yüksek riskli %10’luk hücrelerle örtüşmektedir.

- **Recall@10:** 0.621  
  Gerçek en yüksek riskli %10’luk hücrelerin %62,1’i, modelin en üst sıralı tahminleri içerisinde başarıyla yakalanmıştır.

- **Precision@5:** 0.704  
  En kritik ilk %5’lik yüksek riskli hücrelere odaklanıldığında, modelin tahmin doğruluğu artmaktadır.

- **Recall@5:** 0.366  
  Tahminler yalnızca ilk %5’lik hücrelerle sınırlandığında, toplam yüksek riskli alanların daha küçük bir kısmı kapsanmaktadır. Bu, **hassasiyet (precision) ile kapsama (recall)** arasındaki dengeyi gösterir.


### Results

### Results

Modelin performansı hem regresyon doğruluk metrikleri (RMSE, MAE, R²) hem de mekansal sıralama metrikleri (Precision@K, Recall@K) üzerinden değerlendirdiğimizde

- **Genel doğruluk:** Model, sabit ve süreklilik bazlı temel modellere kıyasla daha doğru tahminler üretmektedir.
- **Varyans açıklama gücü:** R² = 0.3859, modelin risk skorlarındaki yaklaşık %39 varyansı açıklayabildiğini gösterir. Bu, modelin anlamlı bir öngörü yeteneğine sahip olduğunu gösterir.
- **Yüksek riskli alanların tespiti:** Precision@10 = 0.603 ve Recall@10 = 0.621, modelin en yüksek riskli %10’luk grid hücrelerinin çoğunu doğru şekilde tahmin ettiğini ortaya koyar.
- **En kritik bölgeler:** Precision@5 = 0.704, en yüksek riskli %5’lik hücrelerde yüksek doğruluk sağlarken, Recall@5 = 0.366, en kritik bölgelerin tamamını kapsamada sınırlı olduğunu gösterir. Bu, hassasiyet ile kapsama arasındaki klasik trade-off'u yansıtır.
- **Genel yorum:** Model prototip aşamasında başarılı risk tahminleri sunar. Operasyonel kullanım öncesi ek doğrulama ve periyodik yeniden eğitim önerilmektedir.


#### Summary

Model, mekânsal önceliklendirme konusunda güçlü bir yetenek sergilemekte ve hem regresyon doğruluğu hem de yüksek riskli alanların tespitinde temel modellere kıyasla belirgin şekilde daha iyi performans göstermektedir. Sonuçlar, modelin risk odaklı devriye planlaması gibi karar destek uygulamaları için etkili olduğunu göstermektedir; ancak operasyonel kullanımdan önce ek doğrulama yapılması önerilir.

## Model Examination

Model açıklanabilirliği sınırlıdır; ağaç tabanlı feature importance ve SHAP gibi yöntemlerle yorum yapılabilir.

## Technical Specifications

### Model Architecture and Objective

- **XGBoost Regresyon:** Grid/hex + zaman + türetilmiş özellikler üzerinden risk skoru tahmini

### Compute Infrastructure

- Lokal CPU ortamı yeterlidir.

#### Hardware

- CPU (tek makine), GPU gerektirmez

#### Software

- Python 3.11
- xgboost
- scikit-learn
- numpy / pandas

## Citation



## Glossary

- **Adaptive KDE:** Yerel yoğunluğa göre değişen bandwidth ile risk yoğunluğu hesaplama.
- **Grid/Hex Features:** Koordinatların grid/hex hücrelerine dağıtılmasıyla oluşturulan mekansal temsil.

## More Information

- Backend servisleri: `backend/app/services/ml/`
- Eğitim scriptleri: `ml/training/`
