---
# Dataset Card
---

# Dataset Card for Crime Events (crime_fixed.jsonl)

Küçükçekmece ilçesi sınırları içinde konumlandırılmış suç olayları kayıtlarını içeren, risk tahmini ve devriye rotası optimizasyonu için kullanılan JSONL formatında veri seti.

## Dataset Details

### Dataset Description

Bu veri seti, olay bazlı suç kayıtlarını zaman ve konum bilgisiyle birlikte sağlar. Her kayıt tek bir suç olayını temsil eder ve spatio‑temporal risk haritası üretimi ile rota planlama bileşenlerinin giriş verisi olarak kullanılır.

- **Curated by:** Görkem Palalı, Abdullah Belli
- **License:** UNDP & Samsung Innovation AI Campus

### Dataset Sources

- **Repository:** https://github.com/GorkemPalali/police-patrol/tree/main/docs/dataset
- **Paper:** 
- **Demo:** Yok

## Uses

### Direct Use

- Risk haritası üretimi (Adaptive KDE + grid/hex)
- Zaman serisi ve spatio‑temporal tahmin
- Devriye rotası optimizasyonu ve senaryo testleri

## Dataset Structure

Dosya: `docs/dataset/crime_fixed.jsonl`

Format: JSON Lines (her satır bir JSON nesnesi)

Alanlar:
- `timestamp` (string, ISO‑8601, UTC): Olay zamanı
- `crime_type` (string): Suç türü
- `severity` (int, 1‑5): Şiddet seviyesi
- `latitude` (float): Enlem
- `longitude` (float): Boylam

Örnek kayıt:

```json
{"timestamp":"2025-10-16T05:58:00Z","crime_type":"Gasp","severity":4,"latitude":41.037876,"longitude":28.78833}
```

İstatistikler (crime_fixed.jsonl):
- Kayıt sayısı: 1770
- Tarih aralığı: 2025-01-10T02:03:39Z → 2025-12-31T21:06:30Z
- Suç türleri: Hırsızlık, Gasp, Sokak Satıcılığı, Şiddet, Soygun, Kundaklama, Silahlı Saldırı, Cinayet, Uyuşturucu, Kavga, Saldırı, Çete Olayı, Terör Olayı

## Dataset Creation

### Source Data

Veri seti, proje kapsamında kullanılan olay kayıtlarının JSONL formatına normalize edilmiş bir sürümüdür. Kayıtlar Küçükçekmece bölgesi için konumlandırılmıştır ve zaman bilgisi UTC formatında tutulur.

#### Data Collection and Processing

- Zaman alanı ISO‑8601 UTC standardına dönüştürülmüştür.
- Konum alanları `latitude` / `longitude` olarak tek tip isimlendirilmiştir.
- Şiddet seviyeleri 1‑5 aralığında normalize edilmiştir.

#### Features and the target

- **Features:** timestamp, crime_type, severity, latitude, longitude
- **Target:** Risk skoru / olay yoğunluğu (modelleme aşamasında türetilir)

### Annotations

Veri seti ek anotasyon içermemektedir.

#### Annotation process

Yok.

#### Who are the annotators?

Yok.

## Bias, Risks, and Limitations

- Veri yalnızca Küçükçekmece ilçesi için temsili olarak oluşturuldu, olaylar bölgenin sosyodemografik yapısından bağımsız bir şekilde  sınırlıdır(girilen lat/lon değerleri küçükçekmecenin sınırları dışındaysa uyarı veren ).
- Suç türleri ve şiddet seviyeleri kategorik olduğundan sınıflar arası dağılımlarda şiddet seviyesine bağlı olarak dengesizlikler olabilir.
- Konum doğruluğu veri kaynağına bağlıdır; küçük sapmalar risk haritalarında lokal etkiler yaratabilir.

## Citation


