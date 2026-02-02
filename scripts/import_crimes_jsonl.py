"""
JSONL formatındaki suç verilerini veritabanına aktarır.
Kullanım: python scripts/import_crimes_jsonl.py /Users/gorkempalali/crime.jsonl
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Backend modüllerini import etmek için path ekle
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.models.crime_event import CrimeEvent
from app.services.utils import lat_lng_to_geography


def parse_timestamp(timestamp_str: str) -> datetime:
    """ISO format timestamp'i datetime'a çevirir"""
    # UTC formatını işle
    if timestamp_str.endswith('Z'):
        timestamp_str = timestamp_str.replace('Z', '+00:00')
    
    try:
        return datetime.fromisoformat(timestamp_str)
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            raise ValueError(f"Geçersiz timestamp formatı: {timestamp_str}")


def import_crimes_from_jsonl(
    jsonl_file_path: str,
    batch_size: int = 1000,
    skip_errors: bool = True
) -> dict:
    """
    JSONL dosyasından suç olaylarını veritabanına aktarır.
    
    Args:
        jsonl_file_path: JSONL dosyasının yolu
        batch_size: Her batch'te kaç kayıt eklenecek
        skip_errors: Hatalı kayıtları atla (True) veya durdur (False)
    
    Returns:
        İstatistikler: {'total': int, 'added': int, 'skipped': int, 'errors': list}
    """
    db: Session = SessionLocal()
    
    stats = {
        'total': 0,
        'added': 0,
        'skipped': 0,
        'errors': []
    }
    
    try:
        # Dosya varlığını kontrol et
        file_path = Path(jsonl_file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {jsonl_file_path}")
        
        print(f"📂 Dosya: {jsonl_file_path}")
        print(f"📊 Batch boyutu: {batch_size}")
        print("-" * 50)
        
        # JSONL dosyasını satır satır oku
        with open(file_path, 'r', encoding='utf-8') as f:
            batch = []
            
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                
                # Boş satırları atla
                if not line:
                    continue
                
                stats['total'] += 1
                
                try:
                    # JSON parse et
                    item = json.loads(line)
                    
                    # Veriyi doğrula
                    required_fields = ['timestamp', 'crime_type', 'severity', 'latitude', 'longitude']
                    missing_fields = [f for f in required_fields if f not in item]
                    
                    if missing_fields:
                        error_msg = f"Satır {line_num}: Eksik alanlar: {missing_fields}"
                        stats['errors'].append(error_msg)
                        stats['skipped'] += 1
                        if not skip_errors:
                            raise ValueError(error_msg)
                        continue
                    
                    # Timestamp'i parse et
                    event_time = parse_timestamp(item['timestamp'])
                    
                    # Severity kontrolü (1-5 arası olmalı)
                    severity = int(item['severity'])
                    if not (1 <= severity <= 5):
                        error_msg = f"Satır {line_num}: Geçersiz severity değeri: {severity} (1-5 arası olmalı)"
                        stats['errors'].append(error_msg)
                        stats['skipped'] += 1
                        if not skip_errors:
                            raise ValueError(error_msg)
                        continue
                    
                    # Koordinat kontrolü
                    lat = float(item['latitude'])
                    lng = float(item['longitude'])
                    
                    if not (-90 <= lat <= 90):
                        error_msg = f"Satır {line_num}: Geçersiz latitude: {lat}"
                        stats['errors'].append(error_msg)
                        stats['skipped'] += 1
                        if not skip_errors:
                            raise ValueError(error_msg)
                        continue
                    
                    if not (-180 <= lng <= 180):
                        error_msg = f"Satır {line_num}: Geçersiz longitude: {lng}"
                        stats['errors'].append(error_msg)
                        stats['skipped'] += 1
                        if not skip_errors:
                            raise ValueError(error_msg)
                        continue
                    
                    # Crime event oluştur
                    crime_event = CrimeEvent(
                        crime_type=str(item['crime_type'])[:100],  # Max 100 karakter
                        severity=severity,
                        event_time=event_time.replace(tzinfo=None),  # PostgreSQL için timezone olmadan
                        geom=lat_lng_to_geography(lat, lng),
                        street_name=None,
                        confidence_score=1.0  # Varsayılan
                    )
                    
                    batch.append(crime_event)
                    stats['added'] += 1
                    
                    # Batch dolduğunda commit et
                    if len(batch) >= batch_size:
                        db.bulk_save_objects(batch)
                        db.commit()
                        print(f"İşlenen: {stats['total']} satır ({stats['added']} eklendi, {stats['skipped']} atlandı)")
                        batch = []
                        
                except json.JSONDecodeError as e:
                    error_msg = f"Satır {line_num}: JSON parse hatası: {e}"
                    stats['errors'].append(error_msg)
                    stats['skipped'] += 1
                    if not skip_errors:
                        raise
                    continue
                    
                except Exception as e:
                    error_msg = f"Satır {line_num}: Hata: {e}"
                    stats['errors'].append(error_msg)
                    stats['skipped'] += 1
                    if not skip_errors:
                        raise
                    continue
            
            # Kalan kayıtları commit et
            if batch:
                db.bulk_save_objects(batch)
                db.commit()
                print(f"Son batch işlendi: {len(batch)} kayıt")
        
        print("-" * 50)
        print(f"İşlem tamamlandı!")
        print(f"Toplam satır: {stats['total']}")
        print(f"Başarıyla eklendi: {stats['added']}")
        print(f"Atlandı: {stats['skipped']}")
        
        if stats['errors']:
            print(f"\n Hatalar ({len(stats['errors'])}):")
            for error in stats['errors'][:10]:  # İlk 10 hatayı göster
                print(f"   - {error}")
            if len(stats['errors']) > 10:
                print(f"   ... ve {len(stats['errors']) - 10} hata daha")
        
        return stats
        
    except Exception as e:
        db.rollback()
        print(f"\n Kritik hata: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python scripts/import_crimes_jsonl.py <jsonl_dosya_yolu> [batch_size]")
        print("\nÖrnek:")
        print("  python scripts/import_crimes_jsonl.py /Users/gorkempalali/crime.jsonl")
        print("  python scripts/import_crimes_jsonl.py /Users/gorkempalali/crime.jsonl 500")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    try:
        import_crimes_from_jsonl(jsonl_file, batch_size=batch_size)
    except KeyboardInterrupt:
        print("\n\n  İşlem kullanıcı tarafından durduruldu.")
        sys.exit(1)
    except Exception as e:
        print(f"\n Hata: {e}")
        sys.exit(1)





