from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import get_db

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    summary="Sistem Sağlık Kontrolü",
    description="Veritabanı bağlantısı dahil sistem durumunu kontrol eder",
    response_description="Sistem ve veritabanı durum bilgisi"
)
def health_check(db: Session = Depends(get_db)):
    """
    Sistem sağlık kontrolü endpoint'i.
    
    Veritabanı bağlantısını test eder ve sistem durumunu döndürür.
    """
    try:
        # Check database connection
        db.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception:
        db_status = "error"
    
    return {
        "status": "ok",
        "database": db_status
    }


