# BIST30 Risk Portalı

Yüksek lisans tezi kapsamında geliştirilmiş, BIST30 hisselerinin canlı risk analizi ve 2026 yıl sonu tahminini sunan tek sayfalık web uygulaması.

## Özellikler

- 28 BIST30 hissesi için canlı veri (Yahoo Finance)
- Yıllık volatilite ve maksimum drawdown bazlı risk skorlaması
- Ridge Regression ile 2026 yıl sonu fiyat tahmini
- Portföy simülatörü (kovaryans matrisi bazlı volatilite hesabı)
- Hisse karşılaştırma (radar grafik)
- Excel dışa aktarma
- Gerçek zamanlı arama ve kategori filtreleme

## Kurulum

**Python 3.9+ gereklidir.**

```bash
git clone https://github.com/DogusSen/bist30-risk-dashboard.git
cd bist30-risk-dashboard

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Çalıştırma

```bash
uvicorn main:app --reload
```

Tarayıcıda aç: [http://localhost:8000](http://localhost:8000)

> İlk açılışta veriler Yahoo Finance'den indiriliyor (~30-60 saniye). Yükleme ekranı kapanana kadar bekleyin.

## Teknolojiler

| Katman | Teknoloji |
|--------|-----------|
| Backend | FastAPI, yfinance, scikit-learn, pandas, numpy |
| Frontend | Vanilla JS, Plotly.js, SheetJS |
| Model | Ridge Regression (10 yıllık geçmiş veri) |

## Veri Kaynağı

Tüm fiyat verileri **Yahoo Finance** üzerinden canlı olarak çekilmektedir. Sonuçlar yalnızca akademik amaçlıdır; yatırım tavsiyesi niteliği taşımaz.
