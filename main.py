"""
BIST30 Risk Portalı — Backend (FastAPI)
========================================
Bu modül, BIST30 hisselerine ait risk metriklerini hesaplar ve
bir REST API aracılığıyla frontend'e sunar.

Kullanılan yöntemler:
  - Yıllık volatilite  : Logaritmik günlük getirilerin standart sapması × √252
  - Maksimum drawdown  : Kümülatif getiri serisinin tepe noktasından en büyük düşüş
  - Risk skoru         : Volatilite (%60) + drawdown (%40) ağırlıklı, min-max normalize
  - Kategori           : Çeyreklik dilimler (Q25/Q50/Q75) ile 4 risk seviyesine ayrılır
  - Yıl sonu tahmini   : Ridge Regression modeli, lag + hareketli ortalama özellikleri
  - Portföy volatilitesi: w^T × Σ × w (kovaryans matrisi bazlı Modern Portföy Teorisi)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import threading
from datetime import datetime, date, timezone, timedelta

# Sunucu UTC çalıştığından Türkiye saatine (UTC+3) çevirmek için
TZ_TR = timezone(timedelta(hours=3))
import traceback
import os

# index.html dosyasını mutlak yol ile açmak için (Render deploy uyumluluğu)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI()

# Analizde kullanılan 28 BIST30 hissesi (Yahoo Finance formatı: TICKER.IS)
TICKERS = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS",
    "DOHOL.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS",
    "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HALKB.IS",
    "ISCTR.IS", "KCHOL.IS", "KRDMD.IS", "MGROS.IS",
    "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS",
    "SISE.IS", "TAVHL.IS", "TCELL.IS", "THYAO.IS",
    "TKFEN.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS",
]

# Uygulama genelinde paylaşılan veri nesneleri (thread-safe erişim için _lock kullanılır)
_risk_data = None   # Her hisse için hesaplanmış risk metrikleri listesi
_price_data = None  # Temiz sütun adlı (örn. AKBNK, GARAN) tam fiyat DataFrame'i
_last_update = None # Son veri güncelleme zamanı (Türkiye saati ile)
_loading = False    # Veri yükleme işlemi devam ediyorsa True
_lock = threading.Lock()


def calculate_risk_metrics():
    """
    Tüm BIST30 hisselerinin risk metriklerini hesaplar ve global değişkenlere yazar.

    Adımlar:
      1. Yahoo Finance'den 10 yıllık günlük kapanış fiyatları indirilir.
      2. Her hisse için volatilite ve maksimum drawdown hesaplanır.
      3. Volatilite ve drawdown min-max normalize edilip ağırlıklı risk skoru oluşturulur.
      4. Çeyreklik dilimlerle hisseler 4 risk kategorisine atanır.
      5. Ridge Regression ile her hisse için 2026 yıl sonu fiyat tahmini yapılır.
      6. Sonuçlar thread-safe biçimde global değişkenlere yazılır.
    """
    global _risk_data, _price_data, _last_update, _loading
    _loading = True
    try:
        print("Yahoo Finance'den veri indiriliyor...")
        # 10 yıllık geçmiş fiyat verisi, düzeltilmiş kapanış fiyatları ile
        raw = yf.download(TICKERS, period="10y", auto_adjust=True, progress=False)

        # yfinance bazen MultiIndex döner (birden fazla hisse olunca); sadece kapanış sütunu alınır
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw["Close"]
        else:
            df = raw

        results = []
        for ticker in TICKERS:
            try:
                # Sütun adı ".IS" uzantılı veya uzantısız olabilir, ikisini de dene
                col = ticker if ticker in df.columns else ticker.replace(".IS", "")
                if col not in df.columns:
                    continue
                prices = df[col].dropna()

                # En az 1 yıl (252 işlem günü) veri yoksa hesaplama yapılmaz
                if len(prices) < 252:
                    continue

                # --- Volatilite Hesabı ---
                # Logaritmik günlük getiriler: ln(P_t / P_{t-1})
                # Yıllıklaştırma: günlük std × √252 (bir yıldaki ortalama işlem günü)
                log_ret = np.log(prices / prices.shift(1)).dropna()
                volatility = float(log_ret.std() * np.sqrt(252))

                # --- Maksimum Drawdown Hesabı ---
                # Kümülatif getiri serisi oluşturulur
                # Drawdown = (mevcut değer - tarihi tepe) / tarihi tepe
                # Max drawdown: bu serinin mutlak minimum değeri
                cum = (1 + log_ret).cumprod()
                drawdown = (cum - cum.cummax()) / cum.cummax()
                max_dd = float(abs(drawdown.min()))

                # Son 60 günlük fiyatlar minigrafikte (sparkline) gösterilir
                sparkline = [round(float(v), 2) for v in prices.iloc[-60:].values]
                # 52 haftalık (252 işlem günü) yüksek ve düşük değerler
                last_252 = prices.iloc[-252:]

                results.append({
                    "ticker": ticker.replace(".IS", ""),
                    "full_ticker": ticker,
                    "volatility": round(volatility, 4),
                    "max_drawdown": round(max_dd, 4),
                    "current_price": round(float(prices.iloc[-1]), 2),
                    "high_52w": round(float(last_252.max()), 2),
                    "low_52w": round(float(last_252.min()), 2),
                    "sparkline": sparkline,
                })
            except Exception:
                continue

        if not results:
            print("Hiç veri alınamadı.")
            return

        # --- Risk Skoru Hesabı ---
        # Volatilite ve drawdown ayrı ayrı [0,1] aralığına normalize edilir (min-max)
        # Risk skoru = 0.60 × normalize_volatilite + 0.40 × normalize_drawdown
        # Epsilon (1e-10) sıfıra bölünmeyi önler
        vol = np.array([r["volatility"] for r in results])
        dd = np.array([r["max_drawdown"] for r in results])
        vol_n = (vol - vol.min()) / (vol.max() - vol.min() + 1e-10)
        dd_n = (dd - dd.min()) / (dd.max() - dd.min() + 1e-10)
        scores = 0.6 * vol_n + 0.4 * dd_n

        # --- Kategori Sınıflandırması ---
        # Hisseler risk skoruna göre çeyreklik dilimlere ayrılır:
        #   ≤ Q25 → Çok Güvenilir (Defansif)
        #   ≤ Q50 → Güvenilir (Dengeli)
        #   ≤ Q75 → Az Güvenilir (Dinamik)
        #   > Q75 → Güvenilmez (Agresif)
        q25, q50, q75 = np.percentile(scores, [25, 50, 75])

        for i, r in enumerate(results):
            s = float(scores[i])
            r["risk_score"] = round(s, 4)
            if s <= q25:
                r["category"] = "Çok Güvenilir"
                r["category_sub"] = "Defansif"
                r["category_level"] = 1
            elif s <= q50:
                r["category"] = "Güvenilir"
                r["category_sub"] = "Dengeli"
                r["category_level"] = 2
            elif s <= q75:
                r["category"] = "Az Güvenilir"
                r["category_sub"] = "Dinamik"
                r["category_level"] = 3
            else:
                r["category"] = "Güvenilmez"
                r["category_sub"] = "Agresif"
                r["category_level"] = 4

        # En düşük riskten en yükseğe doğru sırala
        results.sort(key=lambda x: x["risk_score"])

        # Sütun adlarından ".IS" uzantısını temizle (API genelinde tutarlı kullanım için)
        clean_df = df.rename(columns=lambda c: c.replace(".IS", "") if ".IS" in str(c) else c)

        # --- 2026 Yıl Sonu Fiyat Tahmini (Ridge Regression) ---
        # Bugünden 31 Aralık 2026'ya kadar kalan işlem günü sayısı hesaplanır
        # (takvim günleri × 252/365 oranıyla işlem gününe çevrilir)
        today = date.today()
        year_end = date(2026, 12, 31)
        trading_days = max(1, int((year_end - today).days * 252 / 365))
        for r in results:
            try:
                col = r["ticker"]
                if col not in clean_df.columns:
                    r["year_end_forecast"] = None
                    continue
                prices_s = clean_df[col].dropna()

                # Özellik mühendisliği: gecikmeli fiyatlar (lag) ve hareketli ortalamalar
                # Lag özellikler: t-1, t-2, t-3, t-5, t-10, t-20 günlük fiyatlar
                # MA özellikler: 20, 50, 200 günlük hareketli ortalamalar
                df_f = pd.DataFrame({"price": prices_s})
                for lag in [1, 2, 3, 5, 10, 20]:
                    df_f[f"lag_{lag}"] = df_f["price"].shift(lag)
                df_f["ma_20"] = df_f["price"].rolling(20).mean()
                df_f["ma_50"] = df_f["price"].rolling(50).mean()
                df_f["ma_200"] = df_f["price"].rolling(200).mean()
                df_f = df_f.dropna()

                if len(df_f) < 50:
                    r["year_end_forecast"] = None
                    continue

                X = df_f.drop("price", axis=1).values
                y = df_f["price"].values

                # Son 6 ay (126 işlem günü) test seti olarak ayrılır, geri kalanı eğitim
                split = max(len(X) - 126, int(len(X) * 0.8))
                scaler = StandardScaler()  # Özellikleri standartlaştır (ortalama=0, std=1)
                model = Ridge(alpha=1.0)   # L2 regularizasyon ile doğrusal regresyon
                model.fit(scaler.fit_transform(X[:split]), y[:split])

                # Yinelemeli tahmin: her adımda bir sonraki günün fiyatı tahmin edilip
                # buffer'a eklenir ve bir sonraki adımın özelliği olarak kullanılır
                price_buf = list(df_f["price"].iloc[-200:].values)
                for _ in range(trading_days):
                    lags = [price_buf[-i] for i in [1, 2, 3, 5, 10, 20]]
                    ma20 = float(np.mean(price_buf[-20:]))
                    ma50 = float(np.mean(price_buf[-50:])) if len(price_buf) >= 50 else float(np.mean(price_buf))
                    ma200 = float(np.mean(price_buf[-200:])) if len(price_buf) >= 200 else float(np.mean(price_buf))
                    feat = np.array(lags + [ma20, ma50, ma200]).reshape(1, -1)
                    pred = float(model.predict(scaler.transform(feat))[0])
                    price_buf.append(pred)
                r["year_end_forecast"] = round(price_buf[-1], 2)
            except Exception:
                r["year_end_forecast"] = None

        # Thread-safe biçimde global değişkenleri güncelle
        with _lock:
            _risk_data = results
            _price_data = clean_df
            _last_update = datetime.now(TZ_TR).strftime("%d.%m.%Y %H:%M")

        print(f"{len(results)} hisse için risk metrikleri hesaplandı.")
    except Exception:
        traceback.print_exc()
    finally:
        _loading = False


@app.on_event("startup")
async def startup():
    """Uygulama başlarken veri hesaplamayı arka planda başlat (ana thread'i bloke etmez)."""
    t = threading.Thread(target=calculate_risk_metrics, daemon=True)
    t.start()


@app.get("/")
@app.head("/")
async def root():
    """Ana sayfa: index.html dosyasını HTML olarak döner. HEAD isteği Render health check için."""
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/refresh")
async def refresh():
    """Manuel veri yenileme: arka planda yeni bir hesaplama thread'i başlatır."""
    global _loading
    if _loading:
        return {"status": "already_loading"}
    t = threading.Thread(target=calculate_risk_metrics, daemon=True)
    t.start()
    return {"status": "started"}


@app.get("/api/status")
async def status():
    """Frontend'in yükleme ekranında polling yapabilmesi için anlık durum bilgisi döner."""
    return {"loading": _loading, "ready": _risk_data is not None, "last_update": _last_update}


@app.get("/api/risk")
async def risk():
    """Tüm hisselerin risk metriklerini JSON olarak döner. Veri hazır değilse 503 döner."""
    if _loading and _risk_data is None:
        return JSONResponse({"loading": True, "data": []})
    if _risk_data is None:
        raise HTTPException(503, "Veri henüz hazır değil")
    return {"loading": False, "data": _risk_data, "last_update": _last_update}


@app.get("/api/portfolio")
async def portfolio_calc(tickers: str, weights: str):
    """
    Kullanıcının seçtiği hisseler ve ağırlıklarla portföy analizi yapar.

    Modern Portföy Teorisi'ne göre:
      - Portföy volatilitesi : σ_p = √(w^T × Σ × w)
        (w: ağırlık vektörü, Σ: yıllıklaştırılmış kovaryans matrisi)
      - Maksimum drawdown    : Portföy kümülatif getiri serisinden hesaplanır
      - Risk skoru           : Hisse risk skorlarının ağırlıklı ortalaması
    """
    if _risk_data is None or _price_data is None:
        raise HTTPException(503, "Veri henüz hazır değil")

    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    try:
        weight_list = [float(w.replace(",", ".")) for w in weights.split(",")]
    except ValueError:
        raise HTTPException(400, "Geçersiz ağırlıklar")

    if len(ticker_list) != len(weight_list) or len(ticker_list) < 2:
        raise HTTPException(400, f"Ticker/ağırlık sayısı uyuşmuyor: {len(ticker_list)} vs {len(weight_list)}")

    # Veri tabanında bulunan hisseleri filtrele
    avail = [t for t in ticker_list if t in _price_data.columns]
    if len(avail) < 2:
        missing = [t for t in ticker_list if t not in _price_data.columns]
        raise HTTPException(400, f"Hisseler bulunamadı: {missing}")

    # Eksik veri olan hisseleri çıkar ve ağırlıkları yeniden normalize et
    idx_map = {t: i for i, t in enumerate(ticker_list)}
    w_raw = np.array([weight_list[idx_map[t]] for t in avail], dtype=float)
    if w_raw.sum() == 0:
        raise HTTPException(400, "Ağırlıklar sıfır olamaz")

    # İleri doldurma (forward fill) ile eksik günleri kapat, ardından NaN satırları at
    prices = _price_data[avail].ffill().dropna(how="all")
    # En az 1 yıl (252 gün) verisi olan hisseleri tut
    good_cols = [c for c in avail if prices[c].notna().sum() >= 252]
    if len(good_cols) < 2:
        raise HTTPException(400, "Yeterli geçerli fiyat verisi bulunamadı")

    # Kalan hisseler için ağırlıkları yeniden normalize et (toplamı 1 olsun)
    good_idx = [avail.index(c) for c in good_cols]
    w = w_raw[good_idx]
    w = w / w.sum()
    avail = good_cols

    returns = prices[avail].pct_change().dropna()
    if len(returns) < 10:
        raise HTTPException(400, "Yeterli getiri verisi yok")

    # Yıllıklaştırılmış kovaryans matrisi (günlük kovaryans × 252)
    cov = returns.cov() * 252
    cov_vals = np.nan_to_num(cov.values, nan=0.0)
    # Portföy varyansı: w^T × Σ × w (matris çarpımı)
    port_var = float(w @ cov_vals @ w)
    port_vol = float(np.sqrt(max(port_var, 0)))

    # Portföy günlük getirisi = her hissenin getirisinin ağırlıklı toplamı
    port_ret = (returns[avail] * w).sum(axis=1)
    cum = (1 + port_ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = float(abs(dd.min()))

    # Ağırlıklı risk skoru: her hissenin risk skorunun portföy ağırlığıyla çarpımının toplamı
    w_score = 0.0
    for i, ticker in enumerate(avail):
        stock = next((r for r in _risk_data if r["ticker"] == ticker), None)
        if stock:
            w_score += float(w[i]) * stock["risk_score"]

    # Portföy risk kategorisi belirlenir (bireysel hisse kategorileriyle aynı eşikler)
    if w_score <= 0.25:
        cat, cat_sub, cat_lv = "Çok Güvenilir", "Defansif", 1
    elif w_score <= 0.50:
        cat, cat_sub, cat_lv = "Güvenilir", "Dengeli", 2
    elif w_score <= 0.75:
        cat, cat_sub, cat_lv = "Az Güvenilir", "Dinamik", 3
    else:
        cat, cat_sub, cat_lv = "Güvenilmez", "Agresif", 4

    # Grafik için son 2 yıllık (504 işlem günü) kümülatif getiri, baz=100 normalize
    hist = cum.iloc[-504:]
    base = float(hist.iloc[0])
    normalized = [round(float(v) / base * 100, 2) for v in hist.values]

    return {
        "tickers": avail,
        "weights": [round(float(wi) * 100, 1) for wi in w],
        "portfolio_volatility": round(port_vol, 4),
        "max_drawdown": round(max_dd, 4),
        "weighted_risk_score": round(w_score, 4),
        "category": cat,
        "category_sub": cat_sub,
        "category_level": cat_lv,
        "dates": hist.index.strftime("%Y-%m-%d").tolist(),
        "values": normalized,
    }


@app.get("/api/forecast/{ticker}")
async def forecast(ticker: str):
    """
    Belirli bir hisse için detaylı fiyat tahmini döner.

    Dönen veriler:
      - hist_*      : 10 yıllık gerçek fiyat serisi
      - backtest_*  : Modelin eğitim dışı test kümesindeki tahminleri (model doğrulaması)
      - forecast_*  : Bugünden 2026 yıl sonuna kadar günlük tahmin fiyatları
      - year_end_forecast : 31 Aralık 2026 tahmini kapanış fiyatı
    """
    if _risk_data is None or _price_data is None:
        raise HTTPException(503, "Veri henüz hazır değil")

    t = ticker.upper()
    stock = next((r for r in _risk_data if r["ticker"] == t), None)
    if not stock:
        raise HTTPException(404, "Hisse bulunamadı")

    if t not in _price_data.columns:
        raise HTTPException(404, "Fiyat verisi bulunamadı")

    try:
        prices = _price_data[t].dropna()

        # Aynı özellik mühendisliği: lag değerleri + hareketli ortalamalar
        df_f = pd.DataFrame({"price": prices})
        for lag in [1, 2, 3, 5, 10, 20]:
            df_f[f"lag_{lag}"] = df_f["price"].shift(lag)
        df_f["ma_20"] = df_f["price"].rolling(20).mean()
        df_f["ma_50"] = df_f["price"].rolling(50).mean()
        df_f["ma_200"] = df_f["price"].rolling(200).mean()
        df_f = df_f.dropna()

        X = df_f.drop("price", axis=1).values
        y = df_f["price"].values
        # Son 6 ay (126 işlem günü) test seti; öncesi eğitim seti
        split = max(len(X) - 126, int(len(X) * 0.8))

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[:split])
        X_te = scaler.transform(X[split:])
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y[:split])
        # Backtest: modelin gerçek test verisindeki tahminleri (grafik doğrulama için)
        backtest = model.predict(X_te).tolist()

        today = date.today()
        year_end = date(2026, 12, 31)
        trading_days = max(1, int((year_end - today).days * 252 / 365))

        # Yinelemeli ileriye dönük tahmin: her gün tahmin bir sonraki günün girdisi olur
        price_buf = list(df_f["price"].iloc[-200:].values)
        forecast_prices = []
        for _ in range(trading_days):
            lags = [price_buf[-i] for i in [1, 2, 3, 5, 10, 20]]
            ma20 = float(np.mean(price_buf[-20:]))
            ma50 = float(np.mean(price_buf[-50:])) if len(price_buf) >= 50 else float(np.mean(price_buf))
            ma200 = float(np.mean(price_buf[-200:])) if len(price_buf) >= 200 else float(np.mean(price_buf))
            feat = np.array(lags + [ma20, ma50, ma200]).reshape(1, -1)
            pred = float(model.predict(scaler.transform(feat))[0])
            forecast_prices.append(round(pred, 2))
            price_buf.append(pred)

        hist = df_f  # Tüm 10 yıllık gerçek fiyat serisi
        bt = df_f.iloc[split:]  # Test kümesi (backtest için)
        # İş günleri bazında tahmin tarihleri oluştur (hafta sonları atlanır)
        forecast_dates = pd.bdate_range(pd.Timestamp.today(), periods=trading_days).strftime("%Y-%m-%d").tolist()

        return {
            "ticker": ticker.upper(),
            "hist_dates": hist.index.strftime("%Y-%m-%d").tolist(),
            "hist_prices": [round(float(p), 2) for p in hist["price"]],
            "backtest_dates": bt.index.strftime("%Y-%m-%d").tolist(),
            "backtest_prices": [round(p, 2) for p in backtest],
            "forecast_dates": forecast_dates,
            "forecast_prices": forecast_prices,
            "year_end_forecast": forecast_prices[-1] if forecast_prices else None,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))
