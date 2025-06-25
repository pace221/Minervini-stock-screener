import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import StringIO
import time
import altair as alt

# Page configuration
st.set_page_config(page_title="S&P 500 Stock Screener - Minervini Criteria",
                   page_icon="📈",
                   layout="wide")


# Get current data date for display
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_data_date():
    try:
        # Get a sample stock to determine data date
        sample_data = yf.download("AAPL", period="1d", progress=False)
        if not sample_data.empty:
            return sample_data.index[-1].strftime("%d.%m.%Y")
    except:
        pass
    return pd.Timestamp.now().strftime("%d.%m.%Y")


# Title and description with data date
col1, col2 = st.columns([3, 1])
with col1:
    st.title("📈 S&P 500 Stock Screener")
    st.markdown("### Minervini Criteria & Pattern Recognition")
with col2:
    data_date = get_data_date()
    st.markdown(f"**Stichtag Daten:**  \n{data_date}")
    st.caption("Letzte Aktualisierung")
st.markdown("""
Dieser Screener wendet Mark Minervinis Trend Template auf S&P 500 Aktien mit Mustererkennung an (**basierend auf Tagesendkursen**):
- **Kurs über 50-Tage MA**: Aktueller Kurs > 50-Tage gleitender Durchschnitt
- **Kurs über 200-Tage MA**: Aktueller Kurs > 200-Tage gleitender Durchschnitt  
- **Nahe 52-Wochen-Hoch**: Aktueller Kurs >= 90% des 52-Wochen-Hochs
- **Erhöhtes Volumen**: 3-Tage Durchschnittsvolumen > eigener historischer Durchschnitt
- **CRV >= 2:1**: Chance-Risiko-Verhältnis muss mindestens 2:1 betragen
- **Mustererkennung**: Identifiziert Breakout-Muster und Trading-Setups
- **KO-Schein Berechnung**: Automatische Knock-Out Empfehlung für 1R Position (10.000€ Portfolio)

**Was bedeutet "KO Investment"?**  
Die Spalte "KO Investment €" zeigt den **Betrag in Euro**, den Sie in einen Knock-Out Schein investieren müssten, 
um die gleiche Marktexposition wie beim direkten Aktienkauf zu erhalten, aber mit geringerem Kapitaleinsatz durch den Hebel.
""")


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_sp500_tickers():
    """Load S&P 500 ticker symbols from Wikipedia"""
    try:
        tables = pd.read_html(
            'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500_df = tables[0]
        tickers = sp500_df['Symbol'].tolist()

        # Clean up ticker symbols - replace problematic characters
        cleaned_tickers = []
        for ticker in tickers:
            # Handle Class B shares and other special cases
            if '.' in ticker:
                # Replace dots with dashes for Yahoo Finance compatibility
                ticker = ticker.replace('.', '-')
            cleaned_tickers.append(ticker)

        return cleaned_tickers
    except Exception as e:
        st.error(f"Failed to load S&P 500 tickers: {e}")
        return []


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_nasdaq100_tickers():
    """Load NASDAQ 100 ticker symbols"""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
        nasdaq_df = tables[4]  # Usually the 5th table contains the components
        return nasdaq_df['Ticker'].tolist()
    except Exception as e:
        # Fallback to major NASDAQ stocks
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA',
            'NFLX', 'ADBE', 'CRM', 'PYPL', 'INTC', 'CMCSA', 'PEP', 'COST',
            'TMUS', 'AVGO', 'TXN', 'QCOM', 'CHTR', 'SBUX', 'GILD', 'INTU',
            'ISRG', 'BKNG', 'REGN', 'FISV', 'ADP', 'AMD'
        ]


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_eurostoxx50_tickers():
    """Load Euro Stoxx 50 ticker symbols"""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/EURO_STOXX_50')
        euro_df = tables[1]  # Usually the 2nd table contains components
        # Add .DE, .PA, .AS suffixes for German, French, Dutch stocks
        tickers = []
        for ticker in euro_df['Ticker'].tolist():
            # Add appropriate suffix based on common patterns
            if any(x in ticker for x in
                   ['SAP', 'SIE', 'ALV', 'BAS', 'BMW', 'DAI', 'DTE', 'MUV2']):
                tickers.append(f"{ticker}.DE")
            elif any(x in ticker
                     for x in ['MC', 'OR', 'SU', 'BNP', 'SAN', 'AI', 'CAP']):
                tickers.append(f"{ticker}.PA")
            elif any(x in ticker for x in ['ASML', 'PHIA', 'UNA']):
                tickers.append(f"{ticker}.AS")
            else:
                tickers.append(ticker)
        return tickers
    except Exception as e:
        # Fallback to major European stocks
        return [
            'SAP.DE', 'ASML.AS', 'LVMH.PA', 'NVO', 'TTE.PA', 'RMS.PA',
            'SAN.PA', 'BNP.PA', 'SIE.DE', 'ALV.DE', 'BAS.DE', 'BMW.DE',
            'VOW3.DE', 'DTE.DE', 'MUV2.DE', 'ADS.DE'
        ]


def screen_stock(ticker, market_avg_volume):
    """Screen a single stock against Minervini criteria"""
    try:
        # Download 1.5 years of data to ensure we have enough for calculations
        data = yf.download(ticker,
                           period="18mo",
                           progress=False,
                           auto_adjust=True)

        if data.empty:
            return None, f"No data available for {ticker}"

        if len(data) < 200:
            return None, f"Insufficient data (only {len(data)} days, need 200+)"

        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Ensure we have the required columns
        required_columns = ['Close', 'Volume']
        missing_columns = [
            col for col in required_columns if col not in data.columns
        ]
        if missing_columns:
            return None, f"Missing required data columns for {ticker}: {missing_columns}"

        close = data['Close'].dropna()
        volume = data['Volume'].dropna()

        if len(close) < 200:
            return None, f"Insufficient close price data ({len(close)} days)"

        if len(volume) < 200:
            return None, f"Insufficient volume data ({len(volume)} days)"

        # Calculate indicators
        ma50 = close.rolling(window=50, min_periods=50).mean()
        ma200 = close.rolling(window=200, min_periods=200).mean()
        high52 = close.rolling(
            window=252,
            min_periods=100).max()  # Use min_periods for flexibility

        # Calculate volume indicators
        recent_volume_3d = volume.tail(
            3).mean()  # Average volume over last 3 days
        stock_avg_volume = volume.mean()  # This stock's average volume

        # Get latest values
        latest_close = close.iloc[-1]
        latest_ma50 = ma50.iloc[-1]
        latest_ma200 = ma200.iloc[-1]
        latest_high52 = high52.iloc[-1]

        # Check for valid data
        if pd.isna(latest_close):
            return None, f"Missing latest close price for {ticker}"
        if pd.isna(latest_ma50):
            return None, f"Cannot calculate 50-day MA for {ticker}"
        if pd.isna(latest_ma200):
            return None, f"Cannot calculate 200-day MA for {ticker}"
        if pd.isna(latest_high52):
            return None, f"Cannot calculate 52-week high for {ticker}"
        if pd.isna(recent_volume_3d) or pd.isna(stock_avg_volume):
            return None, f"Cannot calculate volume data for {ticker}"

        # Calculate volume ratios
        volume_vs_market = recent_volume_3d / market_avg_volume if market_avg_volume > 0 else 0
        volume_vs_own_avg = recent_volume_3d / stock_avg_volume if stock_avg_volume > 0 else 0

        # Pattern recognition and trading setup analysis
        pattern, buy_idea = analyze_pattern(close, ma50, ma200, volume,
                                            latest_close, latest_ma50,
                                            latest_ma200, latest_high52,
                                            recent_volume_3d, stock_avg_volume)

        # Calculate entry price and CRV (Chance-Risk-Verhältnis)
        # Entry price: slightly above current price (1% buffer for market orders)
        entry_price = latest_close * 1.01

        # Stop loss: below MA50 or 8% below current price (whichever is closer)
        stop_loss_ma50 = latest_ma50 * 0.99  # Slightly below MA50
        stop_loss_8pct = latest_close * 0.92  # 8% stop loss
        stop_loss = max(stop_loss_ma50, stop_loss_8pct)

        # Target: 52-week high or 20% above current price (whichever is higher)
        target_52w = latest_high52
        target_20pct = latest_close * 1.20
        target_price = max(target_52w, target_20pct)
        
        # Take Profit levels (conservative and aggressive)
        take_profit_1 = entry_price + (target_price - entry_price) * 0.5  # 50% of move to target
        take_profit_2 = entry_price + (target_price - entry_price) * 0.75  # 75% of move to target

        # Calculate CRV (Chance-Risk-Verhältnis)
        risk = entry_price - stop_loss
        reward = target_price - entry_price
        crv = reward / risk if risk > 0 else 0

        # Calculate position sizing for 1R risk with 10,000 EUR portfolio
        portfolio_value = 10000  # EUR
        risk_per_trade_eur = portfolio_value * 0.01  # 1% risk = 100 EUR
        risk_per_share = entry_price - stop_loss
        max_shares = int(risk_per_trade_eur /
                         risk_per_share) if risk_per_share > 0 else 0
        position_value_eur = max_shares * entry_price

        # Calculate knockout certificate details
        knockout_barrier = stop_loss * 0.98  # Barrier slightly below stop loss
        leverage = round(entry_price / (entry_price - knockout_barrier), 1)
        knockout_investment = position_value_eur / leverage if leverage > 0 else 0

        # Check Minervini criteria + elevated volume + CRV
        criteria_met = (
            latest_close > latest_ma50 and latest_close > latest_ma200
            and latest_close >= 0.9 * latest_high52
            and volume_vs_own_avg >= 1.0
            and  # Recent volume must be above stock's own average
            crv >= 2.0  # CRV must be at least 2:1
        )

        result = {
            'Ticker': ticker,
            'Close': round(float(latest_close), 2),
            'Entry_Price': round(float(entry_price), 2),
            'Stop_Loss': round(float(stop_loss), 2),
            'Target': round(float(target_price), 2),
            'CRV': round(float(crv), 2),
            'Pattern': pattern,
            'Buy_Idea': buy_idea,
            'Position_Size': max_shares,
            'Position_Value_EUR': round(float(position_value_eur), 0),
            'KO_Leverage': round(float(leverage), 1),
            'KO_Investment_EUR': round(float(knockout_investment), 0),
            'KO_Barrier': round(float(knockout_barrier), 2),
            'MA50': round(float(latest_ma50), 2),
            'MA200': round(float(latest_ma200), 2),
            '52W_High': round(float(latest_high52), 2),
            'Distance_from_High': round(((latest_close / latest_high52) * 100),
                                        1),
            'Stock_Avg_Volume': int(stock_avg_volume),
            'Recent_Volume_3d': int(recent_volume_3d),
            'Volume_vs_Market': round(float(volume_vs_market), 2),
            'Volume_vs_Own': round(float(volume_vs_own_avg), 2),
            'Criteria_Met': criteria_met
        }

        return result, None

    except Exception as e:
        return None, f"Error processing {ticker}: {str(e)}"


def analyze_pattern(close, ma50, ma200, volume, current_price, current_ma50,
                    current_ma200, high52w, recent_volume, avg_volume):
    """Analyze price pattern and identify trading setup"""

    # Get recent price action (last 20 days)
    recent_closes = close.tail(20)
    recent_ma50 = ma50.tail(20)
    recent_volume_data = volume.tail(20)

    # Calculate some key metrics
    volume_surge = recent_volume / avg_volume if avg_volume > 0 else 1
    distance_from_ma50 = ((current_price - current_ma50) / current_ma50) * 100
    distance_from_high = ((current_price / high52w) * 100)

    # Pattern detection logic
    patterns = []
    buy_ideas = []

    # Breakout patterns
    if distance_from_high >= 95 and volume_surge >= 1.5:
        patterns.append("52W Breakout")
        buy_ideas.append("New high breakout with volume")
    elif distance_from_high >= 90 and current_price > recent_closes.iloc[-5]:
        patterns.append("Near High")
        buy_ideas.append("Approaching 52-week high")

    # Moving average patterns
    if current_price > current_ma50 > current_ma200:
        if distance_from_ma50 < 5:
            patterns.append("MA50 Support")
            buy_ideas.append("Pullback to MA50 support")
        else:
            patterns.append("MA Uptrend")
            buy_ideas.append("Strong uptrend above MAs")

    # Volume patterns
    if volume_surge >= 2.0:
        patterns.append("Volume Surge")
        buy_ideas.append("High volume breakout")
    elif volume_surge >= 1.5:
        patterns.append("Volume Pickup")
        buy_ideas.append("Increasing volume interest")

    # Momentum patterns
    price_change_5d = (
        (current_price - recent_closes.iloc[-6]) /
        recent_closes.iloc[-6]) * 100 if len(recent_closes) >= 6 else 0
    if price_change_5d > 5:
        patterns.append("Strong Momentum")
        buy_ideas.append("5-day momentum breakout")
    elif price_change_5d > 2:
        patterns.append("Moderate Momentum")
        buy_ideas.append("Steady upward momentum")

    # Consolidation patterns
    volatility = recent_closes.std() / recent_closes.mean() * 100
    if volatility < 3 and distance_from_high >= 85:
        patterns.append("High Consolidation")
        buy_ideas.append("Tight consolidation near highs")

    # Default if no specific pattern
    if not patterns:
        patterns.append("Base Building")
        buy_ideas.append("Building base above MAs")

    # Combine patterns and ideas
    pattern_str = " + ".join(patterns[:2])  # Max 2 patterns
    buy_idea_str = buy_ideas[0] if buy_ideas else "Momentum trade"

    return pattern_str, buy_idea_str


def get_chart_data(ticker, days=120):
    """Get chart data for a specific ticker"""
    try:
        data = yf.download(ticker,
                           period=f"{days}d",
                           progress=False,
                           auto_adjust=True)

        if data.empty:
            return None

        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Calculate moving averages
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()

        # Calculate 52-week high for reference
        data['High52W'] = data['Close'].rolling(window=252,
                                                min_periods=1).max()
        data['High90Pct'] = data['High52W'] * 0.9

        # Reset index to make Date a column
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])

        return data[[
            'Date', 'Close', 'MA50', 'MA200', 'High52W', 'High90Pct', 'Volume'
        ]].dropna()

    except Exception as e:
        return None


def create_stock_chart(ticker, chart_data):
    """Create an interactive chart for a stock with moving averages"""
    if chart_data is None or chart_data.empty:
        return None

    try:
        # Ensure all numeric columns are properly converted
        numeric_cols = ['Close', 'MA50', 'MA200', 'High90Pct', 'Volume']
        for col in numeric_cols:
            if col in chart_data.columns:
                chart_data[col] = pd.to_numeric(chart_data[col],
                                                errors='coerce')

        # Remove any rows with NaN values
        chart_data = chart_data.dropna()

        if chart_data.empty:
            return None

        # Price chart with moving averages
        price_data = chart_data.melt(
            id_vars=['Date'],
            value_vars=['Close', 'MA50', 'MA200', 'High90Pct'],
            var_name='Indicator',
            value_name='Price')

        # Remove NaN values from melted data
        price_data = price_data.dropna()

        # Main price chart
        base_chart = alt.Chart(price_data)

        price_chart = base_chart.mark_line().encode(
            x=alt.X('Date:T', title='Datum'),
            y=alt.Y('Price:Q', title='Kurs ($)', scale=alt.Scale(zero=False)),
            color=alt.Color(
                'Indicator:N',
                scale=alt.Scale(
                    domain=['Close', 'MA50', 'MA200', 'High90Pct'],
                    range=['#2E86C1', '#F39C12', '#E74C3C', '#9B59B6']),
                legend=alt.Legend(title="Indikatoren")),
            strokeWidth=alt.condition(alt.datum.Indicator == 'Close',
                                      alt.value(2.5), alt.value(1.5)),
            strokeDash=alt.condition(
                alt.datum.Indicator == 'High90Pct', alt.value([5, 5]),
                alt.value([]))).properties(
                    width=600,
                    height=300,
                    title=f"{ticker} - Kursverlauf (120 Tage)").interactive()

        # Volume chart
        volume_chart = alt.Chart(chart_data).mark_bar(opacity=0.6).encode(
            x=alt.X('Date:T', title=''),
            y=alt.Y('Volume:Q', title='Volumen', axis=alt.Axis(format='.0s')),
            color=alt.value('#95A5A6')).properties(
                width=600, height=80, title="Volumen").interactive()

        # Combine charts vertically
        final_chart = alt.vconcat(price_chart,
                                  volume_chart).resolve_scale(x='shared')

        return final_chart

    except Exception as e:
        st.error(f"Fehler beim Erstellen des Charts für {ticker}: {str(e)}")
        return None


def calculate_market_avg_volume(tickers, market_name, progress_bar,
                                status_text):
    """Calculate average volume across selected market stocks"""
    volumes = []
    status_text.text(f"Calculating {market_name} average volume...")

    # Sample 50 stocks to calculate average volume (for speed)
    sample_tickers = tickers[:50] if len(tickers) > 50 else tickers

    for i, ticker in enumerate(sample_tickers):
        try:
            progress = (
                i + 1) / len(sample_tickers) * 0.3  # Use 30% of progress bar
            progress_bar.progress(progress)

            data = yf.download(ticker,
                               period="3mo",
                               progress=False,
                               auto_adjust=True)
            if not data.empty and 'Volume' in data.columns:
                avg_vol = data['Volume'].mean()
                if not pd.isna(avg_vol) and avg_vol > 0:
                    volumes.append(avg_vol)
        except:
            continue

    return np.median(volumes) if volumes else 1000000  # Fallback value


def run_screening(tickers, progress_bar, status_text, market_avg_volume):
    """Run screening on selected tickers with progress tracking"""
    results = []
    errors = []

    total_tickers = len(tickers)

    for i, ticker in enumerate(tickers):
        # Update progress (start from 30% if we calculated avg volume)
        progress = 0.3 + (i + 1) / total_tickers * 0.7
        progress_bar.progress(progress)
        status_text.text(f"Screening {ticker} ({i + 1}/{total_tickers})")

        result, error = screen_stock(ticker, market_avg_volume)

        if result:
            results.append(result)
        if error:
            errors.append(f"{ticker}: {error}")

    return results, errors


# Sidebar controls
st.sidebar.header("S&P 500 Screening Parameters")

# Load S&P 500 tickers
with st.spinner("Loading S&P 500 ticker list..."):
    tickers = load_sp500_tickers()

if not tickers:
    st.error("Could not load S&P 500 ticker list. Please refresh the page.")
    st.stop()

st.sidebar.success(f"Loaded {len(tickers)} S&P 500 tickers")

# Stock selection options
stock_count_options = {
    "Test with major tech stocks (AAPL, MSFT, GOOGL, TSLA, NVDA, META)":
    "test",
    "First 10 stocks": 10,
    "First 50 stocks": 50,
    "First 100 stocks": 100,
    "All S&P 500 stocks": len(tickers)
}

selected_option = st.sidebar.selectbox("Select number of stocks to screen:",
                                       list(stock_count_options.keys()))

if stock_count_options[selected_option] == "test":
    # Use major US tech stocks for testing
    selected_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META"]
else:
    num_stocks = stock_count_options[selected_option]
    selected_tickers = tickers[:num_stocks]

st.sidebar.info(f"Will screen {len(selected_tickers)} stocks")

# Additional filters
st.sidebar.subheader("Additional Filters")
min_price = st.sidebar.number_input("Minimum stock price ($)",
                                    min_value=0.0,
                                    value=5.0,
                                    step=0.5)
min_distance_from_high = st.sidebar.slider(
    "Minimum distance from 52-week high (%)", 80, 100, 90)
min_volume_vs_market = st.sidebar.slider("Minimum volume vs market average",
                                         0.5,
                                         3.0,
                                         1.0,
                                         step=0.1)
min_volume_vs_own = st.sidebar.slider("Minimum volume vs stock's own average",
                                      1.0,
                                      5.0,
                                      1.2,
                                      step=0.1)
min_crv = st.sidebar.slider("Minimum CRV (Chance-Risk-Verhältnis)",
                            2.0,
                            10.0,
                            2.0,
                            step=0.5)

# Debug section
st.sidebar.subheader("Debug")
debug_ticker = st.sidebar.text_input("Check specific ticker (e.g., META):",
                                     "").upper()
if debug_ticker and st.sidebar.button("Check Ticker"):
    debug_result, debug_error = screen_stock(debug_ticker,
                                             15000000)  # Use default volume
    if debug_result:
        st.sidebar.success(
            f"{debug_ticker} meets criteria: {debug_result['Criteria_Met']}")
        st.sidebar.write(f"Volume ratio: {debug_result['Volume_vs_Own']}")
    elif debug_error:
        st.sidebar.error(f"{debug_ticker}: {debug_error}")

# Run screening button
if st.sidebar.button("🔍 Run Screening", type="primary"):
    st.header("Screening Results")

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Time tracking
    start_time = time.time()

    with st.spinner(f"Screening {len(selected_tickers)} stocks..."):
        # Calculate S&P 500 average volume first
        market_avg_volume = calculate_market_avg_volume(
            selected_tickers, "S&P 500", progress_bar, status_text)
        st.sidebar.info(f"S&P 500 Average Volume: {market_avg_volume:,.0f}")

        results, errors = run_screening(selected_tickers, progress_bar,
                                        status_text, market_avg_volume)

    end_time = time.time()
    screening_time = round(end_time - start_time, 1)

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    # Process and display results
    if results:
        df_all_results = pd.DataFrame(results)

        # Filter stocks that meet Minervini criteria
        df_passing = df_all_results[df_all_results['Criteria_Met'] == True]

        # Apply additional filters to passing stocks
        df_filtered = df_passing.copy()
        df_filtered = df_filtered[df_filtered['Close'] >= min_price]
        df_filtered = df_filtered[df_filtered['Distance_from_High'] >=
                                  min_distance_from_high]
        df_filtered = df_filtered[df_filtered['Volume_vs_Market'] >=
                                  min_volume_vs_market]
        df_filtered = df_filtered[df_filtered['Volume_vs_Own'] >=
                                  min_volume_vs_own]
        df_filtered = df_filtered[df_filtered['CRV'] >= min_crv]

        # Sort by CRV first, then by volume vs market
        df_filtered = df_filtered.sort_values(['CRV', 'Volume_vs_Market'],
                                              ascending=[False, False])

        # Display results summary
        st.success(f"Screening completed in {screening_time}s")
        st.info(
            f"Analyzed {len(df_all_results)} stocks - {len(df_passing)} meet Minervini criteria - {len(df_filtered)} pass all filters"
        )

        # Show breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyzed", len(df_all_results))
        with col2:
            st.metric("Meet Minervini Criteria", len(df_passing))
        with col3:
            st.metric("Pass All Filters", len(df_filtered))

        if len(df_filtered) > 0:
            # Display results table with key columns first
            st.subheader("Filtered Results")

            # Reorder columns to show most important first
            column_order = [
                'Ticker', 'Pattern', 'Buy_Idea', 'Close', 'Entry_Price',
                'Stop_Loss', 'Target', 'CRV', 'Position_Size', 'KO_Leverage',
                'KO_Investment_EUR', 'Distance_from_High'
            ]
            display_df = df_filtered[column_order]

            st.dataframe(display_df,
                         use_container_width=True,
                         hide_index=True,
                         column_config={
                             'Ticker':
                             st.column_config.TextColumn('Symbol',
                                                         width='small'),
                             'Pattern':
                             st.column_config.TextColumn('Muster',
                                                         width='medium'),
                             'Buy_Idea':
                             st.column_config.TextColumn('Trading Idee',
                                                         width='medium'),
                             'Close':
                             st.column_config.NumberColumn('Aktuell',
                                                           format='$%.2f'),
                             'Entry_Price':
                             st.column_config.NumberColumn('Einstieg',
                                                           format='$%.2f'),
                             'Stop_Loss':
                             st.column_config.NumberColumn('Stop Loss',
                                                           format='$%.2f'),
                             'Target':
                             st.column_config.NumberColumn('Ziel',
                                                           format='$%.2f'),
                             'CRV':
                             st.column_config.NumberColumn('CRV',
                                                           format='%.2f'),
                             'Position_Size':
                             st.column_config.NumberColumn('Aktien Anzahl',
                                                           format='%d'),
                             'KO_Leverage':
                             st.column_config.NumberColumn('KO Hebel',
                                                           format='%.1fx'),
                             'KO_Investment_EUR':
                             st.column_config.NumberColumn('KO Investment €',
                                                           format='%.0f €'),
                             'Distance_from_High':
                             st.column_config.NumberColumn('% v. 52W-Hoch',
                                                           format='%.1f%%')
                         })

            # Summary statistics for filtered results
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Final Results", len(df_filtered))

            with col2:
                avg_price = df_filtered['Close'].mean()
                st.metric("Avg Price", f"${avg_price:.2f}")

            with col3:
                avg_distance = df_filtered['Distance_from_High'].mean()
                st.metric("Avg Distance from High", f"{avg_distance:.1f}%")

            with col4:
                avg_crv = df_filtered['CRV'].mean()
                st.metric("Durchschnittliches CRV", f"{avg_crv:.1f}")

            # Charts section
            st.subheader("Chartübersicht")

            # Show charts for top 5 results
            chart_limit = min(5, len(df_filtered))
            if chart_limit > 0:
                st.info(f"Zeige Charts für die besten {chart_limit} Aktien")

                # Show each chart in an expander
                for i, (_, row) in enumerate(
                        df_filtered.head(chart_limit).iterrows()):
                    ticker = row['Ticker']
                    price = row['Close']
                    distance = row['Distance_from_High']
                    volume_ratio = row['Volume_vs_Own']

                    pattern = row['Pattern']
                    buy_idea = row['Buy_Idea']

                    ko_investment = row['KO_Investment_EUR']
                    ko_leverage = row['KO_Leverage']

                    with st.expander(
                            f"📊 {ticker} - {pattern} | CRV: {row['CRV']} | KO: {ko_investment}€ bei {ko_leverage}x Hebel",
                            expanded=i == 0):
                        # Trading setup summary
                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.markdown(f"**Erkanntes Muster:** {pattern}")
                            st.markdown(f"**Trading Idee:** {buy_idea}")

                        with col2:
                            st.markdown("**Knock-Out Schein Details:**")
                            st.markdown(f"• Hebel: {ko_leverage}x")
                            st.markdown(f"• Investment: {ko_investment}€")
                            st.markdown(f"• Barriere: ${row['KO_Barrier']}")

                        with st.spinner(f"Lade Chart für {ticker}..."):
                            chart_data = get_chart_data(ticker)
                            if chart_data is not None and not chart_data.empty:
                                chart = create_stock_chart(ticker, chart_data)
                                if chart is not None:
                                    st.altair_chart(chart,
                                                    use_container_width=True)

                                    # Add trading metrics below the chart
                                    st.markdown("**Trading Setup Details:**")
                                    col1, col2, col3, col4, col5 = st.columns(
                                        5)
                                    with col1:
                                        st.metric("Einstieg",
                                                  f"${row['Entry_Price']}")
                                    with col2:
                                        st.metric("Stop Loss",
                                                  f"${row['Stop_Loss']}")
                                    with col3:
                                        st.metric("Ziel", f"${row['Target']}")
                                    with col4:
                                        st.metric("CRV", f"{row['CRV']}:1")
                                    with col5:
                                        st.metric("Aktien",
                                                  f"{row['Position_Size']}")
                                else:
                                    st.warning(
                                        f"Chart für {ticker} konnte nicht angezeigt werden"
                                    )
                            else:
                                st.warning(
                                    f"Keine Chart-Daten für {ticker} verfügbar"
                                )

            # Export functionality
            st.subheader("Export Results")

            # Prepare CSV data
            csv_data = df_filtered.drop('Criteria_Met',
                                        axis=1).to_csv(index=False)

            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name=
                f"minervini_screening_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv")

        else:
            st.warning(
                "No stocks found meeting the criteria with current filters.")
            st.info(
                "Try adjusting the filters in the sidebar to find more results."
            )

        # Show sample of all analyzed stocks for debugging
        with st.expander("Debug: View All Analyzed Stocks"):
            st.dataframe(df_all_results.drop('Criteria_Met', axis=1),
                         use_container_width=True,
                         hide_index=True)

    else:
        st.warning("No stocks could be analyzed successfully.")

    # Display errors if any
    if errors:
        st.subheader("Screening Errors")
        with st.expander(f"View {len(errors)} errors"):
            for error in errors:
                st.text(error)

# Information section
with st.expander("ℹ️ About Minervini Criteria"):
    st.markdown("""
    **Mark Minervini's Trend Template** is a stock screening methodology that identifies stocks in strong uptrends:
    
    **The Five Key Criteria:**
    1. **Price above 50-day MA**: Indicates short-term momentum
    2. **Price above 200-day MA**: Indicates long-term trend strength  
    3. **Near 52-week high**: Stock is at or near recent peak performance (≥90%)
    4. **Elevated Volume**: Recent 3-day average volume must be > stock's own average
    5. **CRV >= 2:1**: Chance-Risk-Verhältnis (reward/risk ratio) must be at least 2:1
    
    **Volume Analysis:**
    - Volume vs Market: Recent 3-day avg ÷ selected market average (for comparison)
    - Volume vs Own Average: Recent 3-day avg ÷ stock's historical average (must be > 1.0)
    - Elevated volume confirms price movements and indicates increased interest
    
    **CRV (Chance-Risk-Verhältnis) Calculation:**
    - Entry Price: Current price + 1% buffer for market orders
    - Stop Loss: Below MA50 or 8% below current price (whichever is closer to current price)
    - Target: 52-week high or 20% above current price (whichever is higher)
    - CRV = (Target - Entry) ÷ (Entry - Stop Loss) - must be ≥ 2:1
    
    **Disclaimer:** This tool is for educational purposes only. Always conduct your own research before making investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("*Data provided by Yahoo Finance via yfinance library*")
