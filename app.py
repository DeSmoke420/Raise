# === app.py ===

from flask import Flask, request, jsonify, send_file, send_from_directory
import pandas as pd
import io
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from flask_cors import CORS
from datetime import datetime, timedelta
from prophet import Prophet
try:
    import pmdarima as pm
except ImportError:
    pm = None
import csv
import logging
from typing import Optional, Tuple, List, Dict, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_ROWS = 100000
MAX_PERIODS = 120  # Maximum forecast periods
VALID_TIME_UNITS = ['monthly', 'weekly', 'daily']
SEASONAL_PERIODS = {
    'monthly': 12,
    'weekly': 52,
    'daily': 7
}

# Column synonyms for robust detection
COLUMN_SYNONYMS = {
    'date': ['date', 'time', 'period', 'day', 'dt'],
    'item': ['item', 'product', 'sku', 'id', 'name', 'product_id', 'product name', 'item_id'],
    'quantity': ['quantity', 'qty', 'amount', 'volume', 'sales', 'demand', 'forecast quantity', 'forecast', 'predicted', 'value', 'forecasted']
}

app = Flask(__name__, static_folder='.', static_url_path='')

# Configure CORS more securely
CORS(app, origins=['http://localhost:3000', 'http://localhost:5000'], 
     methods=['GET', 'POST'], 
     allow_headers=['Content-Type'])

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find a column in df matching any of the candidate names (case-insensitive, strip spaces)."""
    norm_cols = {col.strip().lower(): col for col in df.columns}
    for candidate in candidates:
        for col in norm_cols:
            if candidate == col or candidate.replace(' ', '') == col.replace(' ', ''):
                return norm_cols[col]
    # Try substring match as fallback
    for candidate in candidates:
        for col in norm_cols:
            if candidate in col or candidate.replace(' ', '') in col.replace(' ', ''):
                return norm_cols[col]
    return None

def identify_columns_robust(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """Identify required columns in the dataframe using robust matching."""
    df.columns = [col.strip() for col in df.columns]
    missing = []
    date_col = find_column(df, COLUMN_SYNONYMS['date'])
    if not date_col:
        missing.append('date (e.g. Date, period, time, day)')
    item_col = find_column(df, COLUMN_SYNONYMS['item'])
    if not item_col:
        missing.append('item (e.g. Item, product, sku, id, name)')
    qty_col = find_column(df, COLUMN_SYNONYMS['quantity'])
    if not qty_col:
        missing.append('quantity (e.g. Quantity, qty, amount, sales, forecast)')
    return date_col, item_col, qty_col, missing

def parse_csv_safely(csv_text: str) -> Optional[pd.DataFrame]:
    """Safely parse CSV with multiple delimiter attempts."""
    delimiters = [',', ';', '\t']
    for delim in delimiters:
        try:
            df = pd.read_csv(io.StringIO(csv_text), delimiter=delim, engine='python')
            if df.shape[1] > 1 and df.shape[0] <= MAX_ROWS:
                return df
        except Exception as e:
            logger.warning(f"Failed to parse CSV with delimiter '{delim}': {e}")
            continue
    return None

def validate_time_series_data(ts: pd.Series, time_unit: str) -> Dict[str, Any]:
    """Check for sufficient data, missing periods, and outliers."""
    result = {'sufficient': True, 'warnings': []}
    if len(ts) < 2:
        result['sufficient'] = False
        result['warnings'].append('Not enough data points for forecasting.')
        return result
    # Check for missing periods
    freq = {'monthly': 'M', 'weekly': 'W', 'daily': 'D'}[time_unit]
    full_range = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq=freq)
    missing = set(full_range) - set(ts.index)
    if missing:
        result['warnings'].append(f"Missing {len(missing)} periods in the time series.")
    # Check for outliers (simple z-score)
    if len(ts) > 10:
        z = (ts - ts.mean()) / ts.std()
        outliers = (abs(z) > 3).sum()
        if outliers > 0:
            result['warnings'].append(f"Detected {outliers} outlier(s) in the data.")
    return result

def create_forecast_model_with_diagnostics(ts: pd.Series, time_unit: str, period_count: int) -> Tuple[Optional[List[float]], str, str]:
    """Try all models, select the best one, and return diagnostics."""
    seasonal_periods = SEASONAL_PERIODS.get(time_unit, 12)
    diagnostics = []
    best_model = None
    best_aic = float('inf')
    best_forecast = None
    best_name = None
    # Holt-Winters
    try:
        model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
        fit = model.fit()
        forecast_values = fit.forecast(period_count)
        diagnostics.append(f"Holt-Winters: AIC={fit.aic:.2f}")
        if fit.aic < best_aic:
            best_aic = fit.aic
            best_forecast = forecast_values.tolist()
            best_name = "Holt-Winters"
    except Exception as e:
        diagnostics.append(f"Holt-Winters failed: {e}")
    # Prophet
    try:
        model = Prophet()
        df_prophet = ts.reset_index()
        df_prophet.columns = ['ds', 'y']
        model.fit(df_prophet)
        freq_map = {'monthly': 'ME', 'weekly': 'W', 'daily': 'D'}
        freq = freq_map.get(time_unit, 'ME')
        future = model.make_future_dataframe(periods=period_count, freq=freq)
        forecast_df = model.predict(future)
        forecast_values = forecast_df.tail(period_count)['yhat'].values
        # Prophet doesn't have AIC, so use MSE as a proxy
        y_true = df_prophet['y']
        y_pred = model.predict(df_prophet)['yhat']
        mse = ((y_true - y_pred) ** 2).mean()
        diagnostics.append(f"Prophet: MSE={mse:.2f}")
        if mse < best_aic:  # Use MSE as a proxy for AIC
            best_aic = mse
            best_forecast = forecast_values.tolist()
            best_name = "Prophet"
    except Exception as e:
        diagnostics.append(f"Prophet failed: {e}")
    # ARIMA
    if pm is not None:
        try:
            model = pm.auto_arima(ts, seasonal=True, m=seasonal_periods, suppress_warnings=True)
            forecast_values = model.predict(n_periods=period_count)
            diagnostics.append(f"ARIMA: AIC={model.aic():.2f}")
            if model.aic() < best_aic:
                best_aic = model.aic()
                best_forecast = forecast_values.tolist()
                best_name = "ARIMA"
        except Exception as e:
            diagnostics.append(f"ARIMA failed: {e}")
    if best_forecast is not None:
        return best_forecast, best_name or '', ' | '.join(diagnostics)
    return None, '', ' | '.join(diagnostics)

def validate_input_data(data: dict) -> Tuple[bool, str]:
    """Validate input data for forecast endpoint."""
    if not data:
        return False, "No data provided"
    csv_text = data.get('csv', '')
    if not csv_text or len(csv_text) > MAX_FILE_SIZE:
        return False, f"CSV data is empty or too large (max {MAX_FILE_SIZE} bytes)"
    time_unit = data.get('timeUnit', 'monthly')
    if time_unit not in VALID_TIME_UNITS:
        return False, f"Invalid time unit. Must be one of: {', '.join(VALID_TIME_UNITS)}"
    try:
        period_count = int(data.get('periods', 1))
        if period_count <= 0 or period_count > MAX_PERIODS:
            return False, f"Period count must be between 1 and {MAX_PERIODS}"
    except (ValueError, TypeError):
        return False, "Invalid period count"
    return True, ""

def find_forecast_column(df: pd.DataFrame) -> Optional[str]:
    """Find a forecast quantity column in the forecast file."""
    candidates = COLUMN_SYNONYMS['quantity']
    return find_column(df, candidates)

def calculate_accuracy_safely(actual: float, forecast: float) -> float:
    """Calculate accuracy safely avoiding division by zero."""
    if actual == 0:
        return 1.0 if forecast == 0 else 0.0
    
    accuracy = 1 - (abs(forecast - actual) / actual)
    return max(0.0, min(1.0, accuracy))  # Clip between 0 and 1

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        if request.content_type != 'application/json':
            return jsonify({'error': 'Unsupported content type'}), 400
        data = request.get_json()
        # Validate input
        is_valid, error_msg = validate_input_data(data)
        if not is_valid:
            return jsonify({'error': error_msg}), 400
        csv_text = data.get('csv', '')
        time_unit = data.get('timeUnit', 'monthly')
        period_count = int(data.get('periods', 1))
        # Parse CSV safely
        df = parse_csv_safely(csv_text)
        if df is None:
            return jsonify({'error': 'CSV format not recognized or file too large'}), 400
        # Identify columns robustly
        date_col, item_col, qty_col, missing = identify_columns_robust(df)
        if missing:
            return jsonify({'error': f'Missing required columns: {", ".join(missing)}'}), 400
        # Process data
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, qty_col, item_col])
        try:
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
        except Exception:
            return jsonify({'error': f'Could not convert {qty_col} to numeric.'}), 400
        # Warn about negative values but allow them
        neg_count = (df[qty_col] < 0).sum()
        if neg_count > 0:
            logger.warning(f"{neg_count} negative values found in {qty_col}. These will be kept.")
        # Warn about missing values
        missing_qty = df[qty_col].isna().sum()
        if missing_qty > 0:
            logger.warning(f"{missing_qty} missing values in {qty_col}. These rows will be dropped.")
        df = df.dropna(subset=[qty_col])
        if df.empty:
            return jsonify({'error': 'No valid data after processing'}), 400
        # Create periods based on time unit
        if time_unit == 'monthly':
            df['period'] = df[date_col].dt.to_period('M').dt.to_timestamp()
        elif time_unit == 'weekly':
            df['period'] = df[date_col] - pd.to_timedelta(df[date_col].dt.dayofweek, unit='D')
        else:  # daily
            df['period'] = df[date_col]
        forecasts = []
        diagnostics_log = []
        for item_id, group in df.groupby(item_col):
            ts: pd.Series = group.groupby('period')[qty_col].sum().sort_index()
            # Data validation
            val_result = validate_time_series_data(ts, time_unit)
            if not val_result['sufficient']:
                diagnostics_log.append(f"Item {item_id}: {val_result['warnings']}")
                continue
            # Model selection with diagnostics
            forecast_values, model_name, diag = create_forecast_model_with_diagnostics(ts, time_unit, period_count)
            diagnostics_log.append(f"Item {item_id}: {diag}")
            if forecast_values is None:
                continue
            # Generate future dates
            last_date = ts.index.max()
            if pd.isna(last_date) or not isinstance(last_date, pd.Timestamp):
                continue
            future_dates = []
            for i in range(1, period_count + 1):
                if time_unit == 'monthly':
                    next_date = (last_date + pd.DateOffset(months=i)).strftime('%Y-%m')
                elif time_unit == 'weekly':
                    next_date = (last_date + timedelta(weeks=i)).strftime('%Y-%m-%d')
                else:  # daily
                    next_date = (last_date + timedelta(days=i)).strftime('%Y-%m-%d')
                future_dates.append(next_date)
            # Add forecasts to results
            for date_str, val in zip(future_dates, forecast_values):
                val = round(float(val), 2)
                forecasts.append([date_str, item_id, val, model_name])
        if not forecasts:
            return jsonify({'error': 'No forecasts could be generated.\n' + '\n'.join(diagnostics_log)}), 400
        # Create output CSV
        output = io.StringIO()
        result_df = pd.DataFrame(forecasts, columns=("Date", "Item ID", "Forecast Quantity", "Model"))
        result_df.to_csv(
            output,
            index=False,
            float_format="%.2f",
            quoting=csv.QUOTE_MINIMAL
        )
        output.seek(0)
        # Attach diagnostics as a log file
        log_output = io.StringIO()
        for line in diagnostics_log:
            log_output.write(line + '\n')
        log_output.seek(0)
        # Return as a zip file if diagnostics exist
        import zipfile
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'AI_generated_Forecast.csv')
            log_path = os.path.join(tmpdir, 'diagnostics.txt')
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(output.getvalue())
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(log_output.getvalue())
            zip_path = os.path.join(tmpdir, 'forecast_results.zip')
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.write(csv_path, arcname='AI_generated_Forecast.csv')
                zf.write(log_path, arcname='diagnostics.txt')
            with open(zip_path, 'rb') as f:
                return send_file(
                    io.BytesIO(f.read()),
                    download_name="forecast_results.zip",
                    as_attachment=True,
                    mimetype='application/zip'
                )
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/accuracy', methods=['POST'])
def accuracy():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        actual_csv = data.get('actuals', '')
        forecast_csv = data.get('forecast', '')
        if not actual_csv or not forecast_csv:
            return jsonify({'error': 'Both actuals and forecast data are required'}), 400
        # Parse CSVs safely
        df_actuals = parse_csv_safely(actual_csv)
        df_forecast = parse_csv_safely(forecast_csv)
        if df_actuals is None or df_forecast is None:
            return jsonify({'error': 'CSV format not recognized'}), 400
        # Identify columns robustly
        date_col, item_col, qty_col, missing = identify_columns_robust(df_actuals)
        forecast_qty_col = find_forecast_column(df_forecast)
        if not all([date_col, item_col, qty_col, forecast_qty_col]):
            return jsonify({'error': 'Missing required columns for accuracy calculation. Expecting date, item, quantity in actuals and forecast quantity in forecast.'}), 400
        # Process data
        df_actuals[date_col] = pd.to_datetime(df_actuals[date_col], errors='coerce')
        df_forecast[date_col] = pd.to_datetime(df_forecast[date_col], errors='coerce')
        df_actuals = df_actuals.groupby([date_col, item_col])[qty_col].sum().reset_index()
        df_forecast = df_forecast.groupby([date_col, item_col])[forecast_qty_col].sum().reset_index()
        # Merge and calculate accuracy
        merged = pd.merge(df_actuals, df_forecast, on=[date_col, item_col], how='inner')
        if merged.empty:
            return jsonify({'error': 'No matching data found between actuals and forecasts'}), 400
        # Calculate accuracy safely
        merged['accuracy'] = merged.apply(
            lambda row: calculate_accuracy_safely(row[qty_col], row[forecast_qty_col]), 
            axis=1
        )
        # Create output CSV
        output = io.StringIO()
        merged[[date_col, item_col, 'accuracy']].to_csv(output, index=False)
        output.seek(0)
        return send_file(
            io.BytesIO(output.read().encode()),
            download_name="Forecast_Accuracy.csv",
            as_attachment=True,
            mimetype='text/csv'
        )
    except Exception as e:
        logger.error(f"Accuracy calculation error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Use environment variable for debug mode
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
