# === app.py ===

import logging
import os
from typing import Optional, Tuple, List, Dict, Any

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from flask import Flask, request, jsonify, send_file, send_from_directory
    from flask_cors import CORS
    import pandas as pd
    import io
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from datetime import datetime, timedelta, date
    from prophet import Prophet
    import csv
    
    try:
        import pmdarima as pm
    except ImportError:
        pm = None
        logger.warning("pmdarima not available, ARIMA forecasting will be disabled")
        
    logger.info("All imports successful")
except Exception as e:
    logger.error(f"Import error: {e}")
    raise



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
CORS(app, origins=['*'], 
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





@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'AI Forecast Intelligence Platform is running'})

# Add a simple root health check
@app.route('/api/health')
def api_health():
    return jsonify({'status': 'healthy'})

@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        logger.info("Forecast request received")
        if request.content_type != 'application/json':
            logger.error(f"Unsupported content type: {request.content_type}")
            return jsonify({'error': 'Unsupported content type'}), 400
        
        data = request.get_json()
        logger.info(f"Request data keys: {list(data.keys()) if data else 'None'}")
        
        # Validate input
        is_valid, error_msg = validate_input_data(data)
        if not is_valid:
            logger.error(f"Validation failed: {error_msg}")
            return jsonify({'error': error_msg}), 400
        csv_text = data.get('csv', '')
        time_unit = data.get('timeUnit', 'monthly')
        period_count = int(data.get('periods', 1))
        export_format = data.get('exportFormat', 'csv').lower()  # Default to CSV
        
        logger.info(f"Processing forecast: time_unit={time_unit}, periods={period_count}, format={export_format}")
        logger.info(f"CSV data length: {len(csv_text)} characters")
        
        # Parse CSV safely
        df = parse_csv_safely(csv_text)
        if df is None:
            logger.error("CSV parsing failed")
            return jsonify({'error': 'CSV format not recognized or file too large'}), 400
        
        logger.info(f"CSV parsed successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Identify columns robustly
        date_col, item_col, qty_col, missing = identify_columns_robust(df)
        logger.info(f"Identified columns: date={date_col}, item={item_col}, qty={qty_col}")
        if missing:
            logger.error(f"Missing columns: {missing}")
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
        
        logger.info(f"Starting forecast generation for {len(df[item_col].unique())} unique items")
        
        for item_id, group in df.groupby(item_col):
            ts = group.groupby('period')[qty_col].sum().sort_index()
            
            # Ensure ts is a Series for type checking
            if not isinstance(ts, pd.Series):
                continue
            
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
            # Type checking for last_date - handle pandas scalar properly
            try:
                # Convert to scalar if it's a pandas object
                if hasattr(last_date, 'item'):  # type: ignore
                    last_date = last_date.item()  # type: ignore
                
                # Check if it's a valid date - handle pandas scalar properly
                if last_date is None or (hasattr(last_date, '__bool__') and not bool(last_date)):  # type: ignore
                    continue
                
                # Convert to pandas Timestamp for consistent handling
                last_date = pd.Timestamp(last_date)  # type: ignore
            except (ValueError, TypeError, AttributeError):
                continue
            
            future_dates = []
            for i in range(1, period_count + 1):
                try:
                    if time_unit == 'monthly':
                        next_date = (last_date + pd.DateOffset(months=i)).strftime('%Y-%m')  # type: ignore
                    elif time_unit == 'weekly':
                        next_date = (last_date + timedelta(weeks=i)).strftime('%Y-%m-%d')  # type: ignore
                    else:  # daily
                        next_date = (last_date + timedelta(days=i)).strftime('%Y-%m-%d')  # type: ignore
                    future_dates.append(next_date)
                except (TypeError, AttributeError):
                    # Skip if date arithmetic fails
                    continue
            
            # Add forecasts to results
            for date_str, val in zip(future_dates, forecast_values):
                val = round(float(val), 2)
                forecasts.append([date_str, item_id, val, model_name])
        
        if not forecasts:
            logger.error(f"No forecasts generated. Diagnostics: {diagnostics_log}")
            return jsonify({'error': 'No forecasts could be generated.\n' + '\n'.join(diagnostics_log)}), 400
        
        logger.info(f"Generated {len(forecasts)} forecast entries")
        
        # Create result dataframe
        result_df = pd.DataFrame(forecasts, columns=pd.Index(["Date", "Item ID", "Forecast Quantity", "Model"]))
        
        # Return single file based on export format
        if export_format == 'xlsx':
            try:
                # Export to Excel using temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                    result_df.to_excel(tmp_file.name, index=False, float_format="%.2f", engine='openpyxl')
                    with open(tmp_file.name, 'rb') as f:
                        excel_data = f.read()
                    os.unlink(tmp_file.name)  # Clean up temp file
                
                response = send_file(
                    io.BytesIO(excel_data),
                    download_name="AI_generated_Forecast.xlsx",
                    as_attachment=True,
                    mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                return response
            except ImportError:
                logger.warning("openpyxl not available, falling back to CSV")
                # Fall back to CSV if openpyxl is not available
                output = io.StringIO()
                result_df.to_csv(output, index=False, float_format="%.2f", quoting=csv.QUOTE_MINIMAL)
                output.seek(0)
                response = send_file(
                    io.BytesIO(output.read().encode()),
                    download_name="AI_generated_Forecast.csv",
                    as_attachment=True,
                    mimetype='text/csv'
                )
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                return response
        else:  # Default to CSV
            # Export to CSV
            output = io.StringIO()
            result_df.to_csv(output, index=False, float_format="%.2f", quoting=csv.QUOTE_MINIMAL)
            output.seek(0)
            response = send_file(
                io.BytesIO(output.read().encode()),
                download_name="AI_generated_Forecast.csv",
                as_attachment=True,
                mimetype='text/csv'
            )
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        return jsonify({'error': 'Internal server error'}), 500



# For Railway deployment
app.debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

if __name__ == '__main__':
    # Use environment variable for debug mode
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info("Starting Flask application...")
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
