# Trigger redeploy: Railway forced update
# === app.py ===

import logging
import os
import time
import math
import pandas as pd
import io
from typing import Optional, Tuple, List, Dict, Any
from functools import wraps

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from flask import Flask, request, jsonify, send_file, send_from_directory, redirect
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

# Import authentication module
try:
    from auth import (
        require_auth, 
        optional_auth, 
        get_current_user, 
        create_user_session,
        sign_in_with_supabase,
        sign_up_with_supabase
    )
    AUTH_AVAILABLE = True
    logger.info("Authentication module imported successfully")
except ImportError as e:
    logger.warning(f"Authentication module not available: {e}")
    AUTH_AVAILABLE = False
    
    # Import Flask's g for fallback functions
    from flask import g
    
    # Create fallback decorators when auth is not available
    def require_auth(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            return jsonify({'error': 'Authentication required'}), 401
        return decorated_function
    
    def optional_auth(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            g.current_user = None
            return f(*args, **kwargs)
        return decorated_function
    
    def get_current_user() -> Optional[Dict[str, Any]]:
        return None
    
    def create_user_session(user_info: Dict[str, Any]) -> Dict[str, Any]:
        return {'error': 'Authentication not available'}
    
    def sign_in_with_supabase(email: str, password: str) -> Optional[Dict[str, Any]]:
        return None
    
    def sign_up_with_supabase(email: str, password: str) -> Optional[Dict[str, Any]]:
        return None

# Force disable authentication for development
AUTH_AVAILABLE = False

# Import payment and subscription modules
try:
    from stripe_config import payment_manager, PAYMENT_ENABLED
    from subscription_manager import subscription_manager
    PAYMENT_AVAILABLE = True and PAYMENT_ENABLED
    logger.info(f"Payment modules imported successfully. Payment enabled: {PAYMENT_AVAILABLE}")
except ImportError as e:
    logger.warning(f"Payment modules not available: {e}")
    PAYMENT_AVAILABLE = False

# Force disable payment for development
    PAYMENT_AVAILABLE = False

# Configure CORS more securely
CORS(app, origins=['*'], 
     methods=['GET', 'POST'], 
     allow_headers=['Content-Type', 'Authorization', 'X-Supabase-Access-Token'])

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

def create_forecast_model_with_diagnostics(
    ts: pd.Series,
    time_unit: str,
    period_count: int,
    skip_arima: bool = False,
    use_arima: bool = True,
    use_hw: bool = True,
    use_prophet: bool = True
) -> dict:
    """
    For each model, fit on 80% train, evaluate on 20% test (MAPE preferred, RMSE fallback),
    select the best model, then fit the best model on the full data for the actual forecast.
    Returns a dict:
      {
        'forecasts': {model_name: [forecast values]},
        'scores': {model_name: {'MAPE': ..., 'RMSE': ...}},
        'best_model': model_name,
        'diagnostics': {model_name: ...}
      }
    """
    import numpy as np
    start_time = time.time()
    logger.info(f"Starting model selection for time series with {len(ts)} points (train/test for best model)")
    seasonal_periods = SEASONAL_PERIODS.get(time_unit, 12)
    diagnostics = {}
    forecasts = {}
    scores = {}
    best_model = None
    best_metric = float('inf')
    best_metric_name = None
    best_forecast = None
    model_names = []

    def mape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        if not np.any(mask):
            return None
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    def rmse(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    n = len(ts)
    if n < 6:
        return {
            'forecasts': {},
            'scores': {},
            'best_model': None,
            'diagnostics': {'all': 'Not enough data for evaluation.'}
        }
    split_idx = int(n * 0.8)
    train_ts = ts.iloc[:split_idx]
    test_ts = ts.iloc[split_idx:]
    test_len = len(test_ts)

    # --- Holt-Winters ---
    if use_hw:
        logger.info('Attempting Holt-Winters fit...')
        try:
            if len(train_ts) >= seasonal_periods * 2:
                model = ExponentialSmoothing(train_ts, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
            elif len(train_ts) >= 4:
                model = ExponentialSmoothing(train_ts, trend='add', seasonal=None)
            else:
                raise ValueError("Insufficient data for Holt-Winters")
            fit = model.fit()
            hw_forecast = fit.forecast(test_len)
            mape_score = mape(test_ts, hw_forecast)
            rmse_score = rmse(test_ts, hw_forecast)
            scores['Holt-Winters'] = {'MAPE': mape_score, 'RMSE': rmse_score}
            diagnostics['Holt-Winters'] = f"fit params: {fit.params}"
            model_names.append('Holt-Winters')
            # Forecast for output (fit on full data)
            if len(ts) >= seasonal_periods * 2:
                model_full = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
            elif len(ts) >= 4:
                model_full = ExponentialSmoothing(ts, trend='add', seasonal=None)
            else:
                model_full = None
            if model_full:
                fit_full = model_full.fit()
                forecasts['Holt-Winters'] = fit_full.forecast(period_count).tolist()
            else:
                forecasts['Holt-Winters'] = [None] * period_count
            logger.info('Holt-Winters fit and forecast successful.')
        except Exception as e:
            diagnostics['Holt-Winters'] = f"Failed: {e}"
            logger.warning(f"Holt-Winters skipped or failed: {e}")
            forecasts['Holt-Winters'] = [None] * period_count

    # --- Prophet ---
    if use_prophet:
        logger.info('Attempting Prophet fit...')
        try:
            from prophet import Prophet
            df_prophet = train_ts.reset_index()
            df_prophet.columns = ['ds', 'y']
            model = Prophet()
            model.fit(df_prophet)
            if time_unit == 'monthly':
                freq = 'M'
            elif time_unit == 'weekly':
                freq = 'W'
            else:
                freq = 'D'
            future = model.make_future_dataframe(periods=test_len, freq=freq)
            forecast_df = model.predict(future)
            prophet_forecast = forecast_df.tail(test_len)['yhat'].values
            mape_score = mape(test_ts, prophet_forecast)
            rmse_score = rmse(test_ts, prophet_forecast)
            scores['Prophet'] = {'MAPE': mape_score, 'RMSE': rmse_score}
            diagnostics['Prophet'] = f"fit params: {model.params if hasattr(model, 'params') else 'n/a'}"
            model_names.append('Prophet')
            # Forecast for output (fit on full data)
            df_prophet_full = ts.reset_index()
            df_prophet_full.columns = ['ds', 'y']
            model_full = Prophet()
            model_full.fit(df_prophet_full)
            future_full = model_full.make_future_dataframe(periods=period_count, freq=freq)
            forecast_full = model_full.predict(future_full)
            forecasts['Prophet'] = forecast_full.tail(period_count)['yhat'].values.tolist()
            logger.info('Prophet fit and forecast successful.')
        except Exception as e:
            diagnostics['Prophet'] = f"Failed: {e}"
            logger.warning(f"Prophet skipped or failed: {e}")
            forecasts['Prophet'] = [None] * period_count

    # --- ARIMA ---
    if use_arima and pm is not None and not skip_arima:
        logger.info('Attempting ARIMA fit...')
        ARIMA_TIMEOUT = 20
        try:
            if len(train_ts) > 50:  # Lowered from 100
                diagnostics['ARIMA'] = "Skipped: individual item too long (>50 points)"
                logger.warning(f"ARIMA skipped for item: too long ({len(train_ts)} points, stricter)")
                forecasts['ARIMA'] = [None] * period_count
            else:
                import concurrent.futures
                def fit_arima(ts, seasonal, seasonal_periods, period_count):
                    if pm is None:
                        raise RuntimeError("pmdarima is not installed")
                    if seasonal:
                        model = pm.auto_arima(
                            ts, 
                            seasonal=True, 
                            m=seasonal_periods, 
                            suppress_warnings=True, 
                            start_p=0, max_p=1,
                            start_q=0, max_q=1,
                            max_d=1,
                            start_P=0, max_P=1,
                            start_Q=0, max_Q=1,
                            max_D=1,
                            stepwise=True,
                            n_jobs=1,
                            random_state=42
                        )
                    else:
                        model = pm.auto_arima(
                            ts, 
                            seasonal=False, 
                            suppress_warnings=True, 
                            start_p=0, max_p=1,
                            start_q=0, max_q=1,
                            max_d=1,
                            stepwise=True,
                            n_jobs=1,
                            random_state=42
                        )
                    forecast_values = model.predict(n_periods=period_count)
                    return model, forecast_values
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    if len(train_ts) >= seasonal_periods * 2:
                        future = executor.submit(fit_arima, train_ts, True, seasonal_periods, test_len)
                    elif len(train_ts) >= 4:
                        future = executor.submit(fit_arima, train_ts, False, seasonal_periods, test_len)
                    else:
                        raise ValueError("Insufficient data for ARIMA")
                    try:
                        model, arima_forecast = future.result(timeout=ARIMA_TIMEOUT)
                        mape_score = mape(test_ts, arima_forecast)
                        rmse_score = rmse(test_ts, arima_forecast)
                        scores['ARIMA'] = {'MAPE': mape_score, 'RMSE': rmse_score}
                        diagnostics['ARIMA'] = f"fit params: {model.get_params() if hasattr(model, 'get_params') else 'n/a'}"
                        model_names.append('ARIMA')
                        # Forecast for output (fit on full data)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor2:
                            if len(ts) >= seasonal_periods * 2:
                                future2 = executor2.submit(fit_arima, ts, True, seasonal_periods, period_count)
                            elif len(ts) >= 4:
                                future2 = executor2.submit(fit_arima, ts, False, seasonal_periods, period_count)
                            else:
                                future2 = None
                            if future2:
                                try:
                                    model_full, arima_forecast_full = future2.result(timeout=ARIMA_TIMEOUT)
                                    forecasts['ARIMA'] = arima_forecast_full.tolist()
                                    logger.info('ARIMA fit and forecast successful.')
                                except Exception as e:
                                    logger.warning(f"ARIMA full fit failed: {e}")
                                    forecasts['ARIMA'] = [None] * period_count
                    except concurrent.futures.TimeoutError:
                        diagnostics['ARIMA'] = f"Skipped: fitting timed out after {ARIMA_TIMEOUT} seconds."
                        logger.warning(f"ARIMA fitting timed out after {ARIMA_TIMEOUT} seconds.")
                        forecasts['ARIMA'] = [None] * period_count
                    except Exception as e:
                        diagnostics['ARIMA'] = f"Failed: {e}"
                        logger.warning(f"ARIMA failed: {e}")
                        forecasts['ARIMA'] = [None] * period_count
        except Exception as e:
            diagnostics['ARIMA'] = f"Failed: {e} (outer catch)"
            logger.warning(f"ARIMA failed (outer catch): {e}")
            forecasts['ARIMA'] = [None] * period_count
    elif not use_arima:
        diagnostics['ARIMA'] = "Skipped: not selected by user (default is off)."
        logger.info("ARIMA skipped: not selected by user (default is off).")
        forecasts['ARIMA'] = [None] * period_count
    elif skip_arima:
        diagnostics['ARIMA'] = "Skipped: global skip for performance (stricter)."
        logger.info(f"ARIMA skipped for performance (global skip, stricter).")
        forecasts['ARIMA'] = [None] * period_count
    elif pm is None:
        diagnostics['ARIMA'] = "Not available (pmdarima not installed)."
        logger.info("ARIMA not available: pmdarima not installed.")
        forecasts['ARIMA'] = [None] * period_count

    # --- Best model selection ---
    best_model = None
    best_metric = float('inf')
    best_metric_name = None
    for model in model_names:
        mape_score = scores[model]['MAPE']
        rmse_score = scores[model]['RMSE']
        metric = mape_score if mape_score is not None else rmse_score
        if metric is not None and metric < best_metric:
            best_metric = metric
            best_model = model
            best_metric_name = 'MAPE' if mape_score is not None else 'RMSE'

    # --- Forecast for best model (fit on full data) ---
    best_forecast = None
    if best_model:
        if best_model == 'Holt-Winters':
            if len(ts) >= seasonal_periods * 2:
                model = ExponentialSmoothing(ts, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
            elif len(ts) >= 4:
                model = ExponentialSmoothing(ts, trend='add', seasonal=None)
            else:
                model = None
            if model:
                fit = model.fit()
                best_forecast = fit.forecast(period_count).tolist()
        elif best_model == 'Prophet':
            from prophet import Prophet
            df_prophet = ts.reset_index()
            df_prophet.columns = ['ds', 'y']
            model = Prophet()
            model.fit(df_prophet)
            if time_unit == 'monthly':
                freq = 'M'
            elif time_unit == 'weekly':
                freq = 'W'
            else:
                freq = 'D'
            future = model.make_future_dataframe(periods=period_count, freq=freq)
            forecast_df = model.predict(future)
            best_forecast = forecast_df.tail(period_count)['yhat'].values.tolist()
        elif best_model == 'ARIMA' and pm is not None:
            ARIMA_TIMEOUT = 20
            import concurrent.futures
            def fit_arima(ts, seasonal, seasonal_periods, period_count):
                if pm is None:
                    raise RuntimeError("pmdarima is not installed")
                if seasonal:
                    model = pm.auto_arima(
                        ts,
                        seasonal=True,
                        m=seasonal_periods,
                        suppress_warnings=True,
                        start_p=0, max_p=1,
                        start_q=0, max_q=1,
                        max_d=1,
                        start_P=0, max_P=1,
                        start_Q=0, max_Q=1,
                        max_D=1,
                        stepwise=True,
                        n_jobs=1,
                        random_state=42
                    )
                else:
                    model = pm.auto_arima(
                        ts,
                        seasonal=False,
                        suppress_warnings=True,
                        start_p=0, max_p=1,
                        start_q=0, max_q=1,
                        max_d=1,
                        stepwise=True,
                        n_jobs=1,
                        random_state=42
                    )
                forecast_values = model.predict(n_periods=period_count)
                return model, forecast_values
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                if len(ts) >= seasonal_periods * 2:
                    future = executor.submit(fit_arima, ts, True, seasonal_periods, period_count)
                elif len(ts) >= 4:
                    future = executor.submit(fit_arima, ts, False, seasonal_periods, period_count)
                else:
                    future = None
                if future:
                    try:
                        model, arima_forecast = future.result(timeout=ARIMA_TIMEOUT)
                        best_forecast = arima_forecast.tolist()
                    except Exception:
                        best_forecast = None
    
    elapsed_time = time.time() - start_time
    logger.info(f"Model selection completed in {elapsed_time:.2f}s. Best model: {best_model} ({best_metric_name}={best_metric:.4f})")
    return {
        'forecasts': forecasts,
        'scores': scores,
        'best_model': best_model,
        'diagnostics': diagnostics
    }

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





# Authentication routes
@app.route('/api/auth/signin', methods=['POST'])
def signin():
    """Handle Supabase email/password sign in."""
    if not AUTH_AVAILABLE:
        return jsonify({'error': 'Authentication not available'}), 503
    
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Sign in with Supabase
        user_info = sign_in_with_supabase(email, password)
        if not user_info:
            return jsonify({'error': 'Invalid email or password'}), 401
        
        # Create user session
        session = create_user_session(user_info)
        
        logger.info(f"User signed in: {user_info['email']} (ID: {user_info['user_id']})")
        
        return jsonify({
            'success': True,
            'user': user_info,
            'token': session['token'],
            'expires_in': session['expires_in']
        })
        
    except Exception as e:
        logger.error(f"Sign in error: {e}")
        return jsonify({'error': 'Authentication failed'}), 500

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Handle Supabase email/password sign up."""
    if not AUTH_AVAILABLE:
        return jsonify({'error': 'Authentication not available'}), 503
    
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password required'}), 400
        
        # Sign up with Supabase
        user_info = sign_up_with_supabase(email, password)
        if not user_info:
            return jsonify({'error': 'Registration failed'}), 400
        
        # Create user session
        session = create_user_session(user_info)
        
        logger.info(f"User signed up: {user_info['email']} (ID: {user_info['user_id']})")
        
        return jsonify({
            'success': True,
            'user': user_info,
            'token': session['token'],
            'expires_in': session['expires_in']
        })
        
    except Exception as e:
        logger.error(f"Sign up error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/verify', methods=['POST'])
def verify_token():
    """Verify JWT token and return user info."""
    if not AUTH_AVAILABLE:
        return jsonify({'error': 'Authentication not available'}), 503
    
    try:
        data = request.get_json()
        token = data.get('token')
        
        if not token:
            return jsonify({'error': 'Token required'}), 400
        
        from auth import get_user_from_token
        user_info = get_user_from_token(token)
        
        if not user_info:
            return jsonify({'error': 'Invalid token'}), 401
        
        return jsonify({
            'success': True,
            'user': user_info
        })
        
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return jsonify({'error': 'Token verification failed'}), 500

@app.route('/api/auth/user', methods=['GET'])
@optional_auth
def get_user():
    """Get current user information."""
    if not AUTH_AVAILABLE:
        return jsonify({'error': 'Authentication not available'}), 503
    
    current_user = get_current_user()
    if current_user:
        return jsonify({
            'authenticated': True,
            'user': current_user
        })
    else:
        return jsonify({
            'authenticated': False,
            'user': None
        })

@app.route('/api/auth/google/callback')
def google_callback():
    """Handle Google OAuth callback - redirect to frontend."""
    # This route handles the OAuth callback and redirects to the frontend
    # The actual OAuth flow is handled by Supabase
    return redirect('https://raiseproject-production.up.railway.app')

@app.route('/')
def index():
    # Always return 200 for health check - Railway might be checking root path
    logger.info("Root endpoint accessed - returning 200 OK")
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        # Fallback: return a simple HTML response
        return """
        <!DOCTYPE html>
        <html>
        <head><title>AI Forecast Intelligence Platform</title></head>
        <body>
        <h1>AI Forecast Intelligence Platform</h1>
        <p>Service is running. Please check the application logs.</p>
        </body>
        </html>
        """, 200

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'AI Forecast Intelligence Platform is running'})

# Add a simple root health check
@app.route('/api/health')
def api_health():
    return jsonify({'status': 'healthy'})

# Payment and subscription endpoints
@app.route('/api/payments/products', methods=['GET'])
def get_products():
    """Get all available products."""
    if not PAYMENT_AVAILABLE:
        return jsonify({'error': 'Payment system not available'}), 503
    
    try:
        products = payment_manager.get_all_products()
        trial_config = payment_manager.get_trial_config()
        return jsonify({
            'products': products,
            'trial': trial_config
        })
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        return jsonify({'error': 'Failed to get products'}), 500

@app.route('/api/payments/create-checkout', methods=['POST'])
def create_checkout():
    """Create a Stripe checkout session."""
    if not PAYMENT_AVAILABLE:
        return jsonify({'error': 'Payment system not available'}), 503
    
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        user_email = data.get('user_email')
        
        if not product_id or not user_email:
            return jsonify({'error': 'Missing product_id or user_email'}), 400
        
        # Create checkout session
        base_url = request.headers.get('Origin', 'https://r4ise.up.railway.app')
        success_url = f"{base_url}/?payment=success&session_id={{CHECKOUT_SESSION_ID}}"
        cancel_url = f"{base_url}/?payment=cancelled"
        
        session_data = payment_manager.create_checkout_session(
            product_id=product_id,
            user_email=user_email,
            success_url=success_url,
            cancel_url=cancel_url
        )
        
        return jsonify(session_data)
        
    except Exception as e:
        logger.error(f"Error creating checkout session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/payments/verify-session', methods=['POST'])
def verify_session():
    """Verify a completed checkout session and activate subscription."""
    if not PAYMENT_AVAILABLE:
        return jsonify({'error': 'Payment system not available'}), 503
    
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Missing session_id'}), 400
        
        # Get session details from Stripe
        session = payment_manager.get_session(session_id)
        if not session:
            return jsonify({'error': 'Invalid session ID'}), 400
        
        if session['status'] != 'complete' or session['payment_status'] != 'paid':
            return jsonify({'error': 'Payment not completed'}), 400
        
        # Get product info
        product_id = session['metadata'].get('product_id')
        user_email = session['metadata'].get('user_email')
        
        if not product_id or not user_email:
            return jsonify({'error': 'Missing product or user information'}), 400
        
        product_info = payment_manager.get_product_info(product_id)
        if not product_info:
            return jsonify({'error': 'Invalid product'}), 400
        
        # Activate subscription
        subscription_data = {
            'type': product_info['type'],
            'forecasts_allowed': product_info['forecasts_allowed'],
            'session_id': session_id,
            'amount_total': session['amount_total'],
            'currency': session['currency']
        }
        
        success = subscription_manager.activate_subscription(user_email, product_id, subscription_data)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Subscription activated successfully',
                'user_access': subscription_manager.get_user_access(user_email)
            })
        else:
            return jsonify({'error': 'Failed to activate subscription'}), 500
        
    except Exception as e:
        logger.error(f"Error verifying session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscription/access', methods=['GET'])
def get_user_access():
    """Get current user's access level and forecast usage."""
    try:
        # Get user email from query parameter or auth
        user_email = request.args.get('user_email')
        
        if not user_email:
            return jsonify({'error': 'Missing user_email'}), 400
        
        # If payment system is not available, provide unlimited access for development
        if not PAYMENT_AVAILABLE:
            logger.info(f"Payment system not available - providing unlimited access for {user_email}")
            return jsonify({
                'access_type': 'development',
                'forecasts_used': 0,
                'forecasts_allowed': 999999,
                'forecasts_remaining': 999999,
                'is_active': True
            })
        
        access = subscription_manager.get_user_access(user_email)
        return jsonify(access)
        
    except Exception as e:
        logger.error(f"Error getting user access: {e}")
        return jsonify({'error': 'Failed to get user access'}), 500

@app.route('/api/subscription/activate-trial', methods=['POST'])
def activate_trial():
    """Activate trial access for a user."""
    if not PAYMENT_AVAILABLE:
        return jsonify({'error': 'Payment system not available'}), 503
    
    try:
        data = request.get_json()
        user_email = data.get('user_email')
        
        if not user_email:
            return jsonify({'error': 'Missing user_email'}), 400
        
        # Check if user already has trial or subscription
        current_access = subscription_manager.get_user_access(user_email)
        if current_access['is_active']:
            return jsonify({'error': 'User already has active access'}), 400
        
        # Activate trial
        trial_config = payment_manager.get_trial_config()
        success = subscription_manager.activate_trial(
            user_email, 
            trial_config['forecasts_allowed'], 
            trial_config['duration_days']
        )
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Trial activated successfully',
                'user_access': subscription_manager.get_user_access(user_email)
            })
        else:
            return jsonify({'error': 'Failed to activate trial'}), 500
        
    except Exception as e:
        logger.error(f"Error activating trial: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/forecast', methods=['POST'])
@optional_auth
def forecast():
    try:
        logger.info("Forecast request received")
        
        # Get user information if authenticated
        current_user = get_current_user()
        user_id = current_user['user_id'] if current_user else 'anonymous'
        user_email = current_user['email'] if current_user else 'anonymous'
        
        logger.info(f"Forecast request from user: {user_email} (ID: {user_id})")
        
        # Check user access if payment system is available and user is authenticated
        if PAYMENT_AVAILABLE and user_email and user_email != 'anonymous':
            can_generate, message = subscription_manager.can_generate_forecast(user_email)
            if not can_generate:
                logger.warning(f"Access denied for {user_email}: {message}")
                return jsonify({'error': message}), 403
        
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
        date_format = data.get('dateFormat', 'EUR')  # Default to EUR
        decimal_places = int(data.get('decimalPlaces', 0))  # Default to 0
        allow_negative = data.get('allowNegative', False)  # Default to False
        
        # Parse scenario data if provided
        scenario = data.get('scenario', None)
        if scenario:
            logger.info(f"Scenario data received: {scenario}")
            if scenario.get('type') != 'multiplier':
                logger.warning(f"Unsupported scenario type: {scenario.get('type')}")
                scenario = None
        else:
            logger.info("No scenario data provided")
        
        logger.info(f"Processing forecast: time_unit={time_unit}, periods={period_count}, format={export_format}, date_format={date_format}, decimal_places={decimal_places}, allow_negative={allow_negative}")
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
        # Ensure item_col is always string to prevent Excel/pandas misinterpretation in any preview
        df[item_col] = df[item_col].astype(str)
        
        # Process data with robust date parsing
        logger.info(f"Processing data with columns: {date_col}, {item_col}, {qty_col}")
        logger.info(f"Sample date values: {df[date_col].head(3).tolist()}")
        
        # --- Robust date parsing: try multiple strategies ---
        sample_dates = df[date_col].head(100).astype(str).tolist()
        logger.info(f"Analyzing date patterns in: {sample_dates}")
        
        # Log the first few raw date values before parsing
        logger.info(f"Raw date values (first 10): {df[date_col].head(10).tolist()}")
        
        # 1. Try pandas flexible parsing with dayfirst True/False
        dt1 = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
        dt2 = pd.to_datetime(df[date_col], dayfirst=False, errors='coerce')
        valid1 = dt1.notna().sum()
        valid2 = dt2.notna().sum()
        logger.info(f"Date parsing: dayfirst=True valid={valid1}, dayfirst=False valid={valid2}")
        logger.info(f"Sample parsed with dayfirst=True: {dt1.head(5).tolist()}")
        logger.info(f"Sample parsed with dayfirst=False: {dt2.head(5).tolist()}")
        
        # Improved logic: Check if the dates look like they're in DD/MM format
        # If the first few dates have day > 12, it's likely DD/MM format
        sample_dates_str = df[date_col].head(10).astype(str).tolist()
        dd_mm_indicators = 0
        mm_dd_indicators = 0
        
        for date_str in sample_dates_str:
            try:
                parts = date_str.split('/')
                if len(parts) == 3:
                    first_part = int(parts[0])
                    second_part = int(parts[1])
                    if first_part > 12:  # Day > 12, likely DD/MM
                        dd_mm_indicators += 1
                    elif second_part > 12:  # Month > 12, likely MM/DD
                        mm_dd_indicators += 1
            except:
                pass
        
        logger.info(f"Format indicators: DD/MM indicators={dd_mm_indicators}, MM/DD indicators={mm_dd_indicators}")
        
        # Choose the better parsing based on indicators
        if dd_mm_indicators > mm_dd_indicators:
            df[date_col] = dt1
            logger.info("Date parsing succeeded with dayfirst=True (DD/MM format detected)")
        else:
            df[date_col] = dt2
            logger.info("Date parsing succeeded with dayfirst=False (MM/DD format detected)")
        
        if valid1 == 0 and valid2 == 0:
            # 2. Try common custom formats
            custom_formats = ['%d/%m/%Y', '%Y-%m', '%m/%Y', '%Y/%m', '%m-%Y', '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%m/%d/%Y', '%d/%m/%y', '%m/%d/%y', '%y/%d/%m', '%y/%m/%d', '%y-%d-%m', '%y-%m-%d']
            for fmt in custom_formats:
                try:
                    dt_custom = pd.to_datetime(df[date_col], format=fmt, errors='coerce')
                    valid_custom = dt_custom.notna().sum()
                    logger.info(f"Date parsing: format={fmt} valid={valid_custom}")
                    if valid_custom > 0:
                        df[date_col] = dt_custom
                        logger.info(f"Date parsing succeeded with format {fmt}")
                        break
                except Exception as e:
                    logger.warning(f"Date parsing failed for format {fmt}: {e}")
            else:
                logger.error("Could not parse date column with any known format.")
                return jsonify({'error': 'Could not parse date column: please use a standard date format like YYYY-MM-DD, DD/MM/YYYY, or MM/DD/YYYY.'}), 400

        
        df = df.dropna(subset=[date_col, qty_col, item_col])
        logger.info(f"Valid rows after date parsing: {len(df)}")
        logger.info(f"Sample parsed dates: {df[date_col].head(3).tolist()}")
        logger.info(f"Date range after parsing: {df[date_col].min()} to {df[date_col].max()}")
        logger.info(f"Unique months in data: {df[date_col].dt.to_period('M').nunique()}")
        
        # CRITICAL DEBUG: Log the actual data structure
        logger.info("=== CRITICAL DEBUG: DATA STRUCTURE ANALYSIS ===")
        logger.info(f"Total rows in dataframe: {len(df)}")
        logger.info(f"Unique items: {df[item_col].nunique()}")
        logger.info(f"Unique dates: {df[date_col].nunique()}")
        logger.info(f"Sample of actual data:")
        for i, row in df.head(10).iterrows():
            logger.info(f"  Row {i}: {row[item_col]} | {row[date_col]} | {row[qty_col]}")
        logger.info("=== END CRITICAL DEBUG ===")
        
        try:
            df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
            # Always round actuals to 2 decimals for preview
            df[qty_col] = df[qty_col].round(2)
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
        logger.info(f"Creating periods for time_unit: {time_unit}")
        logger.info(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        
        if time_unit == 'monthly':
            # Convert to monthly periods - this works for both MM/YYYY and DD/MM/YYYY
            df['period'] = df[date_col].dt.to_period('M').dt.to_timestamp()
            logger.info(f"Monthly periods created. Unique periods: {df['period'].nunique()}")
            logger.info(f"Sample periods: {df['period'].head(5).tolist()}")
        elif time_unit == 'weekly':
            df['period'] = df[date_col] - pd.to_timedelta(df[date_col].dt.dayofweek, unit='D')
        else:  # daily
            df['period'] = df[date_col]

        forecasts = []
        diagnostics_log = []

        unique_items = df[item_col].unique()
        logger.info(f"Starting forecast generation for {len(unique_items)} unique items")

        # --- Skip ARIMA logic ---
        skip_arima = False
        if len(df) > 500 or len(unique_items) > 5:
            skip_arima = True
            logger.info(f"Skipping ARIMA for performance: {len(unique_items)} items, {len(df)} rows (stricter)")
        # --- End skip ARIMA logic ---

        # Read model selection flags from request (default False for ARIMA)
        use_arima = data.get('useARIMA', False)  # ARIMA now opt-in
        use_hw = data.get('useHW', True)
        use_prophet = data.get('useProphet', True)

        # Determine the global date range (all periods across all items)
        all_periods = pd.Series(df['period'].unique()).sort_values().to_list()
        global_min_period = df['period'].min()
        global_max_period = df['period'].max()
        logger.info(f"Global period range: {global_min_period} to {global_max_period}, total periods: {len(all_periods)}")

        # Prepare output rows
        output_rows = []
        model_names = ['ARIMA', 'Holt-Winters', 'Prophet']
        # For Prophet aggregation
        prophet_aggregate = None
        prophet_periods = None
        prophet_dates = None
        for i, item_id in enumerate(unique_items):
            logger.info(f"Processing item {i+1}/{len(unique_items)}: {item_id}")
            group = df[df[item_col] == item_id]
            ts = group.groupby('period')[qty_col].sum().sort_index()
            ts = ts.reindex(all_periods, fill_value=0)
            if not isinstance(ts, pd.Series):
                continue
            val_result = validate_time_series_data(ts, time_unit)
            if not val_result['sufficient']:
                diagnostics_log.append(f"Item {item_id}: {val_result['warnings']}")
                continue
            # Model selection with diagnostics
            # --- General review: add detailed logging ---
            logger.info(f"Item {item_id}: Full time series: {ts.index.tolist()} -> {ts.values.tolist()}")
            n = len(ts)
            split_idx = int(n * 0.8)
            logger.info(f"Item {item_id}: Train indices: 0 to {split_idx-1}, Test indices: {split_idx} to {n-1}")
            logger.info(f"Item {item_id}: Train values: {ts.iloc[:split_idx].values.tolist()}")
            logger.info(f"Item {item_id}: Test values: {ts.iloc[split_idx:].values.tolist()}")
            model_results = create_forecast_model_with_diagnostics(
                ts,
                time_unit,
                period_count,
                skip_arima=skip_arima,
                use_arima=use_arima,
                use_hw=use_hw,
                use_prophet=use_prophet
            )
            forecasts_dict = model_results['forecasts']
            best_model_name = model_results['best_model']
            # Log model diagnostics and test predictions
            for model in ['ARIMA', 'Holt-Winters', 'Prophet']:
                diag = model_results['diagnostics'].get(model, '')
                logger.info(f"Item {item_id}: {model} diagnostics: {diag}")
                if model in model_results['scores']:
                    logger.info(f"Item {item_id}: {model} MAPE: {model_results['scores'][model]['MAPE']}, RMSE: {model_results['scores'][model]['RMSE']}")
            # Generate future dates
            last_date = all_periods[-1]
            try:
                if hasattr(last_date, 'item'):
                    last_date = last_date.item()
                if last_date is None or (hasattr(last_date, '__bool__') and not bool(last_date)):
                    continue
                # Check for NaT (Not a Time)
                if pd.isna(last_date):
                    continue
                last_date = pd.Timestamp(last_date)
            except (ValueError, TypeError, AttributeError):
                continue
            future_dates = []
            for j in range(1, period_count + 1):
                try:
                    # Only perform date arithmetic if last_date is a valid Timestamp
                    if last_date is None or bool(pd.isna(last_date)) or type(last_date).__name__ == 'NaTType' or not isinstance(last_date, pd.Timestamp):
                        continue
                    if time_unit == 'monthly':
                        next_date = last_date + pd.DateOffset(months=j)
                        if next_date is None or bool(pd.isna(next_date)) or type(next_date).__name__ == 'NaTType':
                            continue
                        if date_format == 'EUR':
                            if isinstance(next_date, pd.Timestamp):
                                future_dates.append(next_date.strftime('01/%m/%Y'))
                        else:
                            if isinstance(next_date, pd.Timestamp):
                                future_dates.append(next_date.strftime('%m/01/%Y'))
                    elif time_unit == 'weekly':
                        next_date = last_date + timedelta(weeks=j)
                        if next_date is None or bool(pd.isna(next_date)) or type(next_date).__name__ == 'NaTType':
                            continue
                        if date_format == 'EUR':
                            if isinstance(next_date, pd.Timestamp):
                                future_dates.append(next_date.strftime('%d/%m/%Y'))
                        else:
                            if isinstance(next_date, pd.Timestamp):
                                future_dates.append(next_date.strftime('%m/%d/%Y'))
                    else:
                        next_date = last_date + timedelta(days=j)
                        if next_date is None or bool(pd.isna(next_date)) or type(next_date).__name__ == 'NaTType':
                            continue
                        if date_format == 'EUR':
                            if isinstance(next_date, pd.Timestamp):
                                future_dates.append(next_date.strftime('%d/%m/%Y'))
                        else:
                            if isinstance(next_date, pd.Timestamp):
                                future_dates.append(next_date.strftime('%m/%d/%Y'))
                except (TypeError, AttributeError, ValueError):
                    continue
            # --- Prophet aggregation logic (moved after future_dates is defined) ---
            if use_prophet and 'Prophet' in forecasts_dict and forecasts_dict['Prophet']:
                if prophet_aggregate is None:
                    prophet_aggregate = [0.0] * len(forecasts_dict['Prophet'])
                    prophet_periods = len(forecasts_dict['Prophet'])
                    prophet_dates = future_dates.copy()
                for idx, val in enumerate(forecasts_dict['Prophet']):
                    if val is not None:
                        prophet_aggregate[idx] += float(val)
            # For each period, build a row with all model forecasts and best model
            for idx, date_str in enumerate(future_dates):
                row = {
                    'Date': date_str,
                    'Item ID': item_id,
                    'Forecast (ARIMA)': '',
                    'Forecast (Holt-Winters)': '',
                    'Forecast (Prophet)': '',
                    'Average': '',
                    'Best Model': best_model_name
                }
                for model in model_names:
                    # Only fill forecast if the model was selected by the user
                    model_selected = (
                        (model == 'ARIMA' and use_arima) or
                        (model == 'Holt-Winters' and use_hw) or
                        (model == 'Prophet' and use_prophet)
                    )
                    forecast_list = forecasts_dict.get(model)
                    if model_selected and forecast_list and idx < len(forecast_list):
                        val_raw = forecast_list[idx]
                        if val_raw is not None:
                            val = round(float(val_raw), decimal_places)
                            if not allow_negative and val < 0:
                                val = 0
                            row[f'Forecast ({model})'] = val
                        else:
                            row[f'Forecast ({model})'] = ''
                
                # Calculate average of available forecasts
                forecast_values = []
                for model in model_names:
                    v = row[f'Forecast ({model})']
                    if isinstance(v, (int, float)) and v != '':
                        forecast_values.append(v)
                if forecast_values:
                    row['Average'] = round(sum(forecast_values) / len(forecast_values), decimal_places)
                else:
                    row['Average'] = ''
                output_rows.append(row)
        
        # Apply scenario adjustments if provided
        if scenario and scenario.get('type') == 'multiplier':
            logger.info("Applying scenario adjustments to forecasts")
            target_model = scenario.get('target', 'Prophet')
            adjustments = scenario.get('adjustments', [])
            
            # Convert date strings to datetime for comparison
            for adjustment in adjustments:
                try:
                    adjustment['start_dt'] = pd.to_datetime(adjustment['start'])
                    adjustment['end_dt'] = pd.to_datetime(adjustment['end'])
                except Exception as e:
                    logger.warning(f"Invalid date in adjustment: {e}")
                    continue
            
            # Apply adjustments to each row
            for row in output_rows:
                row_date = pd.to_datetime(row['Date'])
                target_col = f'Forecast ({target_model})'
                
                # Check if this date falls within any adjustment period
                for adjustment in adjustments:
                    if 'start_dt' in adjustment and 'end_dt' in adjustment:
                        if adjustment['start_dt'] <= row_date <= adjustment['end_dt']:
                            if target_col in row and row[target_col] != '':
                                original_val = float(row[target_col])
                                adjusted_val = original_val * adjustment['factor']
                                row[target_col] = round(adjusted_val, decimal_places)
                                logger.info(f"Applied {adjustment['factor']}x adjustment to {row_date} for {target_model}")
                            break
            
            # Recalculate average after adjustments
            for row in output_rows:
                forecast_values = []
                for model in model_names:
                    v = row[f'Forecast ({model})']
                    if isinstance(v, (int, float)) and v != '':
                        forecast_values.append(v)
                if forecast_values:
                    row['Average'] = round(sum(forecast_values) / len(forecast_values), decimal_places)
                else:
                    row['Average'] = ''
            
            logger.info(f"Scenario adjustments applied to {target_model} forecasts")
        
        if not output_rows:
            logger.error(f"No forecasts generated. Diagnostics: {diagnostics_log}")
            return jsonify({'error': 'No forecasts could be generated.\n' + '\n'.join(diagnostics_log)}), 400
        logger.info(f"Generated {len(output_rows)} forecast entries")
        # Record forecast generation for authenticated users
        if PAYMENT_AVAILABLE and user_email and user_email != 'anonymous':
            subscription_manager.record_forecast_generation(user_email)
            logger.info(f"Recorded forecast generation for user: {user_email}")
        # Create result dataframe
        result_df = pd.DataFrame(output_rows, columns=["Date", "Item ID", "Forecast (ARIMA)", "Forecast (Holt-Winters)", "Forecast (Prophet)", "Average", "Best Model"])
        result_df["Item ID"] = result_df["Item ID"].astype(str)
        # Return single file based on export format
        if export_format == 'xlsx':
            try:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                    result_df.to_excel(tmp_file.name, index=False, float_format=f"%.{decimal_places}f", engine='openpyxl')
                    with open(tmp_file.name, 'rb') as f:
                        excel_data = f.read()
                    os.unlink(tmp_file.name)
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
                output = io.StringIO()
                result_df.to_csv(output, index=False, float_format=f"%.{decimal_places}f", quoting=csv.QUOTE_MINIMAL)
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
        else:
            output = io.StringIO()
            result_df.to_csv(output, index=False, float_format=f"%.{decimal_places}f", quoting=csv.QUOTE_MINIMAL)
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
    # Only run Flask development server if explicitly requested
    if os.environ.get('FLASK_DEBUG', 'False').lower() == 'true':
        logger.info("Starting Flask development server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.info("Flask app configured for production (gunicorn)")
