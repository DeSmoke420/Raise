import os
import stripe
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', 'sk_test_51Rk4JLGfh0Kc8OxaVMGTlJOq9gmSEepODyfsD2VmUuHg7SBrjt74vH9Hm0QTooFXbakv2AP0PhjP6AM6GaeOyoXf00xPPNzkmD')

# Product configurations
PRODUCTS = {
    'single_forecast': {
        'name': 'Single Forecast',
        'price': 199,  # 1.99 EUR in cents
        'currency': 'eur',
        'description': 'Download 1 AI-generated forecast',
        'forecasts_allowed': 1,
        'type': 'one_time'
    },
    'monthly_subscription': {
        'name': 'Monthly Subscription',
        'price': 999,  # 9.99 EUR in cents
        'currency': 'eur',
        'description': 'Full access with up to 100 forecasts per month',
        'forecasts_allowed': 100,
        'type': 'subscription',
        'interval': 'month'
    },
    'lifetime_access': {
        'name': 'Lifetime Access',
        'price': 9999,  # 99.99 EUR in cents
        'currency': 'eur',
        'description': 'Lifetime access with up to 500 forecasts per month',
        'forecasts_allowed': 500,
        'type': 'one_time'
    }
}

# Trial configuration
TRIAL_CONFIG = {
    'forecasts_allowed': 10,
    'duration_days': 30
}

class PaymentManager:
    def __init__(self):
        self.stripe = stripe
        
    def create_checkout_session(self, product_id: str, user_email: str, success_url: str, cancel_url: str) -> Dict[str, Any]:
        """Create a Stripe checkout session for the specified product."""
        try:
            if product_id not in PRODUCTS:
                raise ValueError(f"Invalid product ID: {product_id}")
                
            product = PRODUCTS[product_id]
            
            session_data = {
                'payment_method_types': ['card'],
                'customer_email': user_email,
                'success_url': success_url,
                'cancel_url': cancel_url,
                'metadata': {
                    'product_id': product_id,
                    'user_email': user_email
                }
            }
            
            if product['type'] == 'subscription':
                # Create subscription checkout
                session_data['mode'] = 'subscription'
                session_data['line_items'] = [{
                    'price_data': {
                        'currency': product['currency'],
                        'product_data': {
                            'name': product['name'],
                            'description': product['description']
                        },
                        'unit_amount': product['price'],
                        'recurring': {
                            'interval': product['interval']
                        }
                    },
                    'quantity': 1
                }]
            else:
                # Create one-time payment checkout
                session_data['mode'] = 'payment'
                session_data['line_items'] = [{
                    'price_data': {
                        'currency': product['currency'],
                        'product_data': {
                            'name': product['name'],
                            'description': product['description']
                        },
                        'unit_amount': product['price']
                    },
                    'quantity': 1
                }]
            
            session = self.stripe.checkout.Session.create(**session_data)
            logger.info(f"Created checkout session {session.id} for product {product_id}")
            return {
                'session_id': session.id,
                'url': session.url,
                'product_id': product_id
            }
            
        except Exception as e:
            logger.error(f"Error creating checkout session: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a checkout session by ID."""
        try:
            session = self.stripe.checkout.Session.retrieve(session_id)
            return {
                'id': session.id,
                'status': session.status,
                'payment_status': session.payment_status,
                'customer_email': session.customer_email,
                'metadata': session.metadata,
                'amount_total': session.amount_total,
                'currency': session.currency
            }
        except Exception as e:
            logger.error(f"Error retrieving session {session_id}: {e}")
            return None
    
    def get_product_info(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get product information by ID."""
        return PRODUCTS.get(product_id)
    
    def get_all_products(self) -> Dict[str, Any]:
        """Get all available products."""
        return PRODUCTS
    
    def get_trial_config(self) -> Dict[str, Any]:
        """Get trial configuration."""
        return TRIAL_CONFIG

# Global payment manager instance
payment_manager = PaymentManager() 