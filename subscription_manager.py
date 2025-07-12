import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class SubscriptionManager:
    def __init__(self):
        # In-memory storage for demo purposes
        # In production, this should be a database
        self.user_subscriptions = {}
        self.user_forecast_counts = {}
        self.user_trials = {}
    
    def get_user_access(self, user_email: str) -> Dict[str, Any]:
        """Get user's current access level and forecast usage."""
        try:
            # Check if user has an active subscription
            subscription = self.user_subscriptions.get(user_email)
            trial = self.user_trials.get(user_email)
            forecast_count = self.user_forecast_counts.get(user_email, 0)
            
            # Check trial access first
            if trial and self._is_trial_active(trial):
                return {
                    'access_type': 'trial',
                    'forecasts_used': forecast_count,
                    'forecasts_allowed': trial['forecasts_allowed'],
                    'forecasts_remaining': max(0, trial['forecasts_allowed'] - forecast_count),
                    'expires_at': trial['expires_at'],
                    'is_active': True
                }
            
            # Check paid subscription
            if subscription and self._is_subscription_active(subscription):
                return {
                    'access_type': subscription['type'],
                    'forecasts_used': forecast_count,
                    'forecasts_allowed': subscription['forecasts_allowed'],
                    'forecasts_remaining': max(0, subscription['forecasts_allowed'] - forecast_count),
                    'expires_at': subscription.get('expires_at'),
                    'is_active': True
                }
            
            # No active access
            return {
                'access_type': 'none',
                'forecasts_used': 0,
                'forecasts_allowed': 0,
                'forecasts_remaining': 0,
                'is_active': False
            }
            
        except Exception as e:
            logger.error(f"Error getting user access for {user_email}: {e}")
            return {
                'access_type': 'none',
                'forecasts_used': 0,
                'forecasts_allowed': 0,
                'forecasts_remaining': 0,
                'is_active': False
            }
    
    def can_generate_forecast(self, user_email: str) -> Tuple[bool, str]:
        """Check if user can generate a forecast."""
        access = self.get_user_access(user_email)
        
        if not access['is_active']:
            return False, "No active subscription. Please purchase a plan to generate forecasts."
        
        if access['forecasts_remaining'] <= 0:
            return False, f"No forecasts remaining. You've used {access['forecasts_used']} out of {access['forecasts_allowed']} allowed forecasts."
        
        return True, "Access granted"
    
    def record_forecast_generation(self, user_email: str) -> bool:
        """Record that a user has generated a forecast."""
        try:
            current_count = self.user_forecast_counts.get(user_email, 0)
            self.user_forecast_counts[user_email] = current_count + 1
            logger.info(f"Recorded forecast generation for {user_email}. Total: {current_count + 1}")
            return True
        except Exception as e:
            logger.error(f"Error recording forecast generation for {user_email}: {e}")
            return False
    
    def activate_trial(self, user_email: str, forecasts_allowed: int = 10, duration_days: int = 30) -> bool:
        """Activate trial access for a user."""
        try:
            expires_at = datetime.now() + timedelta(days=duration_days)
            self.user_trials[user_email] = {
                'forecasts_allowed': forecasts_allowed,
                'expires_at': expires_at.isoformat(),
                'activated_at': datetime.now().isoformat()
            }
            logger.info(f"Activated trial for {user_email} with {forecasts_allowed} forecasts")
            return True
        except Exception as e:
            logger.error(f"Error activating trial for {user_email}: {e}")
            return False
    
    def activate_subscription(self, user_email: str, product_id: str, subscription_data: Dict[str, Any]) -> bool:
        """Activate a paid subscription for a user."""
        try:
            # Clear any existing trial
            if user_email in self.user_trials:
                del self.user_trials[user_email]
            
            # Set up subscription
            self.user_subscriptions[user_email] = {
                'product_id': product_id,
                'type': subscription_data.get('type', 'one_time'),
                'forecasts_allowed': subscription_data.get('forecasts_allowed', 1),
                'activated_at': datetime.now().isoformat(),
                'stripe_session_id': subscription_data.get('session_id'),
                'amount_paid': subscription_data.get('amount_total'),
                'currency': subscription_data.get('currency', 'eur')
            }
            
            # Set expiration for one-time purchases (lifetime access)
            if subscription_data.get('type') == 'one_time':
                # Lifetime access - no expiration
                pass
            elif subscription_data.get('type') == 'subscription':
                # Monthly subscription - expires in 30 days
                expires_at = datetime.now() + timedelta(days=30)
                self.user_subscriptions[user_email]['expires_at'] = expires_at.isoformat()
            
            logger.info(f"Activated subscription for {user_email}: {product_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error activating subscription for {user_email}: {e}")
            return False
    
    def _is_trial_active(self, trial: Dict[str, Any]) -> bool:
        """Check if trial is still active."""
        try:
            expires_at = datetime.fromisoformat(trial['expires_at'])
            return datetime.now() < expires_at
        except Exception as e:
            logger.error(f"Error checking trial status: {e}")
            return False
    
    def _is_subscription_active(self, subscription: Dict[str, Any]) -> bool:
        """Check if subscription is still active."""
        try:
            # Lifetime access is always active
            if subscription.get('type') == 'one_time':
                return True
            
            # Check expiration for subscriptions
            if 'expires_at' in subscription:
                expires_at = datetime.fromisoformat(subscription['expires_at'])
                return datetime.now() < expires_at
            
            return True
        except Exception as e:
            logger.error(f"Error checking subscription status: {e}")
            return False
    
    def get_user_stats(self, user_email: str) -> Dict[str, Any]:
        """Get comprehensive user statistics."""
        access = self.get_user_access(user_email)
        subscription = self.user_subscriptions.get(user_email)
        trial = self.user_trials.get(user_email)
        
        return {
            'current_access': access,
            'subscription_details': subscription,
            'trial_details': trial,
            'total_forecasts_generated': self.user_forecast_counts.get(user_email, 0)
        }

# Global subscription manager instance
subscription_manager = SubscriptionManager() 