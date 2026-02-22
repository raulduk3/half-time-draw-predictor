"""
Inference script for half-time draw prediction.

This script loads trained models and makes predictions for upcoming matches.
"""

import pandas as pd
import numpy as np
import pickle
import torch
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering functions
from features import compute_rolling_form, add_rest_days, transform_odds, create_target
from utils import normalize_columns

class HalfTimeDrawPredictor:
    """
    Predictor class that loads trained models and makes predictions.
    """

    def __init__(self, model_dir='models'):
        """Initialize the predictor by loading models and metadata."""
        self.model_dir = Path(model_dir)
        self.models_loaded = False

        # Load models and metadata
        self._load_models()
        self._load_dataset()

    def _load_models(self):
        """Load trained models, scaler, and metadata."""
        print("🔄 Loading trained models...")

        try:
            # Load logistic regression model
            with open(self.model_dir / 'logistic_regression_model.pkl', 'rb') as f:
                self.lr_model = pickle.load(f)

            # Load feature scaler
            with open(self.model_dir / 'scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)

            # Load metrics and model info
            with open(self.model_dir / 'metrics.json', 'r') as f:
                self.metrics = json.load(f)

            # Load LSTM model
            from train import HalfTimeDrawLSTM  # Import the model class

            model_info = self.metrics['model_info']['lstm_architecture']
            self.lstm_model = HalfTimeDrawLSTM(
                input_size=model_info['input_size'],
                hidden_size=model_info['hidden_size'],
                num_layers=model_info['num_layers'],
                dropout=model_info['dropout']
            )

            # Load LSTM weights
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.lstm_model.load_state_dict(torch.load(self.model_dir / 'lstm_model.pth', map_location=device))
            self.lstm_model.eval()
            self.device = device

            self.feature_cols = self.metrics['feature_columns']
            self.models_loaded = True

            print("✅ Models loaded successfully!")
            print(f"  📊 Logistic Regression Test AUC: {self.metrics['logistic_regression']['test_auc']:.4f}")
            print(f"  🧠 LSTM Test AUC: {self.metrics['lstm']['test_auc']:.4f}")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

    def _load_dataset(self):
        """Load the processed dataset for feature engineering."""
        try:
            self.dataset = pd.read_parquet('data/processed/dataset.parquet')
            print(f"✅ Dataset loaded: {len(self.dataset):,} matches")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise

    def _get_team_form(self, team_name, reference_date, num_matches=5):
        """Get rolling form statistics for a team before a given date."""
        # Get historical matches for the team before reference date
        team_matches = self.dataset[
            (self.dataset['Date'] < reference_date) &
            ((self.dataset['HomeTeam'] == team_name) | (self.dataset['AwayTeam'] == team_name))
        ].tail(num_matches)

        if len(team_matches) == 0:
            return {
                'gf_r5': 0.0,
                'ga_r5': 0.0,
                'gd_r5': 0.0,
                'days_since_last': 14  # Default rest days
            }

        # Calculate goals for and against from team's perspective
        goals_for = []
        goals_against = []

        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                # Team played at home
                goals_for.append(match['HTHG'])
                goals_against.append(match['HTAG'])
            else:
                # Team played away
                goals_for.append(match['HTAG'])
                goals_against.append(match['HTHG'])

        # Calculate rest days since last match
        last_match_date = team_matches['Date'].max()
        days_since_last = (reference_date - last_match_date).days if pd.notna(last_match_date) else 14

        return {
            'gf_r5': np.mean(goals_for),
            'ga_r5': np.mean(goals_against),
            'gd_r5': np.mean(goals_for) - np.mean(goals_against),
            'days_since_last': days_since_last
        }

    def _get_betting_odds(self, home_team, away_team, reference_date):
        """Get typical betting odds for similar matches (simplified approach)."""
        # In a real implementation, you would fetch live odds from a betting API
        # For now, we'll estimate based on historical data

        # Get recent matches between these teams or similar form teams
        similar_matches = self.dataset[
            (self.dataset['Date'] < reference_date) &
            (
                ((self.dataset['HomeTeam'] == home_team) & (self.dataset['AwayTeam'] == away_team)) |
                ((self.dataset['HomeTeam'] == away_team) & (self.dataset['AwayTeam'] == home_team))
            )
        ].tail(5)

        if len(similar_matches) > 0 and 'log_home_win_odds' in similar_matches.columns:
            return {
                'log_home_win_odds': similar_matches['log_home_win_odds'].median(),
                'log_draw_odds': similar_matches['log_draw_odds'].median(),
                'log_away_win_odds': similar_matches['log_away_win_odds'].median()
            }
        else:
            # Default odds (roughly evenly matched teams)
            return {
                'log_home_win_odds': np.log(2.5),  # Home odds ~2.5
                'log_draw_odds': np.log(3.2),      # Draw odds ~3.2
                'log_away_win_odds': np.log(3.0)   # Away odds ~3.0
            }

    def predict_match(self, home_team, away_team, match_date=None, model='both'):
        """
        Predict half-time draw probability for a specific match.

        Args:
            home_team: Name of home team
            away_team: Name of away team
            match_date: Date of match (datetime or string). If None, uses current date.
            model: Which model to use ('lr', 'lstm', or 'both')

        Returns:
            Dictionary with predictions and confidence scores
        """
        if not self.models_loaded:
            raise RuntimeError("Models not loaded. Initialize predictor first.")

        # Handle match date
        if match_date is None:
            match_date = datetime.now()
        elif isinstance(match_date, str):
            match_date = pd.to_datetime(match_date)

        print(f"\n🔮 Predicting Half-Time Draw Probability")
        print(f"  🏠 Home: {home_team}")
        print(f"  ✈️  Away: {away_team}")
        print(f"  📅 Date: {match_date.strftime('%Y-%m-%d')}")

        # Get team form statistics
        home_form = self._get_team_form(home_team, match_date)
        away_form = self._get_team_form(away_team, match_date)

        # Get betting odds estimate
        odds = self._get_betting_odds(home_team, away_team, match_date)

        # Create feature vector
        features = {
            'home_gf_r5': home_form['gf_r5'],
            'home_ga_r5': home_form['ga_r5'],
            'home_gd_r5': home_form['gd_r5'],
            'away_gf_r5': away_form['gf_r5'],
            'away_ga_r5': away_form['ga_r5'],
            'away_gd_r5': away_form['gd_r5'],
            'log_home_win_odds': odds['log_home_win_odds'],
            'log_draw_odds': odds['log_draw_odds'],
            'log_away_win_odds': odds['log_away_win_odds'],
            'home_days_since_last': home_form['days_since_last'],
            'away_days_since_last': away_form['days_since_last'],
            'month': match_date.month
        }

        # Convert to numpy array in correct order
        X = np.array([[features[col] for col in self.feature_cols]])

        # Scale features
        X_scaled = self.scaler.transform(X)

        results = {}

        # Logistic Regression prediction
        if model in ['lr', 'both']:
            lr_probs = self.lr_model.predict_proba(X_scaled)
            lr_prob = lr_probs[0, 1] if lr_probs.ndim == 2 else lr_probs[1]
            results['logistic_regression'] = {
                'probability': float(lr_prob),
                'confidence': 'High' if abs(lr_prob - 0.5) > 0.2 else 'Medium' if abs(lr_prob - 0.5) > 0.1 else 'Low'
            }

        # LSTM prediction
        if model in ['lstm', 'both']:
            X_lstm = X_scaled.reshape(1, 1, -1)  # (batch_size, seq_len, features)
            X_tensor = torch.FloatTensor(X_lstm).to(self.device)

            with torch.no_grad():
                lstm_output = self.lstm_model(X_tensor).cpu().numpy()
                lstm_prob = lstm_output.item() if lstm_output.shape == () else lstm_output[0]

            results['lstm'] = {
                'probability': float(lstm_prob),
                'confidence': 'High' if abs(lstm_prob - 0.5) > 0.2 else 'Medium' if abs(lstm_prob - 0.5) > 0.1 else 'Low'
            }

        # Add ensemble prediction if both models used
        if model == 'both':
            ensemble_prob = (results['logistic_regression']['probability'] + results['lstm']['probability']) / 2
            results['ensemble'] = {
                'probability': float(ensemble_prob),
                'confidence': 'High' if abs(ensemble_prob - 0.5) > 0.2 else 'Medium' if abs(ensemble_prob - 0.5) > 0.1 else 'Low'
            }

        # Add context information
        results['context'] = {
            'home_form': home_form,
            'away_form': away_form,
            'estimated_odds': {
                'home_win': float(np.exp(odds['log_home_win_odds'])),
                'draw': float(np.exp(odds['log_draw_odds'])),
                'away_win': float(np.exp(odds['log_away_win_odds']))
            }
        }

        # Print results
        print(f"\n📊 PREDICTIONS:")
        if 'logistic_regression' in results:
            lr_prob = results['logistic_regression']['probability']
            print(f"  📈 Logistic Regression: {lr_prob:.1%} ({results['logistic_regression']['confidence']} confidence)")

        if 'lstm' in results:
            lstm_prob = results['lstm']['probability']
            print(f"  🧠 LSTM: {lstm_prob:.1%} ({results['lstm']['confidence']} confidence)")

        if 'ensemble' in results:
            ens_prob = results['ensemble']['probability']
            print(f"  🎯 Ensemble: {ens_prob:.1%} ({results['ensemble']['confidence']} confidence)")

        print(f"\n📋 Context:")
        print(f"  🏠 Home form (last 5): {home_form['gf_r5']:.1f} GF, {home_form['ga_r5']:.1f} GA, {home_form['gd_r5']:.1f} GD")
        print(f"  ✈️  Away form (last 5): {away_form['gf_r5']:.1f} GF, {away_form['ga_r5']:.1f} GA, {away_form['gd_r5']:.1f} GD")

        return results

    def predict_fixtures(self, fixtures_df):
        """
        Predict multiple fixtures from a DataFrame.

        Args:
            fixtures_df: DataFrame with columns ['HomeTeam', 'AwayTeam', 'Date']

        Returns:
            DataFrame with predictions added
        """
        predictions = []

        for _, fixture in fixtures_df.iterrows():
            try:
                result = self.predict_match(
                    fixture['HomeTeam'],
                    fixture['AwayTeam'],
                    fixture['Date'],
                    model='ensemble'
                )

                pred_data = {
                    'HomeTeam': fixture['HomeTeam'],
                    'AwayTeam': fixture['AwayTeam'],
                    'Date': fixture['Date'],
                    'lr_draw_prob': result.get('logistic_regression', {}).get('probability', np.nan),
                    'lstm_draw_prob': result.get('lstm', {}).get('probability', np.nan),
                    'ensemble_draw_prob': result.get('ensemble', {}).get('probability', np.nan),
                    'confidence': result.get('ensemble', {}).get('confidence', 'Unknown')
                }

                predictions.append(pred_data)

            except Exception as e:
                print(f"⚠️  Error predicting {fixture['HomeTeam']} vs {fixture['AwayTeam']}: {e}")

        return pd.DataFrame(predictions)

def main():
    """Demo script showing how to use the predictor."""
    print("🚀 Half-Time Draw Predictor Demo")
    print("=" * 50)

    try:
        # Initialize predictor
        predictor = HalfTimeDrawPredictor()

        # Example single match prediction
        print("\n🎯 Single Match Prediction Example:")
        result = predictor.predict_match(
            home_team="Leicester",
            away_team="Leeds",
            match_date="2025-02-23"
        )

        # Example fixtures prediction
        print("\n📅 Multiple Fixtures Example:")
        fixtures = pd.DataFrame([
            {'HomeTeam': 'Birmingham', 'AwayTeam': 'Norwich', 'Date': '2025-02-23'},
            {'HomeTeam': 'Burnley', 'AwayTeam': 'Sheffield Utd', 'Date': '2025-02-24'},
            {'HomeTeam': 'Cardiff', 'AwayTeam': 'Millwall', 'Date': '2025-02-25'}
        ])

        fixtures['Date'] = pd.to_datetime(fixtures['Date'])
        predictions_df = predictor.predict_fixtures(fixtures)

        print("\n📊 Fixtures Predictions:")
        print(predictions_df[['HomeTeam', 'AwayTeam', 'ensemble_draw_prob', 'confidence']])

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have run the training script first to generate the models.")

if __name__ == "__main__":
    main()