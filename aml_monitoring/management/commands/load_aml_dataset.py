from django.core.management.base import BaseCommand
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class Command(BaseCommand):
    help = 'Load SAML-D AML dataset and train Isolation Forest'

    def handle(self, *args, **options):
        csv_path = 'aml_monitoring/data/transactions.csv'

        if not os.path.exists(csv_path):
            self.stdout.write(f"❌ CSV not found at {csv_path}")
            self.stdout.write(f"   Please place your CSV file at: {csv_path}")
            return

        self.stdout.write(f"📂 Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        self.stdout.write(f"✓ Loaded {len(df):,} transactions")

        # ===== BUILD 18-DIMENSIONAL FEATURE MATRIX =====
        self.stdout.write("🔧 Building features (18 dimensions)...")

        features = []
        location_map = self._build_location_map()

        for idx, row in df.iterrows():
            if idx % 100000 == 0 and idx > 0:
                self.stdout.write(f"  Processing {idx:,}...")

            try:
                amount = float(row.get('Amount', 0))
                payment_type = str(row.get('Payment_type', 'other')).lower()
                sender_loc = str(row.get('Sender_bank_location', '')).strip()
                receiver_loc = str(row.get('Receiver_bank_location', '')).strip()
                payment_curr = str(row.get('Payment_currency', '')).upper()
                received_curr = str(row.get('Received_currency', '')).upper()

                # Parse timestamp
                try:
                    ts = pd.to_datetime(row['Time'])
                    hour = ts.hour
                    dow = ts.weekday()
                except:
                    hour = 12
                    dow = 3

                # 18 features
                feature_vector = [
                    amount,                                       # f1: amount
                    np.log1p(amount),                            # f2: log_amount
                    hour,                                         # f3: hour
                    dow,                                          # f4: day_of_week
                    1.0 if dow >= 5 else 0.0,                   # f5: is_weekend
                    1.0 if hour >= 23 or hour < 5 else 0.0,     # f6: is_odd_hour
                    self._encode_payment_type(payment_type),     # f7: payment_type
                    self._get_location_risk(sender_loc, location_map),     # f8: sender_risk
                    self._get_location_risk(receiver_loc, location_map),   # f9: receiver_risk
                    1.0 if payment_curr != received_curr else 0.0,  # f10: currency_mismatch
                    float(row.get('txn_count_24h', 1)),          # f11: txn_count_24h
                    float(row.get('total_24h', amount)),         # f12: total_24h
                    float(row.get('avg_amount_7d', amount)),     # f13: avg_7d
                    amount / (float(row.get('avg_amount_7d', amount)) + 1),  # f14: amount_vs_avg
                    float(row.get('unique_cp_24h', 1)),          # f15: unique_cp
                    float(row.get('max_24h', amount)),           # f16: max_24h
                    float(row.get('velocity', 1.0)),             # f17: velocity
                    1.0 if 'cross-border' in payment_type else 0.0  # f18: cross_border
                ]
                features.append(feature_vector)
            except Exception as e:
                continue

        features = np.array(features)
        self.stdout.write(f"✓ Built {features.shape[0]:,} × {features.shape[1]} feature matrix")

        # ===== TRAIN ISOLATION FOREST (UNSUPERVISED - OPTION B) =====
        self.stdout.write("\n🤖 Training Isolation Forest (Option B - Unsupervised)...")
        self.stdout.write("   Using ALL transactions to learn normal patterns")

        clf = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination=0.001,      # Dataset has 0.1039% suspicious
            random_state=42,
            n_jobs=-1
        )
        clf.fit(features)

        # ===== SAVE MODEL =====
        model_dir = 'aml_monitoring/models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'isolation_forest.pkl')

        joblib.dump(clf, model_path)
        self.stdout.write(f"✓ Model saved to {model_path}")

        # ===== STATISTICS =====
        anomaly_scores = clf.decision_function(features)
        self.stdout.write(f"\n📊 Model Statistics:")
        self.stdout.write(f"   Mean anomaly score: {anomaly_scores.mean():.4f}")
        self.stdout.write(f"   Std: {anomaly_scores.std():.4f}")
        self.stdout.write(f"   Min: {anomaly_scores.min():.4f}")
        self.stdout.write(f"   Max: {anomaly_scores.max():.4f}")

        # ===== VALIDATION AGAINST LABELS =====
        if 'Is_laundering' in df.columns:
            self.stdout.write(f"\n🎯 Validation against Is_laundering labels:")
            pred_anomalies = (anomaly_scores > np.percentile(anomaly_scores, 99)) * 1
            actual_labels = df['Is_laundering'].values[:len(pred_anomalies)]

            # Simple metrics
            tp = np.sum((pred_anomalies == 1) & (actual_labels == 1))
            fp = np.sum((pred_anomalies == 1) & (actual_labels == 0))
            tn = np.sum((pred_anomalies == 0) & (actual_labels == 0))
            fn = np.sum((pred_anomalies == 0) & (actual_labels == 1))

            if (tp + tn + fp + fn) > 0:
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                self.stdout.write(f"   Accuracy: {accuracy:.4f}")
                self.stdout.write(f"   Precision: {precision:.4f}")
                self.stdout.write(f"   Recall: {recall:.4f}")

        self.stdout.write("\n✅ Training complete!")

    def _build_location_map(self):
        """Map dataset location names to risk levels"""
        high_risk_locations = {
            'Iran', 'North Korea', 'Syria', 'Myanmar', 'Yemen',
            'Iraq', 'Libya', 'Somalia', 'Pakistan', 'Afghanistan',
            'South Sudan', 'Central African Republic', 'Democratic Republic of Congo',
            'Mali', 'Nicaragua', 'Mexico', 'Turkey', 'Morocco', 'UAE',
            'Tajikistan', 'Uzbekistan'
        }
        return high_risk_locations

    def _get_location_risk(self, location, location_map):
        """Return 1.0 if high-risk, 0.0 if normal"""
        if not location:
            return 0.0
        return 1.0 if location.strip() in location_map else 0.0

    def _encode_payment_type(self, payment_type):
        """Encode payment type as numerical risk"""
        mapping = {
            'cross-border': 0.9,      # Highest risk
            'ach': 0.6,               # Medium-high
            'credit card': 0.4,       # Medium
            'debit card': 0.3,        # Low-medium
            'cheque': 0.2,
            'cash': 0.5,              # Medium
            'other': 0.3
        }
        for key, value in mapping.items():
            if key in payment_type:
                return value
        return 0.3
