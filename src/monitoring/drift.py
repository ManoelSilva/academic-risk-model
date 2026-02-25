import pandas as pd
import json
import os
from evidently import Report
from evidently.presets import DataDriftPreset
class ColumnMapping:
    def __init__(self):
        self.target = None
        self.prediction = None
        self.id = None
        self.datetime = None
        self.numerical_features = None
        self.categorical_features = None
        self.datetime_features = None
        self.target_names = None
        self.task = None
        self.pos_label = None
        self.text_features = None


class DriftDetector:
    def __init__(self, reference_data_path, current_data_path, output_dir='reports'):
        self.reference_path = reference_data_path
        self.current_path = current_data_path
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_data(self):
        # Load reference data (e.g., training set)
        ref_df = pd.read_csv(self.reference_path, delimiter=';')
        
        # Load current data (e.g., new batch or production logs)
        # Assuming current data follows the same format
        curr_df = pd.read_csv(self.current_path, delimiter=';')
        
        return ref_df, curr_df

    def run_drift_check(self):
        ref_df, curr_df = self.load_data()
        
        # Define column mapping if necessary
        # For now, we assume automatic detection works well for this dataset
        # column_mapping = ColumnMapping()
        # column_mapping.target = 'ALUNO_EVADIU' # If available in both
        
        # Create report
        report = Report(metrics=[
            DataDriftPreset(),
            # TargetDriftPreset() removed as it may not be in presets
        ])
        
        snapshot = report.run(reference_data=ref_df, current_data=curr_df)
        
        # Save Report
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        html_path = os.path.join(self.output_dir, f"drift_report_{timestamp}.html")
        json_path = os.path.join(self.output_dir, f"drift_report_{timestamp}.json")
        
        snapshot.save_html(html_path)
        snapshot.save_json(json_path)
        
        return {
            "html_report": html_path,
            "json_report": json_path,
            "drift_detected": self._parse_drift_json(json_path)
        }

    def _parse_drift_json(self, json_path):
        """
        Parses the JSON report to return a simple boolean or summary.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Simple check: look for dataset drift metric
        # Structure varies by evidently version, this is a heuristic
        try:
            # This is a simplified check. In production, parse specific metrics.
            metrics = data['metrics']
            for metric in metrics:
                if metric['metric'] == 'DatasetDriftMetric':
                    return metric['result']['dataset_drift']
        except:
            return None
        return False

if __name__ == "__main__":
    # Example usage
    detector = DriftDetector(
        reference_data_path="data/raw/PEDE_PASSOS_DATASET_FIAP.csv",
        # For demo, using the same file as current. In real life, this would be different.
        current_data_path="data/raw/PEDE_PASSOS_DATASET_FIAP.csv" 
    )
    result = detector.run_drift_check()
    print(json.dumps(result, indent=2))
