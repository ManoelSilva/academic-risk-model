import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.api.main import AcademicRiskApp


@pytest.fixture
def app_instance():
    """Fixture to create an app instance with mocked model loading."""
    with patch('src.api.main.joblib.load') as mock_load:
        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 1])
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        mock_load.return_value = mock_model

        with patch('os.path.exists', return_value=True):
            app = AcademicRiskApp()
            app.app.config['TESTING'] = True
            return app


@pytest.fixture
def client(app_instance):
    """Fixture to provide a test client."""
    return app_instance.app.test_client()


class TestAcademicRiskApp:

    def test_health_check(self, client):
        """Test the /health endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'

    def test_predict_success(self, client):
        """Test the /predict endpoint with valid data."""
        # Generic names as expected by the updated API
        payload = [
            {
                "INDE": 7.5,
                "IAA": 8.0,
                "IEG": 6.5,
                "IPS": 7.0,
                "IDA": 9.0,
                "IPP": 7.5,
                "IPV": 8.5,
                "IAN": 6.0,
                "DEFASAGEM": 0,
                "IDADE_ALUNO": 12,
                "ANOS_PM": 2,
                "PEDRA": "Ametista",
                "PONTO_VIRADA": "Não",
                "SINALIZADOR_INGRESSANTE": "Não"
            },
            {
                "INDE": 6.0,
                "IAA": 7.0,
                "IEG": 5.5,
                "IPS": 6.0,
                "IDA": 8.0,
                "IPP": 6.5,
                "IPV": 7.5,
                "IAN": 5.0,
                "DEFASAGEM": 1,
                "IDADE_ALUNO": 13,
                "ANOS_PM": 3,
                "PEDRA": "Quartzo",
                "PONTO_VIRADA": "Sim",
                "SINALIZADOR_INGRESSANTE": "Não"
            }
        ]

        response = client.post('/predict', json=payload)
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['count'] == 2
        assert len(data['predictions']) == 2
        assert 'risk_label' in data['predictions'][0]

    def test_predict_missing_columns(self, client):
        """Test the /predict endpoint with missing columns."""
        payload = [{"INDE": 7.5}]  # Missing everything else
        response = client.post('/predict', json=payload)
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert 'Missing required columns' in data['message']

    def test_predict_no_payload(self, client):
        """Test the /predict endpoint with empty payload."""
        response = client.post('/predict', json={})
        assert response.status_code == 400  # Or 500 depending on implementation, code says 400 for 'not data' if empty list or dict?
        # Code: if not data: return 400. json={} is dict, bool({}) is False. So 400.

    def test_predict_handles_model_not_loaded(self):
        """Test /predict when model is not loaded."""
        with patch('os.path.exists', return_value=False):  # Force model not found
            app = AcademicRiskApp()
            client = app.app.test_client()

            # Payload is irrelevant here, but provide valid one
            response = client.post('/predict', json=[])
            assert response.status_code == 503
            data = json.loads(response.data)
            assert 'No model loaded' in data['message']

    def test_train_endpoint_trigger(self):
        """Test that /train triggers training (mocked)."""
        with patch('src.api.main.ModelTrainer') as MockTrainer:
            mock_instance = MockTrainer.return_value
            mock_instance.train_and_evaluate.return_value = ("BestModel", 0.95)

            app = AcademicRiskApp()
            client = app.app.test_client()

            response = client.post('/train', json={})
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['status'] == 'success'
            assert data['best_model'] == 'BestModel'
