from src.pipelines.training_pipeline import TrainingPipeline


def test_training_pipeline():
    pipeline = TrainingPipeline(
        config_path="config/config.yaml",
        schema_path="config/schema.yaml"
    )

    pipeline.run_pipeline()

    assert True