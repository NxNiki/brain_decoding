from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    class Config:
        extra = "allow"  # Allow arbitrary attributes

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.__dict__["_list_fields"]: Set[str] = set()
        self.__dict__["_alias"]: Dict[str, str] = {}

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def __getattr__(self, name):
        """Handles alias access and custom parameters."""
        if name in self._alias:
            return getattr(self, self._alias[name])

    def __setattr__(self, name, value):
        """Handles alias assignment, field setting, or adding to _param."""
        if name in self._alias:
            name = self._alias[name]
        if name in self._list_fields and not isinstance(value, list):
            value = [value]
        super().__setattr__(name, value)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def set_alias(self, name: str, alias: str) -> None:
        self.__dict__["_alias"][alias] = name

    def ensure_list(self, name: str):
        value = getattr(self, name, None)
        if value is not None and not isinstance(value, list):
            setattr(self, name, [value])
        # Mark the field to always be treated as a list
        self._list_fields.add(name)


class ExperimentConfig(BaseConfig):
    """
    configurations regarding the experiment
    """

    name: Optional[str] = None
    patient: Optional[Union[List[int], int]] = None


class ModelConfig(BaseConfig):
    name: Optional[str] = None
    learning_rate: Optional[float] = Field(1e-4, alias="lr")
    learning_rate_drop: Optional[int] = Field(50, alias="lr_drop")
    batch_size: Optional[int] = 128
    epochs: Optional[int] = 100
    hidden_size: Optional[int] = 192
    num_hidden_layers: Optional[int] = 4
    num_attention_heads: Optional[int] = 6
    patch_size: Optional[Tuple[int, int]] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._alias: Dict[str, str] = {
            "lr": "learning_rate",
            "lr_drop": "learning_rate_drop",
        }


class DataConfig(BaseConfig):
    data_type: Optional[str] = None
    sd: Optional[float] = None
    root_path: Optional[Union[str, Path]] = None
    data_path: Optional[Union[str, Path]] = None


class PipelineConfig(BaseModel):
    experiment: Optional[ExperimentConfig] = ExperimentConfig()
    model: Optional[ModelConfig] = ModelConfig()
    data: Optional[DataConfig] = DataConfig()

    # class Config:
    #     arbitrary_types_allowed = True

    @classmethod
    def read_config(cls, config_file: Union[str, Path]) -> "PipelineConfig":
        """Reads a YAML configuration file and returns an instance of PipelineConfig."""
        with open(config_file, "r") as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

    def export_config(self, output_file: Union[str, Path] = "config.yaml") -> None:
        """Exports current properties to a YAML configuration file."""
        if isinstance(output_file, str):
            output_file = Path(output_file)

        if not output_file.suffix:
            output_file = output_file / "config.yaml"

        # Create new path with the suffix added before the extension
        output_file = output_file.with_name(f"{output_file.stem}{self._file_tag}{output_file.suffix}")

        dir_path = output_file.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as file:
            yaml.safe_dump(self.model_dump(), file)

    @property
    def _file_tag(self) -> str:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d-%H:%M:%S")
        return f"_{self.experiment.name}-{self.model.name}-{self.data.data_type}_{formatted_time}"


if __name__ == "__main__":
    pipeline_config = PipelineConfig()
    pipeline_config.model.name = "vit"
    pipeline_config.model.learning_rate = 0.001
    pipeline_config.experiment.name = "movie-decoding"

    # Access and print properties
    print(f"Experiment Name: {pipeline_config.experiment.name}")
    print(f"Patient ID: {pipeline_config.experiment.patient}")
    print(f"Model Name: {pipeline_config.model.name}")
    print(f"Learning Rate: {pipeline_config.model.learning_rate}")
    print(f"Batch Size: {pipeline_config.model.batch_size}")

    # Access using aliases
    print(f"Learning Rate (alias 'lr'): {pipeline_config.model['lr']}")
    print(f"Learning Rate (alias 'lr'): {pipeline_config.model.lr}")

    # set alias
    pipeline_config.data["original_name"] = "original_name"
    pipeline_config.data.set_alias("original_name", "alias_name")
    print(f"original_name (from alias_name): {pipeline_config.data.alias_name}")

    # Set new custom parameters
    pipeline_config.model["new_param"] = "custom_value"
    print(f"Custom Parameter 'new_param': {pipeline_config.model['new_param']}")
    pipeline_config.model.new_param2 = "custom_value"
    print(f"Custom Parameter 'new_param2': {pipeline_config.model.new_param2}")

    # Try to access a non-existent field (will raise AttributeError)
    print(f"non_exist_field: {pipeline_config.model.some_non_existent_field}")

    # test ensure list:
    pipeline_config.model["a_list"] = 1
    pipeline_config.model.ensure_list("a_list")
    print(f"ensure_list: {pipeline_config.model.a_list}")
    pipeline_config.model["a_list"] = 1
    print(f"ensure_list: {pipeline_config.model.a_list}")

    # Export config:
    pipeline_config.export_config()
