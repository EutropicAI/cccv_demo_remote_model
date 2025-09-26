from typing import Any

from cccv import MODEL_REGISTRY, SRBaseModel


@MODEL_REGISTRY.register(name="demo_srcnn")
class DemoModel(SRBaseModel):
    def load_model(self) -> Any:
        print("Override load_model function here")
        print("We use default load_model function to load the model")
        return super().load_model()


print("New Model Registered: demo_srcnn")
