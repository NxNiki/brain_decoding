from pydantic import BaseModel


class Config(BaseModel):
    class Config:
        # arbitrary_types_allowed = True
        extra = "allow"  # Allow arbitrary attributes

    _list_fields = set()  # A set to track which fields should be treated as lists

    def ensure_list(self, name: str):
        value = getattr(self, name, None)
        if value is not None and not isinstance(value, list):
            setattr(self, name, [value])
        # Mark the field to always be treated as a list
        self._list_fields.add(name)

    def __setattr__(self, name, value):
        if name in self._list_fields and not isinstance(value, list):
            # Automatically convert to a list if it's in the list fields
            value = [value]
        super().__setattr__(name, value)


class SupConfig(Config):
    pass


# Example usage
config = SupConfig()

# Dynamically adding attributes
config.param1 = "a"

# Ensuring param1 is a list
config.ensure_list("param1")
print(config.param1)  # Output: ['a']

# Assigning new value to param1
config.param1 = "ab"
print(config.param1)  # Output: ['ab'] gets automatically converted to ['ab']

# Adding another parameter and ensuring it's a list
config.ensure_list("param2")
config.param2 = 123
print(config.param2)  # Output: [123]
