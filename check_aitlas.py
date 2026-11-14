
import aitlas
import pkgutil
import importlib.metadata

print("AiTLAS version:", importlib.metadata.version("aitlas"))
print("Package location:", aitlas.__file__)

print("\nAvailable top-level attributes:")
for name in dir(aitlas):
    if not name.startswith("_"):
        print(name)

print("\nAvailable submodules:")
for module in pkgutil.walk_packages(
    aitlas.__path__,
    prefix="aitlas."
):
    print(module.name)
