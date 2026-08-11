from setuptools import setup

setup(
    name="exerkine_map",
    version="1.0.0",
    description="EXERKINEMAP: Exercise-responsive molecular signaling mapping.",
    license="MIT",
    package_dir={"exerkine_map": "src"},
    packages=["exerkine_map"],
    include_package_data=True,
    python_requires=">=3.10",
)
