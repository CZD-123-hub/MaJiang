import sys
from pathlib import Path

from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent

if sys.platform == "win32":
    compile_args = ["/O2", "/std:c++20"]
else:
    compile_args = ["-O3", "-std=c++20"]


setup(
    name="gbmj_foresight_cpp",
    version="0.1.0",
    description="C++ foresight feature calculator for Mortal_gbmj_v4",
    ext_modules=[
        Extension(
            "gbmj_foresight_cpp",
            sources=[
                str(ROOT / "gbmj_foresight.cpp"),
                str(ROOT / "calsht.cpp"),
            ],
            language="c++",
            include_dirs=[str(ROOT)],
            extra_compile_args=compile_args,
        )
    ],
)
