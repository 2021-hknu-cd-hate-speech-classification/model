import setuptools

with open("requirements.txt", mode="r", encoding="utf-8") as deps_file:
    dependencies = deps_file.read()

setuptools.setup(
    name="hate-speech-classification-model",
    version="0.0.1",
    description="한경대학교 컴퓨터공학과 캡스톤디자인",
    license="MIT",
    install_requires=dependencies
)
