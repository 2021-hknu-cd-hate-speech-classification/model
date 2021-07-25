import setuptools

with open("requirements.txt", mode="r", encoding="utf-8") as deps_file:
    dependencies = deps_file.read().splitlines()

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setuptools.setup(
    name="hate-speech-classification-model",
    version="0.0.1",
    url="https://github.com/2021-hknu-cd-hate-speech-classification/model",
    description="한경대학교 컴퓨터공학과 캡스톤디자인",
    long_description=readme,
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=dependencies
)
