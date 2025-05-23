from setuptools import setup, find_packages

setup(
    name="PrivateWebSpeech",
    version="0.1.0",
    description="A self-hosted speech interface for web applications",
    author="Charlie Mehlenbeck",
    author_email="charlie_inventor2003@yahoo.com",
    url="https://github.com/Inventor2525/PrivateWebSpeech",
    packages=find_packages(),
    install_requires=[
        "flask",
        "flask-socketio",
        "torch",
        "soundfile",
        "kokoro",
        "faster-whisper",
        "pyannote.audio",
        "numpy",
		"pydub"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)