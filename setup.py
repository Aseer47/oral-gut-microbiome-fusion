from setuptools import setup

setup(
    name="microbiome-site-attribution",
    version="0.1.0",
    description="SHAP-based site-attribution analysis for paired multi-site microbiome data",
    long_description=open("README_tool.md").read() if __import__("os").path.exists("README_tool.md") else "",
    long_description_content_type="text/markdown",
    author="Chowdhury Aseer Ruthbah, Talim Hossain Sadi, Nur E Shiratun Jahan, Abu Nayem Md. Tanzim Adib",
    url="https://github.com/DaemonTargaryen47/oral-gut-microbiome-fusion",
    py_modules=["microbiome_site_attribution"],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "scikit-learn>=1.0",
        "shap>=0.41",
        "matplotlib>=3.4",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
)
