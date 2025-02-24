## Setup

### Create environment and install packages


```
conda create -n "talkarena" python=3.12 ipython -y
conda activate talkarena
pip install -e .
```

### Add .env file

Include the .env file in the following format

```
OPENAI_API_KEY=[KEY]
GEMINI_API_KEY=[KEY]
```
