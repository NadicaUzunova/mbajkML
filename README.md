## Installation guide

To use this code environment you must first have Poetry installed. Please refer to the 
installation guide [here](https://python-poetry.org/docs/).

When you have Poetry installed, you can begin using this package. To begin we first need to 
install all necessary dependencies. To do that, we can use the command:

```poetry install```

However, there are a few limitations with the Poetry package manager, as it does not 
fully function with keras and tensorflow, so those must be installed separately using the
following commands:

```shell
poetry shell
pip install tensorflow==2.15.0
pip install keras==2.15.0
```

After that all our packages and libraries are installed and ready to go! :rocket:

To run scripts you can use the following command:

```shell
poetry run .\src\data\fetch_data.py
```


