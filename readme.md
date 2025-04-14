## House Price prediction

> First run the code present on `*.ipynb` file, so it will generate the `.pkl` file, which is require to run this code.


## Info

- This code is based on the RandomForestRegression model.
- Dataset is of the california housing, available on `sklearn.datasets`

## Run

> First run_all the `.ipynb` file.

**Using docker**
```
docker build -t housing_app .
```

```
docker run -p 5000:5000 housing_app
```

*or*

```
docker compose up
```

**Normal**

```
pip install -r requirements.txt
```

```py
python app.y
```
