# Computer vision course project

Course is outlined [here](http://en.didattica.unipd.it/off/2017/LM/IN/IN0521/000ZZ/INP6075837/N2CN1).

# How to run project
Install needed Python packages, for example creating a virtual environment (`virtualenv` command and `python3` are required).
```bash
virtualenv -p python3 venv/
source venv/bin/activate
pip install -r requirements.txt
```

Then, just run detection for images in `dataset/` folder.
```bash
python detector.py
```

Get `detector.py` parameters with
```bash
python detector.py --help
```
