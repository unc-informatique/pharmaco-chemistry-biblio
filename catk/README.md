# (Mini) Correspondence Analysis Toolkit (catk)

A custom variation on <https://github.com/MaxHalford/prince> for Correspondence Analysis.

We'll mainly use notation from Greenacre's _Correspondence Analysis in Pratice_
<https://www.routledge.com/Correspondence-Analysis-in-Practice/Greenacre/p/book/9780367782511>

With extra help from Abdi and Bera's _Correspondence Analysis_ <https://cedric.cnam.fr/fichiers/art_3066.pdf>

## Install / dev mode

```bash
# install dev mode
pip install -e ./
# some .egg link file is added to site-packages
pip install -e ./ --upgrade

# to remove
pip uninstall catk
```

To activate autoreload in IPython <https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html>

```IPython
%load_ext autoreload
%autoreload 1
%aimport catk

# s'il traine la pate:
%aimport catk.data

# ou selon

%load_ext autoreload
%autoreload 1
%aimport catk.ca
%run demos.py


```

## Notes

Static data in Python package

- <https://stackoverflow.com/questions/11848030/how-include-static-files-to-setuptools-python-package/11848281#11848281>
- <https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package>
