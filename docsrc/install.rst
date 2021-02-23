.. _install:

Installation
============

.. admonition:: Installation Options

    There are three alternatives to install the library:

    1. Install via pip
    2. Install from wheel package
    3. Build from source code

.. _requirements:

Requirements
------------

The library requires Python **3.6+** and depends on standard packages such as ``pandas, numpy``
The ``requirements.txt`` lists the necessary packages. 

Install via pip
--------------------------

After installing the requirements, you can install the library using the following command:

.. code-block:: bash

	pip install jurity


Install from wheel package
--------------------------

After installing the requirements, you can install the library from the provided wheel package using the following commands:

.. code-block:: bash

	pip install dist/jurity-X.X.X-py3-none-any.whl

Note: Don't forget to replace ``X.X.X`` with the current version number.

Install from source code
------------------------

Alternatively, you can build a wheel package on your platform from scratch using the source code:

.. code-block:: bash

	pip install setuptools wheel # if wheel is not installed
	python setup.py bdist_wheel
	pip install dist/jurity-X.X.X-py3-none-any.whl


Test Your Setup
---------------

To confirm that cloning the repo was successful, run the first example in the 
[Quick Start](#quick-start). 
To confirm that the whole installation was successful, run the tests and all should pass. 

.. code-block:: bash

	python -m unittest discover -v tests


Upgrading the Library
---------------------

To upgrade to the latest version of the library, run ``pip install --upgrade jurity``.