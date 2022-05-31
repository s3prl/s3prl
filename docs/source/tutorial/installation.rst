Install S3PRL
=============

Regular installation
--------------------

.. warning::

    This is not yet ready. Please use the approach below.

.. code-block:: bash

    pip install s3prl

Editable installation
---------------------

Installing a package in the editable mode means when you use a imported class/function,
the code supporting the class/function is right in your cloned repository.
So, when you modify the code inside it, the newly imported class/function will reflect
your modification.

.. code-block:: bash

    git clone https://github.com/s3prl/s3prl.git
    cd s3prl
    pip install -e .

Various installation sets
-------------------------

The installation approaches above only cover the most commonly used functionality of S3PRL:
**pre-trained model collection**. So we can keep the dependency simple.
If you wish to run the tasks with our **problem** modules, you will need to install more
dependencies.

.. code-block:: bash

    pip install -e ".[problem]"

If you wish to develop new functions, you will need extra dependencies for unit-test, pre-commit
and documentation. Please use

.. code-block:: bash

    pip install -e ".[all]"
