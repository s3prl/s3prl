Install S3PRL
=============

Minimal installation
--------------------

This installation only enables the **S3PRL Upstream collection** function to
keep the minimal dependency. To enable all the functions including downstream benchmarking,
you need to follow `Full installation`_.

.. code-block:: bash

    pip install s3prl


Editable installation
---------------------

Installing a package in the editable mode means when you use a imported class/function,
the source code of the class/function is right in your cloned repository.
So, when you modify the code inside it, the newly imported class/function will reflect
your modification.

.. code-block:: bash

    git clone https://github.com/s3prl/s3prl.git
    cd s3prl
    pip install -e .


Full installation
------------------

Install all the dependencies to enable all the S3PRL functions. However, there are **LOTS**
of dependencies.

.. code-block:: bash

    pip install s3prl[all]

    # editable
    pip install ".[all]"


Development installation
-------------------------

Install dependencies of full installation and extra packages for development,
including **pytest** for unit-testing and **sphinx** for documentation.

Usually, you will use this installation variant only in editable mode

.. code-block:: bash

    pip install ".[dev]"
