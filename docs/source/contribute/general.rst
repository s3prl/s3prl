.. _general-contribution-guideline:

General Guideline
==================

Thank you for considering contributing to S3PRL, we really appreciate it.

However, due to the increasing difficulty of maintenance, please understand that **we might not want your new feature**.

Hence, **before submitting the implemented pull request**,
please **submit your feature request** to the Github issue page so we can discuss about whether we want it and how to achieve it.

.. warning::

    If we did not go through this discussion, the pull request will not be dealt with and will be directly closed.

.. note::

    If any previous contributor feels uncomfortable about our government system (we believe the system never changes but in case
    anyone misunderstood it due to some historical reasons), feel free to submit the pull request to delete your previous contribution,
    which can even appear as a new contribution.


Discuss
-----------

`Submit your feature request <https://github.com/s3prl/s3prl/issues/new?assignees=&labels=&template=feature_request.md&title=>`_
on our Github Issue page to propose features and changes.

Please wait for our response and do not move on to the following steps before we have a consensus on what is going to be changed.

Setup
-----------

Clone the repository to **S3PRL_ROOT** and install the package

.. code-block:: bash

    S3PRL_ROOT="/home/leo/d/s3prl"
    git clone https://github.com/s3prl/s3prl.git ${S3PRL_ROOT}
    cd ${S3PRL_ROOT}

    pip install -e ".[dev]"
    # This installs the dependencies for the full functionality of S3PRL in the editable mode,
    # including the dependencies for development like testing and building doc


Tests
----------

Add unit tests to **${S3PRL_ROOT}/test/** to test your own new modules

Verify you pass all the tests

.. code-block:: bash

    cd ${S3PRL_ROOT}
    pytest


Documentation
-------------

Make sure you write the documentation on the modules' docstring

Build the documentation and make sure it looks correct

.. code-block:: bash

    cd ${S3PRL_ROOT}/docs

    # build
    ./rebuild_docs.sh

    # launch http server
    python3 -m http.server -d build/html/

    # You can then use a browser to access the doc webpage on: YOUR_IP_OR_LOCALHOST:8000

Coding-style check
------------------

Stage your changes

.. code-block:: bash

    git add "YOUR_CHANGED_OR_ADDED_FILES_ONLY"

.. warning::

    Please do not use **git add .** to add all the files under your repository.
    If there are files not ignored by git (specified in **.gitignore**), like
    temporary experiment result files, they will all be added into git version
    control, which will mess out our repository.

.. note::

    In our **.gitignore**, there are lots of ignored files especially for *.yaml*
    and *.sh* files. If the config files or the shell scripts are important, please
    remember to add them forcely, for example :code::`git add -f asr.yaml`

Run **pre-commit** to apply the standardized coding-style on **YOUR_CHANGED_OR_ADDED_FILES_ONLY**

.. code-block:: bash

    pre-commit run

If the results show there are files modified by **pre-commit**, you need to re-stage
these files following the previous step.


Commit / Push
-------------

Commit and push the changes

.. code-block:: bash

    git commit -m "YOUR_COMMIT_MESSAGE"
    git push origin "YOUR_BRANCH"


(Optional) Test against multiple environments
---------------------------------------------

We leverage **tox** to simulate multiple envs, see the `tox configuration <https://github.com/s3prl/s3prl/blob/main/tox.ini>`_ for more information.
Tox helps automate the pipeline of creating different virtual envs, installing differnet dependencies of S3PRL, running different testing commands.
Our Github Action CI also relies on tox, hence you can debug the CI error locally with tox.

Before using tox, make sure your cli can launch the following python versions. Usually, this can be achieved via `pyenv <https://github.com/pyenv/pyenv>`_

- python3.7
- python3.8
- python3.9
- python3.10

List all the available environments. An environment means a pre-defined routine of packaging S3PRL, installing S3PRL, installing specific dependencies,
test specific commands. See `tox configuration <https://github.com/s3prl/s3prl/blob/main/tox.ini>`_ for more information.

.. code-block:: bash

    tox -l

Suppose there is an environment named :code:`all_upstream-py38-audio0.12.1`, you can also test against this specific env:

.. code-block:: bash

    tox -e all_upstream-py38-audio0.12.1

Test all environments. This simulate the environments you will meet on the Github Action CI

.. code-block:: bash

    tox


Send a pull request
-------------------

Verify your codes are in the proper format

.. code-block:: bash

    ./ci/format.sh --check
    # If this fails, simply remove --check to do the actual formatting

Make sure you add test cases and your change pass the tests

.. code-block:: bash

    pytest

Send a pull request on GitHub
