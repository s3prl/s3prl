Internal S3PRL Development
==========================

Write code
----------

1.  Make sure you have access to `s3prl/s3prl-private <https://github.com/s3prl/s3prl-private/>`_

2.  Clone the repository to **S3PRL_ROOT** and install the package

    .. code-block:: bash

        git clone s3prl/s3prl-private "S3PRL_ROOT"
        cd "S3PRL_ROOT"
        pip install -e ".[dev]"  # This installs the dependencies for the full functionality of S3PRL

3.  Write code into the packages listed in **S3PRL_ROOT/valid_paths.txt**

Unit tests
----------

4.  Add unit tests to **S3PRL_ROOT/test/** to test your own new modules

5.  Verify you pass all the tests

    .. code-block:: bash

        cd "S3PRL_ROOT"
        pytest

Documentation
-------------

6.  Make sure you write the documentation on the modules' docstring

7.  Build the documentation and make sure it looks correct

    .. code-block:: bash

        cd "S3PRL_ROOT"/docs

        # build
        ./rebuild_docs.sh

        # launch http server
        python3 -m http.server -d build/html/

        # You can then use a browser to access the doc webpage on: YOUR_IP_OR_LOCALHOST:8000

8.  OK now your new changes are ready to be commit

Coding-style check
------------------

9.  Stage your changes

    .. code-block:: bash

        git add "YOUR_CHANGED_OR_ADDED_FILES_ONLY"

    .. warning::

        Please do not use **git add .** to add all the files under your repository.
        If there are files not ignored by git (specified in **.gitignore**), like
        temporary experiment result files, they will all be added into git version
        control, which will mess out our repository.

10. Run **pre-commit** to apply the standardized coding-style on **YOUR_CHANGED_OR_ADDED_FILES_ONLY**

    .. code-block:: bash

        pre-commit run

    If the results show there are files modified by **pre-commit**, you need to re-stage
    these files following step 9.

Commit / Push
-------------

12. Commit and push the changes

    .. code-block:: bash

        git commit -m "YOUR_COMMIT_MESSAGE"
        git push origin "YOUR_BRANCH"

Send a pull request
-------------------

Only do this when you are ready to merge your branch. Since once you send a pull request,
every newly pushed commit will cause GitHub to run CI, but we have a limited number of
runnable CI per month, regularized by GitHub. Hence, you should do this only after the
branch is ready.

13. Verify you can pass the CI locally

    .. code-block:: bash

        ./ci/format.sh --check
        # If this fails, simply remove --check to do the actual formatting

        pytest

14. Send a pull request on GitHub
