Adding New Upstream
====================


Discuss
---------

Please make sure that you already go through :doc:`./general`.
Again, we might not always want new contributions, hence please make sure we have consensus on the new feature request.
The best and the most transparent way is to
`submit your feature request <https://github.com/s3prl/s3prl/issues/new?assignees=&labels=&template=feature_request.md&title=>`_.


Copy from the template
-----------------------

To add new upstream, you can start with an `example <https://github.com/s3prl/s3prl/tree/main/s3prl/upstream/example>`_
Suppose your new upstream called :code:`my_awesome_upstream`, the simplest way to start will be the following:

1.

.. code-block:: bash

    cd ${S3PRL_ROOT}
    cp -r s3prl/upstream/example/ s3prl/upstream/my_awesome_upstream

2. In :code:`s3prl/upstream/my_awesome_upstream/hubconf.py`, change :code:`customized_upstream` to :code:`my_entry_1`
3. In :code:`s3prl/hub.py`, add :code:`from s3prl.upstream.my_awesome_upstream.hubconf import *`

4.

.. code-block:: bash

    python3 utility/extract_feat.py my_entry_1 sample_hidden_states
    # this script extract hidden states from an upstream entry to the "sample_hidden_states" folder

This will extract the hidden states from this :code:`my_entry_1` entry.
The default content in :code:`s3prl/upstream/example/` always works, so you can simply edit the files
inside the new :code:`s3prl/upstream/my_awesome_upstream` folder to enable your new upstream.


Implement
----------

The folder is in the following structure:

.. code-block:: bash

    my_awesome_upstream
    |
     ---- expert.py
    |
     ---- hubconf.py

In principle, :code:`hubconf.py` serves as the URL registry, where each callable function is an entry specifying
the source of the checkpoint, while the :code:`expert.py` serves as the wrapper of your model definition to fit
with our upstream API.

During your implementation, please try to remove as many package dependencies as possible, since the upstream
functionality is our core feature, and should have minimal dependencies to be maintainable.


Tests
-------

After you implementation, please make sure all your entries can pass `the tests <https://github.com/s3prl/s3prl/blob/8eac602117003e2bb5cdb7a4d0e94cc9975fd4f2/test/test_upstream.py#L194-L250>`_
The :code:`test_upstream_with_extracted` test case requires you to pre-extract the expected hidden states via:

.. code-block:: bash

    python3 utility/extract_feat.py my_awesome_upstream ./sample_hidden_states

That is, the test case expects there will be a :code:`my_awesome_upstream.pt` in the :code:`sample_hidden_states` folder.

All the existing sampled hidden states are hosted at a `Huggingface Dataset Repo <https://huggingface.co/datasets/s3prl/sample_hidden_states/tree/main>`_,
and we expect you to clone (by :code:`git lfs`) this :code:`sample_hidden_states` repo and add the sampled hidden states for your new entries.

To make changes to this hidden states repo, please follow the steps `here <https://huggingface.co/datasets/s3prl/sample_hidden_states/discussions>`_
to create a pull request, so that our core maintainer can sync the hidden states extracted by you.

In conclusion, to add new upstream one needs to make two pull requests:

- To https://github.com/s3prl/s3prl/pulls
- To https://huggingface.co/datasets/s3prl/sample_hidden_states/tree/main


.. note::

    In fact, due to the huge time cost, most of the upstreams in S3PRL will not be tested in Github Action CI (or else it will take several hours
    to download all the checkpoints for every PRs). However, our core maintainers will still clone the repository and run tox locally to make sure
    everything works fine, and there is a `tox environment <https://github.com/s3prl/s3prl/blob/8eac602117003e2bb5cdb7a4d0e94cc9975fd4f2/tox.ini#L11>`_
    testing all the upstreams.


Documentation
--------------

After all the implementation, make sure your efforts are known by the users by adding documentation of your entries at
the :doc:`../tutorial/upstream_collection` tutorial page. Also, you can add your name at the bottom of the tutorial
page if you like.
